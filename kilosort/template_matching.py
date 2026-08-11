import logging

import numpy as np
import torch 
from torch.nn.functional import conv1d, max_pool2d, max_pool1d
from tqdm import tqdm

from kilosort import CCG
from kilosort.utils import (
    get_spike_buffer_capacity,
    group_indices_by_label,
    log_performance,
)

logger = logging.getLogger(__name__)


def prepare_extract(xc, yc, U, nC, position_limit, device=torch.device('cuda')):
    """Identify desired channels based on distances and template norms.
    
    Parameters
    ----------
    xc : np.ndarray
        X-coordinates of contact positions on probe.
    yc : np.ndarray
        Y-coordinates of contact positions on probe.
    U : torch.Tensor
        TODO
    nC : int
        Number of nearest channels to use.
    position_limit : float
        Max distance (in microns) between channels that are used to estimate
        spike positions in `postprocessing.compute_spike_positions`.

    Returns
    -------
    iCC : np.ndarray
        For each channel, indices of nC nearest channels.
    iCC_mask : np.ndarray
        For each channel, a 1 if the channel is within 100um and a 0 otherwise.
        Used to control spike position estimate in post-processing.
    iU : torch.Tensor
        For each template, index of channel with greatest norm.
    Ucc : torch.Tensor
        For each template, spatial PC features corresponding to iCC.
    
    """
    ds = (xc - xc[:, np.newaxis])**2 +  (yc - yc[:, np.newaxis])**2 
    iCC = np.argsort(ds, 0)[:nC]
    iCC_mask = np.take_along_axis(ds, iCC, axis=0)
    iCC = torch.from_numpy(iCC).to(device)
    iCC_mask = iCC_mask < position_limit**2
    iCC_mask = torch.from_numpy(iCC_mask).to(device)
    iU = torch.argmax((U**2).sum(1), -1)
    Ucc = U[torch.arange(U.shape[0]),:,iCC[:,iU]]

    return iCC, iCC_mask, iU, Ucc


def extract(ops, bfile, U, device=torch.device('cuda'), progress_bar=None):
    nC = ops['settings']['nearest_chans']
    position_limit = ops['settings']['position_limit']
    iCC, iCC_mask, iU, Ucc = prepare_extract(
        ops['xc'], ops['yc'], U, nC, position_limit, device=device
        )
    ops['iCC'] = iCC
    ops['iCC_mask'] = iCC_mask
    ops['iU'] = iU
    nt = ops['nt']
    
    tiwave = torch.arange(-(nt//2), nt//2+1, device=device) 
    ctc = prepare_matching(ops, U)
    spike_capacity = get_spike_buffer_capacity(bfile.n_batches)
    st = np.zeros((spike_capacity, 3), 'float64')
    tF = torch.zeros((spike_capacity, nC, ops['settings']['n_pcs']))
    k = 0
    prog = tqdm(
        np.arange(bfile.n_batches, dtype=np.int64),
        miniters=200 if progress_bar else None, 
        mininterval=60 if progress_bar else None
        )
    
    # Prefetch: a worker thread reads batch i+1 from disk while batch i runs
    # on the GPU. Yields exactly what padded_batch_to_torch(i, ops) returns.
    batches = bfile.iter_batches(ops)
    try:
        for ibatch in prog:
            if ibatch % 100 == 0:
                log_performance(logger, 'debug', f'Batch {ibatch}')

            X = next(batches)
            stt, amps, th_amps, Xres = run_matching(ops, X, U, ctc, device=device)
            xfeat = Xres[iCC[:, iU[stt[:,1:2]]],stt[:,:1] + tiwave] @ ops['wPCA'].T
            xfeat += amps * Ucc[:,stt[:,1]]

            if ibatch == 0:
                # Can sometimes get negative spike times for first batch since
                # we're aligning to nt0min, not nt//2, but these should be discarded.
                neg_spikes = (stt[:,0] - nt - nt//2 + ops['nt0min']) < 0
                stt = stt[~neg_spikes,:]
                xfeat = xfeat[:,~neg_spikes,:]
                amps = amps[~neg_spikes,:]
                th_amps = th_amps[~neg_spikes,:]

            nsp = len(stt) 
            if k+nsp>st.shape[0]:                     
                st = np.concatenate((st, np.zeros_like(st)), 0)
                tF  = torch.cat((tF,  torch.zeros_like(tF)), 0)

            t_shift = ibatch * bfile.batch_downsampling * (ops['batch_size'])
            # Build all three columns on-device and move them in one transfer.
            col0 = (stt[:,0].double() - nt) + t_shift - nt//2 + ops['nt0min']
            col1 = stt[:,1].double()
            col2 = th_amps.squeeze(-1).double()
            st[k:k+nsp] = torch.stack((col0, col1, col2), dim=1).cpu().numpy()

            tF[k:k+nsp]  = xfeat.transpose(0,1).cpu()

            k+= nsp
            
            if progress_bar is not None:
                progress_bar.emit(int((ibatch+1) / bfile.n_batches * 100))
    except:
        logger.exception(f'Error in template_matching.extract on batch {ibatch}')
        logger.debug(f'X shape: {X.shape}')
        logger.debug(f'stt shape: {stt.shape}')
        raise

    log_performance(logger, 'debug', f'Batch {ibatch}')

    isort = np.argsort(st[:k,0])
    st = st[isort]
    tF = tF[isort]

    return st, tF, ops


def align_U(U, ops, device=torch.device('cuda')):
    Uex = torch.einsum('xyz, zt -> xty', U.to(device), ops['wPCA'])
    X = Uex.reshape(-1, ops['Nchan']).T
    X = conv1d(X.unsqueeze(1), ops['wTEMP'].unsqueeze(1), padding=ops['nt']//2)
    Xmax = X.abs().max(0)[0].max(0)[0].reshape(-1, ops['nt'])
    imax = torch.argmax(Xmax, 1)

    Unew = Uex.clone() 
    for j in range(ops['nt']):
        ix = imax==j
        Unew[ix] = torch.roll(Unew[ix], ops['nt']//2 - j, -2)
    Unew = torch.einsum('xty, zt -> xzy', Unew, ops['wPCA'])#.transpose(1,2).cpu()
    return Unew, imax


def postprocess_templates(Wall, ops, clu, st, tF, device=torch.device('cuda')):
    Wall2, _ = align_U(Wall, ops, device=device)
    #Wall3, _= remove_duplicates(ops, Wall2)
    Wall3, _, _, _, _ = merging_function(
        ops, Wall2.transpose(1,2), clu, st, tF,
        0.9, 'mu', check_dt=False, device=device
        )
    Wall3 = Wall3.transpose(1,2).to(device)
    return Wall3


def prepare_matching(ops, U):
    nt = ops['nt']
    W = ops['wPCA'].contiguous()
    WtW = conv1d(W.reshape(-1, 1,nt), W.reshape(-1, 1 ,nt), padding = nt)
    WtW = torch.flip(WtW, [2,])

    UtU = torch.einsum('ikl, jml -> ijkm',  U, U)
    ctc = torch.einsum('ijkm, kml -> ijl', UtU, WtW)

    # Pre-scale by s_i = nm_i**-0.5 along the row axis so run_matching can work
    # on a scaled projection B and skip the per-peel division by nm (ctc is
    # indexed [:, iY, :] and subtracted from the scaled B).
    nm = (U**2).sum(-1).sum(-1)
    s = nm.clamp_min(1e-30).rsqrt()
    ctc = ctc * s.view(-1, 1, 1)

    return ctc


def run_matching(ops, X, U, ctc, device=torch.device('cuda')):
    # `ctc` must come from prepare_matching, which pre-scales it by
    # s_i = nm_i**-0.5. The 1/sqrt(nm) normalisation is folded into the
    # templates (U -> U*s) so the projection B comes out already scaled:
    # relu(B_i)**2/nm_i == relu(B_i*s_i)**2, and because relu and squaring are
    # monotonic on the reduced axis the max over units commutes with both.
    # The peel loop therefore reduces B directly and applies relu/square on
    # the (NT,) result instead of materialising a (n_units, NT) tensor.
    Th = ops['Th_learned']
    nt = ops['nt']
    max_peels = ops['max_peels']
    W = ops['wPCA'].contiguous()

    nm = (U**2).sum(-1).sum(-1)
    s = nm.clamp_min(1e-30).rsqrt()

    Us = U * s.view(-1, 1, 1)
    B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt//2)
    B = torch.einsum('ijk, kjl -> il', Us, B)

    trange = torch.arange(-nt, nt+1, device=device)
    tiwave = torch.arange(-(nt//2), nt//2+1, device=device)

    # Growable peel buffer: dense MEA batches can exceed the historical 1e5
    # cap and crash mid-assign. Double capacity on overflow (same growth rule
    # as outer detect/extract spike buffers). Low-rate batches stay identical.
    peel_cap = 100000
    st = torch.zeros((peel_cap, 2), dtype=torch.int64, device=device)
    amps = torch.zeros((peel_cap, 1), dtype=torch.float, device=device)
    th_amps = torch.zeros((peel_cap, 1), dtype=torch.float, device=device)
    k = 0

    Xres = X.clone()

    for t in range(max_peels):
        # Reduce first, then apply relu/square on the (NT,) result.
        Cfmax, imax = torch.max(B, 0)
        Cfmax = torch.relu(Cfmax)
        Cfmax = Cfmax * Cfmax
        Cfmax[:nt] = 0
        Cfmax[-nt:] = 0

        Cmax = max_pool1d(Cfmax.view(1, 1, -1), (2*nt+1), stride=1, padding=(nt))
        cmax = Cmax[0, 0]

        cnd1 = cmax > Th**2
        cnd2 = torch.abs(cmax - Cfmax) < 1e-9
        xs = torch.nonzero(cnd1 & cnd2)

        if len(xs)==0:
            break

        iX = xs[:,:1]
        iY = imax[iX]

        nsp = len(iX)
        need = k + nsp
        if need > st.shape[0]:
            new_cap = max(need, st.shape[0] * 2)
            st = torch.cat((st, torch.zeros((new_cap - st.shape[0], 2),
                                            dtype=st.dtype, device=device)), 0)
            amps = torch.cat((amps, torch.zeros((new_cap - amps.shape[0], 1),
                                                dtype=amps.dtype, device=device)), 0)
            th_amps = torch.cat(
                (th_amps, torch.zeros((new_cap - th_amps.shape[0], 1),
                                      dtype=th_amps.dtype, device=device)), 0
            )

        st[k:k+nsp, 0] = iX[:,0]
        st[k:k+nsp, 1] = iY[:,0]
        # B is scaled by s, so B_stock[iY,iX]/nm[iY] == B[iY,iX]*s[iY].
        amps[k:k+nsp] = B[iY,iX] * s[iY]
        amp = amps[k:k+nsp]
        th_amps[k:k+nsp] = cmax[iX[:,0], None]**.5

        k+= nsp

        n = 2
        for j in range(n):
            Xres[:, iX[j::n] + tiwave]  -= amp[j::n] * torch.einsum('ijk, jl -> kil', U[iY[j::n,0]], W)
            B[   :, iX[j::n] + trange]  -= amp[j::n] * ctc[:,iY[j::n,0],:]

    st = st[:k]
    amps = amps[:k]
    th_amps = th_amps[:k]

    return  st, amps, th_amps, Xres


def merging_function(ops, Wall, clu, st, tF, r_thresh=0.5, mode='ccg', check_dt=True,
                     device=torch.device('cuda')):
    clu2 = clu.copy()
    clu_unq, ns = np.unique(clu2, return_counts = True)

    Ww = Wall.to(device)
    NN = len(Ww)

    isort = np.argsort(ns)[::-1]

    is_merged = np.zeros(NN, 'bool')
    is_good = np.zeros(NN,)

    acg_threshold = ops['settings']['acg_threshold']
    ccg_threshold = ops['settings']['ccg_threshold']
    if mode == 'ccg':
        is_ref, est_contam_rate = CCG.refract(clu, st[:,0]/ops['fs'],
                                              acg_threshold=acg_threshold,
                                              ccg_threshold=ccg_threshold)

    nt = ops['nt']
    W = ops['wPCA'].contiguous()
    WtW = conv1d(W.reshape(-1, 1,nt), W.reshape(-1, 1 ,nt), padding = nt) 
    WtW = torch.flip(WtW, [2,])

    # Spike index lists per cluster: replace repeated full-vector
    # `clu2 == kk` masks. On merge, reassign labels and concat+sort indices so
    # gather order still matches a boolean mask over `st` (ascending index).
    spike_idx = group_indices_by_label(clu2)

    # Wall renorm only changes when a merge rewrites Ww; skip it while we only
    # advance the outer pointer over non-merge candidates.
    renorm = True
    mu = None
    Wnorm = None

    t = 0
    nmerge = 0
    while t<NN:
        #if t%100==0:
            #print(t, nmerge)

        kk = clu_unq[isort[t]]

        if (mode == 'ccg') and is_ref[kk]==0:
            t += 1
            continue

        if is_merged[kk]:            
            t += 1
            continue

        if renorm:
            mu = (Ww**2).sum((1,2), keepdims=True)**.5
            Wnorm = Ww / (1e-6 + mu)
            renorm = False

        UtU = torch.einsum('lk, jlm -> jkm',  Wnorm[kk], Wnorm)
        ctc = torch.einsum('jkm, kml -> jl', UtU, WtW)

        cmax, imax = ctc.max(1)
        cmax[kk] = 0

        jsort = np.argsort(cmax.cpu().numpy())[::-1]

        if mode == 'ccg':
            st0 = st[spike_idx[kk], 0] / ops['fs']
        
        is_ccg  = 0
        for j in range(NN):
            jj = jsort[j]
            if cmax[jj] < r_thresh:
                break
            # compare with CCG
            if mode == 'ccg':
                st1 = st[spike_idx[jj], 0] / ops['fs']
                _, is_ccg, _ = CCG.check_CCG(st0, st1, acg_threshold=acg_threshold,
                                             ccg_threshold=ccg_threshold)        
            else:
                dmu = 2 * (mu[kk] - mu[jj]) / (mu[kk] + mu[jj])
                is_ccg = dmu.abs() < 0.2

            if is_ccg:
                is_merged[jj] = 1
                dt = (imax[kk] -imax[jj]).item()
                idx = spike_idx[jj]
                if dt != 0 and check_dt:
                    # Update tF and Wall with shifted features
                    tF, Wall = roll_features(W, tF, Ww, idx, jj, dt)
                    # Shift spike times
                    st[idx,0] -= dt
                
                Ww[kk] = ns[kk]/(ns[kk]+ns[jj]) * Ww[kk] + ns[jj]/(ns[kk]+ns[jj]) * Ww[jj]            
                Ww[jj] = 0
                ns[kk] += ns[jj]
                ns[jj] = 0
                clu2[idx] = kk
                # Preserve ascending-index gather order of `clu2 == kk`.
                spike_idx[kk] = np.sort(np.concatenate((spike_idx[kk], idx)))
                del spike_idx[jj]
                renorm = True

                break

        if is_ccg==0:            
            t +=1    
        else:                
            nmerge+=1
    
    imap = np.cumsum((~is_merged).astype('int32')) - 1
    if imap.size > 0:
        # Otherwise, everything has been merged into a single cluster
        clu2 = imap[clu2]

    Ww = Ww[~is_merged]

    if mode == 'ccg':
        is_ref = is_ref[~is_merged]
    else:
        is_ref = None

    sorted_idx = np.argsort(st[:,0])
    st = np.take_along_axis(st, sorted_idx[..., np.newaxis], axis=0)
    clu2 = clu2[sorted_idx]
    tensor_idx = torch.from_numpy(sorted_idx)
    tF = tF[tensor_idx]

    return Ww.cpu(), clu2, is_ref, st, tF


def roll_features(wPCA, tF, Wall, spike_idx, clust_idx, dt):
    W = wPCA.cpu()
    # Project from PC space back to sample time, shift by dt
    feats = torch.roll(tF[spike_idx] @ W, shifts=dt, dims=2)
    temps = torch.roll(Wall[clust_idx:clust_idx+1] @ wPCA, shifts=dt, dims=2)

    # For values that "rolled over the edge," set equal to next closest bin.
    # Lag from WtW can be |dt| >= T (feature length nt); clamp so edge fill
    # never indexes out of range (merge path used to IndexError).
    T = feats.shape[-1]
    if dt > 0 and T > 0:
        d = min(int(dt), T - 1)
        if d > 0:
            feats[:, :, :d] = feats[:, :, d].unsqueeze(-1)
            temps[:, :, :d] = temps[:, :, d].unsqueeze(-1)
    elif dt < 0 and T > 0:
        d = max(int(dt), 1 - T)
        if d < 0:
            feats[:, :, d:] = feats[:, :, d - 1].unsqueeze(-1)
            temps[:, :, d:] = temps[:, :, d - 1].unsqueeze(-1)

    # Project back to PC space and update tF
    tF[spike_idx] = feats @ W.T
    Wall[clust_idx] = temps @ wPCA.T

    return tF, Wall
