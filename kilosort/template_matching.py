import logging
import os

import numpy as np
import torch
from torch.nn.functional import conv1d, max_pool2d, max_pool1d
from tqdm import tqdm

from kilosort import CCG, fused_peel, fused_peel_cond, fused_peel_store
from kilosort.utils import (
    get_spike_buffer_capacity,
    group_indices_by_label,
    log_performance,
)

logger = logging.getLogger(__name__)


def _residual_dump_request():
    """Parse KS4_DUMP_RESIDUAL, which is 'directory:batch_index'.

    Diagnostic only, and OFF unless the variable is set. The peel subtracts a
    template that covers a median of 13 of 519 channels and leaves 17.6-23.0%
    of each cell's signal energy on channels it never touches
    (src/utilities/AXONAL_AND_PEEL_TRUNCATION.md §1). Whether that arithmetic
    gap reaches the OUTPUT is a separate question, and answering it needs the
    residual, which kilosort does not otherwise save.

    Byte identity: when the variable is unset this returns None, no batch is
    ever selected, and the pre-peel clone below is never taken. The only cost
    on a normal run is one integer comparison per batch.
    """
    spec = os.environ.get('KS4_DUMP_RESIDUAL')
    if not spec:
        return None
    path, _, batch = spec.rpartition(':')
    if not path:
        raise ValueError(
            "KS4_DUMP_RESIDUAL must be 'directory:batch_index', got "
            f"{spec!r}")
    os.makedirs(path, exist_ok=True)
    return path, int(batch)


def _to_numpy(v):
    """Anything the dump might hold -> a host numpy array.

    Every array in ops or returned by run_matching may be a CUDA tensor, a CPU
    tensor, or already numpy, and np.asarray raises on the first of those.
    """
    if hasattr(v, 'detach'):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def _write_residual_dump(path, ibatch, X_pre, Xres, stt, U, ops):
    """Save one batch's pre-peel data, post-peel residual, and its spikes.

    Everything is in the WHITENED, FILTERED space the peel works in, which is
    the space that matters for re-detection: a residual only becomes a spurious
    spike if it survives there. `Wrot` is saved alongside so the analysis can
    map back to raw channels, because whitening MIXES channels and a whitened
    channel is not a location.

    Written as float16 for the two big arrays. They are diagnostics, and at
    519 x 60000 float32 a pair costs 250 MB per batch. float16 keeps ~3 decimal
    digits, which is far more than a shape comparison needs, and the analysis
    that reads these files does not feed any sorter output.
    """
    d = os.path.join(path, f'batch{ibatch:05d}')
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, 'pre.npy'),
            X_pre.detach().to(torch.float16).cpu().numpy())
    np.save(os.path.join(d, 'residual.npy'),
            Xres.detach().to(torch.float16).cpu().numpy())
    # stt columns are (time_in_batch, template_index, ...) as run_matching
    # builds them; saved raw so the analysis does not depend on my reading.
    # run_matching returns it as a CUDA tensor, so np.asarray alone raises.
    np.save(os.path.join(d, 'stt.npy'), _to_numpy(stt))
    np.save(os.path.join(d, 'U.npy'),
            U.detach().to(torch.float32).cpu().numpy())
    meta = {'ibatch': int(ibatch), 'nt': int(ops['nt']),
            'nt0min': int(ops['nt0min']),
            'n_pcs': int(ops['settings']['n_pcs']),
            'nearest_chans': int(ops['settings']['nearest_chans']),
            'fs': float(ops['settings']['fs']),
            'shape_pre': list(X_pre.shape)}
    for key in ('Wrot', 'wPCA', 'xc', 'yc'):
        v = ops.get(key)
        if v is None:
            continue
        np.save(os.path.join(d, f'{key}.npy'), _to_numpy(v))
    np.save(os.path.join(d, 'meta.npy'), meta, allow_pickle=True)
    logger.info(f'KS4_DUMP_RESIDUAL: wrote batch {ibatch} to {d}')


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
    n_src = ds.shape[0]
    nC = int(min(nC, n_src))
    # Same top-nC shortcut as spikedetect.nearest_chans (argpartition + sort).
    if nC >= n_src:
        iCC = np.argsort(ds, 0)
    else:
        part = np.argpartition(ds, nC - 1, axis=0)[:nC]
        ds_part = np.take_along_axis(ds, part, axis=0)
        order = np.argsort(ds_part, axis=0)
        iCC = np.take_along_axis(part, order, axis=0)
    iCC_mask = np.take_along_axis(ds, iCC, axis=0)
    iCC = torch.from_numpy(iCC).to(device)
    iCC_mask = iCC_mask < position_limit**2
    iCC_mask = torch.from_numpy(iCC_mask).to(device)
    # Empty-template NaNs → treat as zero so argmax stays finite.
    Unorm = (U**2).sum(1)
    if not torch.isfinite(Unorm).all():
        Unorm = torch.nan_to_num(Unorm, nan=0.0, posinf=0.0, neginf=0.0)
    iU = torch.argmax(Unorm, -1)
    Ucc = U[torch.arange(U.shape[0]),:,iCC[:,iU]]

    return iCC, iCC_mask, iU, Ucc


def extract(ops, bfile, U, device=torch.device('cuda'), progress_bar=None,
            spike_capacity_hint=None):
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
    # wPCA fixed for extract: cache transpose once (detect does the same).
    wPCA_T = ops['wPCA'].T.contiguous()
    # U is fixed for the whole extract pass: scale, time-domain waveforms, and
    # ctc are batch-invariant. Build both in one pass over U.
    ctc, match_cache = prepare_matching(ops, U, return_cache=True)
    dump = _residual_dump_request()
    spike_capacity = get_spike_buffer_capacity(bfile.n_batches)
    # Learned extract often finds a similar spike count to universal detect;
    # size the buffer from that hint so we avoid a mid-pass 2× realloc of tF.
    if spike_capacity_hint is not None:
        spike_capacity = max(spike_capacity, int(spike_capacity_hint))
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
    ibatch = -1
    try:
        for ibatch in prog:
            if ibatch % 100 == 0:
                log_performance(logger, 'debug', f'Batch {ibatch}')

            X = next(batches)
            # run_matching peels IN PLACE on X, so a pre-peel copy has to be
            # taken before the call, not after. Only when dumping.
            X_pre = X.clone() if (dump is not None
                                  and ibatch == dump[1]) else None
            stt, amps, th_amps, Xres = run_matching(
                ops, X, U, ctc, device=device, unit_cache=match_cache
            )
            if X_pre is not None:
                _write_residual_dump(dump[0], ibatch, X_pre, Xres, stt, U, ops)
                X_pre = None
            nsp = len(stt)
            if nsp == 0:
                if progress_bar is not None:
                    progress_bar.emit(int((ibatch+1) / bfile.n_batches * 100))
                continue

            xfeat = Xres[iCC[:, iU[stt[:,1:2]]],stt[:,:1] + tiwave] @ wPCA_T
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
                if nsp == 0:
                    if progress_bar is not None:
                        progress_bar.emit(int((ibatch+1) / bfile.n_batches * 100))
                    continue

            if k+nsp>st.shape[0]:
                # Double capacity: copy only the live prefix, not a full zeros_like.
                new_cap = max(k + nsp, st.shape[0] * 2)
                st2 = np.zeros((new_cap, st.shape[1]), dtype=st.dtype)
                st2[:k] = st[:k]
                st = st2
                tF2 = torch.zeros((new_cap,) + tF.shape[1:], dtype=tF.dtype)
                tF2[:k] = tF[:k]
                tF = tF2

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

    if ibatch >= 0:
        log_performance(logger, 'debug', f'Batch {ibatch}')

    isort = np.argsort(st[:k,0])
    st = st[isort]
    tF = tF[isort]

    return st, tF, ops


def align_U(U, ops, device=torch.device('cuda')):
    U = U.to(device)
    # Empty-cluster mean templates are intentionally NaN; treat as zero so
    # argmax / roll stay defined and do not poison finite units.
    if not torch.isfinite(U).all():
        U = torch.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    Uex = torch.einsum('xyz, zt -> xty', U, ops['wPCA'])
    X = Uex.reshape(-1, ops['Nchan']).T
    X = conv1d(X.unsqueeze(1), ops['wTEMP'].unsqueeze(1), padding=ops['nt']//2)
    Xmax = X.abs().max(0)[0].max(0)[0].reshape(-1, ops['nt'])
    imax = torch.argmax(Xmax, 1)

    Unew = Uex.clone()
    # Only roll unique lag bins that actually appear (nt loop was wasteful when
    # few templates share lags; results identical).
    for j in torch.unique(imax).tolist():
        j = int(j)
        ix = imax == j
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


def prepare_matching(ops, U, return_cache=False):
    """Build scaled cross-template filter bank (and optional unit_cache).

    When ``return_cache=True``, also returns the batch-invariant peel tensors
    (s / Us / U_time / W) so extract can avoid a second pass over U for
    ``_matching_unit_cache``.
    """
    nt = ops['nt']
    W = ops['wPCA'].contiguous()
    WtW = conv1d(W.reshape(-1, 1,nt), W.reshape(-1, 1 ,nt), padding = nt)
    WtW = torch.flip(WtW, [2,])

    # Fuse UtU @ WtW so the (nU, nU, nPC, nPC) intermediate is never retained.
    # Mathematically identical to the two-step form; bit-checked in unit tests.
    # Non-finite empty-template rows (NaN means from clustering) zeroed so peel
    # does not poison B / ctc for the whole batch.
    if not torch.isfinite(U).all():
        U = torch.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    ctc = torch.einsum('ikl, jml, kmt -> ijt', U, U, WtW)

    # Pre-scale by s_i = nm_i**-0.5 along the row axis so run_matching can work
    # on a scaled projection B and skip the per-peel division by nm (ctc is
    # indexed [:, iY, :] and subtracted from the scaled B).
    nm = (U**2).sum(-1).sum(-1)
    s = nm.clamp_min(1e-30).rsqrt()
    ctc = ctc * s.view(-1, 1, 1)

    if not return_cache:
        return ctc

    Us = U * s.view(-1, 1, 1)
    # (n_units, n_chan, nt); peel permutes selected rows to (C, n_sel, nt)
    U_time = torch.einsum('ijk, jl -> ikl', U, W)
    # Peel index windows depend only on nt (+ device of U); cache once per pass.
    device = U.device
    trange = torch.arange(-nt, nt + 1, device=device)
    tiwave = torch.arange(-(nt // 2), nt // 2 + 1, device=device)
    return ctc, {
        's': s, 'Us': Us, 'U_time': U_time, 'W': W,
        'trange': trange, 'tiwave': tiwave,
    }


def _matching_unit_cache(ops, U):
    """Batch-invariant tensors for run_matching (scale + time-domain units).

    U is fixed for the whole extract pass, so s / Us / U_time only need to be
    built once. Returning a small dict keeps run_matching's call surface stable
    for tests that invoke it directly. Prefer ``prepare_matching(...,
    return_cache=True)`` when ctc is also needed (single pass over U).
    """
    # Non-finite empty-template rows must match prepare_matching so B / peel
    # stay finite when tests / callers skip prepare_matching's nan scrub.
    if not torch.isfinite(U).all():
        U = torch.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    nt = ops['nt']
    W = ops['wPCA'].contiguous()
    nm = (U ** 2).sum(-1).sum(-1)
    s = nm.clamp_min(1e-30).rsqrt()
    Us = U * s.view(-1, 1, 1)
    # (n_units, n_chan, nt); peel permutes selected rows to (C, n_sel, nt)
    U_time = torch.einsum('ijk, jl -> ikl', U, W)
    device = U.device
    trange = torch.arange(-nt, nt + 1, device=device)
    tiwave = torch.arange(-(nt // 2), nt // 2 + 1, device=device)
    return {
        's': s, 'Us': Us, 'U_time': U_time, 'W': W,
        'trange': trange, 'tiwave': tiwave,
    }


def run_matching(ops, X, U, ctc, device=torch.device('cuda'), unit_cache=None):
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
    if unit_cache is None:
        unit_cache = _matching_unit_cache(ops, U)
    s = unit_cache['s']
    Us = unit_cache['Us']
    U_time = unit_cache['U_time']
    W = unit_cache['W']
    # Windows are nt/device-fixed; older callers may pass a partial cache.
    trange = unit_cache.get('trange')
    tiwave = unit_cache.get('tiwave')
    if trange is None or tiwave is None:
        trange = torch.arange(-nt, nt + 1, device=device)
        tiwave = torch.arange(-(nt // 2), nt // 2 + 1, device=device)
        unit_cache['trange'] = trange
        unit_cache['tiwave'] = tiwave

    B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt//2)
    B = torch.einsum('ijk, kjl -> il', Us, B)

    # Growable peel buffer. Cap at historical 1e5; start smaller so quiet
    # batches do not reserve a full 100k×(2+1+1) int64/float slab up front.
    NT = int(X.shape[-1])
    peel_cap = min(100000, max(2048, NT // 8))
    st = torch.zeros((peel_cap, 2), dtype=torch.int64, device=device)
    amps = torch.zeros((peel_cap, 1), dtype=torch.float, device=device)
    th_amps = torch.zeros((peel_cap, 1), dtype=torch.float, device=device)
    k = 0

    # Peel in-place on X: callers (extract) never reuse the pre-peel batch, so
    # a full clone (~125 MiB at 519×60k float32) was pure peak-RAM + bandwidth.
    # Spike times / amps / residual features stay bit-identical to clone path.
    Xres = X

    Th2 = Th * Th
    # ctc is indexed [row, unit, t] by the stock subtract; the fused kernel
    # wants [unit, row, t]. A permuted VIEW, not a copy -- the strides are
    # passed to the kernel.
    ctc_p = ctc.permute(1, 0, 2)
    for t in range(max_peels):
        # Reduce first, then apply relu/square on the (NT,) result.
        # In-place square avoids a full (NT,) temporary per peel.
        Cfmax, imax = torch.max(B, 0)
        # relu -> square -> zero the two nt-wide edges -> max_pool1d -> two
        # comparisons -> and, in one kernel. That is ~10 launches on a (NT,)
        # array of 40 KB, so the block is launch-bound rather than
        # bandwidth-bound, and it runs ~48x per batch. `nonzero` stays outside:
        # its length is data-dependent and its row ORDER is load-bearing for
        # the st/amps writes below. See fused_peel_cond.py; falls back to the
        # stock statements unless it proves bit-identical on the first peel.
        cmax, cnd = fused_peel_cond.peak_condition(Cfmax, nt, Th2)
        xs = torch.nonzero(cnd)

        if len(xs)==0:
            break

        iX = xs[:,:1]
        # iY is produced by the fused store below, which needs the grown
        # buffers -- computing it here as well would just be a second gather.
        nsp = len(iX)
        need = k + nsp
        if need > st.shape[0]:
            new_cap = max(need, st.shape[0] * 2)
            # Grow by empty tail only (avoid zeros_like full-buffer copy).
            extra = new_cap - st.shape[0]
            st = torch.cat((st, st.new_zeros((extra, 2))), 0)
            amps = torch.cat((amps, amps.new_zeros((extra, 1))), 0)
            th_amps = torch.cat((th_amps, th_amps.new_zeros((extra, 1))), 0)

        # B is scaled by s, so B_stock[iY,iX]/nm[iY] == B[iY,iX]*s[iY].
        # Four gathers, a multiply, a sqrt and four scatters to move ~26
        # spikes -- ~8 launches, so launch-bound like the condition tail.
        # One program per spike, which also preserves the `nonzero` row order
        # that st/amps/th_amps are indexed by positionally. See
        # fused_peel_store.py.
        iY = fused_peel_store.store_spikes(st, amps, th_amps, k, iX, imax, B,
                                           s, cmax)
        amp = amps[k:k+nsp]

        k+= nsp

        # n=2 splits the peel: advanced-index -= is last-write-wins on
        # overlapping trange windows, so stride is load-bearing for identity
        # (not just GPU memory). Keep stock n=2 on all devices.
        #
        # fused_peel replaces the two subtract statements with one kernel each
        # (5x), but ONLY for phases it has checked are window-disjoint, and
        # only after proving itself bit-identical on the first such phase of
        # the sort. It falls back to exactly the stock statements otherwise --
        # see fused_peel.py for why the disjointness check is not optional.
        n = 2
        fused_peel.peel_subtract(Xres, B, iX, iY, amp, U_time, ctc, ctc_p,
                                 tiwave, trange, nt, n)

    st = st[:k]
    amps = amps[:k]
    th_amps = th_amps[:k]

    return  st, amps, th_amps, Xres


def merging_function(ops, Wall, clu, st, tF, r_thresh=0.5, mode='ccg', check_dt=True,
                     device=torch.device('cuda'), max_sweeps=None):
    clu2 = clu.copy()

    Ww = Wall.to(device)
    NN = len(Ww)

    # Dense spike counts indexed by cluster label 0..NN-1. Historical code used
    # `unique(..., return_counts)` and then `ns[kk]` with kk=label, which only
    # works when labels are exactly 0..K-1 dense with no empty Wall rows. Empty
    # templates (gaps) made `while t<NN: clu_unq[isort[t]]` IndexError once t
    # passed len(unique). bincount is identical for dense full label sets and
    # safe with gaps / empty Wall rows.
    clu2_i = clu2.astype(np.int64, copy=False)
    if clu2_i.size and (clu2_i.min() < 0 or clu2_i.max() >= NN):
        raise ValueError(
            f'merging_function: cluster labels must be in [0, {NN}), '
            f'got min={int(clu2_i.min())} max={int(clu2_i.max())}'
        )
    ns = np.bincount(clu2_i, minlength=NN).astype(np.float64, copy=False)
    isort = np.argsort(ns)[::-1]

    is_merged = np.zeros(NN, 'bool')
    is_good = np.zeros(NN,)

    acg_threshold = ops['settings']['acg_threshold']
    ccg_threshold = ops['settings']['ccg_threshold']
    final_merge_union_acg_veto = False
    if mode == 'ccg':
        final_merge_union_acg_veto = bool(ops['settings'].get(
            'final_merge_union_acg_veto', False))
        is_ref, est_contam_rate = CCG.refract(clu, st[:,0]/ops['fs'],
                                              acg_threshold=acg_threshold,
                                              ccg_threshold=ccg_threshold)
        # refract returns length max(label)+1; pad/truncate to NN for empty tails
        if len(is_ref) < NN:
            pad = np.zeros(NN - len(is_ref), dtype=is_ref.dtype)
            is_ref = np.concatenate([is_ref, pad])
        else:
            is_ref = is_ref[:NN]

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

    # Spike times in seconds once. Extract sorts st by time; spike_idx lists
    # are ascending indices, so st_sec[idx] is sorted until a dt≠0 roll.
    # After any time-shifting merge, fall back to sorting inside compute_CCG.
    st_sec = None
    times_sorted = True
    if mode == 'ccg':
        st_sec = st[:, 0] / ops['fs']

    # Merging is swept to a FIXPOINT, because one pass exits early.
    #
    # `isort` ranks units by spike count and is computed ONCE, before any merge.
    # Absorbing a unit sets `ns[jj] = 0`, but `isort` still lists jj at its
    # original rank. The old loop read
    #
    #     kk = int(isort[t])
    #     if ns[kk] == 0:
    #         break
    #
    # so the first time `t` stepped onto a unit that had already been absorbed,
    # the whole stage terminated -- abandoning every unit ranked below it,
    # unexamined. The zero-count test was meant to skip the empty tail; on a
    # stale ordering it fires in the middle of the list instead. The earlier a
    # high-count unit is absorbed, the more of the list is thrown away.
    #
    # Measured on 20260818B (981 clusters in, 895 out, one pass, 28.8 s): the
    # FINISHED sort still contained 130 pairs that this function's own criteria
    # say to merge -- template similarity >= r_thresh and cross-refractory by
    # check_CCG. Six of them are duplicate cells that visibly break the RF
    # mosaics of the hand-typed types. The single pass made 86 merges and left
    # 130 behind; it was never a threshold problem, it was termination.
    #
    # So: treat the zero-count hit as the end of a SWEEP, not the end of the
    # stage. Re-sort by the updated counts, restart at t=0, and repeat until a
    # sweep completes with no merge. `is_merged` persists across sweeps, so
    # absorbed units are never revisited and each sweep is cheaper than the
    # last. `max_sweeps` bounds the cost; reaching it is not an error, it just
    # leaves the remaining merges unmade, exactly as before. max_merge_sweeps=1
    # reproduces the old single-pass behaviour bug-for-bug.
    if max_sweeps is None:
        try:
            max_sweeps = int(ops['settings'].get('max_merge_sweeps', 10))
        except (KeyError, TypeError, AttributeError):
            max_sweeps = 10
    # max_merge_sweeps=0 is floored to 1, so it has never turned merging OFF --
    # it is the single-pass arm, not a no-merge arm. That matters: the A/B that
    # "cleared" this stage of creating contaminated units compared 10 sweeps
    # against 1, and the first sweep already makes the merges between the
    # highest-count units, which is where a fusion would do the most damage. A
    # negative value is the arm that test never had.
    no_merge = int(max_sweeps) < 0
    max_sweeps = max(1, int(max_sweeps))

    t = 0 if not no_merge else NN
    nmerge = 0
    union_acg_veto_count = 0
    sweep_merges = 0
    sweeps_done = 0
    while True:
        # Sweep boundary: ran off the end, or reached the zero-count tail
        # (merged-away / empty Wall rows sort last under descending ns).
        if t >= NN or ns[int(isort[t])] == 0:
            sweeps_done += 1
            if no_merge or sweep_merges == 0 or sweeps_done >= max_sweeps:
                break
            sweep_merges = 0
            isort = np.argsort(ns)[::-1]
            t = 0
            continue

        kk = int(isort[t])

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
            st0 = st_sec[spike_idx[kk]]
        
        is_ccg  = 0
        for j in range(NN):
            jj = jsort[j]
            if cmax[jj] < r_thresh:
                break
            # compare with CCG
            if mode == 'ccg':
                # Merged-away / empty labels may be missing from spike_idx
                if jj not in spike_idx:
                    continue
                st1 = st_sec[spike_idx[jj]]
                _, is_ccg, _ = CCG.check_CCG(
                    st0, st1,
                    acg_threshold=acg_threshold,
                    ccg_threshold=ccg_threshold,
                    assume_sorted=times_sorted,
                )
            else:
                # Zero-energy templates (empty Wall rows) → 0/0; treat as not
                # mergeable on amplitude criterion (same as non-match).
                denom = mu[kk] + mu[jj]
                if float(denom.abs().min()) <= 0:
                    is_ccg = False
                else:
                    dmu = 2 * (mu[kk] - mu[jj]) / denom
                    is_ccg = dmu.abs() < 0.2

            if is_ccg:
                dt = (imax[kk] - imax[jj]).item()
                idx = spike_idx.get(jj, np.zeros(0, dtype=np.int64))
                if final_merge_union_acg_veto:
                    # Check the exact union that would be committed below:
                    # shift candidate samples in sample space first, then
                    # convert to seconds, matching `st[idx, 0] -= dt`.
                    st0_union = st[spike_idx[kk], 0] / ops['fs']
                    st1_union = np.array(st[idx, 0], copy=True)
                    if dt != 0 and check_dt and idx.size:
                        st1_union -= dt
                    st_union = np.sort(np.concatenate((
                        st0_union, st1_union / ops['fs'])))
                    is_union_ref, _, _ = CCG.check_CCG(
                        st_union,
                        acg_threshold=acg_threshold,
                        ccg_threshold=ccg_threshold,
                        assume_sorted=True,
                    )
                    if not is_union_ref:
                        union_acg_veto_count += 1
                        is_ccg = 0
                        continue

                is_merged[jj] = 1
                if dt != 0 and check_dt and idx.size:
                    # Update tF and Wall with shifted features
                    tF, Wall = roll_features(W, tF, Ww, idx, jj, dt)
                    # Shift spike times (and seconds cache); order no longer
                    # guaranteed sorted by time under ascending index lists.
                    st[idx,0] -= dt
                    if st_sec is not None:
                        st_sec[idx] = st[idx, 0] / ops['fs']
                    times_sorted = False
                
                denom = ns[kk] + ns[jj]
                if denom > 0:
                    Ww[kk] = ns[kk]/denom * Ww[kk] + ns[jj]/denom * Ww[jj]
                Ww[jj] = 0
                ns[kk] += ns[jj]
                ns[jj] = 0
                if idx.size:
                    clu2[idx] = kk
                    # Preserve ascending-index gather order of `clu2 == kk`.
                    prev = spike_idx.get(kk, np.zeros(0, dtype=np.int64))
                    spike_idx[kk] = np.sort(np.concatenate((prev, idx)))
                    spike_idx.pop(jj, None)
                renorm = True

                break

        if is_ccg==0:            
            t +=1    
        else:                
            nmerge+=1
            sweep_merges += 1
    
    if mode == 'ccg':
        ops['final_merge_count'] = int(nmerge)
        ops['final_merge_sweeps'] = int(sweeps_done)
        ops['final_merge_sweep_cap_hit'] = bool(
            not no_merge and sweep_merges > 0 and sweeps_done >= max_sweeps)
        ops['final_merge_union_acg_veto_count'] = int(union_acg_veto_count)

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
    # tF is host-side on the merge path; Wall (Ww) may sit on CUDA. Keep each
    # matmul on its operand's device. On CPU MEA, both share wPCA — no copy.
    if wPCA.device.type == 'cpu':
        W_host = wPCA
    else:
        W_host = wPCA.cpu()
    if Wall.device == wPCA.device:
        W_wall = wPCA
    else:
        W_wall = wPCA.to(Wall.device)

    # Project from PC space back to sample time, shift by dt
    feats = torch.roll(tF[spike_idx] @ W_host, shifts=dt, dims=2)
    temps = torch.roll(Wall[clust_idx:clust_idx+1] @ W_wall, shifts=dt, dims=2)

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

    # Project back to PC space and update tF / Wall
    tF[spike_idx] = feats @ W_host.T
    Wall[clust_idx] = temps @ W_wall.T

    return tF, Wall
