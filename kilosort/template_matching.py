import logging
from collections import deque
import os

from numba import njit
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


def _parse_batch_spec(spec):
    """'150' | '150-179' | '0,150-159' -> a sorted set of batch indices.

    A range exists because one batch of 10122 samples holds a median of 8
    spikes per unit, and an 8-spike average sits only sqrt(8)=2.8x above the
    single-sample noise. Thirty batches take that to ~240 spikes and ~15x.
    """
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition('-')
        if sep:
            a, b = int(lo), int(hi)
            if b < a:
                raise ValueError(
                    f'KS4_DUMP_RESIDUAL: range {part!r} runs backwards')
            out.update(range(a, b + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError('KS4_DUMP_RESIDUAL: no batch index in the spec')
    return out


def _residual_dump_request():
    """Parse KS4_DUMP_RESIDUAL, which is 'directory:batches'.

    Diagnostic only, and OFF unless the variable is set. The peel subtracts a
    template that covers a median of 13 of 519 channels and leaves 17.6-23.0%
    of each cell's signal energy on channels it never touches
    (src/utilities/AXONAL_AND_PEEL_TRUNCATION.md §1). Whether that arithmetic
    gap reaches the OUTPUT is a separate question, and answering it needs the
    residual, which kilosort does not otherwise save.

    Byte identity: when the variable is unset this returns None, no batch is
    ever selected, and the pre-peel clone below is never taken. The only cost
    on a normal run is one set membership test per batch.
    """
    spec = os.environ.get('KS4_DUMP_RESIDUAL')
    if not spec:
        return None
    path, _, batches = spec.rpartition(':')
    if not path:
        raise ValueError(
            "KS4_DUMP_RESIDUAL must be 'directory:batches', got "
            f"{spec!r}")
    want = _parse_batch_spec(batches)
    os.makedirs(path, exist_ok=True)
    return path, want


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
                                  and int(ibatch) in dump[1]) else None
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
    #
    # When lam > 0, the peel uses the ks2.5 amplitude-regularised score:
    #   a = 1 + lam
    #   b = relu(B[i,t]) + lam * mu[i]
    #   Cf[i,t] = b^2/a - lam * mu[i]^2
    # where mu[i] = 1/s[i] is the template norm. The prior centred on mu
    # boosts detection of spikes whose spatial projection is weak but whose
    # amplitude matches the template.
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

    lam = float(ops.get('lam', 0))
    use_lam = lam > 0

    B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt//2)
    B = torch.einsum('ijk, kjl -> il', Us, B)

    if use_lam:
        mu = 1.0 / s
        mu_col = mu.unsqueeze(1)
        a_inv = 1.0 / (1.0 + lam)
        lam_mu2 = lam * mu_col * mu_col

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
        if use_lam:
            b = torch.clamp(B, min=0) + lam * mu_col
            Cf = b * b * a_inv - lam_mu2
            Cfmax, imax = torch.max(Cf, 0)
            Cf_det = Cfmax.clone()
            Cf_det[:nt] = 0
            Cf_det[-nt:] = 0
            cmax = max_pool1d(
                Cf_det.view(1, 1, -1), 2 * nt + 1, stride=1, padding=nt
            )[0, 0]
            cnd = (cmax > Th2) & (torch.abs(cmax - Cf_det) < 1e-9)
        else:
            # Reduce first, then apply relu/square on the (NT,) result.
            Cfmax, imax = torch.max(B, 0)
            # relu -> square -> zero the two nt-wide edges -> max_pool1d -> two
            # comparisons -> and, in one kernel.
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
    isi_threshold = ops['settings'].get('isi_threshold', 0.01)
    isi_min_spikes = ops['settings'].get('isi_min_spikes', 500)
    final_merge_union_acg_veto = False
    if mode == 'ccg':
        final_merge_union_acg_veto = bool(ops['settings'].get(
            'final_merge_union_acg_veto', False))
        is_ref, est_contam_rate = CCG.refract(clu, st[:,0]/ops['fs'],
                                              acg_threshold=acg_threshold,
                                              ccg_threshold=ccg_threshold,
                                              isi_threshold=isi_threshold,
                                              isi_min_spikes=isi_min_spikes)
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


def _peak_shared_fraction_with_matches(st_a, st_b, fs, maxlag_s=0.002,
                                       halfbin_s=0.00025):
    """Fraction of the smaller train sharing spikes at the CCG peak lag.

    Searches all lags within ±maxlag_s and returns the highest bin count
    divided by min(len(a), len(b)). The final count is one-to-one: a spike
    from either train can contribute at most once. Spike times are in SAMPLES
    (int64).
    """
    if len(st_a) == 0 or len(st_b) == 0:
        return 0.0, 0.0, np.zeros(len(st_b), dtype=bool)
    maxlag = int(maxlag_s * fs)
    halfbin = int(halfbin_s * fs)
    frac, best_lag, matched_b = _peak_shared_fraction_kernel(
        np.asarray(st_a, dtype=np.int64), np.asarray(st_b, dtype=np.int64),
        maxlag, halfbin)
    return float(frac), float(best_lag / fs), matched_b


@njit(cache=True)
def _peak_shared_fraction_kernel(st_a, st_b, maxlag, halfbin):
    """Numba implementation of the exact peak/matching contract above.

    This deliberately keeps the old broad-bin and greedy one-to-one rules.
    It removes only Python allocation and per-spike interpreter overhead from
    the hot path; the returned lag and match mask remain the same.
    """
    width = 2 * halfbin + 1
    lo_edge = -maxlag - halfbin
    # Match np.arange(lo_edge, maxlag + halfbin + 1, width): the stop is
    # exclusive, so the rightmost histogram edge may end below the stop.
    n_bins = (maxlag + halfbin - lo_edge) // width
    counts = np.zeros(n_bins, dtype=np.int64)
    window = maxlag + halfbin
    has_pairs = False
    for a in st_a:
        lo = np.searchsorted(st_b, a - window, side='left')
        hi = np.searchsorted(st_b, a + window, side='right')
        for j in range(lo, hi):
            dt = st_b[j] - a
            bin_index = (dt - lo_edge) // width
            # np.histogram excludes values below the first edge and values
            # above the last edge (except the right edge of the last bin).
            if (bin_index == n_bins and
                    dt == lo_edge + n_bins * width):
                bin_index = n_bins - 1
            if 0 <= bin_index < n_bins:
                right_edge = lo_edge + (bin_index + 1) * width
                if dt < right_edge or bin_index == n_bins - 1:
                    counts[bin_index] += 1
                    has_pairs = True

    if not has_pairs:
        return 0.0, 0, np.zeros(len(st_b), dtype=np.bool_)

    k = 0
    for i in range(1, n_bins):
        if counts[i] > counts[k]:
            k = i
    lag_lo = lo_edge + k * width
    lag_hi = lag_lo + width
    last_bin = k == n_bins - 1

    best_count = -1
    best_lag = lag_lo
    midpoint = (lag_lo + lag_hi) / 2.0
    first_lag = lag_lo
    last_lag = lag_hi + (1 if last_bin else 0)
    for lag in range(first_lag, last_lag):
        exact_count = 0
        for a in st_a:
            pos = np.searchsorted(st_b, a + lag, side='left')
            if pos < len(st_b) and st_b[pos] == a + lag:
                exact_count += 1
        distance = abs(lag - midpoint)
        best_distance = abs(best_lag - midpoint)
        # Python max((count, -distance, -abs(lag), lag)) prefers the lower
        # absolute lag only after count and distance are tied.
        if (exact_count > best_count or
                (exact_count == best_count and distance < best_distance) or
                (exact_count == best_count and distance == best_distance and
                 abs(lag) < abs(best_lag))):
            best_count = exact_count
            best_lag = lag

    matched_b = np.zeros(len(st_b), dtype=np.bool_)
    j = 0
    matched = 0
    for a in st_a:
        while j < len(st_b) and st_b[j] - a < lag_lo:
            j += 1
        if (j < len(st_b) and
                (st_b[j] - a < lag_hi or
                 (last_bin and st_b[j] - a <= lag_hi))):
            matched += 1
            matched_b[j] = True
            j += 1
    return matched / max(min(len(st_a), len(st_b)), 1), best_lag, matched_b


def _peak_shared_fraction(st_a, st_b, fs, maxlag_s=0.002, halfbin_s=0.00025):
    """Compatibility wrapper returning only fraction and lag."""
    frac, lag, _ = _peak_shared_fraction_with_matches(
        st_a, st_b, fs, maxlag_s=maxlag_s, halfbin_s=halfbin_s)
    return frac, lag


@njit(cache=True)
def _coincidence_vote_counts(times, labels, cluster_sizes, n_labels,
                             maxlag, halfbin, n_bins):
    """Count all cross-cluster event pairs in the broad CCG lag bins.

    ``times`` is globally sorted.  Numba keeps the local event sweep cheap;
    the Python caller only materializes the usually small set of candidate
    cluster pairs.  This count is a superset of the exact one-to-one count,
    so it cannot hide a pair that the adjudicator would accept.
    """
    counts = np.zeros((n_labels, n_labels, n_bins), dtype=np.uint32)
    width = 2 * halfbin + 1
    lo_edge = -maxlag - halfbin
    window = maxlag + halfbin
    n = len(times)
    for i in range(n - 1):
        j = i + 1
        while j < n and times[j] - times[i] <= window:
            li = labels[i]
            lj = labels[j]
            if li != lj:
                if li < lj:
                    a = li
                    b = lj
                    signed_lag = times[j] - times[i]
                else:
                    a = lj
                    b = li
                    signed_lag = times[i] - times[j]
                # Orient votes toward the larger train, matching the exact
                # adjudicator's target/absorbed ordering. Equal-size trains
                # are allowed both orientations because the queue's tie
                # order is not part of the identity contract.
                if cluster_sizes[li] > cluster_sizes[lj]:
                    first_lag = times[j] - times[i]
                    n_votes = 1
                elif cluster_sizes[lj] > cluster_sizes[li]:
                    first_lag = times[i] - times[j]
                    n_votes = 1
                else:
                    first_lag = signed_lag
                    n_votes = 2
                for vote_index in range(n_votes):
                    vote_lag = (first_lag if vote_index == 0
                                else -first_lag)
                    lag_bin = (vote_lag - lo_edge) // width
                    if (lag_bin == n_bins and
                            vote_lag == lo_edge + n_bins * width):
                        lag_bin = n_bins - 1
                    if 0 <= lag_bin < n_bins:
                        counts[a, b, lag_bin] += 1
            j += 1
    return counts


@njit(cache=True)
def _coincidence_target_vote_counts(target_times, other_times, other_labels,
                                    target_index, n_labels, maxlag, halfbin,
                                    n_bins):
    """Count broad-bin votes from one grown target to all other clusters."""
    counts = np.zeros((n_labels, n_bins), dtype=np.uint32)
    width = 2 * halfbin + 1
    lo_edge = -maxlag - halfbin
    window = maxlag + halfbin
    for target_time in target_times:
        lo = np.searchsorted(other_times, target_time - window, side='left')
        hi = np.searchsorted(other_times, target_time + window, side='right')
        for j in range(lo, hi):
            other_index = other_labels[j]
            if target_index < other_index:
                signed_lag = other_times[j] - target_time
            else:
                signed_lag = target_time - other_times[j]
            lag_bin = (signed_lag - lo_edge) // width
            if (lag_bin == n_bins and
                    signed_lag == lo_edge + n_bins * width):
                lag_bin = n_bins - 1
            if 0 <= lag_bin < n_bins:
                counts[other_index, lag_bin] += 1
    return counts


def _coincidence_candidate_partners(target_label, target_times,
                                    cluster_times, fs, frac_thresh,
                                    maxlag_s=0.002, halfbin_s=0.00025):
    """Refresh only the candidate edges incident to a grown target."""
    labels = sorted(int(k) for k, v in cluster_times.items()
                    if int(k) != int(target_label) and len(v) >= 100)
    if not labels or len(target_times) < 100:
        return set()
    maxlag = int(maxlag_s * fs)
    halfbin = int(halfbin_s * fs)
    label_values = np.asarray(sorted(labels + [int(target_label)]), dtype=np.int64)
    target_index = int(np.searchsorted(label_values, int(target_label)))
    other_times = np.concatenate([
        np.asarray(cluster_times[k], dtype=np.int64) for k in labels])
    other_labels = np.concatenate([
        np.full(len(cluster_times[k]), int(np.searchsorted(label_values, k)),
                dtype=np.int64) for k in labels])
    order = np.argsort(other_times, kind='stable')
    other_times = other_times[order]
    other_labels = other_labels[order]
    edges = np.arange(-maxlag - halfbin, maxlag + halfbin + 1,
                      2 * halfbin + 1)
    counts = _coincidence_target_vote_counts(
        np.asarray(target_times, dtype=np.int64), other_times, other_labels,
        target_index, len(label_values), maxlag, halfbin, len(edges) - 1)
    peak_counts = counts.max(axis=1)
    if counts.shape[1] > 1:
        peak_counts = np.maximum(
            peak_counts, (counts[:, :-1] + counts[:, 1:]).max(axis=1))

    pairs = set()
    for other_index, other_label in enumerate(label_values):
        if other_index == target_index:
            continue
        required = int(np.ceil(
            frac_thresh * min(len(target_times),
                              len(cluster_times[int(other_label)]))))
        if required and peak_counts[other_index] >= required:
            a, b = sorted((int(target_label), int(other_label)))
            pairs.add((a, b))
    return pairs


def _coincidence_candidate_pairs(cluster_times, fs, frac_thresh,
                                 maxlag_s=0.002, halfbin_s=0.00025,
                                 max_samples_per_cluster=None):
    """Find candidate pairs with an exhaustive event-indexed vote pass.

    Every event is considered, so a real duplicate is not lost because its
    shared spikes missed a sampling grid.  The broad-bin count is a cheap
    superset of the exact one-to-one count; callers must still run
    ``_peak_shared_fraction_with_matches`` on every returned pair.

    ``max_samples_per_cluster`` is retained only as an explicit diagnostic
    escape hatch for small benchmarks.  The production caller leaves it
    unset, which is the recall-preserving mode.
    """
    labels = sorted(int(k) for k, v in cluster_times.items() if len(v) >= 100)
    if len(labels) < 2:
        return set()
    maxlag = int(maxlag_s * fs)
    halfbin = int(halfbin_s * fs)
    all_times = np.concatenate([np.asarray(cluster_times[k], dtype=np.int64)
                                for k in labels])
    all_labels = np.concatenate([np.full(len(cluster_times[k]), k,
                                         dtype=np.int64) for k in labels])
    if max_samples_per_cluster is not None:
        sampled_times = []
        sampled_labels = []
        for label in labels:
            train = np.asarray(cluster_times[label], dtype=np.int64)
            count = min(len(train), int(max_samples_per_cluster))
            positions = np.linspace(0, len(train) - 1, count).round().astype(int)
            sampled_times.append(train[positions])
            sampled_labels.append(np.full(count, label, dtype=np.int64))
        all_times = np.concatenate(sampled_times)
        all_labels = np.concatenate(sampled_labels)

    order = np.argsort(all_times, kind='stable')
    all_times = all_times[order]
    all_labels = all_labels[order]
    label_values = np.asarray(labels, dtype=np.int64)
    label_indices = np.searchsorted(label_values, all_labels)
    cluster_sizes = np.asarray([
        min(len(cluster_times[k]), int(max_samples_per_cluster))
        if max_samples_per_cluster is not None else len(cluster_times[k])
        for k in labels], dtype=np.int64)
    edges = np.arange(-maxlag - halfbin, maxlag + halfbin + 1,
                      2 * halfbin + 1)
    counts = _coincidence_vote_counts(
        all_times, label_indices, cluster_sizes, len(label_values), maxlag,
        halfbin, len(edges) - 1)

    # The exact adjudicator includes the final histogram edge.  Treat that
    # edge the same way here, then also join neighboring bins when deciding
    # eligibility.  The latter is conservative for a train whose true lag
    # straddles a broad-bin boundary; the exact matcher still decides it.
    peak_counts = counts.max(axis=2)
    if counts.shape[2] > 1:
        adjacent_counts = counts[:, :, :-1] + counts[:, :, 1:]
        peak_counts = np.maximum(peak_counts, adjacent_counts.max(axis=2))

    candidates = set()
    for i in range(len(label_values)):
        for j in range(i + 1, len(label_values)):
            count_i = len(cluster_times[int(label_values[i])])
            count_j = len(cluster_times[int(label_values[j])])
            if max_samples_per_cluster is not None:
                count_i = min(count_i, int(max_samples_per_cluster))
                count_j = min(count_j, int(max_samples_per_cluster))
            required = int(np.ceil(frac_thresh * min(count_i, count_j)))
            if required and peak_counts[i, j] >= required:
                candidates.add((int(label_values[i]), int(label_values[j])))
    return candidates


def _retained_global_feature_mean(ops, st, tF, spike_idx, n_channels):
    """Place retained local PC features on physical channels and average.

    ``tF`` is stored in the nearest-channel order of each detection template,
    whereas ``Wall`` is indexed by physical channel.  A direct mean of ``tF``
    is therefore only valid for the small legacy fixtures that have no
    template map.  Production data always take the mapped scatter path.
    """
    idx = torch.as_tensor(spike_idx, dtype=torch.long, device=tF.device)
    values = tF[idx]
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError('retained features must be a non-empty 3-D tensor')

    iC = ops.get('iC')
    iCC = ops.get('iCC')
    iU = ops.get('iU')
    st_array = np.asarray(st)
    if ((iC is None and (iCC is None or iU is None)) or
            st_array.ndim < 2 or st_array.shape[1] < 2):
        # Compatibility for minimal unit-test fixtures from before st[:, 5]
        # (the universal detection-template id) was part of this contract.
        if values.shape[1] != n_channels:
            raise ValueError(
                'cannot align retained features without ops["iC"]: '
                f'{values.shape[1]} local channels vs {n_channels} global')
        return values.mean(dim=0)

    # In-memory spike tables have six columns and keep the detection template
    # in column 5.  The compact three-column export keeps that same id in
    # column 1, which is useful for replaying a saved merge checkpoint.
    template_column = 5 if st_array.shape[1] > 5 else 1
    templates = torch.as_tensor(
        st_array[np.asarray(spike_idx), template_column].astype(np.int64),
        dtype=torch.long, device=tF.device)
    if iCC is not None and iU is not None:
        iCC = torch.as_tensor(iCC, dtype=torch.long, device=tF.device)
        iU = torch.as_tensor(iU, dtype=torch.long, device=tF.device)
        if templates.numel() and (templates.min() < 0 or
                                  templates.max() >= iU.shape[0]):
            raise ValueError('learned spike template index is out of range')
        channels = iCC[:, iU[templates]].T.contiguous()
    else:
        iC = torch.as_tensor(iC, dtype=torch.long, device=tF.device)
        if templates.numel() and (templates.min() < 0 or
                                  templates.max() >= iC.shape[1]):
            raise ValueError('spike template index is out of range for ops["iC"]')
        channels = iC[:, templates].T.contiguous()
    if values.shape[:2] != channels.shape:
        raise ValueError('retained features and template maps have incompatible shapes')
    if channels.numel() and (channels.min() < 0 or channels.max() >= n_channels):
        raise ValueError('template channel index is out of range for Wall')

    n_features = int(values.shape[2])
    sums = torch.zeros((n_channels, n_features), dtype=values.dtype,
                       device=tF.device)
    flat_channels = channels.reshape(-1)
    sums.index_add_(0, flat_channels, values.reshape(-1, n_features))
    # Wall is built with zero-filled unmapped channels and divided by the
    # total number of events, not by per-channel observation counts.
    return sums / float(values.shape[0])


def residual_event_sample_reference(detector_samples, nt, nt0min):
    """Convert detector-window positions to the learned spike-time reference."""
    return np.asarray(detector_samples) - nt - nt // 2 + nt0min


def coincidence_merge(ops, Wall, clu, st, tF, frac_thresh=0.20,
                      acg_threshold=0.2, ccg_threshold=0.25,
                      isi_threshold=0.01, isi_min_spikes=500):
    """Merge clusters that share spike-time coincidences above a threshold.

    This pass catches splits that the template-similarity merge misses: the
    same cell detected at different channels produces templates with low
    waveform correlation but high spike-time overlap at a non-zero lag (axonal
    propagation). The CCG peak-lag method finds them.

    Safety gates:
      - The target (larger) cluster must already be refractory (``is_ref``).
      - The prospective union removes only the explicitly matched copy from
        the smaller train; all unmatched and within-source events remain in
        the refractory-quality check.

    Parameters
    ----------
    frac_thresh : float
        Minimum fraction of the smaller cluster's spikes that must coincide
        at the CCG peak lag for a merge. Default 0.20 (same as the lab's
        dup_collapse_figure.py).

    Returns the same tuple as merging_function.
    """
    logger = logging.getLogger(__name__)

    if frac_thresh <= 0:
        return Wall, clu, None, st, tF

    clu2 = clu.copy()
    fs = ops['fs']
    n_clusters = int(clu2.max()) + 1
    if Wall is not None:
        n_clusters = max(n_clusters, int(Wall.shape[0]))

    # Quality labels from the preceding merge step (with ISI fallback)
    is_ref, _ = CCG.refract(clu2, st[:, 0] / fs,
                            acg_threshold=acg_threshold,
                            ccg_threshold=ccg_threshold,
                            isi_threshold=isi_threshold,
                            isi_min_spikes=isi_min_spikes)
    if len(is_ref) < n_clusters:
        is_ref = np.concatenate([is_ref,
                                 np.zeros(n_clusters - len(is_ref), dtype=bool)])

    spike_idx = group_indices_by_label(clu2)
    is_merged = np.zeros(n_clusters, dtype=bool)
    dropped_spikes = np.zeros(len(st), dtype=bool)
    ops['coincidence_merge_deduped_spikes'] = 0

    # Count spikes per cluster, sort descending
    ns = np.bincount(clu2.astype(np.int64), minlength=n_clusters).astype(np.float64)
    isort = np.argsort(ns)[::-1]

    # Precompute sorted spike times per cluster (in samples)
    cluster_times = {}
    for kid in spike_idx:
        cluster_times[kid] = np.sort(st[spike_idx[kid], 0].astype(np.int64))

    nmerge = 0
    nveto = 0

    # Only check clusters with enough spikes to be meaningful
    candidates = [int(isort[i]) for i in range(n_clusters)
                  if ns[int(isort[i])] >= 100 and not is_merged[int(isort[i])]]
    candidate_pairs = _coincidence_candidate_pairs(
        cluster_times, fs, frac_thresh)
    ops['coincidence_candidate_pairs'] = len(candidate_pairs)
    # A target can grow as the merge loop proceeds.  Preserve the original
    # source labels so evidence for B-C remains usable after B is absorbed
    # into A; otherwise the prefilter changes the old chain semantics.
    merge_sources = {int(k): {int(k)} for k in cluster_times}

    # Revisit a target after it grows. This handles aggregate evidence split
    # across several source clusters and partners that were earlier in the
    # original count ordering.
    candidate_queue = deque(candidates)
    while candidate_queue:
        kk = candidate_queue.popleft()
        if is_merged[kk] or kk not in cluster_times:
            continue
        if not is_ref[kk]:
            continue
        for j in range(len(candidates)):
            # The target grows after every accepted merge.  Read it here,
            # inside the loop, so subsequent candidates are tested against
            # the committed train rather than a stale pre-merge snapshot.
            st_a = cluster_times.get(kk)
            if st_a is None or len(st_a) < 100:
                break
            jj = candidates[j]
            if jj == kk:
                continue
            if is_merged[jj] or jj not in cluster_times:
                continue
            # The larger current train absorbs the smaller one. A target can
            # change rank after a merge, hence the queue revisit above.
            if ns[kk] < ns[jj]:
                continue
            pair = (kk, jj) if kk < jj else (jj, kk)
            possible = pair in candidate_pairs
            if not possible:
                possible = any(
                    ((src, jj) if src < jj else (jj, src)) in candidate_pairs
                    for src in merge_sources.get(kk, {kk}))
            if not possible:
                continue
            st_b = cluster_times[jj]
            if len(st_b) < 100:
                continue

            frac, lag, matched_b = _peak_shared_fraction_with_matches(
                st_a, st_b, fs)
            if frac < frac_thresh:
                continue

            dt_samples = int(round(lag * fs))

            # Do not run remove_duplicates here.  That would erase the very
            # refractory violations used to decide whether this is a valid
            # merge.  Only the one-to-one CCG matches are duplicates by
            # evidence; every other event must remain visible to the gate.
            st_union = np.sort(np.concatenate([
                st_a, st_b[~matched_b] - dt_samples]))
            st_union_sec = st_union / fs

            is_union_ref, _, _ = CCG.check_CCG(
                st_union_sec, acg_threshold=acg_threshold,
                ccg_threshold=ccg_threshold, assume_sorted=True)
            if not is_union_ref and isi_threshold > 0:
                n_union = len(st_union_sec)
                if (n_union >= isi_min_spikes
                        and CCG.isi_violation_rate(st_union_sec) < isi_threshold):
                    is_union_ref = True

            if not is_union_ref:
                nveto += 1
                continue

            idx_jj = spike_idx.get(jj, np.zeros(0, dtype=np.int64))
            # ``matched_b`` is aligned to the time-sorted train, while
            # ``idx_jj`` is in the original spike-row order.  Convert the
            # match mask back to row indices before committing the merge.
            jj_order = np.argsort(st[idx_jj, 0])
            idx_jj_sorted = idx_jj[jj_order]
            keep_idx_jj = idx_jj_sorted[~matched_b]
            drop_idx_jj = idx_jj_sorted[matched_b]

            # Merge jj into kk.  The trial union above is the event set we
            # validated, so matched copies must be removed from the returned
            # arrays as well; retaining them would make later merges see a
            # different train from the one that passed the veto.
            is_merged[jj] = True
            merge_sources.setdefault(kk, {kk}).update(
                merge_sources.pop(jj, {jj}))
            dropped_spikes[drop_idx_jj] = True
            if idx_jj.size:
                if dt_samples != 0:
                    # The timestamp shift changes the waveform's reference
                    # time too. Keep features and the averaged template in
                    # the same frame as the committed spike times.
                    tF, Wall = roll_features(
                        ops['wPCA'], tF, Wall, keep_idx_jj, jj, dt_samples)
                    st[keep_idx_jj, 0] -= dt_samples
                clu2[keep_idx_jj] = kk
                prev = spike_idx.get(kk, np.zeros(0, dtype=np.int64))
                spike_idx[kk] = np.sort(np.concatenate((prev, keep_idx_jj)))
                spike_idx.pop(jj, None)
                cluster_times[kk] = np.sort(st[spike_idx[kk], 0].astype(np.int64))

            # Weighted average of Wall templates
            n_jj = len(keep_idx_jj)
            ns[kk] += n_jj
            ns[jj] = 0
            denom = ns[kk]
            if denom > 0 and Wall is not None and n_jj:
                # Wall[jj] was estimated from all source events.  Once the
                # matched rows are removed it is no longer the right source
                # mean; use the surviving event features instead.
                retained_wall_jj = _retained_global_feature_mean(
                    ops, st, tF, keep_idx_jj, Wall.shape[1])
                retained_wall_jj = retained_wall_jj.to(
                    device=Wall.device, dtype=Wall.dtype)
                old_kk = denom - n_jj
                Wall[kk] = ((old_kk / denom) * Wall[kk]
                            + (n_jj / denom) * retained_wall_jj)
                Wall[jj] = 0

            if len(merge_sources.get(kk, {kk})) > 1:
                # A pair can become eligible only after evidence from several
                # already-merged sources is combined. Refresh the conservative
                # event-indexed graph, then queue this target so partners that
                # were already scanned are reconsidered.
                candidate_pairs.update(
                    _coincidence_candidate_partners(
                        kk, cluster_times[kk], cluster_times,
                        fs, frac_thresh))
                candidate_queue.append(kk)

            nmerge += 1
            ops['coincidence_merge_deduped_spikes'] = (
                int(ops.get('coincidence_merge_deduped_spikes', 0))
                + int(matched_b.sum()))
            logger.debug(
                f'coincidence merge: {jj} → {kk} (frac={frac:.2f}, '
                f'lag={lag*1000:.2f}ms)')
            # Keep scanning this target. Its refreshed time cache and size
            # should be used for the remaining candidates. The queue revisit
            # handles candidates earlier in the original count ordering.

    ops['coincidence_merge_count'] = nmerge
    ops['coincidence_merge_veto_count'] = nveto
    logger.info(f'coincidence merge: {nmerge} merges, {nveto} vetoed by ACG')

    if nmerge == 0:
        return Wall, clu2, None, st, tF

    # Renumber clusters to fill gaps
    imap = np.cumsum((~is_merged).astype('int32')) - 1
    if imap.size > 0:
        clu2 = imap[clu2]
    Wall = Wall[~is_merged]

    # Remove the duplicate event rows that were explicitly matched above.
    # This is deliberately after all index-based merge bookkeeping is done.
    if dropped_spikes.any():
        keep_spikes = ~dropped_spikes
        st = st[keep_spikes]
        clu2 = clu2[keep_spikes]
        keep_idx = torch.as_tensor(np.flatnonzero(keep_spikes),
                                   device=tF.device)
        tF = tF[keep_idx]

    # Re-sort by time
    sorted_idx = np.argsort(st[:, 0])
    st = np.take_along_axis(st, sorted_idx[..., np.newaxis], axis=0)
    clu2 = clu2[sorted_idx]
    tensor_idx = torch.as_tensor(sorted_idx, device=tF.device)
    tF = tF[tensor_idx]

    return Wall, clu2, None, st, tF


def extract_residual_spikes(ops, bfile, Wall, device=torch.device('cuda'),
                            progress_bar=None, residual_Th=4):
    """Detect spikes in the residual after peeling the main sort's templates.

    For each batch: peel with Wall, then run universal detect on what remains.
    Returns (st_res, tF_res) in the same format as spikedetect.run.
    """
    from kilosort import spikedetect

    # Residual discovery is diagnostic/augmenting work.  Keep all device
    # conversions and threshold changes in a shallow private copy: the
    # caller immediately reuses `ops` for the learned extraction, and a
    # residual pass must not change that extraction's contract.
    ops_res = dict(ops)
    if isinstance(ops.get('settings'), dict):
        ops_res['settings'] = dict(ops['settings'])
    ops_res['Th_universal'] = residual_Th

    iC = ops_res['iC']
    iC2 = ops_res.get('iC2')
    weigh = ops_res.get('weigh')
    if iC2 is None or weigh is None:
        raise RuntimeError('residual pass requires iC2/weigh from universal detect')

    if not isinstance(iC, torch.Tensor):
        iC = torch.as_tensor(iC, device=device).long()
    if not isinstance(iC2, torch.Tensor):
        iC2 = torch.as_tensor(iC2, device=device).long()
    if not isinstance(weigh, torch.Tensor):
        weigh = torch.as_tensor(weigh, device=device).float()
    wTEMP = ops_res['wTEMP']
    if not isinstance(wTEMP, torch.Tensor):
        wTEMP = torch.as_tensor(wTEMP, device=device).float()
    elif wTEMP.device != device:
        wTEMP = wTEMP.to(device)
    ops_res['wTEMP'] = wTEMP

    nC = ops_res['settings']['nearest_chans']
    nt = ops_res['nt']
    tarange = torch.arange(-(nt // 2), nt // 2 + 1, device=device)
    wPCA_T = ops_res['wPCA'].T.contiguous()
    yc = ops_res['yc']
    yc_t = torch.as_tensor(yc, device=device)

    ctc, match_cache = prepare_matching(ops_res, Wall, return_cache=True)

    spike_capacity = get_spike_buffer_capacity(bfile.n_batches)
    st = np.zeros((spike_capacity, 6), 'float64')
    tF = np.zeros((spike_capacity, nC, ops_res['settings']['n_pcs']), 'float32')
    k = 0

    tm_scratch = {}
    batches = bfile.iter_batches(ops_res)
    prog = tqdm(np.arange(bfile.n_batches, dtype=np.int64),
                miniters=200 if progress_bar else None,
                mininterval=60 if progress_bar else None)

    ibatch = -1
    try:
        for ibatch in prog:
            X = next(batches)
            # Peel the main sort's templates from this batch
            _, _, _, Xres = run_matching(
                ops_res, X, Wall, ctc, device=device, unit_cache=match_cache)

            # Run universal detect on the residual at lower threshold
            xy, imax, amp, adist = spikedetect.template_match(
                Xres, ops_res, iC, iC2, weigh, device=device,
                scratch=tm_scratch)
            nsp = len(xy)
            if nsp == 0:
                continue

            yct = spikedetect.yweighted(yc, iC, adist, xy, device=device,
                                        yc_t=yc_t)

            if k + nsp > st.shape[0]:
                new_cap = max(k + nsp, st.shape[0] * 2)
                st2 = np.zeros((new_cap, st.shape[1]), dtype=st.dtype)
                st2[:k] = st[:k]
                st = st2
                tF2 = np.zeros((new_cap,) + tF.shape[1:], dtype=tF.dtype)
                tF2[:k] = tF[:k]
                tF = tF2

            xsub = Xres[iC[:, xy[:, :1]], xy[:, 1:2] + tarange]
            xfeat = xsub @ wPCA_T
            tF[k:k + nsp] = xfeat.transpose(0, 1).cpu().numpy()

            t_shift = ibatch * bfile.batch_downsampling * (
                ops_res['batch_size'] / ops_res['fs'])
            col1 = xy[:, 1].double()
            cols = torch.stack(
                (col1, yct.double(), amp.double(), imax.double(),
                 torch.full_like(col1, ibatch), xy[:, 0].double()), dim=1)
            cols = cols.cpu().numpy()
            # Match the learned extractor's event reference exactly.  The
            # detector reports the window position, while exported spike
            # times are referenced at nt//2 - nt0min samples in that window.
            cols[:, 0] = residual_event_sample_reference(
                cols[:, 0], nt, ops_res['nt0min']) / ops_res['fs'] + t_shift
            st[k:k + nsp] = cols

            k += nsp

            if progress_bar is not None:
                progress_bar.emit(
                    int((ibatch + 1) / bfile.n_batches * 100))
    except Exception:
        logger.exception(
            f'Error in extract_residual_spikes on batch {ibatch}')
        raise

    st = st[:k]
    tF = tF[:k]
    logger.info(f'Residual detect: {k} spikes at Th={residual_Th}')
    return st, tF


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
