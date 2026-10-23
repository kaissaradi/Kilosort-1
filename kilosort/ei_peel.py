"""Peel with each unit's EI (electrical image) instead of its PC template.

Why
---
A Kilosort template stores every channel's waveform in the same 6 temporal PCs,
learned on trough-centred spikes. That basis assumes every channel sees the
spike at the same instant. A retinal ganglion cell's spike does not: it runs
down the axon, so an electrode 0.3-3 mm away sees it 5-24 samples (0.25-1.2 ms)
after the soma trough. Measured on 20260514A/data003 (60 um, 512 ch), the 6 PCs
keep 92-100 % of the soma channel's energy but only 1-31 % (median ~13 %) of
the axon channel's. The peel therefore subtracts the soma and leaves the axon
in the residual, where another template catches it: 175 of 176 duplicate pairs
on that recording are one neuron detected at its soma and again on its axon
(lag consistent with ~1.3 m/s conduction), ~9.5 % of all spikes.

The EI -- the mean filtered, whitened waveform on all channels at full time
resolution -- has no such basis, so it holds the axon at its own delay. The
peel scores and subtracts the EI; the arithmetic of run_matching is unchanged
(it is written for any (n_units, n_chan, nt) waveform: B = <x, E> / |E|,
ctc = cross-correlations of the E's).

What
----
``unit_eis``   mean whitened snippet per unit on all channels, from the
               spikes the units were given ('detect' or 'peel' frame),
               channels kept only where significant, each EI shifted so the
               largest deflection on its largest channel sits at nt0min (a
               spike's time marks the same feature of its neuron in every unit).
``merge_eis``  a larger unit absorbs a unit whose EI matches at lagged
               cosine >= MERGE_COS (one neuron detected twice: soma + axon, or
               two alignments of one waveform).
``ei_matching`` the ctc and unit cache that run_matching reads, built from E.

Used by run_kilosort.detect_spikes when settings['ei_peel'] is True: EIs from
the first clustering -> merge -> peel -> EIs from the peel's own spikes ->
merge -> peel again. Off by default.
"""
import logging

import numpy as np
import torch
from torch.nn.functional import conv1d

logger = logging.getLogger(__name__)

MAX_SPIKES_PER_UNIT = 300
MIN_SPIKES = 20
SIG_FACTOR = 3.0     # channel kept when its mean's energy > 3 x the energy expected from noise alone
ALIGN_SLACK = 20     # samples the EI may be shifted to put its largest deflection at nt0min
MERGE_COS = 0.9


def _batch_positions(ops, bfile, st, mode):
    """(batch index, sample index inside the padded batch) for every spike."""
    nt = ops['nt']
    bs = int(ops['batch_size']) if 'batch_size' in ops else int(ops['settings']['batch_size'])
    bd = int(getattr(bfile, 'batch_downsampling', 1))
    fs = float(ops['fs'])
    if mode == 'detect':
        # spikedetect.run: col0 = (xy1 - nt)/fs + ibatch*bd*bs/fs, col4 = ibatch
        ib = st[:, 4].astype(np.int64)
        xy1 = np.rint((st[:, 0] - ib * bd * bs / fs) * fs).astype(np.int64) + nt
    else:
        # template_matching.extract: col0 = (stt0 - nt) + ibatch*bd*bs - nt//2 + nt0min (samples)
        c0 = st[:, 0].astype(np.int64)
        ib = np.floor_divide(c0 - ops['nt0min'] + nt // 2, bd * bs)
        ib = np.clip(ib, 0, int(bfile.n_batches) - 1)
        xy1 = c0 - ib * bd * bs + nt + nt // 2 - ops['nt0min']
    return ib, xy1


def unit_eis(ops, bfile, st, clu, n_units, mode, device=torch.device('cuda'), seed=0):
    """Return (E, cnt): E (n_units, n_chan, nt) whitened EIs in the peel's template frame, cnt spikes averaged.

    Units with fewer than MIN_SPIKES sampled spikes get an all-zero EI (cnt tells the caller).
    """
    nt = ops['nt']
    S = ALIGN_SLACK
    clu = np.asarray(clu).astype(np.int64)
    ib, xy1 = _batch_positions(ops, bfile, st, mode)
    NTpad = int(bfile.NT) + 2 * int(bfile.nt)
    ok = (xy1 - nt // 2 - S >= 0) & (xy1 + nt // 2 + S < NTpad) & (clu >= 0) & (clu < n_units)
    # every unit gets up to MAX_SPIKES_PER_UNIT spikes drawn from the whole recording: sampling a subset of batches
    # leaves units with few spikes without an EI, and they would drop out of the peel
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(ok)
    idx = idx[rng.permutation(idx.size)]
    srt = np.argsort(clu[idx], kind='stable')
    cs = clu[idx][srt]
    rank = np.empty(idx.size, np.int64)
    rank[srt] = np.arange(idx.size) - np.searchsorted(cs, cs)
    idx = np.sort(idx[rank < MAX_SPIKES_PER_UNIT])

    n_chan = int(ops['Nchan'])
    L = nt + 2 * S
    S1 = torch.zeros((n_units, n_chan, L), device=device)
    S2 = torch.zeros((n_units, n_chan), device=device)
    cnt = torch.zeros(n_units, device=device)
    tw = torch.arange(-(nt // 2) - S, nt // 2 + 1 + S, device=device)
    for b in np.unique(ib[idx]):
        sel_all = idx[ib[idx] == b]
        X = bfile.padded_batch_to_torch(int(b), ops)
        for k0 in range(0, sel_all.size, 1000):
            sel = sel_all[k0:k0 + 1000]
            t = torch.as_tensor(xy1[sel], device=device)
            clips = X[:, t.unsqueeze(1) + tw].permute(1, 0, 2).contiguous()   # (n_sp, n_chan, L)
            u = torch.as_tensor(clu[sel], device=device)
            S1.index_add_(0, u, clips)
            S2.index_add_(0, u, (clips[:, :, S:S + nt] ** 2).sum(-1))
            cnt.index_add_(0, u, torch.ones(u.numel(), device=device))
    n = cnt.clamp_min(1)
    M = S1 / n[:, None, None]
    # significance on the template window: energy of the mean vs. the energy noise alone leaves in a mean of n clips
    core = M[:, :, S:S + nt]
    e_mean = (core ** 2).sum(-1)
    var = (S2 / n[:, None] - e_mean).clamp_min(0)
    keep = e_mean > SIG_FACTOR * var / n[:, None]
    M = M * keep[..., None]
    # align: the trough (most negative sample) on the largest channel, searched within +-S of nt0min. A spike's
    # time is its inward-current trough; aligning on the largest |deflection| put one 20260514A GT cell on a
    # positive peak 21 samples away.
    main = (core * keep[..., None]).pow(2).sum(-1).argmax(1)
    w = M[torch.arange(n_units, device=device), main]                         # (n_units, L)
    c0 = S + ops['nt0min']
    win = w[:, c0 - S:c0 + S + 1]
    shift = win.argmin(1) - S                                                 # sample of the trough - nt0min
    ar = torch.arange(nt, device=device)
    gi = (S + shift)[:, None] + ar[None]                                      # (n_units, nt)
    E = torch.gather(M, 2, gi[:, None, :].expand(n_units, n_chan, nt)).contiguous()
    good = cnt >= MIN_SPIKES
    E[~good] = 0
    nk = keep[good].sum(1).float()
    logger.info(f'ei_peel EIs ({mode}): {int(good.sum())} of {n_units} units from {int(cnt.sum())} spikes in '
                f'{np.unique(ib[idx]).size} batches; significant channels per unit median '
                f'{float(nk.median()) if nk.numel() else 0:.0f}; |shift| median '
                f'{float(shift[good].abs().float().median()) if good.any() else 0:.0f} samples')
    return E, cnt


def _xcorr(E, chunk=256, reduce_max=False):
    """out[i, j, t] = sum_c,tau E[i, c, tau] * E[j, c, tau - nt + t], t in 0..2nt (j shifted by t - nt).

    reduce_max: return out.amax(-1) (n, n) without holding the full (n, n, 2nt+1) tensor.
    Full fp32: TF32 (cuDNN's default on Ampere+) rounds these 10^4-10^5-term sums to ~1e-3.
    """
    n, C, nt = E.shape
    Ep = torch.nn.functional.pad(E, (nt, nt))
    out = torch.empty((n, n) if reduce_max else (n, n, 2 * nt + 1), device=E.device)
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        for a in range(0, n, chunk):
            # conv1d(input=Ep[j], weight=E[i])[j, i, t] = sum E[j, c, t + tau - nt] E[i, c, tau]
            r = conv1d(Ep, E[a:a + chunk]).permute(1, 0, 2)
            out[a:a + chunk] = r.amax(-1) if reduce_max else r
    return out


def _xcorr_row(E, k):
    """max over lags of the cross-correlation of E[k] with every E[j] (n,)."""
    n, C, nt = E.shape
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        return conv1d(torch.nn.functional.pad(E, (nt, nt)), E[k:k + 1])[:, 0].amax(-1)


def merge_eis(E, cnt, merge_cos=MERGE_COS):
    """cnt: each unit's TOTAL spike count (not the sampled count, which is capped and ties).

    Greedy: units in descending spike count absorb any later unit with lagged EI cosine >= merge_cos.

    Returns keep (bool, n_units) and target (int, n_units: the unit each one's spikes now belong to).
    """
    n = E.shape[0]
    nrm = (E ** 2).sum((1, 2)).sqrt()
    alive = (cnt >= MIN_SPIKES) & (nrm > 0)
    cc = _xcorr(E, reduce_max=True) / (nrm[:, None] * nrm[None, :] + 1e-12)
    cc = cc.cpu().numpy()
    order = np.argsort(-cnt.cpu().numpy(), kind='stable')
    alive = alive.cpu().numpy()
    target = np.arange(n)
    absorbed = np.zeros(n, bool)
    for i in order:
        if not alive[i] or absorbed[i]:
            continue
        m = (cc[i] >= merge_cos) & alive & ~absorbed
        m[i] = False
        absorbed |= m
        target[m] = i
    keep = alive & ~absorbed
    logger.info(f'ei_peel merge: {int(absorbed.sum())} units absorbed at EI cosine >= {merge_cos}; '
                f'{int(keep.sum())} EIs kept')
    return keep, target


def ei_matching(ops, E):
    """ctc and unit cache for run_matching from full-resolution EIs E (n, n_chan, nt)."""
    nt = ops['nt']
    nm = (E ** 2).sum((1, 2))
    s = nm.clamp_min(1e-30).rsqrt()
    ctc = _xcorr(E) * s.view(-1, 1, 1)          # ctc[i, j, t] = s_i * sum E_i[c, nt - t + tau] E_j[c, tau]
    device = E.device
    cache = {
        's': s, 'Es': (E * s.view(-1, 1, 1)).contiguous(), 'U_time': E.contiguous(),
        'W': ops['wPCA'].contiguous(),
        'trange': torch.arange(-nt, nt + 1, device=device),
        'tiwave': torch.arange(-(nt // 2), nt // 2 + 1, device=device),
    }
    return ctc, cache


def ei_pcs(ops, E):
    """PC projection of E in the (n_units, n_pcs, n_chan) layout extract's U uses (for iU, iCC and tF)."""
    return torch.einsum('ict,pt->ipc', E, ops['wPCA'])
