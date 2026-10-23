"""Neuron-level identity after a finished sort (post-sort; off by default).

Why this exists
---------------
Kilosort 4 gives each unit a LOCAL identity: templates are born per clustering
tile (``clustering_qr.run``), are zero outside ``nearest_chans`` around their
main electrode, the peel picks a unit by shape on those truncated templates,
and the final merge (``template_matching.merging_function``) runs its CCG test
only for pairs whose truncated-template similarity is >= 0.5.  Two units of
one neuron whose main electrodes are more than about one pitch apart score
~0 and are never tested (measured on 20260724A/data011: units 883/627 have
template similarity 0.06 but a full raw-waveform cosine of 0.972, size ratio
0.99 and a cross-CCG R12 of 0.03).  An axonal copy is the extreme case: the
same spike, seen 0.5 mm away on the axon, is a second local identity.

What it does
------------
It keeps every spike of the sort and joins units that are one neuron by BOTH
physical signatures, never by one alone:

* split:     the two trains are cross-refractory (Kilosort's own
             ``CCG.check_CCG`` with the sort's ``ccg_threshold``) AND share
             < ``split_max_shared`` of their spikes at any lag within
             +-``dup_maxlag_ms`` (CCG_metrics ignores the +-0.5 ms centre bin,
             so a double detection 0.2-0.5 ms apart also passes the CCG), AND the
             full raw mean waveforms match (cosine >= ``split_cos``) at a
             similar size (``size_lo`` .. ``size_hi``).  Neighbours that avoid
             each other through detection dead time fail the waveform test.
* duplicate: >= ``dup_frac`` of the smaller train repeats at one lag within
             +-``dup_maxlag_ms`` (``_peak_shared_fraction_with_matches``),
             AND the full mean waveforms match at that lag (cosine >=
             ``dup_cos``).  Drop-only: the copies are deleted from the copy
             unit; its other spikes stay as its
             own unit.  Importing them fused neurons on 20260514A/data003.
             The keeper is the unit whose EI trough sits at its spike time (an
             axonal copy or a misaligned twin is offset by its lag); on a tie,
             the unit with the lower refractory contamination.

Waveforms come from the raw data (high-pass 300 Hz, zero phase), sampled from
``n_chunks`` windows spread over the whole recording, compared on the union
of each unit's channels with peak-to-peak >= ``chan_frac`` of its own maximum.
"""
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from kilosort import CCG
from kilosort.postprocessing import remove_duplicates
from kilosort.template_matching import _peak_shared_fraction_with_matches

logger = logging.getLogger(__name__)

WAVE_L, WAVE_R = 30, 50          # samples before / after the spike time


@dataclass
class NeuronMergeSettings:
    split_cos: float = 0.9
    size_lo: float = 0.67
    size_hi: float = 1.5
    split_reach_pitches: float = 3.0
    split_max_lag: int = 15          # samples, waveform alignment search for splits
    split_max_shared: float = 0.1    # a split shares no spikes; more than this is a duplicate question
    dup_frac: float = 0.5
    dup_cos: float = 0.8
    dup_maxlag_ms: float = 2.5
    dup_prefilter_frac: float = 0.3
    chan_frac: float = 0.1
    n_wave: int = 600
    n_chunks: int = 30
    chunk_s: float = 10.0
    min_spikes: int = 100
    highpass_hz: float = 300.0
    seed: int = 0


class RawReader:
    """Read sort channels from a Litke folder or a flat int16 file, float32."""

    def __init__(self, raw, ops):
        raw = Path(raw)
        self.chan_map = np.asarray(ops['chanMap'] if 'chanMap' in ops
                                   else ops['probe']['chanMap']).astype(np.int64)
        if raw.is_dir():
            from kilosort.litke import LitkeRecording
            self.rec = LitkeRecording(raw)
            self.n_samples = int(self.rec.n_samples)
            self._read = lambda a, b: np.asarray(self.rec[a:b], np.float32)
        else:
            n_chan = int(ops['settings']['n_chan_bin'])
            mm = np.memmap(raw, np.int16, 'r')
            mm = mm[: mm.size // n_chan * n_chan].reshape(-1, n_chan)
            self.n_samples = int(mm.shape[0])
            self._read = lambda a, b: np.asarray(mm[a:b], np.float32)

    def read(self, a, b):
        return self._read(a, b)[:, self.chan_map]


def mean_waveforms(spike_times, spike_clusters, n_units, reader, cfg):
    """Mean high-passed waveform (n_units, n_chan, L+R) from spikes spread over the recording."""
    rng = np.random.default_rng(cfg.seed)
    fs = 20000.0
    n_s = reader.n_samples
    clen = int(cfg.chunk_s * fs)
    if n_s <= cfg.n_chunks * clen:
        starts = np.arange(0, n_s, clen)
    else:
        starts = np.linspace(0, n_s - clen, cfg.n_chunks).astype(np.int64)
    pad = 400
    lo_ok, hi_ok = WAVE_L + pad, WAVE_R + pad
    # spikes inside the chunks, then at most n_wave per unit
    sel = np.zeros(spike_times.size, bool)
    for s0 in starts:
        a, b = np.searchsorted(spike_times, [s0 + lo_ok, s0 + clen - hi_ok])
        sel[a:b] = True
    idx = np.flatnonzero(sel)
    keep = []
    for u in range(n_units):
        iu = idx[spike_clusters[idx] == u]
        if iu.size > cfg.n_wave:
            iu = rng.choice(iu, cfg.n_wave, replace=False)
        keep.append(iu)
    keep = np.sort(np.concatenate(keep)) if keep else np.zeros(0, np.int64)
    sos = butter(4, cfg.highpass_hz, btype='high', fs=fs, output='sos')
    n_chan = reader.chan_map.size
    acc = np.zeros((n_units, n_chan, WAVE_L + WAVE_R), np.float64)
    cnt = np.zeros(n_units, np.int64)
    for s0 in starts:
        a0, a1 = max(s0 - pad, 0), min(s0 + clen + pad, n_s)
        a, b = np.searchsorted(spike_times[keep], [s0 + lo_ok, s0 + clen - hi_ok])
        if b <= a:
            continue
        X = sosfiltfilt(sos, reader.read(a0, a1), axis=0).astype(np.float32)
        for i in keep[a:b]:
            t = int(spike_times[i]) - a0
            acc[spike_clusters[i]] += X[t - WAVE_L:t + WAVE_R].T
            cnt[spike_clusters[i]] += 1
    M = acc / np.maximum(cnt, 1)[:, None, None]
    return M.astype(np.float32), cnt


def _shift(B, lag):
    """B(t - lag) with zero fill (no wrap-around)."""
    out = np.zeros_like(B)
    if lag > 0:
        out[:, lag:] = B[:, :-lag]
    elif lag < 0:
        out[:, :lag] = B[:, -lag:]
    else:
        out[:] = B
    return out


def compare_waveforms(Ma, Mb, lags, chan_frac):
    """Best cosine over `lags` on the union of both units' significant channels.

    Returns (cosine, size ratio |b|/|a|, best lag).  Mb is shifted by `lag`, so
    a positive lag means b's waveform comes later than a's.
    """
    pa, pb = np.ptp(Ma, axis=1), np.ptp(Mb, axis=1)
    ch = (pa >= chan_frac * pa.max()) | (pb >= chan_frac * pb.max())
    A = Ma[ch].astype(np.float64)
    na = np.sqrt((A * A).sum())
    best = (-1.0, np.nan, 0)
    for lag in lags:
        B = _shift(Mb[ch].astype(np.float64), int(lag))
        nb = np.sqrt((B * B).sum())
        c = float((A * B).sum() / (na * nb + 1e-12))
        if c > best[0]:
            best = (c, float(nb / (na + 1e-12)), int(lag))
    return best


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def _duplicate_candidates(st, sc, n_units, ok, maxlag, bin_samples=10, max_spikes=3_000_000):
    """Cheap prefilter for duplicates: binned coincidence counts for all unit pairs.

    Counts in 0.5 ms bins at every lag within +-maxlag, summed over two adjacent
    lag bins (a copy jittering across a bin edge is not halved). Returns
    {(a, b): fraction of the smaller train}. Uses an early window of at most
    `max_spikes` spikes (spike times are sorted).
    """
    from scipy import sparse
    take = np.flatnonzero(ok[sc] & (st >= 0))[:max_spikes]       # a few first-batch times can be negative
    t, c = st[take] // bin_samples, sc[take]
    nb = int(t.max()) + 2
    A = sparse.csr_matrix((np.ones(t.size, np.float32), (c, t)), shape=(n_units, nb))
    n = np.asarray(A.sum(1)).ravel()
    K = maxlag // bin_samples + 1
    pos = []
    for lag in range(0, K + 1):
        Bl = sparse.csr_matrix((np.ones(t.size, np.float32), (c, np.minimum(t + lag, nb - 1))), shape=(n_units, nb))
        pos.append((A @ Bl.T).tocsr())
    seq = [pos[l].T.tocsr() for l in range(K, 0, -1)] + pos          # lags -K..K
    best = None
    for l in range(len(seq) - 1):
        s2 = seq[l] + seq[l + 1]
        best = s2 if best is None else best.maximum(s2)
    best = best.maximum(best.T).tocoo()
    m = best.row < best.col
    r, col, v = best.row[m], best.col[m], best.data[m]
    frac = v / np.maximum(np.minimum(n[r], n[col]), 1)
    return {(int(a), int(b)): float(f) for a, b, f in zip(r, col, frac)}


def neuron_merge(spike_times, spike_clusters, ops, raw, cfg=None, channel_positions=None):
    """Join units that are one neuron. Returns (times, clusters, report)."""
    cfg = cfg or NeuronMergeSettings()
    se = ops['settings']
    fs = float(se.get('fs', 20000))
    o = np.argsort(spike_times, kind='stable')
    st = np.asarray(spike_times, np.int64)[o]
    sc = np.asarray(spike_clusters, np.int64)[o]
    n_units = int(sc.max()) + 1
    counts = np.bincount(sc, minlength=n_units)
    ok = counts >= cfg.min_spikes
    reader = RawReader(raw, ops)
    M, ncnt = mean_waveforms(st, sc, n_units, reader, cfg)
    ok &= ncnt >= 30
    ptp = np.ptp(M, axis=2)
    main = ptp.argmax(1)
    # EI trough offset: time of the minimum on the largest-p2p channel, relative to the unit's spike time
    trough_off = np.array([int(np.argmin(M[u, main[u]])) - WAVE_L for u in range(n_units)])
    # Kilosort's own refractory contamination per unit (R12; lower = cleaner), used only to break keeper ties
    is_ref, contam = CCG.refract(sc, st / fs, acg_threshold=se['acg_threshold'], ccg_threshold=se['ccg_threshold'])
    contam = np.nan_to_num(np.asarray(contam, float), nan=np.inf)
    pos = channel_positions
    if pos is None:
        pos = np.c_[np.asarray(ops['xc']), np.asarray(ops['yc'])]
    D = np.hypot(*(pos[:, None] - pos[None]).transpose(2, 0, 1))
    pitch = float(np.median(np.where(D > 0, D, np.inf).min(1)))
    idx = {u: np.flatnonzero(sc == u) for u in range(n_units) if ok[u]}
    uf = _UF(n_units)
    report = dict(settings=asdict(cfg), pitch_um=pitch, n_units_in=n_units, splits=[], duplicates=[])

    # splits: cross-refractory + same waveform, near pairs
    reach = cfg.split_reach_pitches * pitch + 1e-6
    near = [(a, b) for a in idx for b in idx if a < b and D[main[a], main[b]] <= reach]
    for a, b in near:
        ta, tb = st[idx[a]] / fs, st[idx[b]] / fs
        _, cross, r12 = CCG.check_CCG(ta, tb, acg_threshold=se['acg_threshold'],
                                      ccg_threshold=se['ccg_threshold'], assume_sorted=True)
        if not cross:
            continue
        # CCG_metrics zeroes the +-0.5 ms centre bin, so two units that detect the SAME spikes 0.2-0.5 ms apart
        # also look cross-refractory. A true split shares no spikes; shared spikes go to the duplicate rule.
        shared, _, _ = _peak_shared_fraction_with_matches(st[idx[a]], st[idx[b]], fs, maxlag_s=cfg.dup_maxlag_ms / 1000)
        if shared >= cfg.split_max_shared:
            continue
        c, amp, lag = compare_waveforms(M[a], M[b], range(-cfg.split_max_lag, cfg.split_max_lag + 1), cfg.chan_frac)
        if c >= cfg.split_cos and cfg.size_lo <= amp <= cfg.size_hi:
            uf.union(a, b)
            report['splits'].append(dict(a=a, b=b, cos=c, size=amp, r12=float(r12), shared=float(shared),
                                         d_um=float(D[main[a], main[b]])))

    # duplicates: shared spikes at one lag + same waveform at that lag
    maxlag = int(round(cfg.dup_maxlag_ms * fs / 1000))
    pre = _duplicate_candidates(st, sc, n_units, ok, maxlag)
    drop = np.zeros(st.size, bool)
    shift = np.zeros(st.size, np.int64)
    for (a, b), f in sorted(pre.items(), key=lambda kv: -kv[1]):
        if f < cfg.dup_prefilter_frac or not (ok[a] and ok[b]):
            continue
        frac, lag_s, matched_b = _peak_shared_fraction_with_matches(
            st[idx[a]], st[idx[b]], fs, maxlag_s=cfg.dup_maxlag_ms / 1000)
        if frac < cfg.dup_frac:
            continue
        lag = int(round(lag_s * fs))                          # t_b - t_a
        c, amp, wl = compare_waveforms(M[a], M[b], range(lag - 5, lag + 6), cfg.chan_frac)
        if c < cfg.dup_cos:
            continue
        # keeper = the unit with the larger peak (the soma side of an axonal copy)
        # Which unit keeps the shared events? The unit whose spike time marks its neuron's largest trough (EI trough
        # offset nearest 0): an axonal copy is timed on the axon arrival, a same-electrode twin on a misaligned
        # template. On 20260514A/data003 this picks the reference-timed unit in 45 of 48 duplicate cells.
        # On a tie, the CLEANER unit (lower Kilosort refractory contamination) keeps them: a mixture unit that also
        # caught the events violates the refractory period (20260501A/data000 GT 14: precision-1.00 unit 1075 lost 78 %
        # of its spikes to mixture unit 431 under a 'larger train' tie-break). Rules that failed: larger peak, larger
        # train, and 'delete from the unit where the shared spikes are a minority' (an axonal copy catches only part
        # of the soma's spikes: d003 clean 71 -> 59).
        # 0. If exactly one unit passes Kilosort's refractory test (CCG.refract is_ref), the other is a mixture:
        #    the refractory unit keeps the events (20260514A/data000 GT 17: the trough rule kept mixture unit 355,
        #    precision 0.23 for the cell, over its refractory twin 577).
        oa, ob = abs(int(trough_off[a])), abs(int(trough_off[b]))
        if bool(is_ref[a]) != bool(is_ref[b]):
            keep_u, copy_u = (a, b) if is_ref[a] else (b, a)
        elif oa != ob:
            keep_u, copy_u = (a, b) if oa < ob else (b, a)
        else:
            keep_u, copy_u = (a, b) if contam[a] <= contam[b] else (b, a)
        if copy_u == b:
            copy_mask = matched_b
        else:
            _, _, copy_mask = _peak_shared_fraction_with_matches(
                st[idx[b]], st[idx[a]], fs, maxlag_s=cfg.dup_maxlag_ms / 1000)
        # drop-only: delete the copies; the copy unit's other spikes stay its own unit, unshifted.
        # (Joining imported them; on 20260514A/data003 that fused neurons: clean 71 -> 58, contaminated 12 -> 27,
        # because a unit on an axon electrode also holds other neurons' spikes.)
        ci = idx[copy_u]
        drop[ci[copy_mask]] = True
        # The copy unit's remainder (its spikes that are NOT copies) joins the keeper only if it is the same neuron by
        # the split rule: same full waveform and size, and cross-refractory with the keeper. A mixture unit (e.g. an
        # axon-bundle unit holding other neurons) fails this and stays drop-only. (Yger 20170706_patch1: units 21/265,
        # cosine 0.99, 52 % shared, remainder cross-refractory R12 0.07: one neuron both duplicated and split.)
        rem = ci[~copy_mask]
        joined_rem = False
        if rem.size >= 50 and c >= cfg.split_cos and cfg.size_lo <= amp <= cfg.size_hi:
            tk = st[idx[keep_u]] / fs; trm = np.sort(st[rem]) / fs
            _, cross_r, _ = CCG.check_CCG(tk, trm, acg_threshold=se['acg_threshold'], ccg_threshold=se['ccg_threshold'],
                                          assume_sorted=True)
            if cross_r:
                uf.union(copy_u, keep_u); joined_rem = True
        report['duplicates'].append(dict(keeper=keep_u, copy=copy_u, frac=float(frac), lag=lag, cos=c, size=amp,
                                         keeper_offset=int(trough_off[keep_u]), copy_offset=int(trough_off[copy_u]), remainder_joined=joined_rem,
                                         d_um=float(D[main[a], main[b]]), dropped=int(copy_mask.sum())))

    roots = np.array([uf.find(u) for u in range(n_units)])
    _, newid = np.unique(roots, return_inverse=True)
    t2 = (st - shift)[~drop]
    c2 = newid[sc][~drop].astype(np.int32)
    src = o[~drop]                                   # index of each output spike in the caller's original order
    o2 = np.argsort(t2, kind='stable')
    t2, c2, src = t2[o2], c2[o2], src[o2]
    t2, c2, kept = remove_duplicates(t2, c2, dt=int(ops.get('duplicate_spike_bins', 7)))
    src = src[kept]
    _, c2 = np.unique(c2, return_inverse=True)          # dense ids (a pure-copy unit is now empty)
    c2 = c2.astype(np.int32)
    report['trough_offset'] = trough_off.tolist()
    report['_source_index'] = src
    report.update(n_units_out=int(c2.max()) + 1, n_spikes_in=int(st.size), n_spikes_out=int(t2.size),
                  n_dropped_copies=int(drop.sum()), n_split_joins=len(report['splits']),
                  n_duplicate_joins=len(report['duplicates']))
    return t2, c2, report
