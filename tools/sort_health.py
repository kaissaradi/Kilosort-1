"""Reference-free health of a sort: what a good sorter should get right, measured on the sort and its raw data alone.

    python tools/sort_health.py SORT_DIR RAW [--json OUT.json]

Reports (no ground truth needed):
  duplicates  events detected in two units at one fixed lag with the same full waveform (kilosort.neuron_merge detector):
              number of unit pairs and of copy spikes (one neuron counted twice).
  splits      unit pairs that are one neuron by full waveform AND cross-refractory with no shared spikes.
  junk        units holding a separate population of small, poorly matching events: 2-component mixture on (raw size,
              cosine to the unit's mean raw waveform), minor component clearly separate (Ashman D >= 2 on cosine), mean
              cosine < 0.8 x the main one's and mean size < 0.7. Reported: units and estimated spikes.
  refractory  units with > 1 % of inter-spike intervals < 1.5 ms.
RAW is the Litke folder or the flat int16 .bin the sort read.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kilosort.neuron_merge import NeuronMergeSettings, RawReader, neuron_merge  # noqa: E402


def junk_units(st, sc, n, reader, max_per_unit=400, seed=0):
    """Per unit: raw size and cosine of up to max_per_unit spikes to the unit's mean raw waveform (12 strongest channels)."""
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(seed); L, R, NCH = 20, 30, 12; fs = 20000
    sos = butter(4, 300, btype='high', fs=fs, output='sos')
    n_s = reader.n_samples
    o = np.argsort(sc, kind='stable'); b = np.searchsorted(sc[o], np.arange(n + 1))
    pick = np.concatenate([rng.choice(o[b[u]:b[u + 1]], min(max_per_unit, b[u + 1] - b[u]), replace=False)
                           for u in range(n) if b[u + 1] > b[u]])
    pick = pick[(st[pick] > 1000) & (st[pick] < n_s - 1000)]
    pick = pick[np.argsort(st[pick])]
    snips = {}
    CH = 400000
    for s0 in range(0, n_s, CH):
        k = pick[(st[pick] >= s0) & (st[pick] < s0 + CH)]
        if k.size == 0: continue
        a0 = max(s0 - 400, 0); X = sosfiltfilt(sos, reader.read(a0, min(s0 + CH + 400, n_s)), axis=0).astype(np.float32)
        for i in k:
            t = st[i] - a0; snips[i] = X[t - L:t + R]
    by = {}
    for i, s in snips.items(): by.setdefault(int(sc[i]), []).append(s)
    n_units = n_spk = 0; detail = []
    for u, S in by.items():
        if len(S) < 200: continue
        S = np.stack(S); M = S.mean(0); ch = np.argsort(np.ptp(M, 0))[::-1][:NCH]
        T = M[:, ch]; tt = (T * T).sum(); x = S[:, :, ch]
        d = (x * T).sum((1, 2)); a = d / tt; c = d / np.sqrt((x * x).sum((1, 2)) * tt + 1e-9)
        F = np.c_[a, c]; g = GaussianMixture(2, random_state=0).fit(F); m = g.means_
        sd = np.sqrt(np.array([np.diag(cv) for cv in g.covariances_]))
        main, minor = (0, 1) if m[0, 1] >= m[1, 1] else (1, 0)
        D = np.sqrt(2) * abs(m[0, 1] - m[1, 1]) / np.sqrt(sd[0, 1] ** 2 + sd[1, 1] ** 2)
        if D >= 2 and m[minor, 1] < 0.8 * m[main, 1] and m[minor, 0] < 0.7:
            frac = float(g.weights_[minor]); n_units += 1; n_spk += frac * int((sc == u).sum())
            detail.append(dict(unit=u, frac=round(frac, 3), minor_size=round(float(m[minor, 0]), 2), minor_cos=round(float(m[minor, 1]), 2)))
    return n_units, int(n_spk), detail


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('sort_dir'); ap.add_argument('raw'); ap.add_argument('--json')
    a = ap.parse_args(); sd = Path(a.sort_dir)
    ops = np.load(sd / 'ops.npy', allow_pickle=True).item()
    for k in ('xc', 'yc', 'chanMap'):
        if k in ops and hasattr(ops[k], 'cpu'): ops[k] = ops[k].cpu().numpy()
    st = np.load(sd / 'spike_times.npy').ravel().astype(np.int64); sc = np.load(sd / 'spike_clusters.npy').ravel().astype(np.int64)
    o = np.argsort(st, kind='stable'); st, sc = st[o], sc[o]
    n = int(sc.max()) + 1; pos = np.load(sd / 'channel_positions.npy')
    _, _, rep = neuron_merge(st, sc, ops, a.raw, NeuronMergeSettings(), channel_positions=pos)
    ju, js, jd = junk_units(st, sc, n, RawReader(a.raw, ops))
    viol = []
    for u in range(n):
        t = st[sc == u]
        if t.size > 100: viol.append(np.mean(np.diff(t) < 30))
    viol = np.array(viol)
    out = dict(sort=str(sd), units=n, spikes=int(st.size),
               duplicate_pairs=rep['n_duplicate_joins'], duplicate_spikes=rep['n_dropped_copies'],
               duplicate_spike_frac=rep['n_dropped_copies'] / st.size, split_pairs=rep['n_split_joins'],
               junk_units=ju, junk_spikes_est=js, junk_spike_frac=js / st.size, junk_detail=jd,
               refractory_bad_units=int((viol > 0.01).sum()), refractory_units_tested=int(viol.size))
    print(f"units {n} | spikes {st.size:,} | duplicates: {out['duplicate_pairs']} pairs, {out['duplicate_spikes']:,} spikes "
          f"({out['duplicate_spike_frac']:.1%}) | splits: {out['split_pairs']} | junk: {ju} units, ~{js:,} spikes "
          f"({out['junk_spike_frac']:.1%}) | refractory >1%: {out['refractory_bad_units']} of {out['refractory_units_tested']}")
    if a.json: Path(a.json).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
