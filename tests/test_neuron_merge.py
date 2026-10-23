"""Synthetic checks for kilosort.neuron_merge (post-sort neuron-level identity).

One neuron X is split across units 0 and 1 (alternate spikes: a cross-refractory
pair); unit 3 is an axonal-style copy of every X spike 10 samples later; unit 2
is an unrelated neuron Y with its own waveform and an independent train.
Expected: 0 + 1 joined as a split; 3 found as a duplicate, its copies dropped
(drop-only), which empties it; 2 left alone.
"""
import numpy as np

from kilosort.neuron_merge import NeuronMergeSettings, neuron_merge, compare_waveforms, _shift

FS = 20000


def _train(rng, rate, dur_s, refr=60):
    t, out = 0, []
    while True:
        t += refr + int(rng.exponential(FS / rate))
        if t >= dur_s * FS - 200:
            return np.array(out, np.int64)
        if t > 200:
            out.append(t)


def _make(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    n_ch, dur = 16, 60
    x = np.arange(n_ch) % 4 * 30.0; y = np.arange(n_ch) // 4 * 30.0
    n_s = dur * FS
    data = rng.normal(0, 4, (n_s, n_ch))
    tt = np.arange(-20, 30)
    wav = lambda c, w: -np.exp(-(tt / w) ** 2) * c        # noqa: E731
    Xamp = np.exp(-np.hypot(x - 0, y - 0) / 40) * 120      # neuron X near channel 0
    Yamp = np.exp(-np.hypot(x - 90, y - 90) / 40) * 120     # neuron Y near channel 15
    tx, ty = _train(rng, 20, dur), _train(rng, 15, dur)
    for t in tx:
        data[t - 20:t + 30] += np.outer(wav(1, 4), Xamp)
    for t in ty:
        data[t - 20:t + 30] += np.outer(wav(1, 6), Yamp)
    raw = tmp_path / 'raw.bin'
    data.astype(np.int16).tofile(raw)
    st = np.r_[tx[0::2], tx[1::2], ty, tx + 10]
    sc = np.r_[np.zeros(tx[0::2].size), np.ones(tx[1::2].size), np.full(ty.size, 2), np.full(tx.size, 3)].astype(np.int32)
    o = np.argsort(st, kind='stable')
    ops = {'settings': {'fs': FS, 'n_chan_bin': n_ch, 'acg_threshold': 0.2, 'ccg_threshold': 0.25},
           'chanMap': np.arange(n_ch), 'xc': x, 'yc': y, 'duplicate_spike_bins': 7}
    return st[o], sc[o], ops, raw, tx


def test_shift_no_wraparound():
    B = np.arange(10.0)[None]
    assert np.array_equal(_shift(B, 3)[0], [0, 0, 0, 0, 1, 2, 3, 4, 5, 6])
    assert np.array_equal(_shift(B, -3)[0], [3, 4, 5, 6, 7, 8, 9, 0, 0, 0])


def test_compare_waveforms_lag_sign():
    rng = np.random.default_rng(1)
    A = np.zeros((4, 80)); A[:, 30] = -10; A += rng.normal(0, 0.01, A.shape)
    B = _shift(A, -10)                        # b's mean aligned 10 samples LATER: a's event appears 10 earlier
    c, amp, lag = compare_waveforms(A, B, range(0, 16), 0.1)
    assert lag == 10 and c > 0.99 and abs(amp - 1) < 0.05


def test_split_and_duplicate_joined_unrelated_kept(tmp_path):
    st, sc, ops, raw, tx = _make(tmp_path)
    cfg = NeuronMergeSettings(n_chunks=6, chunk_s=10.0, min_spikes=50)
    pos = np.c_[ops['xc'], ops['yc']]
    t2, c2, rep = neuron_merge(st, sc, ops, raw, cfg, channel_positions=pos)
    pairs = {(d['a'], d['b']) for d in rep['splits']}
    assert (0, 1) in pairs, rep['splits']
    assert any(d['copy'] in (0, 1, 3) and d['keeper'] in (0, 1, 3) for d in rep['duplicates']), rep['duplicates']
    # unit 2 (neuron Y) is never joined
    assert all(2 not in (d['a'], d['b']) for d in rep['splits'])
    assert all(2 not in (d['keeper'], d['copy']) for d in rep['duplicates'])
    assert rep['n_units_out'] == 2
    # the joined X unit holds each X spike once
    xu = c2[np.argmin(np.abs(t2 - tx[5]))]
    tX = t2[c2 == xu]
    assert abs(tX.size - tx.size) <= 0.02 * tx.size, (tX.size, tx.size)
