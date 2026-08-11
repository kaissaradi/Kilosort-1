"""CPU-safe identity tests for datashift helpers."""
import numpy as np
from scipy.sparse import coo_matrix

from kilosort.datashift import bin_spikes


def _reference_bin_spikes(ops, st):
    """Historical per-batch coo_matrix loop (pre vectorized bincount)."""
    ymin = ops['yc'].min()
    ymax = ops['yc'].max()
    dd = ops['binning_depth']
    dmin = ymin - 1
    dmax = 1 + np.ceil((ymax - dmin) / dd).astype('int32')
    Nbatches = ops['Nbatches']
    batch_id = st[:, 4].copy()
    F = np.zeros((Nbatches, dmax, 20))
    for t in range(ops['Nbatches']):
        ix = (batch_id == t).nonzero()[0]
        sst = st[ix]
        dep = sst[:, 1] - dmin
        amp = np.log10(np.minimum(99, sst[:, 2])) - np.log10(ops['Th_universal'])
        amp = amp / (np.log10(100) - np.log10(ops['Th_universal']))
        rows = (dep / dd).astype('int32')
        cols = (1e-5 + amp * 20).astype('int32')
        cou = np.ones(len(ix))
        M = coo_matrix((cou, (rows, cols)), (dmax, 20))
        F[t] = np.log2(1 + M.todense())
    ysamp = dmin + dd * np.arange(dmax) - dd / 2
    return F, ysamp


def test_bin_spikes_matches_per_batch_loop():
    rng = np.random.default_rng(0)
    ops = {
        'yc': np.linspace(0, 1000, 50),
        'binning_depth': 20.0,
        'Nbatches': 12,
        'Th_universal': 8,
    }
    n = 5000
    st = np.zeros((n, 6))
    st[:, 1] = rng.uniform(0, 1000, n)
    st[:, 2] = rng.uniform(8, 50, n)
    st[:, 4] = rng.integers(0, 12, n)
    got_F, got_y = bin_spikes(ops, st)
    exp_F, exp_y = _reference_bin_spikes(ops, st)
    np.testing.assert_allclose(got_F, exp_F)
    np.testing.assert_allclose(got_y, exp_y)


def test_bin_spikes_empty():
    ops = {
        'yc': np.array([0.0, 100.0]),
        'binning_depth': 20.0,
        'Nbatches': 3,
        'Th_universal': 8,
    }
    st = np.zeros((0, 6))
    F, ysamp = bin_spikes(ops, st)
    assert F.shape[0] == 3
    assert F.shape[2] == 20
    assert np.all(F == 0)
    assert ysamp.ndim == 1
