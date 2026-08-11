import numpy as np
import torch

from scipy.sparse import csr_matrix

from kilosort.clustering_qr import (
    Mstats,
    kmeans_plusplus,
    mean_cluster_templates,
    neigh_mat,
    x_centers,
)
from kilosort.hierarchical import Mstats as hierarchical_Mstats
from kilosort.hierarchical import prepare as hierarchical_prepare
from kilosort.io import load_probe
from kilosort.utils import PROBE_DIR


def _reference_mean_cluster_templates(Xd, iclust, ichan, n_chan, n_pcs):
    """Historical per-label boolean-mask mean used in clustering_qr.run."""
    if isinstance(iclust, torch.Tensor):
        iclust_t = iclust
    else:
        iclust_t = torch.as_tensor(iclust)
    Nfilt = int(iclust_t.max().item()) + 1 if iclust_t.numel() else 0
    W = torch.zeros((Nfilt, n_chan, n_pcs), dtype=Xd.dtype)
    for j in range(Nfilt):
        w = Xd[iclust_t == j].mean(0)
        W[j, ichan, :] = torch.reshape(w, (-1, n_pcs))
    return W


def test_mean_cluster_templates_matches_mask_loop():
    rng = np.random.default_rng(11)
    n_spikes, n_feat, n_chan_local, n_pcs = 2000, 48, 8, 6
    # Flattened local features (merge_dim=True style)
    Xd = torch.from_numpy(rng.standard_normal((n_spikes, n_feat)).astype(np.float32))
    iclust = rng.integers(0, 25, size=n_spikes).astype(np.int64)
    # Leave a gap so empty-cluster NaN path is exercised
    iclust[iclust == 7] = 8
    ichan = torch.arange(n_chan_local, dtype=torch.long)
    n_chan = 64

    got = mean_cluster_templates(Xd, iclust, ichan, n_chan, n_pcs)
    ref = _reference_mean_cluster_templates(Xd, iclust, ichan, n_chan, n_pcs)
    assert got.shape == ref.shape
    # NaNs for empty labels; equal elsewhere
    both_nan = torch.isnan(got) & torch.isnan(ref)
    assert torch.equal(torch.nan_to_num(got, nan=0.0), torch.nan_to_num(ref, nan=0.0))
    assert both_nan.any()  # gap at label 7


def test_mean_cluster_templates_accepts_torch_iclust():
    Xd = torch.randn(100, 12)
    iclust = torch.zeros(100, dtype=torch.long)
    iclust[40:] = 1
    ichan = torch.tensor([2, 3])
    W = mean_cluster_templates(Xd, iclust, ichan, n_chan=8, n_pcs=6)
    assert W.shape == (2, 8, 6)
    assert torch.isfinite(W[:, [2, 3], :]).all()


def test_neigh_mat_drops_self_edges_like_historical_zeroing():
    """CSR without self edges must match post-zeroing dense adjacency values."""
    rng = np.random.default_rng(0)
    n_samples, dim, n_neigh = 200, 12, 8
    Xd = rng.standard_normal((n_samples, dim)).astype(np.float32)
    kn, M = neigh_mat(Xd, nskip=1, n_neigh=n_neigh, max_sub=None, device=torch.device('cpu'))
    # Historical: ones at kn then zero diagonal of subset (every row when nskip=1)
    ref = np.zeros((n_samples, n_samples), dtype=np.float32)
    for i in range(n_samples):
        ref[i, kn[i]] = 1.0
    np.fill_diagonal(ref, 0.0)
    got = M.toarray()
    np.testing.assert_array_equal(got, ref)
    assert kn.shape == (n_samples, n_neigh)


def test_maketree_empty_and_single_label():
    from kilosort.hierarchical import maketree
    from scipy.sparse import csr_matrix
    # Empty labels
    M0 = csr_matrix((0, 0), dtype=np.float32)
    xt, ts, mc = maketree(M0, np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    assert xt.shape == (0, 3)
    assert ts.shape == (0, 3)
    assert mc == []
    # Single cluster
    M1 = csr_matrix((3, 2), dtype=np.float32)
    xt, ts, mc = maketree(M1, np.zeros(3, dtype=np.int64), np.array([0, 0], dtype=np.int64))
    assert xt.shape == (0, 3)
    assert mc == [[0]]


def test_find_merges_zero_cneg_stays_finite():
    """Incremental crat update must not inject NaN when cneg row is zero."""
    from kilosort.hierarchical import find_merges

    nc = 4
    # Positive off-diagonal so merges progress; zero one cneg column entirely
    # after a contrived setup that forces divide-by-zero on a merged row.
    cc = np.array(
        [
            [1.0, 0.5, 0.1, 0.0],
            [0.5, 1.0, 0.2, 0.0],
            [0.1, 0.2, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    cneg = np.array(
        [
            [1.0, 0.5, 0.1, 0.0],
            [0.5, 1.0, 0.2, 0.0],
            [0.1, 0.2, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],  # zero row/col — divide would NaN
        ],
        dtype=np.float64,
    )
    crat = np.divide(cc, cneg, out=np.zeros_like(cc), where=cneg != 0)
    crat = crat - np.diag(np.diag(crat)) - np.eye(nc)
    xtree, tstat = find_merges(crat.copy(), cc.copy(), cneg.copy())
    assert xtree.shape == (nc - 1, 3)
    assert np.isfinite(tstat).all()


def test_mstats_zero_adjacency_finite():
    """Empty neighbor graph must not yield NaN ki/kj (0/0)."""
    M = csr_matrix((5, 3), dtype=np.float32)
    m, ki, kj = Mstats(M, device=torch.device('cpu'))
    assert float(m) == 0.0
    assert torch.isfinite(ki).all()
    assert torch.isfinite(kj).all()
    assert torch.count_nonzero(ki) == 0
    assert torch.count_nonzero(kj) == 0

    m2, ki2, kj2 = hierarchical_Mstats(M)
    assert m2 == 0.0
    assert np.all(np.isfinite(ki2))
    assert np.all(np.isfinite(kj2))


def test_hierarchical_prepare_zero_m_no_divide():
    M = csr_matrix((4, 2), dtype=np.float32)
    iclust = np.array([0, 0, 1, 1], dtype=np.int64)
    iclust0 = np.array([0, 1], dtype=np.int64)
    cc, cneg = hierarchical_prepare(M, iclust, iclust0)
    assert np.all(np.isfinite(cc))
    assert np.all(np.isfinite(cneg))
    assert np.allclose(cneg, 0.001)


def test_kmeans_plusplus_handles_low_rank_residual():
    """Identical / low-rank rows used to crash multinomial (n_pos < ntry)."""
    device = torch.device('cpu')
    # 1200 copies of the same 6-D feature → residual mass collapses fast
    row = torch.randn(6, device=device)
    Xg = row.unsqueeze(0).expand(1200, -1).contiguous()
    # Need the internal vtot; kmeans_plusplus expects already-augmented Xg
    # as used by cluster() — append ones column matching production call site.
    # Looking at callers: kmeans_plusplus(Xg, ...) where Xg comes from cluster
    # after feature prep. Call with raw features: function uses vtot from Xg.
    # Read kmeans_plusplus signature usage...
    # Actually kmeans_plusplus computes vtot from Xg inside? Check.
    iclust = kmeans_plusplus(Xg, niter=50, seed=1, device=device)
    assert iclust.shape == (1200,)
    assert int(iclust.min()) >= 0
    assert int(iclust.max()) < 50


def random_np2(n_chans=384, n_shanks=4):
    # Generates xc,yc for a probe containing *all* neuropixels 2 contact positions,
    # then randomly subsamples from those positions to get a probe layout
    # corresponding to 384-channel output data.

    # 12um square contacts with 32um lateral spacing,
    # 15um vertical spacing,
    # 1280 contacts per shank

    # Want alternating 6um, 38um for lateral positions
    xc0 = np.empty(1280)
    xc0[::2] = 6
    xc0[1::2] = 38
    # Then add 250um for each additional shank
    xc = np.concatenate([xc0 + (250*i) for i in range(4)])

    # For vertical positions, start at 6 and increase by 15
    yc0 = (np.arange(640)*15) + 6
    # Each position appears twice (two columns on each shank)
    yc0 = np.repeat(yc0, 2)
    yc = np.concatenate([yc0 for i in range(4)])

    # Repeat 0 1280 times, then repeat 1 1280 times, etc
    kcoords = np.repeat(np.arange(4), 1280)

    # Pick n_chans out of n_shanks
    shanks_used = np.random.choice(range(4), n_shanks, replace=False)
    shank_indices = np.argwhere(np.isin(kcoords, shanks_used))[:,0]
    contact_indices = np.random.choice(shank_indices, n_chans, replace=False)

    return {'xc': xc[contact_indices], 'yc': yc[contact_indices]}


class TestCenters:
    ops = {'dminx': 32}

    def test_linear(self, data_directory):
        # NOTE: The `data_directory` argument is only there to make sure probes are
        # downloaded before these tests are run.
        probe = load_probe(PROBE_DIR/'Linear16x1_test.mat')
        self.ops['xc'] = probe['xc']
        centers = x_centers(self.ops)
        # X positions are all 1um
        assert len(centers) == 1
        assert np.abs(centers[0] - 1) < 5

    def test_np1(self):
        probe = load_probe(PROBE_DIR/'NeuroPix1_default.mat')
        self.ops['xc'] = probe['xc']
        centers = x_centers(self.ops)
        # One shank from 11um to 59um, should be 1 center near 35um
        assert len(centers) == 1
        assert np.abs(centers[0] - 35) < 5

    def test_np2_1shank(self):
        probe = load_probe(PROBE_DIR/'NeuroPix2_default.mat')
        self.ops['xc'] = probe['xc']
        centers = x_centers(self.ops)
        # One shank from 0 to 32um, should be 1 center near 16um
        assert len(centers) == 1
        assert np.abs(centers[0] - 16) < 5

    def test_np2_3shank(self):
        probe = random_np2(n_shanks=3)
        self.ops['xc'] = probe['xc']
        centers = x_centers(self.ops)
        assert len(centers) == 3
        true = np.array([22, 272, 522, 772])
        for c in centers:
            # Each center is within 2 microns of exactly one true center
            print(f'center: {c}')
            assert (np.abs(c - true) < 5).sum() == 1

    def test_np2_4shank(self):
        probe = random_np2(n_shanks=4)
        self.ops['xc'] = probe['xc']
        centers = x_centers(self.ops)
        # All centers should be within 2 microns of the true values
        print(f'centers: {centers}')
        assert np.allclose(np.sort(centers), np.sort([22, 272, 522, 772]), atol=5)
