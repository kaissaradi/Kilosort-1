import numpy as np
import torch

from kilosort.clustering_qr import mean_cluster_templates, x_centers
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
