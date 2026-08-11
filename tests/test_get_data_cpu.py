"""Pure synthetic tests for get_data_cpu scatter (CPU-safe, no MEA e2e)."""
import numpy as np
import pytest
import torch

from kilosort.clustering_qr import get_data_cpu


def reference_get_data_cpu(
    ops, xy, iC, PID, tF, ycenter, xcenter, dmin=20, dminx=32,
    ix=None, merge_dim=True,
):
    """Independent historical-loop reference (not the optimized path)."""
    PID = torch.from_numpy(np.asarray(PID)).long()

    y0 = ycenter
    x0 = xcenter

    if ix is None:
        ix = torch.logical_and(
            torch.abs(xy[1] - y0) < dmin,
            torch.abs(xy[0] - x0) < dminx,
        )
    igood = ix[PID].nonzero()[:, 0]

    if len(igood) == 0:
        return None, None, None

    pid = PID[igood]
    data = tF[igood]
    nspikes, nchanraw, nfeatures = data.shape
    ichan, imap = torch.unique(iC[:, ix], return_inverse=True)
    nchan = ichan.nelement()

    dd = torch.zeros((nspikes, nchan, nfeatures), dtype=data.dtype)
    for k, j in enumerate(ix.nonzero()[:, 0]):
        ij = torch.nonzero(pid == j)[:, 0]
        dd[ij.unsqueeze(-1), imap[:, k]] = data[ij]

    if merge_dim:
        Xd = torch.reshape(dd, (nspikes, -1))
    else:
        Xd = dd

    return Xd, igood, ichan


def _synthetic_unique_imap(n_templates=40, nchanraw=16, n_chans_total=64, seed=0):
    """Nearest-chan style iC: unique channel slots per template column."""
    rng = np.random.default_rng(seed)
    cols = []
    for _ in range(n_templates):
        cols.append(rng.choice(n_chans_total, size=nchanraw, replace=False))
    iC = torch.from_numpy(np.stack(cols, axis=1).astype(np.int64))
    # Template xy on a grid so spatial ix selection is controllable
    xy = torch.stack(
        (
            torch.arange(n_templates, dtype=torch.float) * 10.0,
            torch.zeros(n_templates, dtype=torch.float),
        ),
        dim=0,
    )
    return iC, xy


def _spikes_for_templates(n_templates, nspikes, nchanraw, nfeatures, seed=1):
    rng = np.random.default_rng(seed)
    PID = rng.integers(0, n_templates, size=nspikes).astype(np.int64)
    tF = torch.from_numpy(
        rng.standard_normal((nspikes, nchanraw, nfeatures)).astype(np.float32)
    )
    return PID, tF


def test_get_data_cpu_accepts_tensor_pid():
    """PID as torch.Tensor must not crash (from_numpy only accepted ndarray)."""
    n_templates, nchanraw, nfeatures, nspikes = 10, 5, 3, 40
    iC, xy = _synthetic_unique_imap(n_templates, nchanraw, n_chans_total=40, seed=3)
    PID_np, tF = _spikes_for_templates(n_templates, nspikes, nchanraw, nfeatures, seed=4)
    ops = {}
    Xd_np, ig_np, ch_np = get_data_cpu(
        ops, xy, iC, PID_np, tF, ycenter=0.0, xcenter=50.0,
        dmin=100.0, dminx=100.0, merge_dim=True,
    )
    Xd_t, ig_t, ch_t = get_data_cpu(
        ops, xy, iC, torch.from_numpy(PID_np), tF, ycenter=0.0, xcenter=50.0,
        dmin=100.0, dminx=100.0, merge_dim=True,
    )
    if Xd_np is None:
        assert Xd_t is None
    else:
        assert torch.equal(Xd_np, Xd_t)
        assert torch.equal(ig_np, ig_t)
        assert torch.equal(ch_np, ch_t)


def test_empty_igood_returns_none():
    n_templates, nchanraw, nfeatures = 8, 4, 3
    iC, xy = _synthetic_unique_imap(n_templates, nchanraw, n_chans_total=32, seed=2)
    # All spikes on templates far from the query center
    PID = np.array([5, 6, 7], dtype=np.int64)
    tF = torch.randn(3, nchanraw, nfeatures)
    # Center near templates 0-1 only
    Xd, igood, ichan = get_data_cpu(
        {}, xy, iC, PID, tF, ycenter=0.0, xcenter=0.0, dmin=5, dminx=5
    )
    assert Xd is None and igood is None and ichan is None


def test_identity_unique_imap_merge_dim():
    n_templates, nchanraw, nfeatures, nspikes = 40, 16, 6, 5000
    iC, xy = _synthetic_unique_imap(n_templates, nchanraw, seed=3)
    PID, tF = _spikes_for_templates(n_templates, nspikes, nchanraw, nfeatures, seed=4)

    # Select a contiguous block of templates via spatial mask
    ycenter, xcenter = 0.0, 150.0
    dmin, dminx = 5.0, 80.0

    got = get_data_cpu(
        {}, xy, iC, PID, tF, ycenter, xcenter, dmin=dmin, dminx=dminx, merge_dim=True
    )
    ref = reference_get_data_cpu(
        {}, xy, iC, PID, tF, ycenter, xcenter, dmin=dmin, dminx=dminx, merge_dim=True
    )
    assert got[0] is not None
    assert torch.equal(got[0], ref[0])
    assert torch.equal(got[1], ref[1])
    assert torch.equal(got[2], ref[2])


def test_identity_unique_imap_no_merge_and_explicit_ix():
    n_templates, nchanraw, nfeatures, nspikes = 20, 8, 4, 1200
    iC, xy = _synthetic_unique_imap(n_templates, nchanraw, n_chans_total=40, seed=5)
    PID, tF = _spikes_for_templates(n_templates, nspikes, nchanraw, nfeatures, seed=6)

    ix = torch.zeros(n_templates, dtype=torch.bool)
    ix[[1, 4, 7, 12, 15]] = True

    got = get_data_cpu(
        {}, xy, iC, PID, tF, None, None, ix=ix, merge_dim=False
    )
    ref = reference_get_data_cpu(
        {}, xy, iC, PID, tF, None, None, ix=ix, merge_dim=False
    )
    assert got[0] is not None
    assert got[0].shape[1] == got[2].nelement()  # nchan axis preserved
    assert torch.equal(got[0], ref[0])
    assert torch.equal(got[1], ref[1])
    assert torch.equal(got[2], ref[2])


def test_pid_accepts_numpy_array():
    n_templates, nchanraw, nfeatures = 6, 4, 2
    iC, xy = _synthetic_unique_imap(n_templates, nchanraw, n_chans_total=16, seed=7)
    PID = np.array([0, 0, 2, 2, 1], dtype=np.int64)
    tF = torch.randn(5, nchanraw, nfeatures)
    ix = torch.tensor([True, True, True, False, False, False])

    Xd, igood, ichan = get_data_cpu(
        {}, xy, iC, PID, tF, None, None, ix=ix, merge_dim=False
    )
    assert isinstance(PID, np.ndarray)  # caller still holds numpy
    assert Xd is not None
    assert igood.numel() == 5
    assert ichan.numel() > 0


def test_identity_duplicate_imap_slots_last_write_wins():
    """If a template column maps two raw slots to the same unique channel,
    advanced-index assignment is last-write-wins — same as the historical loop.
    """
    # iC columns with intentional within-column duplicates
    iC = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],  # col0: channel 0 twice
            [1, 1, 4],  # col1: channel 1 twice
        ],
        dtype=torch.long,
    )
    n_templates = 3
    xy = torch.zeros(2, n_templates)
    PID = np.array([0, 0, 1, 1], dtype=np.int64)
    tF = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
    ix = torch.tensor([True, True, False])

    got = get_data_cpu({}, xy, iC, PID, tF, None, None, ix=ix, merge_dim=False)
    ref = reference_get_data_cpu(
        {}, xy, iC, PID, tF, None, None, ix=ix, merge_dim=False
    )
    assert torch.equal(got[0], ref[0])
    assert torch.equal(got[1], ref[1])
    assert torch.equal(got[2], ref[2])


def test_shapes_merge_dim_true_vs_false():
    n_templates, nchanraw, nfeatures, nspikes = 10, 5, 3, 200
    iC, xy = _synthetic_unique_imap(n_templates, nchanraw, n_chans_total=20, seed=8)
    PID, tF = _spikes_for_templates(n_templates, nspikes, nchanraw, nfeatures, seed=9)
    ix = torch.ones(n_templates, dtype=torch.bool)

    Xd_m, igood_m, ichan_m = get_data_cpu(
        {}, xy, iC, PID, tF, None, None, ix=ix, merge_dim=True
    )
    Xd_s, igood_s, ichan_s = get_data_cpu(
        {}, xy, iC, PID, tF, None, None, ix=ix, merge_dim=False
    )
    assert torch.equal(igood_m, igood_s)
    assert torch.equal(ichan_m, ichan_s)
    nchan = ichan_m.nelement()
    assert Xd_m.shape == (igood_m.numel(), nchan * nfeatures)
    assert Xd_s.shape == (igood_s.numel(), nchan, nfeatures)
    assert torch.equal(Xd_m, Xd_s.reshape(Xd_s.shape[0], -1))
