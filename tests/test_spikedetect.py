import numpy as np
import torch

from kilosort.spikedetect import extract_wPCA_wTEMP, nearest_chans
from kilosort.template_matching import prepare_extract
from kilosort.utils import get_clip_buffer_capacity, get_spike_buffer_capacity


def test_spike_buffer_capacity_scales_with_recording_length():
    assert get_spike_buffer_capacity(1) == 10_000
    assert get_spike_buffer_capacity(50) == 500_000
    assert get_spike_buffer_capacity(100) == 1_000_000
    assert get_spike_buffer_capacity(1_000) == 1_000_000


def test_clip_buffer_capacity_scales_with_recording_length():
    # Short runs: n_used batches * 5000, floored at 10k.
    assert get_clip_buffer_capacity(1, nskip=25) == 10_000
    assert get_clip_buffer_capacity(25, nskip=25) == 10_000  # 1 used batch
    assert get_clip_buffer_capacity(50, nskip=25) == 10_000  # 2 used → 10k
    assert get_clip_buffer_capacity(100, nskip=25) == 20_000  # 4 used
    assert get_clip_buffer_capacity(2500, nskip=25) == 500_000  # 100 used → cap
    assert get_clip_buffer_capacity(10_000, nskip=25) == 500_000


def test_extract_wpca_grows_when_first_batch_exceeds_capacity(monkeypatch):
    """Shipped extract path must not break when peak count > initial capacity.

    Regression for the short-run capacity shrink: a single dense batch can yield
    more isolated peaks than the initial 10k allocation. Grow-on-overflow (or
    partial fill at the hard cap) must retain clips so TruncatedSVD can run.
    """
    nt = 7
    n_peaks = 48  # well above the forced initial capacity of 8
    n_pcs = 3
    n_templates = 2

    monkeypatch.setattr(
        'kilosort.spikedetect.get_clip_buffer_capacity',
        lambda *args, **kwargs: 8,
    )

    def fake_snippets(X, nt=61, twav_min=20, Th_single_ch=6, loc_range=None,
                      long_range=None, device=None):
        # Deterministic non-zero waveforms so normalization/SVD are well-posed.
        base = torch.linspace(0.1, 1.0, steps=nt, dtype=torch.float32)
        return base.unsqueeze(0).repeat(n_peaks, 1) + 0.01 * torch.arange(
            n_peaks, dtype=torch.float32
        ).unsqueeze(1)

    monkeypatch.setattr('kilosort.spikedetect.extract_snippets', fake_snippets)

    class _TinyBfile:
        n_batches = 1

        def padded_batch_to_torch(self, j, ops):
            return torch.zeros(4, 64)

    ops = {'settings': {'n_pcs': n_pcs, 'n_templates': n_templates}}
    wPCA, wTEMP = extract_wPCA_wTEMP(
        ops, _TinyBfile(), nt=nt, nskip=1, device=torch.device('cpu')
    )

    assert wPCA.shape == (n_pcs, nt)
    assert wTEMP.shape == (n_templates, nt)
    assert torch.isfinite(wPCA).all()
    assert torch.isfinite(wTEMP).all()


def test_extract_wpca_partial_fills_at_hard_cap(monkeypatch):
    """When already at the hard cap, keep a partial batch — never 0 clips."""
    nt = 5
    n_pcs = 2
    n_templates = 2
    hard_cap = 16
    # Start at hard cap so grow cannot expand further; return more peaks than room.
    monkeypatch.setattr('kilosort.spikedetect.CLIP_BUFFER_HARD_CAP', hard_cap)
    monkeypatch.setattr(
        'kilosort.spikedetect.get_clip_buffer_capacity',
        lambda *args, **kwargs: hard_cap,
    )

    def fake_snippets(X, nt=61, twav_min=20, Th_single_ch=6, loc_range=None,
                      long_range=None, device=None):
        n = hard_cap + 10
        base = torch.ones(n, nt, dtype=torch.float32)
        base[:, 0] = torch.linspace(0.5, 1.5, steps=n)
        return base

    monkeypatch.setattr('kilosort.spikedetect.extract_snippets', fake_snippets)

    class _TinyBfile:
        n_batches = 1

        def padded_batch_to_torch(self, j, ops):
            return torch.zeros(2, 32)

    ops = {'settings': {'n_pcs': n_pcs, 'n_templates': n_templates}}
    wPCA, wTEMP = extract_wPCA_wTEMP(
        ops, _TinyBfile(), nt=nt, nskip=1, device=torch.device('cpu')
    )
    assert wPCA.shape == (n_pcs, nt)
    assert wTEMP.shape[0] == n_templates


def test_nearest_chans_matches_independent_sorted_distances():
    ys = np.array([0, 30, 60])
    xs = np.array([0, 30, 0])
    yc = np.array([0, 0, 30, 60])
    xc = np.array([0, 30, 0, 0])
    expected_distances = (ys - yc[:, np.newaxis])**2 + \
        (xs - xc[:, np.newaxis])**2
    expected_indices = np.argsort(expected_distances, axis=0)[:3]
    expected_distances = np.sort(expected_distances, axis=0)[:3]

    indices, distances = nearest_chans(ys, yc, xs, xc, nC=3,
                                       device=torch.device('cpu'))

    np.testing.assert_array_equal(indices.numpy(), expected_indices)
    np.testing.assert_array_equal(distances, expected_distances)


def test_prepare_extract_matches_independent_sorted_distances():
    xc = np.array([0, 30, 0, 30])
    yc = np.array([0, 0, 30, 30])
    templates = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    expected_distances = (xc - xc[:, np.newaxis])**2 + \
        (yc - yc[:, np.newaxis])**2
    expected_indices = np.argsort(expected_distances, axis=0)[:3]
    expected_mask = np.sort(expected_distances, axis=0)[:3] < 31**2

    indices, mask, _, _ = prepare_extract(
        xc, yc, templates, nC=3, position_limit=31,
        device=torch.device('cpu')
    )

    np.testing.assert_array_equal(indices.numpy(), expected_indices)
    np.testing.assert_array_equal(mask.numpy(), expected_mask)


def test_wpca_wtemp(bfile, saved_ops, torch_device):
    # Make sure extracting templates from data works, and with
    # differnt values than the default for n_templates, n_pcs
    ops = saved_ops.copy()
    ops['n_templates'] = 3
    ops['n_pcs'] = 5

    wPCA, wTEMP = extract_wPCA_wTEMP(ops, bfile, device=torch_device)
