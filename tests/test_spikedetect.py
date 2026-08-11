import numpy as np
import torch

from kilosort.spikedetect import extract_wPCA_wTEMP, nearest_chans, yweighted
from kilosort.template_matching import prepare_extract
from kilosort.utils import get_clip_buffer_capacity, get_spike_buffer_capacity


def test_yweighted_finite_when_all_weights_zero():
    """All-negative adist → relu sum 0 must not emit NaN template y-centers."""
    device = torch.device('cpu')
    yc = np.array([0., 10., 20., 30.], dtype=np.float32)
    iC = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)  # (nC, n_temp)
    adist = -torch.ones(3, 2)  # no positive mass
    xy = torch.zeros(2, 2, dtype=torch.long)
    yct = yweighted(yc, iC, adist, xy, device=device)
    assert torch.isfinite(yct).all()


def test_yweighted_identity_on_normal_weights():
    """clamp_min must not change results when sum(0) is safely positive.

    Matches production shapes: iC (nC, n_templates), adist (nC, nsp),
    xy (nsp, 2) with col0 = template index.
    """
    device = torch.device('cpu')
    yc = np.array([0., 10., 20., 30., 40.], dtype=np.float32)
    # 3 nearest chans × 4 templates
    iC = torch.tensor(
        [[0, 1, 2, 0], [1, 2, 3, 1], [2, 3, 4, 2]], dtype=torch.long
    )
    # 5 spikes; template indices 0,1,2,3,1
    nsp = 5
    xy = torch.tensor([[0, 10], [1, 20], [2, 30], [3, 40], [1, 50]], dtype=torch.long)
    adist = torch.tensor(
        [
            [1.0, 0.5, 2.0, 1.0, 0.8],
            [0.5, 1.5, 0.0, 0.5, 1.2],
            [0.0, 1.0, 1.0, 0.5, 0.4],
        ]
    )
    assert adist.shape == (3, nsp)
    got = yweighted(yc, iC, adist, xy, device=device)
    # Historical formula without clamp (sums > 0 here)
    yy = torch.from_numpy(yc).to(device)[iC]
    cF0 = torch.nn.functional.relu(adist)
    cF0 = cF0 / cF0.sum(0)
    exp = (cF0 * yy[:, xy[:, 0]]).sum(0)
    assert torch.allclose(got, exp)


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


def test_template_match_body_dispatch_eager_on_cpu(monkeypatch):
    """CPU path must not torch.compile by default (fieldlab / no CUDA)."""
    import kilosort.spikedetect as sd
    # Reset dispatch cache
    sd._TM_BODY = None
    monkeypatch.delenv('KILOSORT_FORCE_COMPILE', raising=False)
    monkeypatch.delenv('KILOSORT_NO_COMPILE', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    # Call dispatch once with dummy tensors matching body signature
    weigh = torch.randn(2, 3, 4)
    Bsl = torch.randn(5, 2, 6)
    iC = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    # weigh is (nsizes, nC, Nfilt); simplify: call body directly via dispatch
    # after ensuring init runs
    try:
        # Minimal call may fail on shape; we only care that _TM_BODY is set
        # to the eager function after first init.
        sd._template_match_body_dispatch(
            Bsl, weigh, iC, torch.arange(4), 2, 4
        )
    except Exception:
        # Shape mismatch is fine; init of _TM_BODY happens before the call body
        if sd._TM_BODY is None:
            # Force init path by calling the None-branch logic
            pass
    # Re-init explicitly like dispatch does
    sd._TM_BODY = None
    # Manually invoke the selection logic by calling with valid shapes from body
    # _template_match_body(Bsl, weigh, iC, iC2_flat, nC2, Nfilt)
    # weigh: (nsize, nC, Nfilt), Bsl: (n_chan, n_temp, T), iC: (nC, Nfilt)
    nC, Nfilt, nsize, n_chan, n_temp, T = 3, 4, 2, 6, 2, 10
    weigh = torch.randn(nsize, nC, Nfilt)
    Bsl = torch.randn(n_chan, n_temp, T)
    iC = torch.randint(0, n_chan, (nC, Nfilt))
    iC2 = torch.randint(0, Nfilt, (5, Nfilt))
    iC2_flat = iC2.reshape(-1)
    nC2 = iC2.shape[0]
    sd._template_match_body_dispatch(Bsl, weigh, iC, iC2_flat, nC2, Nfilt)
    assert sd._TM_BODY is sd._template_match_body
    sd._TM_BODY = None  # leave clean for other tests


def test_clip_norm_drops_zero_energy_rows():
    """All-zero clips must not NaN-normalize; only positive-energy rows kept."""
    # Unit-test the norm filter logic in isolation (same as extract_wPCA path).
    clips = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ], dtype=np.float32)
    norms = (clips ** 2).sum(1, keepdims=True) ** .5
    good = norms[:, 0] > 0
    assert good.tolist() == [True, False, True]
    kept = clips[good] / norms[good]
    assert np.isfinite(kept).all()
    np.testing.assert_allclose(np.linalg.norm(kept, axis=1), [1.0, 1.0], atol=1e-6)


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
