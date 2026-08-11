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
