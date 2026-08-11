"""Pure unit tests for kilosort.utils helpers (CPU-safe, no e2e)."""
import numpy as np
import pytest

from kilosort.utils import (
    get_clip_buffer_capacity,
    get_spike_buffer_capacity,
    group_indices_by_label,
)


def reference_group_indices_by_label(labels):
    """Independent oracle: flatnonzero per unique label (sorted keys)."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return {}
    out = {}
    for k in np.unique(labels):
        out[int(k)] = np.flatnonzero(labels == k)
    return out


def test_group_indices_empty():
    assert group_indices_by_label(np.array([], dtype=np.int32)) == {}


def test_group_indices_matches_flatnonzero_oracle():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 40, size=8_000, dtype=np.int32)
    # Inject gaps and negatives (merge/export paths can carry sparse ids).
    labels[::97] = -3
    labels[::113] = 200

    got = group_indices_by_label(labels)
    exp = reference_group_indices_by_label(labels)

    assert list(got.keys()) == list(exp.keys())
    for k in exp:
        np.testing.assert_array_equal(got[k], exp[k])
        # Ascending index order (boolean-mask gather semantics).
        assert np.all(got[k][1:] > got[k][:-1])


def test_group_indices_single_label():
    labels = np.full(50, 7, dtype=np.int32)
    got = group_indices_by_label(labels)
    assert list(got.keys()) == [7]
    np.testing.assert_array_equal(got[7], np.arange(50))


def test_group_indices_merge_reassign_matches_mask():
    """Simulates merging_function index-bookkeeping after clu2[jj] -> kk."""
    rng = np.random.default_rng(1)
    clu = rng.integers(0, 12, size=3_000, dtype=np.int32)
    groups = group_indices_by_label(clu)
    kk, jj = 3, 8
    if jj not in groups or kk not in groups:
        pytest.skip("rng did not produce both labels")

    idx = groups[jj]
    clu2 = clu.copy()
    clu2[idx] = kk
    groups[kk] = np.sort(np.concatenate((groups[kk], idx)))
    del groups[jj]

    # Oracle: boolean mask after reassignment.
    exp_kk = np.flatnonzero(clu2 == kk)
    np.testing.assert_array_equal(groups[kk], exp_kk)
    assert jj not in groups
    # Other labels untouched.
    for lab, inds in group_indices_by_label(clu2).items():
        if lab == kk:
            continue
        np.testing.assert_array_equal(groups[lab], inds)


def test_buffer_capacity_helpers_bounds():
    assert get_spike_buffer_capacity(1) == 10_000
    assert get_spike_buffer_capacity(200) == 10**6
    assert get_clip_buffer_capacity(1, nskip=25) == 10_000
    assert get_clip_buffer_capacity(10_000, nskip=25) == 500_000
