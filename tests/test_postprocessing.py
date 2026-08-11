"""Unit tests for kilosort.postprocessing (CPU-safe)."""
import numpy as np
import pytest

from kilosort.postprocessing import remove_duplicates


def reference_remove_duplicates(spike_times, spike_clusters, dt=15):
    """Independent implementation of the historical keep rule.

    First spike per cluster is kept. Later same-cluster spikes are kept only
    when at least `dt` samples after the previous *kept* spike of that cluster.
    """
    spike_times = np.asarray(spike_times, dtype=np.int64)
    spike_clusters = np.asarray(spike_clusters, dtype=np.int32)
    keep = np.zeros(spike_times.size, dtype=bool)
    last_kept = {}
    for i in range(spike_times.size):
        t = int(spike_times[i])
        c = int(spike_clusters[i])
        t0 = last_kept[c] if c in last_kept else t - dt
        if t >= t0 + dt:
            last_kept[c] = t
            keep[i] = True
    return spike_times[keep], spike_clusters[keep], keep


def test_remove_duplicates_matches_reference_rule():
    rng = np.random.default_rng(0)
    n = 5_000
    # Interleaved multi-unit stream with many near-duplicates.
    spike_times = np.sort(rng.integers(0, 50_000, size=n)).astype(np.int64)
    spike_clusters = rng.integers(0, 40, size=n, dtype=np.int32)
    # Inject exact same-cluster collisions within dt.
    for _ in range(200):
        i = int(rng.integers(1, n))
        spike_times[i] = spike_times[i - 1] + int(rng.integers(0, 10))
        spike_clusters[i] = spike_clusters[i - 1]
    order = np.argsort(spike_times, kind='stable')
    spike_times = spike_times[order]
    spike_clusters = spike_clusters[order]

    exp_t, exp_c, exp_keep = reference_remove_duplicates(
        spike_times, spike_clusters, dt=15
    )
    got_t, got_c, got_keep = remove_duplicates(
        spike_times.copy(), spike_clusters.copy(), dt=15
    )

    np.testing.assert_array_equal(got_keep, exp_keep)
    np.testing.assert_array_equal(got_t, exp_t)
    np.testing.assert_array_equal(got_c, exp_c)
    # Must actually drop some spikes on this synthetic collision stream.
    assert got_keep.sum() < n
    assert got_keep.sum() == exp_keep.sum()


def test_remove_duplicates_empty_input():
    times = np.zeros(0, dtype=np.int64)
    clusters = np.zeros(0, dtype=np.int32)
    out_t, out_c, keep = remove_duplicates(times, clusters, dt=15)
    assert out_t.size == 0
    assert out_c.size == 0
    assert keep.size == 0
    assert keep.dtype == np.bool_ or keep.dtype == bool


def test_remove_duplicates_independent_clusters_do_not_suppress_each_other():
    # Same sample times, different clusters: both kept.
    spike_times = np.array([100, 100, 100], dtype=np.int64)
    spike_clusters = np.array([0, 1, 2], dtype=np.int32)
    out_t, out_c, keep = remove_duplicates(spike_times, spike_clusters, dt=15)
    assert keep.all()
    np.testing.assert_array_equal(out_t, spike_times)
    np.testing.assert_array_equal(out_c, spike_clusters)


def test_remove_duplicates_suppresses_within_window_same_cluster():
    spike_times = np.array([0, 5, 14, 15, 30], dtype=np.int64)
    spike_clusters = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    out_t, out_c, keep = remove_duplicates(spike_times, spike_clusters, dt=15)
    # Keep 0; drop 5 and 14 (within 15 of 0); keep 15 (exactly t0+dt); keep 30.
    np.testing.assert_array_equal(keep, np.array([True, False, False, True, True]))
    np.testing.assert_array_equal(out_t, np.array([0, 15, 30], dtype=np.int64))


def test_remove_duplicates_gapped_cluster_ids():
    # Dense table must still work when labels are not 0..K-1 contiguous usage.
    spike_times = np.array([0, 1, 20, 21], dtype=np.int64)
    spike_clusters = np.array([0, 7, 0, 7], dtype=np.int32)
    out_t, out_c, keep = remove_duplicates(spike_times, spike_clusters, dt=15)
    np.testing.assert_array_equal(keep, np.array([True, True, True, True]))
    np.testing.assert_array_equal(out_c, spike_clusters)
