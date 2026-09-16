import numpy as np

from kilosort import CCG


def reference_refract(cluster_ids, spike_times, acg_threshold=0.2,
                      ccg_threshold=0.25, isi_threshold=0.01,
                      isi_min_spikes=500):
    n_clusters = cluster_ids.max() + 1
    is_refractory = np.zeros(n_clusters)
    contamination = np.zeros(n_clusters)

    for cluster_id in range(n_clusters):
        unit_times = spike_times[cluster_ids == cluster_id]
        if len(unit_times) > 10 and unit_times.max() != unit_times.min():
            is_refractory[cluster_id], _, contamination[cluster_id] = \
                CCG.check_CCG(
                    unit_times, acg_threshold=acg_threshold,
                    ccg_threshold=ccg_threshold
                )
        if (not is_refractory[cluster_id] and isi_threshold > 0
                and len(unit_times) >= isi_min_spikes):
            st_sorted = np.sort(unit_times)
            if CCG.isi_violation_rate(st_sorted) < isi_threshold:
                is_refractory[cluster_id] = True

    return is_refractory.astype(bool), contamination


def test_refract_matches_reference_for_sorted_spikes():
    rng = np.random.default_rng(42)
    cluster_ids = rng.choice(np.array([0, 1, 3, 7]), size=2_000)
    spike_times = np.sort(rng.uniform(0, 120, size=cluster_ids.size))

    expected_labels, expected_contamination = reference_refract(
        cluster_ids, spike_times
    )
    labels, contamination = CCG.refract(cluster_ids, spike_times)

    assert labels.dtype == bool
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(contamination, expected_contamination)


def test_refract_matches_reference_for_unordered_spikes():
    rng = np.random.default_rng(43)
    cluster_ids = rng.integers(0, 5, size=1_000)
    spike_times = rng.uniform(0, 120, size=cluster_ids.size)

    expected_labels, expected_contamination = reference_refract(
        cluster_ids, spike_times
    )
    labels, contamination = CCG.refract(cluster_ids, spike_times)

    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(contamination, expected_contamination)


def test_refract_distinguishes_clean_and_refractory_violating_units():
    clean_times = np.arange(0, 10, 0.05)
    contaminated_times = np.sort(
        np.column_stack((clean_times, clean_times + 0.001)).ravel()
    )
    spike_times = np.concatenate((clean_times, contaminated_times))
    cluster_ids = np.concatenate((
        np.zeros(clean_times.size, dtype=np.int64),
        np.ones(contaminated_times.size, dtype=np.int64),
    ))
    order = np.argsort(spike_times, kind='stable')

    labels, contamination = CCG.refract(
        cluster_ids[order], spike_times[order]
    )

    assert labels.tolist() == [True, False]
    assert contamination[0] < contamination[1]


def test_refract_accepts_empty_spike_vectors():
    labels, contamination = CCG.refract(np.array([], dtype=np.int64),
                                         np.array([]))

    assert labels.dtype == bool
    assert labels.size == 0
    assert contamination.size == 0


def test_refract_accepts_negative_cluster_ids():
    """Negative labels used to crash bincount; must return finite tables."""
    # Two clusters at -1 and 0 with enough spikes for CCG.
    st_a = np.arange(0.0, 5.0, 0.05)
    st_b = np.arange(0.01, 5.0, 0.05)
    cluster_ids = np.concatenate([
        np.full(st_a.size, -1, dtype=np.int64),
        np.zeros(st_b.size, dtype=np.int64),
    ])
    spike_times = np.concatenate([st_a, st_b])
    order = np.argsort(spike_times, kind='stable')
    labels, contam = CCG.refract(cluster_ids[order], spike_times[order])
    assert labels.dtype == bool
    assert labels.size == 2  # min=-1, max=0 → offset table length 2
    assert np.all(np.isfinite(contam))


def test_isi_fallback_rescues_clean_units():
    """A cluster with low ISI violations but borderline ACG gets rescued."""
    rng = np.random.default_rng(99)
    # Clean cell: 5000 spikes, ~28 Hz, no refractory violations
    st_clean = np.sort(rng.uniform(0, 180, size=5000))
    # Remove any ISI < 2ms
    while True:
        isi = np.diff(st_clean)
        bad = np.where(isi < 0.002)[0]
        if len(bad) == 0:
            break
        st_clean = np.delete(st_clean, bad + 1)
    cluster_ids = np.zeros(len(st_clean), dtype=np.int64)

    # Without ISI fallback
    labels_old, _ = CCG.refract(cluster_ids, st_clean,
                                isi_threshold=0)
    # With ISI fallback
    labels_new, _ = CCG.refract(cluster_ids, st_clean,
                                isi_threshold=0.01, isi_min_spikes=500)

    # If ACG already passes, both agree. If ACG fails, ISI should rescue.
    if not labels_old[0]:
        assert labels_new[0], (
            'ISI fallback should rescue a clean unit that fails the ACG test')
    else:
        assert labels_new[0]


def test_isi_fallback_does_not_rescue_contaminated_units():
    """A contaminated cluster stays MUA even with the ISI fallback on."""
    rng = np.random.default_rng(100)
    clean = np.sort(rng.uniform(0, 180, size=2000))
    # Add 5% ISI violations (spikes 0.5ms after each existing spike)
    violations = clean[:100] + 0.0005
    contaminated = np.sort(np.concatenate([clean, violations]))
    cluster_ids = np.zeros(len(contaminated), dtype=np.int64)

    labels, _ = CCG.refract(cluster_ids, contaminated,
                            isi_threshold=0.01, isi_min_spikes=500)
    assert not labels[0], 'contaminated unit should stay MUA'


def test_isi_fallback_disabled_when_threshold_zero():
    """isi_threshold=0 reproduces the old ACG-only behavior."""
    rng = np.random.default_rng(42)
    cluster_ids = rng.choice(np.array([0, 1, 3, 7]), size=2_000)
    spike_times = np.sort(rng.uniform(0, 120, size=cluster_ids.size))

    old_ref, old_contam = reference_refract(
        cluster_ids, spike_times, isi_threshold=0)
    labels, contam = CCG.refract(cluster_ids, spike_times, isi_threshold=0)

    np.testing.assert_array_equal(labels, old_ref)
    np.testing.assert_array_equal(contam, old_contam)


def test_compute_ccg_empty_trains_no_crash():
    K, T = CCG.compute_CCG(np.zeros(0), np.zeros(0))
    assert T == 0.0
    assert K.shape[0] == 1001
    assert K.sum() == 0


def test_check_ccg_empty_and_zero_span():
    # Empty
    is_ref, cross, R12 = CCG.check_CCG(np.array([]))
    assert is_ref is False and cross is False
    # All equal times → T==0
    is_ref, cross, R12 = CCG.check_CCG(np.array([1.0, 1.0, 1.0]))
    assert is_ref is False and cross is False
    # Non-degenerate still returns a finite R12
    times = np.arange(0, 5, 0.05)
    is_ref, cross, R12 = CCG.check_CCG(times)
    assert np.isfinite(R12)
