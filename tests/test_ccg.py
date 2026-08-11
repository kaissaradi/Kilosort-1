import numpy as np

from kilosort import CCG


def reference_refract(cluster_ids, spike_times, acg_threshold=0.2,
                      ccg_threshold=0.25):
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
