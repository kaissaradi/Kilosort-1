"""Tests for the optional Kilosort-to-Vision export path."""

import numpy as np

from kilosort.vision_export import compute_ei_from_spikes, load_kilosort_spikes


def _sort_dir(tmp_path):
    sort = tmp_path / "kilosort4"
    sort.mkdir()
    np.save(sort / "spike_times.npy", np.array([20, 30, 10, 40], dtype=np.int64))
    np.save(sort / "spike_clusters.npy", np.array([1, 0, 1, 0], dtype=np.int64))
    (sort / "cluster_KSLabel.tsv").write_text(
        "cluster_id\tKSLabel\n0\tgood\n1\tmua\n", encoding="utf-8")
    return sort


def test_load_kilosort_spikes_returns_one_based_sorted_ids(tmp_path):
    ids, times, dense = load_kilosort_spikes(_sort_dir(tmp_path))
    assert ids.tolist() == [1, 2]
    assert times.tolist() == [30, 40, 10, 20]
    assert dense.tolist() == [0, 0, 1, 1]


def test_good_only_filters_zero_based_quality_labels(tmp_path):
    ids, times, dense = load_kilosort_spikes(_sort_dir(tmp_path), good_only=True)
    assert ids.tolist() == [1]
    assert times.tolist() == [30, 40]
    assert dense.tolist() == [0, 0]


def test_ei_uses_native_recording_ttl_and_preserves_cell_ids(monkeypatch):
    class FakeRecording:
        array_id = 504
        num_electrodes = 3
        n_samples = 40

        def __init__(self, _path, drop_ttl=False):
            assert drop_ttl is False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def detect_ttl_pipeline_edges(self):
            return np.array([4, 24], dtype=np.int64)

        def __getitem__(self, key):
            start, stop, step = key.indices(self.n_samples)
            assert step == 1
            return np.zeros((stop - start, self.num_electrodes), dtype=np.int16)

    monkeypatch.setattr("kilosort.vision_export.LitkeRecording", FakeRecording)
    ids, counts, avg, err, ttl, n_samples, array_id = compute_ei_from_spikes(
        np.array([1, 2]), np.array([20, 2, 10]), np.array([0, 0, 1]),
        "unused", left=1, right=1, chunk=16, cell_block=1)
    assert ids.tolist() == [1, 2]
    assert counts.tolist() == [2, 1]
    assert avg.shape == (2, 3, 3)
    assert err.shape == avg.shape
    assert ttl.tolist() == [4, 24]
    assert n_samples == 40
    assert array_id == 504
