"""Tests for the optional Kilosort-to-Vision export path."""

import json
import logging
import sys
import types

import numpy as np
import pytest

from kilosort.vision_export import (
    VisionExportError, compute_ei_from_spikes, export_vision, load_kilosort_spikes)


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


def _edge_sort_dir(tmp_path, times, clusters):
    sort = tmp_path / "kilosort4"
    sort.mkdir()
    np.save(sort / "spike_times.npy", np.array(times, dtype=np.int64))
    np.save(sort / "spike_clusters.npy", np.array(clusters, dtype=np.int64))
    (sort / "cluster_KSLabel.tsv").write_text(
        "cluster_id\tKSLabel\n0\tgood\n1\tmua\n", encoding="utf-8")
    return sort


def test_out_of_range_spike_times_are_dropped_not_shifted(tmp_path):
    # 20260724A/data003: stock and fork each emitted one spike at sample -1/-2.
    sort = _edge_sort_dir(tmp_path, [-2, 30, 10, 40, 45], [1, 0, 1, 0, 1])
    ids, times, dense, dropped = load_kilosort_spikes(
        sort, n_samples=45, return_dropped=True)
    assert dropped == 2
    assert ids.tolist() == [1, 2]
    assert times.tolist() == [30, 40, 10]
    assert dense.tolist() == [0, 0, 1]
    # Without a recording length only the negative time can be judged.
    ids, times, dense = load_kilosort_spikes(sort)
    assert times.tolist() == [30, 40, 10, 45]


def test_dropped_count_is_taken_after_good_only(tmp_path):
    sort = _edge_sort_dir(tmp_path, [-1, 30, 40], [1, 0, 0])
    _ids, times, _dense, dropped = load_kilosort_spikes(
        sort, good_only=True, return_dropped=True)
    assert dropped == 0
    assert times.tolist() == [30, 40]


def test_negative_cluster_ids_are_still_an_error(tmp_path):
    sort = _edge_sort_dir(tmp_path, [10, 20], [-1, 0])
    with pytest.raises(VisionExportError, match="cluster IDs must be non-negative"):
        load_kilosort_spikes(sort)


def test_all_spikes_out_of_range_is_an_error(tmp_path):
    sort = _edge_sort_dir(tmp_path, [-3, -1], [0, 1])
    with pytest.raises(VisionExportError, match="inside the recording"):
        load_kilosort_spikes(sort)


def _fake_recording(n_samples):
    rng = np.random.default_rng(7)
    data = rng.integers(-300, 300, size=(n_samples, 4)).astype(np.int16)

    class FakeRecording:
        array_id = 504
        num_electrodes = 4

        def __init__(self, _path, drop_ttl=False):
            assert drop_ttl is False
            self.n_samples = n_samples

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def detect_ttl_pipeline_edges(self):
            return np.array([4, 60], dtype=np.int64)

        def __getitem__(self, key):
            return data[key]

    return FakeRecording


def _fake_visionwriter(calls):
    class _Context:
        def __init__(self, *_args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Neurons(_Context):
        def write_neuron_file(self, spikes, ttl, n_samples):
            calls["neurons"] = (spikes, ttl, n_samples)

    class Globals(_Context):
        def write_simplified_litke_array_globals_file(self, *args):
            calls["globals"] = args

    class EI:
        def __init__(self, *_args, **_kwargs):
            pass

        def write_eis_by_cell_id(self, eis):
            calls["ei"] = eis

        def close(self):
            pass

    return types.SimpleNamespace(
        NeuronsFileWriter=Neurons, GlobalsFileWriter=Globals, EIWriter=EI,
        WriteableEIData=lambda ei, err, n: (ei, err, n))


def test_export_drops_edge_spikes_and_keeps_in_range_ei_identical(
        tmp_path, monkeypatch, caplog):
    n_samples = 120
    monkeypatch.setattr("kilosort.vision_export.LitkeRecording", _fake_recording(n_samples))
    calls = {}
    monkeypatch.setitem(sys.modules, "visionwriter", _fake_visionwriter(calls))
    raw = tmp_path / "data003"
    raw.mkdir()
    sort = _edge_sort_dir(
        tmp_path, [-1, 20, 50, 30, 70, 100, n_samples], [1, 0, 0, 1, 1, 0, 0])
    with caplog.at_level(logging.WARNING, logger="kilosort.vision_export"):
        manifest = export_vision(sort, raw, output_dir=tmp_path / "out",
                                 left=3, right=5, chunk=32, cell_block=1)
    assert "dropping 2 spike(s)" in caplog.text
    assert manifest["dropped_out_of_range_spikes"] == 2
    assert manifest["spike_count"] == 5
    saved = json.loads((tmp_path / "out" / "vision_export.json").read_text())
    assert saved["dropped_out_of_range_spikes"] == 2
    spikes = calls["neurons"][0]
    assert spikes[1].tolist() == [20, 50, 100]
    assert spikes[2].tolist() == [30, 70]
    # The EI of the kept spikes equals the EI of a sort that never had the
    # out-of-range spikes: the drop does not touch in-range arithmetic.
    ids, counts, avg, err, *_ = compute_ei_from_spikes(
        np.array([1, 2]), np.array([20, 50, 100, 30, 70]),
        np.array([0, 0, 0, 1, 1]), raw, left=3, right=5, chunk=32, cell_block=1)
    for i, cid in enumerate(ids):
        ei, ei_err, n = calls["ei"][int(cid)]
        assert n == counts[i]
        assert ei.tobytes() == np.asarray(avg[i, 1:], dtype=np.float32).tobytes()
        assert ei_err.tobytes() == np.asarray(err[i, 1:], dtype=np.float32).tobytes()
