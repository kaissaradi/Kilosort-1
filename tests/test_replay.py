import json

import numpy as np
import pytest
import torch

from kilosort import clustering_qr
from kilosort.replay import (
    load_center_capture,
    replay_center,
    save_center_capture,
)


def fixture_arrays(n_events=24):
    Xd = torch.arange(n_events * 4, dtype=torch.float32).reshape(n_events, 4)
    igood = np.arange(n_events, dtype=np.int64)
    ichan = np.array([1, 3], dtype=np.int64)
    st0 = np.arange(n_events, dtype=np.float64) / 20_000
    return Xd, igood, ichan, st0


def test_capture_roundtrip_and_small_center_replay(tmp_path):
    Xd, igood, ichan, st0 = fixture_arrays()
    metadata = {"commit": "abc", "center_id": 7, "settings_sha256": "def"}
    directory = tmp_path / "center"
    manifest = save_center_capture(
        directory, Xd=Xd, igood=igood, ichan=ichan, st0=st0, metadata=metadata
    )

    capture = load_center_capture(directory, expected_metadata=metadata)
    result = replay_center(
        capture,
        settings={"cluster_downsampling": 1},
        n_channels=5,
        n_pcs=2,
    )

    assert capture.manifest_sha256 == manifest["manifest_sha256"]
    assert np.array_equal(capture.Xd.numpy(), Xd.numpy())
    assert np.array_equal(result.labels, np.zeros(len(igood), dtype=np.int32))
    assert result.Wall.shape == (1, 5, 2)


def test_capture_rejects_array_mutation(tmp_path):
    Xd, igood, ichan, st0 = fixture_arrays()
    directory = tmp_path / "center"
    save_center_capture(
        directory, Xd=Xd, igood=igood, ichan=ichan, st0=st0, metadata={}
    )
    changed = np.load(directory / "Xd.npy")
    changed[0, 0] += 1
    np.save(directory / "Xd.npy", changed)

    with pytest.raises(ValueError, match="identity mismatch"):
        load_center_capture(directory)


def test_capture_rejects_manifest_or_expected_metadata_mutation(tmp_path):
    Xd, igood, ichan, st0 = fixture_arrays()
    directory = tmp_path / "center"
    save_center_capture(
        directory, Xd=Xd, igood=igood, ichan=ichan, st0=st0, metadata={"seed": 1}
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_center_capture(directory, expected_metadata={"seed": 2})

    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["metadata"]["seed"] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="seal mismatch"):
        load_center_capture(directory)


def test_capture_validates_aligned_rows_before_writing(tmp_path):
    Xd, igood, ichan, st0 = fixture_arrays()
    with pytest.raises(ValueError, match="igood"):
        save_center_capture(
            tmp_path / "center",
            Xd=Xd,
            igood=igood[:-1],
            ichan=ichan,
            st0=st0,
            metadata={},
        )


def test_run_callback_observes_exact_post_get_data_boundary(monkeypatch):
    n_events = 24
    Xd, igood, ichan, _st0 = fixture_arrays(n_events)
    xy = torch.zeros((2, n_events), dtype=torch.float32)
    iC = torch.zeros((1, n_events), dtype=torch.long)
    monkeypatch.setattr(clustering_qr, "xy_templates", lambda ops: (xy, iC))
    monkeypatch.setattr(clustering_qr, "x_centers", lambda ops: np.array([0.0]))
    monkeypatch.setattr(clustering_qr, "y_centers", lambda ops: np.array([0.0]))
    monkeypatch.setattr(
        clustering_qr,
        "get_nearest_centers",
        lambda *args: (
            torch.zeros(n_events, dtype=torch.long),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
        ),
    )
    monkeypatch.setattr(
        clustering_qr, "get_data_cpu", lambda *args, **kwargs: (Xd, igood, ichan)
    )

    observed = []
    ops = {
        "dmin": 20,
        "dminx": 32,
        "Nchan": 5,
        "fs": 20_000,
        "xcup": np.array([0.0]),
        "ycup": np.array([0.0]),
        "settings": {
            "cluster_downsampling": 1,
            "cluster_neighbors": 10,
            "max_cluster_subset": None,
            "cluster_init_seed": 1,
            "n_pcs": 2,
        },
    }
    st = np.column_stack((np.arange(n_events), np.zeros(n_events)))
    tF = torch.zeros((n_events, 1, 1), dtype=torch.float32)

    plain_clu, plain_wall = clustering_qr.run(
        ops,
        st,
        tF,
        mode="template",
        device=torch.device("cpu"),
    )

    callback_clu, callback_wall = clustering_qr.run(
        ops,
        st,
        tF,
        mode="template",
        device=torch.device("cpu"),
        center_callback=lambda **kwargs: observed.append(kwargs),
    )

    np.testing.assert_array_equal(callback_clu, plain_clu)
    torch.testing.assert_close(callback_wall, plain_wall, rtol=0, atol=0)
    assert len(observed) == 1
    assert observed[0]["Xd"] is Xd
    assert observed[0]["igood"] is igood
    assert observed[0]["ichan"] is ichan
    np.testing.assert_array_equal(observed[0]["st0"], st[:, 0] / ops["fs"])
    assert observed[0]["metadata"]["small_center_bypass"] is True
