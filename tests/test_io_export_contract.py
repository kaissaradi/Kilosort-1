import numpy as np
import pytest
import torch

from kilosort import io


def _export_inputs():
    st = np.array([[10, 0, 1.0], [20, 1, 1.0]], dtype=np.float64)
    clu = np.array([0, 1], dtype=np.float64)
    tF = torch.ones((2, 1, 1))
    Wall = torch.ones((2, 1, 1))
    probe = {
        'chanMap': np.array([0]),
        'xc': np.array([0.0]),
        'yc': np.array([0.0]),
        'kcoords': np.array([0.0]),
    }
    ops = {
        'Wrot': torch.eye(1),
        'wPCA': torch.ones((1, 1)),
        'fs': 30_000,
        'nt': 1,
        'duplicate_spike_bins': 15,
        'settings': {
            'n_chan_bin': 1,
            'fs': 30_000,
            'acg_threshold': 0.2,
            'ccg_threshold': 0.25,
            'filename': 'recording.bin',
        },
    }
    return st, clu, tF, Wall, probe, ops


def _set_nonintegral_time(st, clu):
    st[0, 0] = 10.5


def _set_nonfinite_template(st, clu):
    st[0, 1] = np.inf


def _set_nonintegral_cluster(st, clu):
    clu[0] = 0.5


def _set_invalid_template(st, clu):
    st[0, 1] = 2


def _set_invalid_cluster(st, clu):
    clu[0] = 2


@pytest.fixture
def patched_export_dependencies(monkeypatch):
    monkeypatch.setattr(
        io,
        'compute_spike_positions',
        lambda st, tF, ops: (np.zeros(len(st)), np.zeros(len(st))),
    )
    monkeypatch.setattr(
        io,
        'remove_duplicates',
        lambda times, clusters, dt: (
            times,
            clusters,
            np.ones(len(times), dtype=bool),
        ),
    )
    monkeypatch.setattr(
        io.CCG,
        'similarity',
        lambda Wall, wPCA, nt: np.eye(Wall.shape[0]),
    )
    monkeypatch.setattr(
        io.CCG,
        'refract',
        lambda clusters, times, acg_threshold, ccg_threshold: (
            np.ones(2, dtype=bool),
            np.zeros(2),
        ),
    )
    monkeypatch.setattr(
        io,
        'make_pc_features',
        lambda ops, templates, clusters, features: (
            features.permute(0, 2, 1),
            np.zeros((2, 1), dtype=np.uint32),
        ),
    )


@pytest.mark.parametrize(
    ('mutate', 'message'),
    [
        (lambda st, clu, tF, Wall: (st[:, :1], clu, tF, Wall), 'st must'),
        (lambda st, clu, tF, Wall: (st, clu[:1], tF, Wall), 'same number'),
        (lambda st, clu, tF, Wall: (st, clu, tF[:1], Wall), 'same number'),
    ],
)
def test_save_to_phy_rejects_shape_mismatches_before_writing(
    tmp_path, patched_export_dependencies, mutate, message
):
    st, clu, tF, Wall, probe, ops = _export_inputs()
    st, clu, tF, Wall = mutate(st, clu, tF, Wall)
    result_dir = tmp_path / 'results'

    with pytest.raises(ValueError, match=message):
        io.save_to_phy(
            st, clu, tF, Wall, probe, ops, imin=0, results_dir=result_dir
        )

    assert not result_dir.exists()


@pytest.mark.parametrize(
    ('mutate', 'message'),
    [
        (_set_nonintegral_time, 'finite integer'),
        (_set_nonfinite_template, 'finite integers'),
        (_set_nonintegral_cluster, 'finite integers'),
        (_set_invalid_template, 'valid indices'),
        (_set_invalid_cluster, 'valid indices'),
    ],
)
def test_save_to_phy_rejects_invalid_times_and_labels(
    tmp_path, patched_export_dependencies, mutate, message
):
    st, clu, tF, Wall, probe, ops = _export_inputs()
    mutate(st, clu)
    result_dir = tmp_path / 'results'

    with pytest.raises(ValueError, match=message):
        io.save_to_phy(
            st, clu, tF, Wall, probe, ops, imin=0, results_dir=result_dir
        )

    assert not result_dir.exists()


def test_save_to_phy_rejects_unsorted_times(
    tmp_path, patched_export_dependencies
):
    st, clu, tF, Wall, probe, ops = _export_inputs()
    st = st[[1, 0]]
    result_dir = tmp_path / 'results'

    with pytest.raises(ValueError, match='sorted'):
        io.save_to_phy(
            st, clu, tF, Wall, probe, ops, imin=0, results_dir=result_dir
        )

    assert not result_dir.exists()


def test_save_to_phy_normalizes_numpy_dtype_in_params(
    tmp_path, patched_export_dependencies
):
    st, clu, tF, Wall, probe, ops = _export_inputs()
    result_dir = tmp_path / 'results'

    io.save_to_phy(
        st,
        clu,
        tF,
        Wall,
        probe,
        ops,
        imin=0,
        results_dir=result_dir,
        data_dtype=np.uint16,
    )

    params = (result_dir / 'params.py').read_text()
    assert "dtype = 'uint16'" in params


def test_save_ops_does_not_mutate_nested_live_ops(tmp_path):
    settings = {
        'results_dir': tmp_path / 'original',
        'filename': [tmp_path / 'recording.bin'],
        'data_dir': tmp_path / 'data',
    }
    preprocessing = {'whitening': torch.ones((1, 1))}
    ops = {
        'settings': settings,
        'filename': tmp_path / 'recording.bin',
        'data_dir': tmp_path / 'data',
        'preprocessing': preprocessing,
        'Wrot': torch.eye(1),
    }

    io.save_ops(ops, tmp_path / 'saved')

    assert ops['settings'] is settings
    assert ops['settings']['results_dir'] == tmp_path / 'original'
    assert ops['settings']['filename'] == [tmp_path / 'recording.bin']
    assert ops['settings']['data_dir'] == tmp_path / 'data'
    assert ops['filename'] == tmp_path / 'recording.bin'
    assert ops['data_dir'] == tmp_path / 'data'
    assert ops['preprocessing'] is preprocessing
    assert isinstance(ops['preprocessing']['whitening'], torch.Tensor)
    assert 'is_tensor' not in ops
