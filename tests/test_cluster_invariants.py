"""Focused CPU characterization tests for clustering/splitting audit findings.

The strict xfails in this file are intentional: they describe invariants we
want to make true in a future production change, while the passing tests
record the current behavior that motivates each change.
"""

import numpy as np
import pytest
import torch

from kilosort import clustering_qr, swarmsplitter


def _same_numpy_rng_state(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_bimod_score_currently_depends_on_fixed_coordinate_range():
    """A clear mixture stops looking bimodal after large affine transforms."""
    rng = np.random.default_rng(2026)
    mixture = np.concatenate((
        rng.normal(-1.0, 0.08, 1_000),
        rng.normal(1.0, 0.08, 1_000),
    ))

    base = swarmsplitter.bimod_score(mixture)
    scaled = swarmsplitter.bimod_score(10.0 * mixture)
    translated = swarmsplitter.bimod_score(mixture + 10.0)

    assert base > 0.9
    assert scaled < 0.1
    assert translated < 0.1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Desired invariant: bimod_score should preserve a clear two-mode "
        "decision under positive scaling and translation; current scoring "
        "uses a fixed [-2, 2] histogram and a fixed central valley window."
    ),
)
def test_bimod_score_should_be_affine_invariant_for_clear_mixture():
    rng = np.random.default_rng(2026)
    mixture = np.concatenate((
        rng.normal(-1.0, 0.08, 1_000),
        rng.normal(1.0, 0.08, 1_000),
    ))
    scores = np.array([
        swarmsplitter.bimod_score(mixture),
        swarmsplitter.bimod_score(10.0 * mixture),
        swarmsplitter.bimod_score(mixture + 10.0),
    ])
    np.testing.assert_allclose(scores, scores[0], rtol=0.0, atol=1e-12)


def test_split_local_modularity_fallback_is_unreachable(monkeypatch):
    """The local-modularity block cannot see criterion == 0 after check_split.

    With a score above the bimodality split threshold, ``check_split`` sets
    criterion to -1. If the later local-modularity fallback were reachable,
    changing tstat[:, -1] from 0 to 1 would change the returned tree.
    """
    calls = []

    def fake_check_split(*args, **kwargs):
        calls.append(True)
        return np.zeros(4), 0.8

    monkeypatch.setattr(swarmsplitter, "check_split", fake_check_split)

    Xd = np.zeros((4, 2), dtype=np.float64)
    iclust = np.array([0, 0, 1, 1], dtype=np.int64)
    my_clus = [[0], [1], [0, 1]]
    xtree = np.array([[0, 1, 2]], dtype=np.int32)

    low_mod = np.array([[1.0, 2.0, 0.0]], dtype=np.float32)
    high_mod = np.array([[1.0, 2.0, 1.0]], dtype=np.float32)
    low_result = swarmsplitter.split(
        Xd, xtree, low_mod, iclust, my_clus, meta=None
    )
    high_result = swarmsplitter.split(
        Xd, xtree, high_mod, iclust, my_clus, meta=None
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(low_result[0], high_result[0])
    assert low_result[0].shape == (1, 3)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Desired invariant: kmeans_plusplus should use local RNG generators "
        "and leave caller NumPy/Torch global RNG state unchanged; current "
        "implementation calls torch.manual_seed and np.random.seed."
    ),
)
def test_kmeans_plusplus_should_preserve_global_rng_state():
    Xg = torch.from_numpy(
        np.random.default_rng(7).normal(size=(200, 4)).astype(np.float32)
    )
    torch.manual_seed(1234)
    np.random.seed(5678)
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()

    try:
        clustering_qr.kmeans_plusplus(
            Xg, niter=4, seed=9, device=torch.device("cpu")
        )
        torch_after = torch.random.get_rng_state()
        numpy_after = np.random.get_state()
    finally:
        torch.random.set_rng_state(torch_before)
        np.random.set_state(numpy_before)

    assert torch.equal(torch_before, torch_after)
    assert _same_numpy_rng_state(numpy_before, numpy_after)


def test_symmetric_center_tie_currently_depends_on_torch_rng():
    """The current 1e-20 tie noise selects different equal-distance centers."""
    xy = torch.tensor([[0.0], [0.0]], dtype=torch.float64)
    xcent = np.array([-0.001, 0.001], dtype=np.float64)
    ycent = np.array([0.0], dtype=np.float64)
    original_state = torch.random.get_rng_state().clone()

    try:
        winners = set()
        for seed in (0, 1):
            torch.manual_seed(seed)
            nearest, _, _ = clustering_qr.get_nearest_centers(
                xy, xcent, ycent
            )
            winners.add(int(nearest[0]))
    finally:
        torch.random.set_rng_state(original_state)

    assert winners == {0, 1}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Desired invariant: exactly symmetric center ties should have a "
        "documented deterministic winner (the lowest flattened center index); "
        "current tie-breaking consumes Torch RNG state."
    ),
)
def test_symmetric_center_tie_should_choose_lowest_flattened_index():
    xy = torch.tensor([[0.0], [0.0]], dtype=torch.float64)
    xcent = np.array([-0.001, 0.001], dtype=np.float64)
    ycent = np.array([0.0], dtype=np.float64)
    original_state = torch.random.get_rng_state().clone()

    try:
        for seed in (0, 1):
            torch.manual_seed(seed)
            nearest, _, _ = clustering_qr.get_nearest_centers(
                xy, xcent, ycent
            )
            assert int(nearest[0]) == 0
    finally:
        torch.random.set_rng_state(original_state)


def test_run_exercises_small_center_one_cluster_shortcut(monkeypatch):
    """A 999-spike center takes the actual run() shortcut, not cluster()."""
    n_spikes = 999
    n_pcs = 2
    xy = torch.zeros((2, n_spikes), dtype=torch.float32)
    iC = torch.zeros((1, n_spikes), dtype=torch.long)
    Xd = torch.ones((n_spikes, 2), dtype=torch.float32)
    igood = torch.arange(n_spikes, dtype=torch.long)
    ichan = torch.tensor([0], dtype=torch.long)

    monkeypatch.setattr(
        clustering_qr, "xy_templates", lambda ops: (xy, iC)
    )
    monkeypatch.setattr(
        clustering_qr, "x_centers", lambda ops: np.array([0.0])
    )
    monkeypatch.setattr(
        clustering_qr, "y_centers", lambda ops: np.array([0.0])
    )
    monkeypatch.setattr(
        clustering_qr,
        "get_nearest_centers",
        lambda xy_arg, x_arg, y_arg: (
            torch.zeros(n_spikes, dtype=torch.long),
            torch.tensor([0.0]),
            torch.tensor([0.0]),
        ),
    )
    monkeypatch.setattr(
        clustering_qr,
        "get_data_cpu",
        lambda *args, **kwargs: (Xd, igood, ichan),
    )
    monkeypatch.setattr(
        clustering_qr,
        "mean_cluster_templates",
        lambda Xd_arg, iclust_arg, ichan_arg, n_chan_arg, n_pcs_arg: (
            torch.ones((1, n_chan_arg, n_pcs_arg), dtype=Xd_arg.dtype)
        ),
    )

    def cluster_must_not_run(*args, **kwargs):
        raise AssertionError("small center did not take the shortcut")

    monkeypatch.setattr(clustering_qr, "cluster", cluster_must_not_run)

    ops = {
        "dmin": 20,
        "dminx": 32,
        "Nchan": 1,
        "xcup": np.array([0.0]),
        "ycup": np.array([0.0]),
        "settings": {
            "cluster_downsampling": 1,
            "cluster_neighbors": 10,
            "max_cluster_subset": None,
            "cluster_init_seed": 1,
            "n_pcs": n_pcs,
        },
    }
    st = np.zeros((n_spikes, 2), dtype=np.float32)
    tF = torch.zeros((n_spikes, 1, 1), dtype=torch.float32)

    clu, wall = clustering_qr.run(
        ops, st, tF, mode="template", device=torch.device("cpu")
    )

    assert clu.shape == (n_spikes,)
    np.testing.assert_array_equal(clu, np.zeros(n_spikes, dtype=np.int32))
    assert wall.shape == (1, 1, n_pcs)
