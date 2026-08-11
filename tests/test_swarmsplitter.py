import numpy as np

from kilosort.swarmsplitter import (
    bimod_score,
    check_CCG,
    check_split,
    labels_in,
    new_clusters,
)


def test_labels_in_matches_numpy_isin():
    labels = np.array([0, 1, 2, 1, 4, 0, 3], dtype=np.int64)
    members = [1, 3, 9]
    np.testing.assert_array_equal(labels_in(labels, members),
                                  np.isin(labels, members))
    # Empty edges
    np.testing.assert_array_equal(labels_in(labels, []),
                                  np.isin(labels, []))
    np.testing.assert_array_equal(labels_in(np.array([], dtype=np.int64), members),
                                  np.isin(np.array([], dtype=np.int64), members))


def test_labels_in_handles_negative_ids_like_isin():
    # Skeptic case: negatives must match np.isin, not be silently dropped.
    labels = np.array([-1, 0, 1], dtype=np.int64)
    members = [-1, 1]
    np.testing.assert_array_equal(labels_in(labels, members),
                                  np.isin(labels, members))
    labels2 = np.array([-5, -1, 2, 7], dtype=np.int64)
    members2 = [-5, 2, 99]
    np.testing.assert_array_equal(labels_in(labels2, members2),
                                  np.isin(labels2, members2))


def test_labels_in_large_spike_vector_matches_isin():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 500, size=50_000)
    members = rng.choice(500, size=40, replace=False)
    np.testing.assert_array_equal(labels_in(labels, members),
                                  np.isin(labels, members))
    # Mixed-sign random ids still match isin
    labels_n = rng.integers(-50, 50, size=20_000)
    members_n = rng.choice(np.arange(-50, 50), size=30, replace=False)
    np.testing.assert_array_equal(labels_in(labels_n, members_n),
                                  np.isin(labels_n, members_n))


def _reference_new_clusters(iclust, my_clus, xtree):
    """Isin-based remapping oracle for non-empty trees.

    Empty-tree policy: preserve labels (stock zeros was a multi-cluster collapse
    bug; tests assert the fixed behavior explicitly).
    """
    if len(xtree) == 0:
        return np.asarray(iclust).copy()

    xtree = np.array(xtree, copy=True)
    nc = xtree.max() + 1
    isleaf = np.zeros(2 * nc - 1,)
    isleaf[xtree[:, 0]] = 1
    isleaf[xtree[:, 1]] = 1
    isleaf[xtree[:, 2]] = 0
    ind = np.nonzero(isleaf)[0]
    iclust1 = np.asarray(iclust).copy()
    for j in range(len(ind)):
        ix = np.isin(iclust1 if False else iclust, my_clus[ind[j]])
        # Match original: mask from the input labels, write into copy
        ix = np.isin(iclust, my_clus[ind[j]])
        iclust1[ix] = j
        xtree[xtree[:, 0] == ind[j], 0] = j
        xtree[xtree[:, 1] == ind[j], 1] = j
    return iclust1, xtree


def test_new_clusters_matches_isin_reference():
    # Keep both base merges as the current tree. Leaves are original children
    # 0,1,2,3 (parents 4,5 are non-leaves). Remap is identity-shaped but still
    # exercises the dense remap path vs historical isin.
    xtree = np.array([
        [0, 1, 4],
        [2, 3, 5],
    ], dtype=np.int32)
    my_clus = [
        [0], [1], [2], [3],
        [1, 0], [3, 2],
    ]
    iclust = np.array([0, 0, 1, 1, 2, 2, 3, 3, 0, 2], dtype=np.int64)

    expected, expected_tree = _reference_new_clusters(iclust, my_clus, xtree)
    xtree_shipped = xtree.copy()
    got = new_clusters(iclust, my_clus, xtree_shipped, tstat=None)

    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(xtree_shipped, expected_tree)


def test_new_clusters_collapses_merged_leaf_membership():
    # Pruned tree keeps only the deeper merge. Leaves are intermediate node 3
    # (originals 0 and 1) and original 2 — matching post-split cleanup shape.
    xtree = np.array([[3, 2, 4]], dtype=np.int32)
    my_clus = [
        [0], [1], [2],
        [1, 0],
        [2, 1, 0],
    ]
    iclust = np.array([0, 1, 1, 2, 0, 2], dtype=np.int64)

    expected, expected_tree = _reference_new_clusters(iclust, my_clus, xtree)
    xtree_shipped = xtree.copy()
    got = new_clusters(iclust, my_clus, xtree_shipped, tstat=None)

    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(xtree_shipped, expected_tree)
    # Originals 0 and 1 share the leaf that owns [1, 0]; 2 is the other leaf.
    assert got[0] == got[1] == got[2] == got[4]
    assert got[3] == got[5]
    assert got[0] != got[3]


def test_new_clusters_empty_tree_preserves_labels():
    """Fully-split tree must not collapse multi-cluster labels to zero."""
    iclust = np.array([0, 1, 0, 1, 2], dtype=np.int64)
    got = new_clusters(iclust, my_clus=[[0], [1], [2]], xtree=[], tstat=None)
    np.testing.assert_array_equal(got, iclust)
    assert got is not iclust  # copy, not alias
    # Single-cluster still all zeros after preserve (already zeros).
    single = np.zeros(4, dtype=np.int64)
    np.testing.assert_array_equal(
        new_clusters(single, my_clus=[[0]], xtree=[], tstat=None), single
    )


def test_check_CCG_zero_span_returns_false():
    # All equal times → T==0; must not nan / throw.
    st = np.array([1.0, 1.0, 1.0])
    is_ref, cross = check_CCG(st, st)
    assert bool(is_ref) is False
    assert bool(cross) is False


def test_check_CCG_empty_returns_false():
    is_ref, cross = check_CCG(np.array([]), np.array([0.1, 0.2]))
    assert bool(is_ref) is False
    assert bool(cross) is False


def test_check_CCG_acg_no_copy_matches_explicit_pair():
    # Clean refractory unit: regular spikes with large ISI.
    st = np.arange(0.0, 2.0, 0.02)
    is_ref_acg, cross_acg = check_CCG(st)
    is_ref_pair, cross_pair = check_CCG(st, st.copy())
    assert is_ref_acg == is_ref_pair
    assert cross_acg == cross_pair
    assert bool(is_ref_acg) is True


def test_check_CCG_contaminated_not_refractory():
    # Burst-like contamination near 0 lag.
    rng = np.random.default_rng(0)
    base = np.sort(rng.uniform(0, 5, size=200))
    contaminated = np.sort(np.concatenate([base, base + 0.0005]))
    is_ref, cross = check_CCG(contaminated)
    assert bool(is_ref) is False


def test_check_CCG_assume_sorted_matches_unsorted_path():
    st1 = np.array([0.1, 0.5, 1.2, 2.0, 3.1])
    st2 = np.array([0.11, 0.49, 1.25, 2.05])
    a = check_CCG(st1, st2, assume_sorted=True)
    b = check_CCG(st1[::-1].copy(), st2[::-1].copy(), assume_sorted=False)
    assert a == b


def test_bimod_score_finite_on_empty_and_normal():
    # All mass outside hist range / flat → no crash, finite non-bimodal score.
    s0 = bimod_score(np.zeros(50))
    assert np.isfinite(s0)
    # Clear bimodal projection: two peaks around ±1
    left = np.random.default_rng(0).normal(-1.0, 0.15, size=400)
    right = np.random.default_rng(1).normal(1.0, 0.15, size=400)
    s1 = bimod_score(np.concatenate([left, right]))
    assert np.isfinite(s1)
    assert s1 > s0


def test_check_split_empty_membership_returns_zero_score():
    """No spikes in parent leaf → score 0, no linalg crash."""
    Xd = np.random.default_rng(0).standard_normal((20, 5)).astype(np.float64)
    iclust = np.zeros(20, dtype=np.int64)
    # Tree node 2 is parent of leaves 0,1 but members list empty for leaf 0
    xtree = np.array([[0, 1, 2]], dtype=np.int32)
    my_clus = [[], [1], []]  # parent membership empty → ixy all False
    xproj, score = check_split(Xd, 0, xtree, iclust, my_clus)
    assert score == 0.0
    assert xproj.size == 0


def test_check_split_single_class_returns_zero_score():
    """All spikes on one side of a proposed split → non-bimodal."""
    rng = np.random.default_rng(2)
    Xd = rng.standard_normal((30, 6)).astype(np.float64)
    iclust = np.zeros(30, dtype=np.int64)  # all label 0
    xtree = np.array([[0, 1, 2]], dtype=np.int32)
    my_clus = [[0], [1], [0, 1]]  # parent has 0 and 1; only 0 present in data
    xproj, score = check_split(Xd, 0, xtree, iclust, my_clus)
    assert score == 0.0
    assert xproj.shape == (30,)
