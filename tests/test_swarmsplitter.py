import numpy as np

from kilosort.swarmsplitter import check_CCG, new_clusters


def _reference_new_clusters(iclust, my_clus, xtree):
    """Historical isin-based remapping (identity oracle for the shipped path)."""
    if len(xtree) == 0:
        return np.zeros_like(iclust)

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


def test_new_clusters_empty_tree_returns_zeros():
    iclust = np.array([0, 1, 0], dtype=np.int64)
    got = new_clusters(iclust, my_clus=[[0], [1]], xtree=[], tstat=None)
    np.testing.assert_array_equal(got, np.zeros_like(iclust))


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
