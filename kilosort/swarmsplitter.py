import os

import numpy as np
from numba import njit
import math
from scipy.ndimage import gaussian_filter1d
from scipy.stats import poisson
from kilosort.CCG import compute_CCG, CCG_metrics


# Fixed histogram edges for bimod_score (stock linspace(-2, 2, 400)).
_BIMOD_EDGES = np.linspace(-2.0, 2.0, 400)


def labels_in(labels, members):
    """Boolean membership of integer labels in `members` (np.isin-compatible).

    Hierarchical split/merge repeatedly tests spike labels against small leaf
    member lists. A dense boolean table over the closed integer range
    ``[min, max]`` of both arrays (including negatives) plus a gather matches
    ``np.isin`` while avoiding per-element hashing on dense cluster ids.
    Pathologically huge sparse ranges fall back to ``np.isin``.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return np.zeros(0, dtype=bool)
    members = np.asarray(members, dtype=np.int64).ravel()
    if members.size == 0:
        return np.zeros(labels.shape, dtype=bool)

    lo = int(min(labels.min(), members.min()))
    hi = int(max(labels.max(), members.max()))
    size = hi - lo + 1
    # Guard sparse huge spans (e.g. a single very large id): hashing is fine.
    if size > max(2_000_000, 8 * (labels.size + members.size)):
        return np.isin(labels, members)

    table = np.zeros(size, dtype=bool)
    table[members - lo] = True
    return table[labels - lo]


def _member_bool_tables(my_clus, n_labels):
    """Dense bool tables: table[node][label] for original labels in ``0..n_labels-1``.

    Production clustering uses non-negative dense original ids. Building once at
    the start of ``split`` turns every ``labels_in`` into a gather. Nodes whose
    members fall outside that range (or are empty) still get a valid table
    (empty → all-False; out-of-range → fall back via labels_in at use sites
    is not needed if we expand n_labels).
    """
    tables = []
    for members in my_clus:
        t = np.zeros(int(n_labels), dtype=bool)
        if members:
            m = np.asarray(members, dtype=np.int64).ravel()
            if m.size:
                # Clip to table; production members are always in range.
                good = (m >= 0) & (m < n_labels)
                if np.any(good):
                    t[m[good]] = True
        tables.append(t)
    return tables


def count_elements(kk, iclust, my_clus, xtree):
    n1 = labels_in(iclust, my_clus[xtree[kk, 0]]).sum()
    n2 = labels_in(iclust, my_clus[xtree[kk, 1]]).sum()
    return n1, n2

def check_split(Xd, kk, xtree, iclust, my_clus, member_tables=None):
    if member_tables is not None:
        ixy = member_tables[int(xtree[kk, 2])][iclust]
    else:
        ixy = labels_in(iclust, my_clus[xtree[kk, 2]])
    # Empty membership (remapped labels / fully pruned branch): not bimodal.
    if not np.any(ixy):
        return np.zeros(0, dtype=np.float64), 0.0

    iclu = iclust[ixy]
    if member_tables is not None:
        # 2*bool - 1 → ±1 int (same dtype promotion as 2*labels_in(...) - 1)
        labels = 2 * member_tables[int(xtree[kk, 0])][iclu] - 1
    else:
        labels = 2*labels_in(iclu, my_clus[xtree[kk, 0]]) - 1

    Xs = Xd[ixy]
    pos = labels > 0
    neg = labels < 0
    if not np.any(pos) or not np.any(neg):
        return np.zeros(Xs.shape[0], dtype=np.float64), 0.0

    Xs[:,-1] = 1

    n = float(labels.shape[0])
    n_pos = float(np.count_nonzero(pos))
    n_neg = float(np.count_nonzero(neg))
    w = np.empty((Xs.shape[0], 1), dtype=np.float64)
    w[pos, 0] = n_neg / n
    w[neg, 0] = n_pos / n

    Xw = Xs * w
    CC = Xs.T @ Xw
    CC = CC + .01 * np.eye(CC.shape[0])
    b = np.linalg.solve(CC, labels @ Xw)
    xproj = Xs @ b

    score = bimod_score(xproj)
    return xproj, score

def _parent_edge_lists(xtree):
    """Map parent node id (xtree[:, 2]) → edge indices into xtree / valid_merge.

    Built once per ``split`` so ``clean_tree`` is O(edges visited) instead of
    rescanning the full tree on every prune.
    """
    n_edges = int(xtree.shape[0])
    if n_edges == 0:
        return []
    max_node = int(xtree[:, 2].max())
    buckets = [[] for _ in range(max_node + 1)]
    for i in range(n_edges):
        buckets[int(xtree[i, 2])].append(i)
    return buckets


def clean_tree(valid_merge, xtree, inode, parent_edges=None):
    # Iterative (was recursive): deep trees on large centers could overflow.
    stack = [int(inode)]
    while stack:
        node = stack.pop()
        if parent_edges is not None:
            if node < 0 or node >= len(parent_edges):
                continue
            ix_list = parent_edges[node]
            if not ix_list:
                continue
            for i in ix_list:
                valid_merge[i] = 0
                stack.append(int(xtree[i, 0]))
                stack.append(int(xtree[i, 1]))
        else:
            ix = (xtree[:, 2] == node).nonzero()[0]
            if ix.size == 0:
                continue
            valid_merge[ix] = 0
            for i in ix:
                stack.append(int(xtree[i, 0]))
                stack.append(int(xtree[i, 1]))
    return

def bimod_score(xproj):
    xbin, _ = np.histogram(xproj, _BIMOD_EDGES)
    xbin = gaussian_filter1d(xbin.astype('float32'), 4)

    imin = np.argmin(xbin[175:225])
    xmin = np.min(xbin[175:225])
    xm1  = np.max(xbin[:imin+175])
    xm2  = np.max(xbin[imin+175:])

    # The valley is only ever looked for in bins 175:225, i.e. xproj in about
    # [-0.25, +0.25] of a fixed [-2, 2] axis. A minority second cell pulls the
    # true valley off centre, and mass outside [-2, 2] is not binned at all --
    # either way the score describes the wrong place. Record both so the audit
    # can say whether that is happening rather than assuming it is not.
    _LAST_BIMOD['imin'] = int(imin)
    _LAST_BIMOD['edge'] = int(imin == 0 or imin == 49)
    _LAST_BIMOD['outside'] = float(np.mean((xproj < -2.0) | (xproj > 2.0)))

    # Empty half of the projection (or all mass outside [-2, 2]) used to make
    # xmin/xm1 → inf/nan and poison split criterion. Treat as non-bimodal.
    if xm1 <= 0 or xm2 <= 0:
        return 0.0
    score = 1 - np.maximum(xmin / xm1, xmin / xm2)
    return float(score)

# The cross-correlogram threshold that decides "these two are one neuron, never
# split them". It was hardcoded at .25 here, unreachable from settings: the
# `ccg_threshold` setting feeds kilosort.CCG, which is a different function, so
# sweeping it moved split rate 20.3% -> 20.1% and looked like a dead knob.
# Raising this makes the splitter more willing to call two candidates one unit.
SPLIT_CCG_THRESHOLD = 0.25


# Per-decision audit of the splitter, off unless KS4_SPLIT_STATS names a file.
#
# The splitter is the only stage whose job is to notice that one cluster holds
# two neurons, and it is demonstrably letting ~45 units per sort through. Which
# of its three gates does the letting through is not something the sorted output
# can answer -- a unit that was never split looks identical to one that was
# never a candidate. So record every decision: the gate that fired, the score it
# fired on, and the sizes involved.
_LAST_CCG = {}
_LAST_BIMOD = {}
_STATS_PATH = os.environ.get('KS4_SPLIT_STATS')


# Veto a merge whose union cannot be one neuron, whatever the other gates say.
#
# The clustering stage hands over MORE contaminated spike mass than its own tree
# leaves carry -- 10.2% -> 13.0% on d007, 10.4% -> 11.8% on d005 -- so its
# accepted merges are building contaminated units out of cleaner pieces. And
# they are visible: 25% of the merges split() keeps produce a union with
# sub-refractory ISIs far above chance, nearly all of them waved through by the
# CCG gate. That gate asks whether the two halves fire together; it never asks
# whether the result could be a single cell.
#
# On by default; the `refractory_merge_veto` setting carries the measurements.
# The bar is the same one the GT-free contamination metric uses, so a unit this
# vetoes is a unit that metric would have counted.
REFRAC_VETO = True

# The bar itself. Both halves matter and neither is arbitrary: RATIO is how much
# of the chance-expected violation count a single cell is allowed to produce,
# ALPHA is how sure we have to be that the excess is not Poisson noise. They are
# module-level and settable so the bar can be swept rather than asserted -- the
# shipped 0.35/0.01 was inherited from the GT-free contamination metric, which
# makes the headline partly self-referential and makes an independent sweep of
# these two the obvious next measurement.
REFRAC_VETO_RATIO = 0.35
REFRAC_VETO_ALPHA = 0.01


def _impossible(obs, exp, ratio=REFRAC_VETO_RATIO, alpha=REFRAC_VETO_ALPHA):
    """Is this violation count too high to be one neuron, with power to say so?

    Two conditions, not one: the count has to be a real fraction of what
    independence would produce (a handful of violations on a huge train is a
    clean cell), and it has to be unlikely under Poisson at that rate (a
    handful on a small train is noise). Either alone misfires -- the raw
    violation percentage is diluted by unit size, which is how an earlier ISI
    test produced five false over-splits.
    """
    if exp <= 0:
        return False
    return obs >= ratio * exp and poisson.sf(obs - 1, exp) < alpha


def _rvi(st, refrac=0.0015):
    """Sub-refractory ISI count and its chance expectation for one train.

    Same convention as the GT-free contamination metric: a train whose observed
    count sits well above the rate-matched expectation holds more than one
    neuron. Returned raw so the audit, not this function, decides the bar.
    """
    st = np.asarray(st)
    n = int(st.size)
    if n < 50:
        return n, 0, 0.0
    st = np.sort(st)
    T = float(st[-1] - st[0])
    if T <= 0:
        return n, 0, 0.0
    obs = int(np.count_nonzero(np.diff(st) < refrac))
    return n, obs, (n - 1) * (n / T) * refrac


_TRACE_PATH = os.environ.get('KS4_CLUSTER_TRACE')
_TRACE_CENTER = [0]


def write_trace_stats(meta, snapshots, iclust_final):
    """Is there a partition on the way down that is finer AND cleaner?

    cluster() collapses 200 k-means++ seeds to ~13 labels, and those labels are
    already 12.6% refractorily impossible -- a fusion nothing downstream can
    undo, because maketree only agglomerates and split() only prunes merges.
    So the collapse is the suspect. Record the partition at a ladder of
    iterations, plus the fixed point as t=-1, with per-leaf refractory counts,
    and let the analysis say whether an intermediate granularity exists at all.

    Leaves are written whole, including tiny ones: a leaf below the 300-spike
    floor cannot be judged clean or dirty, and dropping it here would hide how
    much of the spike mass the finer partitions shatter -- which is the cost
    side of the trade and the thing that killed the init-label probe.
    """
    if not _TRACE_PATH:
        return
    _TRACE_CENTER[0] += 1
    c = _TRACE_CENTER[0]
    meta = np.asarray(meta)
    new = not os.path.exists(_TRACE_PATH)
    with open(_TRACE_PATH, 'a') as f:
        if new:
            f.write('center\tt\tnclust\tleaf\tn\tobs\texp\n')
        for t, lab in list(snapshots) + [(-1, np.asarray(iclust_final))]:
            u = np.unique(lab)
            for j in u:
                n, obs, exp = _rvi(meta[lab == j])
                f.write('%d\t%d\t%d\t%d\t%d\t%d\t%.6g\n'
                        % (c, t, u.size, int(j), n, obs, exp))


def write_init_stats(meta, iclust, iclust_init):
    """Does the partition the sorter THREW AWAY separate what its leaves fuse?

    cluster() seeds 200 k-means++ centres and then collapses them to ~13 labels
    in the alternating-assignment loop. The pre-collapse labels are returned and
    dropped on the floor at the call site. If a refractorily impossible leaf
    breaks into refractorily clean pieces under those labels, the information
    needed to separate two neurons was computed and discarded, and the fix is
    architectural. If it does not, the features themselves cannot tell the two
    cells apart and no amount of re-partitioning will help.
    """
    if not _STATS_PATH:
        return
    path = _STATS_PATH + '.init'
    new = not os.path.exists(path)
    with open(path, 'a') as f:
        if new:
            f.write('leaf\tn\tobs\texp\tsub\tsub_n\tsub_obs\tsub_exp\n')
        for c in np.unique(iclust):
            m = iclust == c
            n, obs, exp = _rvi(meta[m])
            if n < 300:
                continue
            sub = iclust_init[m]
            for j in np.unique(sub):
                sn, so, se = _rvi(meta[m][sub == j])
                f.write('%d\t%d\t%d\t%.6g\t%d\t%d\t%d\t%.6g\n'
                        % (c, n, obs, exp, int(j), sn, so, se))


def write_post_stats(meta, iclust):
    """The units the CLUSTERING stage hands over, before template matching.

    The leaves get cleaner when the collapse is cut short, but the final output
    gets dirtier -- so the contamination in the output is not simply inherited
    from the partition. Something between the two redistributes it. Scoring the
    labels at this exact point splits the pipeline in half: contamination
    already here is the clustering stage's, contamination only in
    spike_clusters.npy is template matching assigning two neurons' spikes to one
    template.
    """
    if not _STATS_PATH:
        return
    path = _STATS_PATH + '.post'
    new = not os.path.exists(path)
    with open(path, 'a') as f:
        if new:
            f.write('unit\tn\tobs\texp\n')
        for c in np.unique(iclust):
            f.write('%d\t%d\t%d\t%.6g\n' % ((int(c),) + _rvi(meta[iclust == c])))


def _write_leaf_stats(meta, iclust):
    """Is the partition already impure BEFORE any tree merge is accepted?

    hierarchical.maketree only ever agglomerates and split() only ever prunes
    merges, so cluster()'s labels are the finest partition the sorter will ever
    hold. If those leaves already carry sub-refractory ISIs, no downstream gate
    can help and the defect is upstream of everything measured so far.
    """
    rows = []
    for c in np.unique(iclust):
        rows.append((int(c),) + _rvi(meta[iclust == c]))
    path = _STATS_PATH + '.leaves'
    new = not os.path.exists(path)
    with open(path, 'a') as f:
        if new:
            f.write('leaf\tn\tobs\texp\n')
        for r in rows:
            f.write('%d\t%d\t%d\t%.6g\n' % r)


def _write_stats(rows):
    if not rows:
        return
    new = not os.path.exists(_STATS_PATH)
    with open(_STATS_PATH, 'a') as f:
        if new:
            f.write('kk\tn1\tn2\tmod\tlocalmod\tgate\tscore\t'
                    'R12\tQ12\tQ00\timin\tedge\toutside\t'
                    'o1\te1\to2\te2\tuo\tue\tsplit\n')
        for r in rows:
            f.write('\t'.join('%.6g' % x if isinstance(x, float) else str(x)
                               for x in r) + '\n')


def check_CCG(st1, st2=None, nbins = 500, tbin  = 1/1000, assume_sorted=False,
              split_ccg_threshold=SPLIT_CCG_THRESHOLD):
    # ACG path: reuse the same array. compute_CCG rebinds sorted views and does
    # not mutate spike times in place, so a defensive copy is wasted memory.
    if st2 is None:
        st2 = st1
    st1 = np.asarray(st1)
    st2 = np.asarray(st2)
    # Correct empty / zero-span guard (upstream 4.1.3 wrote `len(st2 == 0)`,
    # which is always truthy for non-empty st2 and disabled all CCG checks).
    # T==0 (all equal times) still divides by T in CCG_metrics → nans → both
    # flags false; short-circuit instead of computing garbage.
    if st1.size == 0 or st2.size == 0:
        return False, False
    K, T = compute_CCG(st1, st2, nbins=nbins, tbin=tbin,
                       assume_sorted=assume_sorted)
    if T == 0:
        return False, False
    R12, Q12, Q00 = CCG_metrics(st1, st2, K, T,  nbins = nbins, tbin = tbin)
    is_refractory    = R12<.1  and (Q12<.2  or Q00<.25)
    cross_refractory = R12<split_ccg_threshold and (Q12<.05 or Q00<.25)
    # The veto's own numbers, for KS4_SPLIT_STATS. R12 is a ratio, so a pair
    # with few spikes can land under the threshold on noise alone; the audit
    # cannot tell that from a real refractory dip without seeing the counts.
    _LAST_CCG['R12'], _LAST_CCG['Q12'], _LAST_CCG['Q00'] = R12, Q12, Q00
    _LAST_CCG['n1'], _LAST_CCG['n2'] = int(st1.size), int(st2.size)
    return is_refractory, cross_refractory

def refractoriness(st1, st2, assume_sorted=False,
                   split_ccg_threshold=SPLIT_CCG_THRESHOLD):
    # compute goodness of st1, st2, and both
    # Production clustering passes time-ordered spike times (global detect order
    # + increasing igood), and boolean masks preserve that order — so callers
    # can set assume_sorted=True to skip two O(n log n) sorts per CCG check.

    is_refractory = check_CCG(st1, st2, assume_sorted=assume_sorted,
                              split_ccg_threshold=split_ccg_threshold)[1]
    if is_refractory:
        criterion = 1 # never split
        #print('this is refractory')
    else:
        criterion = 0
        #good_0 = check_CCG(np.hstack((st1,st2)))[0]
        #good_1 = check_CCG(st1)[0]
        #good_2 = check_CCG(st2)[0]
        #print(good_0, good_1, good_2)
        #if (good_0==1) and (good_1==0) and (good_2==0):
        #    criterion = 1 # don't split
        #    print('good cluster becomes bad')
    return criterion

def split(Xd, xtree, tstat, iclust, my_clus, verbose = False, meta = None,
          meta_sorted=True, split_ccg_threshold=SPLIT_CCG_THRESHOLD,
          refrac_veto=REFRAC_VETO, refrac_veto_ratio=REFRAC_VETO_RATIO,
          refrac_veto_alpha=REFRAC_VETO_ALPHA):
    xtree = np.array(xtree)
    iclust = np.asarray(iclust)

    kk = xtree.shape[0]-1
    nc = xtree.shape[0] + 1
    valid_merge = np.ones((nc-1,), 'bool')

    # Precompute dense membership tables once. Original labels from hierarchical
    # clustering are non-negative and dense in 0..nc0-1; my_clus nodes only
    # contain those originals. Gather-based membership matches labels_in.
    n_labels = 0
    for members in my_clus:
        if members:
            n_labels = max(n_labels, int(max(members)) + 1)
    if iclust.size:
        n_labels = max(n_labels, int(iclust.max()) + 1)
    member_tables = _member_bool_tables(my_clus, max(n_labels, 1))
    # Parent→edge lists for O(visited) clean_tree (was O(n_edges) scan/prune).
    parent_edges = _parent_edge_lists(xtree)

    stats = [] if _STATS_PATH else None
    if stats is not None and meta is not None:
        _write_leaf_stats(np.asarray(meta), iclust)

    for kk in range(nc-2,-1,-1):
        if not valid_merge[kk]:
            continue

        score = np.nan
        gate = 'mod'
        ix1 = ix2 = None
        _LAST_CCG.clear()
        _LAST_BIMOD.clear()
        # first mutation is global modularity — reject before membership gathers
        if tstat[kk, 0] < 0.2:
            criterion = -1
        else:
            criterion = 0
            left = int(xtree[kk, 0])
            right = int(xtree[kk, 1])
            ix1 = member_tables[left][iclust]
            ix2 = member_tables[right][iclust]

            if meta is not None and criterion == 0:
                # second mutation is based on meta_data
                criterion = refractoriness(
                    meta[ix1], meta[ix2], assume_sorted=meta_sorted,
                    split_ccg_threshold=split_ccg_threshold
                )
                if criterion == 1:
                    gate = 'ccg'

            if criterion == 0:
                xproj, score = check_split(
                    Xd, kk, xtree, iclust, my_clus, member_tables=member_tables
                )
                # third mutation is bimodality
                criterion = 2 * (score < .6) - 1
                gate = 'bimod'

        # Whatever the gates concluded, a union that cannot be one neuron is
        # not one unit. This runs after them and only ever turns a KEEP-MERGED
        # into a split, so it can add fragments but never fuse anything.
        if (refrac_veto and criterion == 1 and meta is not None
                and ix1 is not None):
            _, _uo, _ue = _rvi(meta[ix1 | ix2])
            if _impossible(_uo, _ue, refrac_veto_ratio, refrac_veto_alpha):
                criterion = -1
                gate = 'refrac'

        if stats is not None:
            n1 = int(ix1.sum()) if ix1 is not None else -1
            n2 = int(ix2.sum()) if ix2 is not None else -1
            # Would a refractory counter-veto have anything to fire on? Record
            # the union's violation count against each half's, for every node,
            # split or not -- that is the evidence a "geometrically unimodal but
            # refractorily impossible" rule would need, and it is not currently
            # consulted anywhere in the pipeline.
            uo = ue = o1 = e1 = o2 = e2 = -1.0
            if meta is not None and ix1 is not None:
                _, o1, e1 = _rvi(meta[ix1])
                _, o2, e2 = _rvi(meta[ix2])
                _, uo, ue = _rvi(meta[ix1 | ix2])
            stats.append((kk, n1, n2, float(tstat[kk, 0]), float(tstat[kk, -1]),
                          gate, float(score),
                          float(_LAST_CCG.get('R12', np.nan)),
                          float(_LAST_CCG.get('Q12', np.nan)),
                          float(_LAST_CCG.get('Q00', np.nan)),
                          int(_LAST_BIMOD.get('imin', -1)),
                          int(_LAST_BIMOD.get('edge', -1)),
                          float(_LAST_BIMOD.get('outside', np.nan)),
                          int(o1), float(e1), int(o2), float(e2),
                          int(uo), float(ue),
                          int(criterion != 1)))

        if criterion == 0:
            # fourth mutation is local modularity (not reachable)
            score = tstat[kk, -1]
            criterion = score > .15

        if verbose:
            n1, n2 = int(ix1.sum()), int(ix2.sum())
                # print('%3.0d, %6.0d, %6.0d, %6.0d, %2.2f,%4.2f, %2.2f'%(kk, n1, n2,n1+n2,
                # tstat[kk,0], tstat[kk,-1], score))

        if criterion == 1:
            valid_merge[kk] = 0
            clean_tree(valid_merge, xtree, xtree[kk, 0], parent_edges)
            clean_tree(valid_merge, xtree, xtree[kk, 1], parent_edges)

    if stats is not None:
        _write_stats(stats)

    tstat = tstat[valid_merge]
    xtree = xtree[valid_merge]

    return xtree, tstat


def new_clusters(iclust, my_clus, xtree, tstat):

    # Empty tree after split() means every hierarchical merge was rejected:
    # leaves are the original labels. Returning zeros (historical stock) silently
    # collapses multi-cluster centers into a single cluster — a real correctness
    # bug on fully-split trees. Preserve labels instead.
    if len(xtree) == 0:
        return np.asarray(iclust).copy()

    nc = xtree.max() + 1

    isleaf = np.zeros(2*nc-1,)
    isleaf[xtree[:,0]] = 1
    isleaf[xtree[:,1]] = 1
    isleaf[xtree[:,2]] = 0

    ind = np.nonzero(isleaf)[0]
    iclust_arr = np.asarray(iclust)
    if ind.size == 0:
        return iclust_arr.copy()

    # One pass over leaf membership builds a dense remap so each spike is
    # reassigned with a single integer gather instead of O(n_leaves) isin scans.
    max_label = -1
    for leaf in ind:
        members = my_clus[leaf]
        if members:
            m = max(members)
            if m > max_label:
                max_label = m
    if max_label < 0:
        return iclust_arr.copy()

    remap = np.full(max_label + 1, -1, dtype=np.int64)
    # Node id → new leaf id for xtree child remapping (vectorized after loop).
    max_node = int(max(ind.max(), xtree.max()))
    node_remap = np.arange(max_node + 1, dtype=np.int64)
    for j, leaf in enumerate(ind):
        leaf = int(leaf)
        for orig in my_clus[leaf]:
            remap[orig] = j
        node_remap[leaf] = j

    # Vectorized leaf→new-id rewrite of both tree child columns.
    xtree[:, 0] = node_remap[xtree[:, 0]]
    xtree[:, 1] = node_remap[xtree[:, 1]]

    # Preserve original labels for any spike id not present in a leaf (same as
    # the historical isin loop, which only wrote matched membership).
    iclust1 = iclust_arr.copy()
    known = (iclust_arr >= 0) & (iclust_arr <= max_label)
    if np.any(known):
        mapped = remap[iclust_arr[known]]
        take = mapped >= 0
        idx = np.flatnonzero(known)
        iclust1[idx[take]] = mapped[take]
    return iclust1
