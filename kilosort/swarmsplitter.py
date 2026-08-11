import numpy as np
from numba import njit
import math
from kilosort.CCG import compute_CCG, CCG_metrics


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


def count_elements(kk, iclust, my_clus, xtree):
    n1 = labels_in(iclust, my_clus[xtree[kk, 0]]).sum()
    n2 = labels_in(iclust, my_clus[xtree[kk, 1]]).sum()
    return n1, n2

def check_split(Xd, kk, xtree, iclust, my_clus):
    ixy = labels_in(iclust, my_clus[xtree[kk, 2]])
    # Empty membership (remapped labels / fully pruned branch): not bimodal.
    if not np.any(ixy):
        return np.zeros(0, dtype=np.float64), 0.0

    iclu = iclust[ixy]
    labels = 2*labels_in(iclu, my_clus[xtree[kk, 0]]) - 1

    Xs = Xd[ixy]
    # One class empty → weighted LS / bimod_score are meaningless; treat as
    # non-bimodal rather than solving a singular/zero-weight system.
    pos = labels > 0
    neg = labels < 0
    if not np.any(pos) or not np.any(neg):
        return np.zeros(Xs.shape[0], dtype=np.float64), 0.0

    Xs = Xs.copy()
    Xs[:,-1] = 1

    w = np.ones((Xs.shape[0],1))
    w[pos] = np.mean(neg)
    w[neg] = np.mean(pos)

    CC = Xs.T @ (Xs * w)
    CC = CC + .01 * np.eye(CC.shape[0])
    b = np.linalg.solve(CC, labels @ (Xs * w))
    xproj = Xs @ b

    score = bimod_score(xproj)
    return xproj, score

def clean_tree(valid_merge, xtree, inode):
    ix = (xtree[:,2]==inode).nonzero()[0]
    if len(ix)==0:
        return
    valid_merge[ix] = 0
    clean_tree(valid_merge, xtree, xtree[ix, 0])
    clean_tree(valid_merge, xtree, xtree[ix, 1])
    return

def bimod_score(xproj):
    from scipy.ndimage import gaussian_filter1d
    xbin, _ = np.histogram(xproj, np.linspace(-2,2,400))
    xbin = gaussian_filter1d(xbin.astype('float32'), 4)

    imin = np.argmin(xbin[175:225])
    xmin = np.min(xbin[175:225])
    xm1  = np.max(xbin[:imin+175])
    xm2  = np.max(xbin[imin+175:])

    # Empty half of the projection (or all mass outside [-2, 2]) used to make
    # xmin/xm1 → inf/nan and poison split criterion. Treat as non-bimodal.
    if xm1 <= 0 or xm2 <= 0:
        return 0.0
    score = 1 - np.maximum(xmin / xm1, xmin / xm2)
    return float(score)

def check_CCG(st1, st2=None, nbins = 500, tbin  = 1/1000, assume_sorted=False):
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
    cross_refractory = R12<.25 and (Q12<.05 or Q00<.25)
    return is_refractory, cross_refractory

def refractoriness(st1, st2):
    # compute goodness of st1, st2, and both

    is_refractory = check_CCG(st1, st2)[1]
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

def split(Xd, xtree, tstat, iclust, my_clus, verbose = True, meta = None):
    xtree = np.array(xtree)

    kk = xtree.shape[0]-1
    nc = xtree.shape[0] + 1
    valid_merge = np.ones((nc-1,), 'bool')


    for kk in range(nc-2,-1,-1):
        if not valid_merge[kk]:
            continue;

        ix1 = labels_in(iclust, my_clus[xtree[kk, 0]])
        ix2 = labels_in(iclust, my_clus[xtree[kk, 1]])

        criterion = 0
        score = np.nan
        if criterion==0:
            # first mutation is global modularity
            if tstat[kk,0] < 0.2:
                criterion = -1


        if meta is not None and criterion==0:
            # second mutation is based on meta_data
            criterion = refractoriness(meta[ix1],meta[ix2])
            #criterion = 0
        
        if criterion==0:
            xproj, score = check_split(Xd, kk, xtree, iclust, my_clus)
            # third mutation is bimodality
            #xproj, score = check_split(Xd, kk, xtree, iclust, my_clus)
            criterion = 2 * (score <  .6) - 1

        if criterion==0:
            # fourth mutation is local modularity (not reachable)
            score = tstat[kk,-1]
            criterion = score > .15

        if verbose:
            n1,n2 = ix1.sum(), ix2.sum()
            #print('%3.0d, %6.0d, %6.0d, %6.0d, %2.2f,%4.2f, %2.2f'%(kk, n1, n2,n1+n2,
            #tstat[kk,0], tstat[kk,-1], score))

        if criterion==1:
            valid_merge[kk] = 0
            clean_tree(valid_merge, xtree, xtree[kk,0])
            clean_tree(valid_merge, xtree, xtree[kk,1])

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
    for j, leaf in enumerate(ind):
        for orig in my_clus[leaf]:
            remap[orig] = j
        xtree[xtree[:, 0] == leaf, 0] = j
        xtree[xtree[:, 1] == leaf, 1] = j

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
