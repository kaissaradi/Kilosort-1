from scipy.sparse import csr_matrix
import numpy as np


def cluster_qr(M, iclust, iclust0):
    NN = M.shape[0]
    nr = M.shape[1]

    nc = iclust.max()+1
    q = csr_matrix((np.ones(NN,), (iclust, np.arange(NN))), (nc, NN))
    r  = csr_matrix((np.ones(nr,), (np.arange(nr), iclust0)), (nr, nc))
    return q,r

def Mstats(M):
    m = M.sum()
    ki = np.array(M.sum(1)).flatten()
    kj = np.array(M.sum(0)).flatten()
    # Guard 0/0 when adjacency is empty after self-edges are zeroed.
    ki_sum = float(ki.sum())
    kj_sum = float(kj.sum())
    if ki_sum <= 0 or kj_sum <= 0 or float(m) == 0:
        return 0.0, np.zeros_like(ki, dtype=np.float64), np.zeros_like(kj, dtype=np.float64)
    ki = m * ki / ki_sum
    kj = m * kj / kj_sum
    return m, ki, kj

def prepare(M, iclust, iclust0, lam=1):
    m, ki, kj = Mstats(M)
    q,r = cluster_qr(M, iclust, iclust0)
    cc = (q @ M @ r).toarray()
    nc = cc.shape[0]
    # m==0 → no edges; keep cneg as the small prior only (avoid /0).
    if m == 0:
        cneg = .001 + np.zeros((nc, nc), dtype=np.float64)
    else:
        cneg = .001 + np.outer(q @ ki , kj @ r)/m
    return cc, cneg

def merge_reduce(cc, cneg, iclust):
    nmerges = 0
    nc = cc.shape[0]

    cc = cc + cc.T
    cneg = cneg + cneg.T

    # Prefer divide-where so zero cneg never injects NaN into the merge tree
    # (stock `cc/cneg` did; empty graphs already short-circuit via m==0).
    crat = np.divide(cc, cneg, out=np.zeros_like(cc, dtype=np.float64),
                     where=cneg != 0)
    crat = crat -np.diag(np.diag(crat)) - np.eye(crat.shape[0])

    xtree, tstat = find_merges(crat, cc, cneg)

    my_clus = get_my_clus(xtree, tstat)
    return xtree, tstat, my_clus

def find_merges(crat, cc, cneg):
    nc = cc.shape[0]
    xtree = np.zeros((nc-1,3), 'int32')
    tstat = np.zeros((nc-1,3), 'float32')
    xnow = np.arange(nc)
    ntot = np.ones(nc,)
    # Flat argmax is identical to unravel_index(argmax, shape) for C-order
    # matrices (numpy default) and avoids building an index tuple each merge.
    ncols = int(cc.shape[1])

    for nmerges in range(nc-1):
        flat = int(np.argmax(crat))
        y, x = divmod(flat, ncols)
        lam = crat[y, x]

        # Stock mass formula (MouseLand); keep exact expression for identity.
        m      = cc[y,x] + cc[x,x] + cc[x,y] + cc[y,x]
        ki = cc[x,x] + cc[x,y]
        kj = cc[y,y] + cc[y,x]
        cpos_l = cc[y,x] + cc[x,y]
        # Empty 2x2 block (no edges) → 0/0 NaNs used to poison tstat and
        # downstream split decisions. Treat as zero modularity ratio.
        if m == 0:
            M = 0.0
        else:
            cneg_l = .5 * (ki * kj + (m-ki) * (m-kj)) / m
            M = cpos_l / cneg_l if cneg_l != 0 else 0.0

        cc[y]   = cc[y] + cc[x]
        cc[:,y] = cc[:,y] + cc[:,x]
        cc[x]   = -1
        cc[:,x] = -1
        cneg[y]   = cneg[y]   + cneg[x]
        cneg[:,y] = cneg[:,y] + cneg[:,x]

        # divide-where: zero cneg after prior merges must not inject NaN into
        # the remaining tree (matches initial crat build in merge_reduce).
        crat_y = np.divide(
            cc[y], cneg[y],
            out=np.zeros_like(cc[y], dtype=np.float64),
            where=cneg[y] != 0,
        )
        crat[y] = crat_y
        crat[:,y] = crat[y]
        crat[y,y] = -1
        crat[x] = -1
        crat[:,x]=-1

        xtree[nmerges,:] = [xnow[x], xnow[y], nmerges + nc]
        tstat[nmerges,:] = [lam, ntot[x]+ntot[y], M]

        ntot[y] +=ntot[x]
        xnow[y] = nc+nmerges

    return xtree, tstat

def get_my_clus(xtree, tstat):
    nc = xtree.shape[0]+1
    my_clus = [[j] for j in range(nc)]
    for t in range(nc-1):
        # New list = right-child members + left-child members (same order as
        # historical copy(right).extend(left)).
        my_clus.append(my_clus[xtree[t, 1]] + my_clus[xtree[t, 0]])
    return my_clus

def maketree(M, iclust, iclust0):
    iclust = np.asarray(iclust)
    if iclust.size == 0:
        # No spikes → empty tree / no leaves (caller should not split).
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
            [],
        )
    nc = int(np.max(iclust)) + 1
    if nc <= 1:
        # Single leaf: nothing to agglomerate.
        return (
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0, 3), dtype=np.float32),
            [[0]],
        )

    cc, cneg        = prepare(M, iclust, iclust0, lam = 1)
    xtree, tstat, my_clus  = merge_reduce(cc, cneg, iclust)

    return xtree, tstat, my_clus
