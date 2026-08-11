from numba import njit
import numpy as np
import torch
from torch.nn.functional import conv1d
import math 
from tqdm import trange 

@njit()
def compute_CCG(st1, st2, tbin = 1/1000, nbins = 500,
                assume_sorted=False):

    K = np.zeros(2*nbins+1,)
    # Empty trains: avoid st1.max() / st2.max() on zero-length arrays.
    # Callers treat T==0 as non-refractory (see check_CCG).
    if len(st1) == 0 or len(st2) == 0:
        return K, 0.0

    if not assume_sorted:
        st1 = np.sort(st1)
        st2 = np.sort(st2)

    dt = nbins * tbin
    T = np.maximum(st1.max(), st2.max()) - np.minimum(st1.min(), st2.min())

    ilow = 0
    ihigh = 0
    j = 0

    while j<len(st2):
        while (ihigh<len(st1)) and (st1[ihigh] <= st2[j]+dt):
            ihigh += 1
        while (ilow<len(st1))  and (st1[ilow] <= st2[j]-dt):
            ilow += 1
        if ilow >=len(st1):
            break
        if st1[ilow]>st2[j]+dt:
            j += 1
            continue;
        for k in range(ilow, ihigh):
            ibin = int(np.round((st2[j] - st1[k])/tbin))
            K[ibin+nbins] += 1
        j += 1
    return K, T

#@njit()
def CCG_metrics(st1, st2, K, T, nbins=None, tbin=None):
    irange1 = np.hstack((np.arange(1,nbins//2), np.arange(3*nbins//2, 2*nbins)))
    irange2 = np.arange(nbins-50, nbins-10)
    irange3 = np.arange(nbins+10, nbins+50)

    R00 = K[irange1].sum() / (len(irange1) * tbin * len(st1) * len(st2) /T)
    R1 = K[irange2].sum() / (len(irange2) * tbin * len(st1) * len(st2) /T)
    R2 = K[irange3].sum() / (len(irange3) * tbin * len(st1) * len(st2) /T)
    R01 = np.maximum(R1, R2)

    Q00 = np.maximum(K[irange2].mean(), K[irange3].mean())
    Q00 = np.maximum(Q00, K[irange1].mean())

    a = K[nbins]
    K[nbins] = 0

    Ri = np.zeros(10,)
    Qi = np.zeros(10,)
    for i in range(1,11):
        irange = np.arange(nbins-i, nbins+i+1)
        Ri0 = K[irange].sum() / (2*i*tbin*len(st1)*len(st2)/T)
        Ri[i-1] = Ri0

        n = K[irange].sum()/2
        lam = Q00 * i

        p = 1/2 * (1 + math.erf((n-lam)/(1e-10 + 2*lam)**.5))

        Qi[i-1] = p

    K[nbins] = a

    # NOTE: The variable names R12 and Q12 are reversed here compared to past
    #       versions of Kilosort, but the code remains unchanged. This change
    #       was made to be consistent with the methods in the Kilosort4 paper.
    R12 = np.min(Ri) / (1e-10 + np.maximum(R00, R01))
    Q12 = np.min(Qi)

    #print('%4.2f, %4.2f, %4.2f'%(R00, Q12, R12))
    return R12, Q12, Q00

def check_CCG(st1, st2=None, nbins = 500, tbin  = 1/1000, acg_threshold=0.2,
              ccg_threshold=0.25, assume_sorted=False):
    # NOTE: The default `acg_threshold=0.2` is different from the value of 0.1
    #       used for the Kilosort4 paper. We felt this better reflects common
    #       practice for determining 'good' units, but you can set
    #       `acg_threshold=0.1` in your run settings for stricter criteria.
    # ACG path: reuse the same array. compute_CCG rebinds sorted views and does
    # not mutate spike times in place, so a defensive copy is wasted memory.
    st1 = np.asarray(st1)
    if st2 is None:
        st2 = st1
    else:
        st2 = np.asarray(st2)
    # Empty / zero-span: CCG_metrics divides by T and by len(st)*len(st2).
    if st1.size == 0 or st2.size == 0:
        return False, False, np.nan
    K, T = compute_CCG(st1, st2, nbins=nbins, tbin=tbin,
                       assume_sorted=assume_sorted)
    if T == 0:
        return False, False, np.nan
    R12, Q12, Q00 = CCG_metrics(st1, st2, K, T,  nbins = nbins, tbin = tbin)
    is_refractory    = R12<acg_threshold  and (Q12<.2)#  or Q00<.25)
    cross_refractory = R12<ccg_threshold and (Q12<.05)# or Q00<.25)
    return is_refractory, cross_refractory, R12

def similarity(Wall, W, nt=61):
    WtW = conv1d(W.reshape(-1, 1,nt), W.reshape(-1, 1 ,nt), padding = nt) 
    WtW = torch.flip(WtW, [2,])
    mu = (Wall**2).sum((1,2), keepdims=True)**.5
    Wnorm = Wall / (1e-6 + mu)
    UtU = torch.einsum('ilk, jlm -> ijkm',  Wnorm, Wnorm)
    similar_templates = torch.einsum('ijkm, kml -> ijl', UtU.cpu(), WtW.cpu()).numpy()
    similar_templates = similar_templates.max(axis=-1)
    return similar_templates

def refract(iclust2, st0, acg_threshold=0.2, ccg_threshold=0.25):
    """Estimate refractory labels and contamination for every cluster.

    Kilosort's export path supplies spike times in chronological order. Grouping
    clusters with a stable sort preserves that order, avoiding both a full
    spike-vector comparison per cluster and redundant per-cluster time sorts.
    Unordered callers retain the original sorting behavior.
    """
    iclust2 = np.asarray(iclust2)
    st0 = np.asarray(st0)
    if iclust2.size == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)

    # Dense 0..max table (export path). Negative / non-int labels used to make
    # bincount raise; shift into a non-negative table like remove_duplicates.
    labels = iclust2.astype(np.int64, copy=False)
    min_lab = int(labels.min())
    max_lab = int(labels.max())
    offset = -min_lab if min_lab < 0 else 0
    Nfilt = max_lab + offset + 1

    is_refractory = np.zeros(Nfilt, dtype=bool)
    R12 = np.zeros(Nfilt)

    shifted = labels + offset
    counts = np.bincount(shifted, minlength=Nfilt)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    order = np.argsort(shifted, kind='stable')
    assume_sorted = st0.size < 2 or np.all(st0[:-1] <= st0[1:])

    for kk in range(Nfilt):
        start, stop = offsets[kk], offsets[kk + 1]
        if stop - start <= 10:
            continue
        st1 = st0[order[start:stop]]

        if (st1.max() - st1.min()) != 0:
            is_refractory[kk], _, R12[kk] = check_CCG(
                st1, acg_threshold=acg_threshold,
                ccg_threshold=ccg_threshold, assume_sorted=assume_sorted
            )

    # When labels were non-negative and dense, Nfilt == max+1 as before.
    # With an offset, index 0 is min_lab; callers that index by raw label and
    # expect max+1 length only work for min_lab>=0 (production export).
    if offset == 0:
        return is_refractory, R12
    # Negative labels: return tables sized so index = label - min_lab.
    # Documented via tests; merging_function uses non-negative clu only.
    return is_refractory, R12
