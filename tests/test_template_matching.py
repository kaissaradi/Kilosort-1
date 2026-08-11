"""CPU-safe unit tests for template_matching hot paths."""
import numpy as np
import torch
from torch.nn.functional import conv1d

from kilosort.template_matching import merging_function, roll_features


def reference_merging_function(ops, Wall, clu, st, tF, r_thresh=0.5, mode='ccg',
                               check_dt=True, device=torch.device('cpu')):
    """Historical mask-based merge (pre index-map / renorm-cache).

    Kept here as an independent oracle so the optimised path stays bit-identical
    on synthetic Wall/st streams without running a full MEA sort.
    """
    clu2 = clu.copy()
    clu_unq, ns = np.unique(clu2, return_counts=True)

    Ww = Wall.to(device)
    NN = len(Ww)

    isort = np.argsort(ns)[::-1]
    is_merged = np.zeros(NN, 'bool')

    acg_threshold = ops['settings']['acg_threshold']
    ccg_threshold = ops['settings']['ccg_threshold']
    if mode == 'ccg':
        from kilosort import CCG
        is_ref, _ = CCG.refract(
            clu, st[:, 0] / ops['fs'],
            acg_threshold=acg_threshold, ccg_threshold=ccg_threshold
        )

    nt = ops['nt']
    W = ops['wPCA'].contiguous()
    WtW = conv1d(W.reshape(-1, 1, nt), W.reshape(-1, 1, nt), padding=nt)
    WtW = torch.flip(WtW, [2, ])

    t = 0
    while t < NN:
        kk = clu_unq[isort[t]]

        if (mode == 'ccg') and is_ref[kk] == 0:
            t += 1
            continue
        if is_merged[kk]:
            t += 1
            continue

        mu = (Ww ** 2).sum((1, 2), keepdims=True) ** .5
        Wnorm = Ww / (1e-6 + mu)

        UtU = torch.einsum('lk, jlm -> jkm', Wnorm[kk], Wnorm)
        ctc = torch.einsum('jkm, kml -> jl', UtU, WtW)

        cmax, imax = ctc.max(1)
        cmax[kk] = 0
        jsort = np.argsort(cmax.cpu().numpy())[::-1]

        if mode == 'ccg':
            st0 = st[:, 0][clu2 == kk] / ops['fs']

        is_ccg = 0
        for j in range(NN):
            jj = jsort[j]
            if cmax[jj] < r_thresh:
                break
            if mode == 'ccg':
                from kilosort import CCG
                st1 = st[:, 0][clu2 == jj] / ops['fs']
                _, is_ccg, _ = CCG.check_CCG(
                    st0, st1, acg_threshold=acg_threshold,
                    ccg_threshold=ccg_threshold
                )
            else:
                dmu = 2 * (mu[kk] - mu[jj]) / (mu[kk] + mu[jj])
                is_ccg = dmu.abs() < 0.2

            if is_ccg:
                is_merged[jj] = 1
                dt = (imax[kk] - imax[jj]).item()
                if dt != 0 and check_dt:
                    idx = (clu2 == jj)
                    tF, Wall = roll_features(W, tF, Ww, idx, jj, dt)
                    st[idx, 0] -= dt

                Ww[kk] = (ns[kk] / (ns[kk] + ns[jj]) * Ww[kk]
                          + ns[jj] / (ns[kk] + ns[jj]) * Ww[jj])
                Ww[jj] = 0
                ns[kk] += ns[jj]
                ns[jj] = 0
                clu2[clu2 == jj] = kk
                break

        if is_ccg == 0:
            t += 1

    imap = np.cumsum((~is_merged).astype('int32')) - 1
    if imap.size > 0:
        clu2 = imap[clu2]

    Ww = Ww[~is_merged]
    if mode == 'ccg':
        is_ref = is_ref[~is_merged]
    else:
        is_ref = None

    sorted_idx = np.argsort(st[:, 0])
    st = np.take_along_axis(st, sorted_idx[..., np.newaxis], axis=0)
    clu2 = clu2[sorted_idx]
    tF = tF[torch.from_numpy(sorted_idx)]
    return Ww.cpu(), clu2, is_ref, st, tF


def _synthetic_merge_case(seed=0, n_spikes=400, n_units=8):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    nC, nPC, nt = 4, 3, 21
    device = torch.device('cpu')

    W = torch.randn(nPC, nt, dtype=torch.float32)
    W = W / (torch.norm(W, dim=1, keepdim=True) + 1e-6)

    # Pair of near-duplicate units so template-mode dmu merge fires.
    base_a = torch.randn(nC, nPC)
    base_b = torch.randn(nC, nPC)
    Wall = torch.randn(n_units, nC, nPC)
    Wall[0] = base_a
    Wall[1] = base_a * 1.04
    Wall[2] = base_b
    Wall[3] = base_b * 0.97

    clu = rng.integers(0, n_units, size=n_spikes).astype(np.int32)
    st = np.zeros((n_spikes, 2), dtype=np.float64)
    st[:, 0] = np.sort(rng.integers(0, 200_000, size=n_spikes)).astype(np.float64)
    st[:, 1] = clu
    tF = torch.randn(n_spikes, nC, nPC)

    ops = {
        'settings': {'acg_threshold': 0.2, 'ccg_threshold': 0.25},
        'nt': nt,
        'wPCA': W,
        'fs': 20_000.0,
    }
    return ops, Wall, clu, st, tF, device


def test_merging_function_template_mode_matches_reference():
    ops, Wall, clu, st, tF, device = _synthetic_merge_case(seed=2)

    got = merging_function(
        ops, Wall.clone(), clu.copy(), st.copy(), tF.clone(),
        r_thresh=0.4, mode='template', check_dt=True, device=device
    )
    exp = reference_merging_function(
        ops, Wall.clone(), clu.copy(), st.copy(), tF.clone(),
        r_thresh=0.4, mode='template', check_dt=True, device=device
    )

    Ww_g, clu_g, ref_g, st_g, tF_g = got
    Ww_e, clu_e, ref_e, st_e, tF_e = exp

    assert ref_g is None and ref_e is None
    np.testing.assert_array_equal(clu_g, clu_e)
    np.testing.assert_allclose(st_g, st_e, rtol=0, atol=0)
    assert torch.allclose(Ww_g, Ww_e, rtol=1e-5, atol=1e-5)
    assert torch.allclose(tF_g, tF_e, rtol=1e-5, atol=1e-5)
    # At least one merge should fire on this near-duplicate Wall pair.
    assert Ww_g.shape[0] < Wall.shape[0]


def test_merging_function_ccg_mode_matches_reference():
    ops, Wall, clu, st, tF, device = _synthetic_merge_case(seed=5, n_spikes=600)

    # Clean-ish times: large inter-spike gaps so ACG refract can pass for some.
    st = st.copy()
    st[:, 0] = np.arange(st.shape[0], dtype=np.float64) * 2000.0

    got = merging_function(
        ops, Wall.clone(), clu.copy(), st.copy(), tF.clone(),
        r_thresh=0.5, mode='ccg', check_dt=False, device=device
    )
    exp = reference_merging_function(
        ops, Wall.clone(), clu.copy(), st.copy(), tF.clone(),
        r_thresh=0.5, mode='ccg', check_dt=False, device=device
    )

    Ww_g, clu_g, ref_g, st_g, tF_g = got
    Ww_e, clu_e, ref_e, st_e, tF_e = exp

    np.testing.assert_array_equal(clu_g, clu_e)
    np.testing.assert_array_equal(ref_g.astype(bool), ref_e.astype(bool))
    np.testing.assert_allclose(st_g, st_e, rtol=0, atol=0)
    assert torch.allclose(Ww_g, Ww_e, rtol=1e-5, atol=1e-5)
    assert torch.allclose(tF_g, tF_e, rtol=1e-5, atol=1e-5)
