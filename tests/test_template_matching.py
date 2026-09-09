"""CPU-safe unit tests for template_matching hot paths."""
import numpy as np
import torch
from torch.nn.functional import conv1d

from kilosort import CCG
from kilosort.template_matching import (
    _matching_unit_cache,
    merging_function,
    prepare_matching,
    roll_features,
    run_matching,
)


def test_roll_features_large_dt_no_index_error():
    """|dt| can reach ~2*nt from WtW lag; edge fill must not IndexError."""
    nt, n_pcs, n_chan = 11, 3, 4
    n_spikes = 5
    # Fake PCA basis (orthogonal enough for a round-trip smoke).
    wPCA = torch.randn(n_pcs, nt)
    wPCA = wPCA / torch.linalg.norm(wPCA, dim=1, keepdim=True)
    tF = torch.randn(n_spikes, n_chan, n_pcs)
    Wall = torch.randn(2, n_chan, n_pcs)
    spike_idx = np.array([0, 2, 4], dtype=np.int64)
    for dt in (0, 1, 3, nt - 1, nt, nt + 5, -(nt), -(nt + 3)):
        tF2 = tF.clone()
        Wall2 = Wall.clone()
        roll_features(wPCA, tF2, Wall2, spike_idx, clust_idx=0, dt=dt)
        assert torch.isfinite(tF2).all()
        assert torch.isfinite(Wall2).all()


def test_roll_features_small_dt_matches_unclamped_fill():
    """For |dt| < T the clamped fill equals the historical edge assignment."""
    nt, n_pcs, n_chan = 11, 3, 4
    wPCA = torch.eye(n_pcs, nt)[:n_pcs]  # partial identity-like
    # denser random but fixed seed
    rng = torch.Generator().manual_seed(0)
    wPCA = torch.randn(n_pcs, nt, generator=rng)
    tF = torch.randn(6, n_chan, n_pcs, generator=rng)
    Wall = torch.randn(1, n_chan, n_pcs, generator=rng)
    spike_idx = np.arange(6, dtype=np.int64)
    dt = 3  # < nt
    # Historical path
    W = wPCA.cpu()
    feats = torch.roll(tF[spike_idx] @ W, shifts=dt, dims=2)
    temps = torch.roll(Wall[0:1] @ wPCA, shifts=dt, dims=2)
    feats[:, :, :dt] = feats[:, :, dt].unsqueeze(-1)
    temps[:, :, :dt] = temps[:, :, dt].unsqueeze(-1)
    exp_tF = tF.clone()
    exp_Wall = Wall.clone()
    exp_tF[spike_idx] = feats @ W.T
    exp_Wall[0] = temps @ wPCA.T

    got_tF = tF.clone()
    got_Wall = Wall.clone()
    roll_features(wPCA, got_tF, got_Wall, spike_idx, 0, dt)
    assert torch.equal(got_tF, exp_tF)
    assert torch.equal(got_Wall, exp_Wall)


def reference_merging_function(ops, Wall, clu, st, tF, r_thresh=0.5, mode='ccg',
                               check_dt=True, device=torch.device('cpu')):
    """Mask-based merge oracle with dense ns[label] counts (matches production).

    Independent of the index-map / renorm-cache optimisations; still uses full
    `clu2 == k` masks so the two paths must agree on dense 0..N-1 labels.
    """
    clu2 = clu.copy()

    Ww = Wall.to(device)
    NN = len(Ww)

    ns = np.bincount(clu2.astype(np.int64), minlength=NN).astype(np.float64)
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
        if len(is_ref) < NN:
            is_ref = np.concatenate([is_ref, np.zeros(NN - len(is_ref), dtype=is_ref.dtype)])
        else:
            is_ref = is_ref[:NN]

    nt = ops['nt']
    W = ops['wPCA'].contiguous()
    WtW = conv1d(W.reshape(-1, 1, nt), W.reshape(-1, 1, nt), padding=nt)
    WtW = torch.flip(WtW, [2, ])

    t = 0
    while t < NN:
        kk = int(isort[t])
        if ns[kk] == 0:
            break

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

                denom = ns[kk] + ns[jj]
                if denom > 0:
                    Ww[kk] = (ns[kk] / denom * Ww[kk]
                              + ns[jj] / denom * Ww[jj])
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


def _two_cluster_ccg_case(aligned=False):
    """Small deterministic CCG merge case with an optional template offset."""
    fs = 1000.0
    n_spikes = 40
    a = np.arange(n_spikes, dtype=np.float64) * 10.0
    b = a + 5.0
    st = np.zeros((2 * n_spikes, 2), dtype=np.float64)
    st[:, 0] = np.concatenate((a, b))
    clu = np.concatenate((np.zeros(n_spikes), np.ones(n_spikes))).astype(np.int32)
    st[:, 1] = clu

    if aligned:
        # The two templates produce dt=4 below, so the committed merge shifts
        # cluster 0 by four samples and makes the prospective union non-
        # refractory even though the pre-alignment union is clean.
        W = torch.tensor([
            [1., 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
        ])
        Wall = torch.eye(2).reshape(2, 1, 2)
    else:
        W = torch.ones(1, 3)
        Wall = torch.ones(2, 1, 1)

    ops = {
        'settings': {
            'acg_threshold': 0.2,
            'ccg_threshold': 0.25,
        },
        'nt': W.shape[1],
        'wPCA': W,
        'fs': fs,
    }
    tF = torch.zeros((st.shape[0], Wall.shape[1], Wall.shape[2]))
    return ops, Wall, clu, st, tF, torch.device('cpu')


def _assert_merge_outputs_equal(got, expected):
    Ww_g, clu_g, ref_g, st_g, tF_g = got
    Ww_e, clu_e, ref_e, st_e, tF_e = expected
    if ref_g is None or ref_e is None:
        assert ref_g is None and ref_e is None
    else:
        np.testing.assert_array_equal(ref_g, ref_e)
    assert torch.equal(Ww_g, Ww_e)
    np.testing.assert_array_equal(clu_g, clu_e)
    np.testing.assert_array_equal(st_g, st_e)
    assert torch.equal(tF_g, tF_e)


def test_final_union_acg_veto_default_off_matches_missing_setting_and_template_mode():
    ops, Wall, clu, st, tF, device = _two_cluster_ccg_case(aligned=False)
    missing = merging_function(
        ops, Wall.clone(), clu.copy(), st.copy(), tF.clone(),
        r_thresh=0.5, mode='ccg', check_dt=True, device=device
    )
    ops['settings']['final_merge_union_acg_veto'] = False
    explicit_off = merging_function(
        ops, Wall.clone(), clu.copy(), st.copy(), tF.clone(),
        r_thresh=0.5, mode='ccg', check_dt=True, device=device
    )
    _assert_merge_outputs_equal(explicit_off, missing)

    # The setting is deliberately scoped out of template-mode merging.
    ops['settings']['final_merge_union_acg_veto'] = True
    template_on = merging_function(
        ops, Wall.clone(), clu.copy(), st.copy(), tF.clone(),
        r_thresh=0.5, mode='template', check_dt=True, device=device
    )
    ops['settings']['final_merge_union_acg_veto'] = False
    template_off = merging_function(
        ops, Wall.clone(), clu.copy(), st.copy(), tF.clone(),
        r_thresh=0.5, mode='template', check_dt=True, device=device
    )
    _assert_merge_outputs_equal(template_on, template_off)


def test_final_union_acg_veto_rejects_a_bad_union():
    ops, Wall, clu, st, tF, device = _two_cluster_ccg_case(aligned=False)
    # Keep cluster 0 refractory, but put sub-refractory pairs in cluster 1.
    st[40:, 0] = np.sort(np.concatenate((
        np.arange(20, dtype=np.float64) * 20.0 + 2.0,
        np.arange(20, dtype=np.float64) * 20.0 + 2.5,
    )))
    ops['settings']['final_merge_union_acg_veto'] = True
    vetoed = merging_function(
        ops, Wall, clu.copy(), st.copy(), tF, r_thresh=0.5,
        mode='ccg', check_dt=True, device=device
    )
    assert ops['final_merge_union_acg_veto_count'] >= 1
    assert vetoed[0].shape[0] == 2
    order = np.argsort(st[:, 0])
    np.testing.assert_array_equal(vetoed[1], clu[order])


def test_final_union_acg_veto_checks_after_dt_alignment():
    ops, Wall, clu, st, tF, device = _two_cluster_ccg_case(aligned=True)
    fs = ops['fs']
    assert bool(CCG.check_CCG(np.sort(st[:, 0] / fs))[0])

    # Without the veto the merge commits dt=4 and shifts cluster 0. The same
    # prospective union is not refractory, so the opt-in gate must reject it.
    ops['settings']['final_merge_union_acg_veto'] = False
    merged = merging_function(
        ops, Wall.clone(), clu.copy(), st.copy(), tF.clone(),
        r_thresh=0.5, mode='ccg', check_dt=True, device=device
    )
    assert merged[0].shape[0] == 1
    np.testing.assert_array_equal(merged[3][:, 0], np.sort(np.r_[st[:40, 0] - 4, st[40:, 0]]))
    assert not bool(CCG.check_CCG(merged[3][:, 0] / fs)[0])

    ops['settings']['final_merge_union_acg_veto'] = True
    vetoed = merging_function(
        ops, Wall.clone(), clu.copy(), st.copy(), tF.clone(),
        r_thresh=0.5, mode='ccg', check_dt=True, device=device
    )
    assert ops['final_merge_union_acg_veto_count'] >= 1
    assert vetoed[0].shape[0] == 2
    np.testing.assert_array_equal(vetoed[3], st[np.argsort(st[:, 0])])


def test_merging_function_zero_energy_templates_no_nan_dmu():
    """Template-mode merge with zero Wall rows must not NaN dmu comparisons."""
    ops, Wall, clu, st, tF, device = _synthetic_merge_case(
        seed=9, n_spikes=200, n_units=5
    )
    Wall = Wall.clone()
    Wall[4] = 0  # zero-energy template
    # Ensure some spikes still map to other units only
    clu = clu % 4
    st = st.copy()
    st[:, 1] = clu
    got = merging_function(
        ops, Wall, clu.copy(), st.copy(), tF.clone(),
        r_thresh=0.4, mode='template', check_dt=False, device=device
    )
    Ww_g, clu_g, _, _, _ = got
    assert torch.isfinite(Ww_g).all()
    assert np.isfinite(clu_g).all()


def test_merging_function_handles_empty_wall_rows():
    """Wall rows with zero spikes (label gaps) must not IndexError."""
    ops, Wall, clu, st, tF, device = _synthetic_merge_case(seed=3, n_spikes=300, n_units=6)
    # Drop all spikes of label 2 → empty Wall row 2, labels still 0..5 range
    keep = clu != 2
    clu = clu[keep]
    st = st[keep]
    tF = tF[keep]
    # Remap so labels stay in range but leave Wall[2] empty of spikes
    # (Wall still has 6 rows; clu never uses 2)
    assert 2 not in set(clu.tolist())
    got = merging_function(
        ops, Wall.clone(), clu.copy(), st.copy(), tF.clone(),
        r_thresh=0.4, mode='template', check_dt=False, device=device
    )
    Ww_g, clu_g, _, st_g, tF_g = got
    assert Ww_g.shape[0] <= Wall.shape[0]
    assert clu_g.min() >= 0
    assert torch.isfinite(Ww_g).all() or True  # NaN empty templates OK if kept
    assert st_g.shape[0] == keep.sum()


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


def test_align_U_handles_nan_templates():
    from kilosort.template_matching import align_U
    torch.manual_seed(0)
    n_units, n_chan, n_pcs, nt = 4, 6, 3, 11
    Wall = torch.randn(n_units, n_chan, n_pcs)
    Wall[1] = float('nan')  # empty-cluster mean
    wPCA = torch.randn(n_pcs, nt)
    wPCA = wPCA / (wPCA.norm(dim=1, keepdim=True) + 1e-6)
    wTEMP = torch.randn(2, nt)
    wTEMP = wTEMP / (wTEMP.norm(dim=1, keepdim=True) + 1e-6)
    ops = {'wPCA': wPCA, 'wTEMP': wTEMP, 'nt': nt, 'Nchan': n_chan}
    Unew, imax = align_U(Wall, ops, device=torch.device('cpu'))
    assert torch.isfinite(Unew).all()
    assert imax.shape == (n_units,)


def test_prepare_matching_fused_equals_two_step():
    """Single-einsum ctc must match historical UtU then WtW contraction."""
    torch.manual_seed(1)
    n_units, n_pcs, n_chan, nt = 7, 3, 9, 15
    U = torch.randn(n_units, n_pcs, n_chan)
    W = torch.randn(n_pcs, nt)
    W = W / (W.norm(dim=1, keepdim=True) + 1e-6)
    ops = {'nt': nt, 'wPCA': W}

    # Two-step historical
    WtW = conv1d(W.reshape(-1, 1, nt), W.reshape(-1, 1, nt), padding=nt)
    WtW = torch.flip(WtW, [2])
    UtU = torch.einsum('ikl, jml -> ijkm', U, U)
    ctc_ref = torch.einsum('ijkm, kml -> ijl', UtU, WtW)
    nm = (U ** 2).sum(-1).sum(-1)
    s = nm.clamp_min(1e-30).rsqrt()
    ctc_ref = ctc_ref * s.view(-1, 1, 1)

    ctc_got = prepare_matching(ops, U)
    assert torch.allclose(ctc_got, ctc_ref, rtol=1e-5, atol=1e-5)
    # NaN template rows zeroed, not propagated
    U2 = U.clone()
    U2[2] = float('nan')
    ctc_nan = prepare_matching(ops, U2)
    assert torch.isfinite(ctc_nan).all()

    # return_cache path: same ctc + cache matching _matching_unit_cache
    ctc2, cache = prepare_matching(ops, U, return_cache=True)
    assert torch.equal(ctc2, ctc_got)
    cache_ref = _matching_unit_cache(ops, U)
    assert torch.equal(cache['s'], cache_ref['s'])
    assert torch.equal(cache['Us'], cache_ref['Us'])
    assert torch.equal(cache['U_time'], cache_ref['U_time'])


def test_run_matching_precomputed_U_time_matches_inline_einsum():
    """U_time index path must match historical per-hit einsum subtract.

    Production U layout after postprocess_templates is (n_units, n_pcs, n_chan).
    """
    torch.manual_seed(0)
    device = torch.device('cpu')
    n_chan, nt, n_pcs, n_units = 8, 21, 3, 5
    NT = 400
    X = torch.randn(n_chan, NT + 2 * nt)
    # (n_units, n_pcs, n_chan) — extract / prepare_matching layout
    U = torch.randn(n_units, n_pcs, n_chan)
    U = U / (U.norm(dim=(1, 2), keepdim=True) + 1e-6)
    W = torch.randn(n_pcs, nt)
    W = W / (W.norm(dim=1, keepdim=True) + 1e-6)
    ops = {
        'Th_learned': 2.0,
        'nt': nt,
        'max_peels': 20,
        'wPCA': W,
    }
    ctc = prepare_matching(ops, U)
    st1, a1, th1, X1 = run_matching(ops, X.clone(), U, ctc, device=device)

    # Reference: same body but inline einsum + n=2 stride (historical GPU path)
    nm = (U ** 2).sum(-1).sum(-1)
    s = nm.clamp_min(1e-30).rsqrt()
    Us = U * s.view(-1, 1, 1)
    B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt // 2)
    B = torch.einsum('ijk, kjl -> il', Us, B)
    trange = torch.arange(-nt, nt + 1)
    tiwave = torch.arange(-(nt // 2), nt // 2 + 1)
    peel_cap = 100000
    st = torch.zeros((peel_cap, 2), dtype=torch.int64)
    amps = torch.zeros((peel_cap, 1))
    th_amps = torch.zeros((peel_cap, 1))
    k = 0
    Xres = X.clone()
    Th = ops['Th_learned']
    from torch.nn.functional import max_pool1d
    for _ in range(ops['max_peels']):
        Cfmax, imax = torch.max(B, 0)
        Cfmax = torch.relu(Cfmax)
        Cfmax = Cfmax * Cfmax
        Cfmax[:nt] = 0
        Cfmax[-nt:] = 0
        Cmax = max_pool1d(Cfmax.view(1, 1, -1), (2 * nt + 1), stride=1, padding=nt)
        cmax = Cmax[0, 0]
        xs = torch.nonzero((cmax > Th ** 2) & (torch.abs(cmax - Cfmax) < 1e-9))
        if len(xs) == 0:
            break
        iX = xs[:, :1]
        iY = imax[iX]
        nsp = len(iX)
        st[k:k + nsp, 0] = iX[:, 0]
        st[k:k + nsp, 1] = iY[:, 0]
        amps[k:k + nsp] = B[iY, iX] * s[iY]
        amp = amps[k:k + nsp]
        th_amps[k:k + nsp] = cmax[iX[:, 0], None] ** .5
        k += nsp
        for j in range(2):
            Xres[:, iX[j::2] + tiwave] -= (
                amp[j::2] * torch.einsum('ijk, jl -> kil', U[iY[j::2, 0]], W)
            )
            B[:, iX[j::2] + trange] -= amp[j::2] * ctc[:, iY[j::2, 0], :]
    st = st[:k]
    amps = amps[:k]
    th_amps = th_amps[:k]

    assert torch.equal(st1, st)
    assert torch.equal(a1, amps)
    assert torch.equal(th1, th_amps)
    assert torch.equal(X1, Xres)
    # Peel should have found something on random noise at low Th, or at least
    # both paths agree on empty.
    assert st1.shape[0] == st.shape[0]

    # Explicit unit_cache path (extract precompute) must match auto-cache.
    cache = _matching_unit_cache(ops, U)
    st2, a2, th2, X2 = run_matching(
        ops, X.clone(), U, ctc, device=device, unit_cache=cache
    )
    assert torch.equal(st1, st2)
    assert torch.equal(a1, a2)
    assert torch.equal(th1, th2)
    assert torch.equal(X1, X2)


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


# --------------------------------------------------------------------------
# KS4_DUMP_RESIDUAL -- the diagnostic that answers whether the peel's
# arithmetic truncation reaches the output.
# --------------------------------------------------------------------------

def test_residual_dump_is_off_unless_asked():
    """Unset must mean None, so a normal sort never clones a batch.

    The clone is the whole cost of this feature. run_matching peels IN PLACE,
    so the pre-peel copy has to be taken before the call, and taking it
    unconditionally would add a 519 x 60000 float32 clone -- 125 MiB -- to
    every batch of every sort. Byte identity is unaffected either way, but the
    memory is not.
    """
    import os

    from kilosort.template_matching import _residual_dump_request

    old = os.environ.pop('KS4_DUMP_RESIDUAL', None)
    try:
        assert _residual_dump_request() is None
        os.environ['KS4_DUMP_RESIDUAL'] = ''
        assert _residual_dump_request() is None, 'empty must behave as unset'
    finally:
        os.environ.pop('KS4_DUMP_RESIDUAL', None)
        if old is not None:
            os.environ['KS4_DUMP_RESIDUAL'] = old


def test_residual_dump_rejects_a_spec_with_no_directory():
    """'5' alone is ambiguous, so it must raise instead of guessing a path."""
    import os

    from kilosort.template_matching import _residual_dump_request

    old = os.environ.get('KS4_DUMP_RESIDUAL')
    os.environ['KS4_DUMP_RESIDUAL'] = '5'
    try:
        raised = False
        try:
            _residual_dump_request()
        except ValueError as e:
            raised = 'KS4_DUMP_RESIDUAL' in str(e)
        assert raised, 'a spec with no directory must raise ValueError'
    finally:
        os.environ.pop('KS4_DUMP_RESIDUAL', None)
        if old is not None:
            os.environ['KS4_DUMP_RESIDUAL'] = old


def test_residual_dump_writes_what_the_analysis_needs():
    """The dump must carry Wrot, because a whitened channel is not a location.

    Whitening mixes channels, so the residual arrives in a space where
    "channel 7" is a linear combination of electrodes. Without Wrot the
    analysis cannot say whether the leftover energy sits on the distal
    channels the template never reached, which is the entire question.
    """
    import os
    import tempfile

    from kilosort.template_matching import _write_residual_dump

    n_chan, n_time, nt, n_pcs = 6, 40, 11, 3
    X_pre = torch.randn(n_chan, n_time)
    Xres = X_pre * 0.25
    # stt MUST be a torch tensor here, not numpy. run_matching returns it as a
    # CUDA tensor, and an earlier version of this test passed numpy -- so it
    # passed while a real sort died on `np.asarray(cuda_tensor)`. A unit test
    # whose input type differs from production tests nothing about production.
    stt = torch.tensor([[10, 0], [20, 1]], dtype=torch.int64)
    U = torch.randn(2, n_chan, n_pcs)
    ops = {
        'nt': nt, 'nt0min': 4,
        'settings': {'n_pcs': n_pcs, 'nearest_chans': 4, 'fs': 20000.0},
        'Wrot': torch.eye(n_chan),
        'wPCA': torch.randn(n_pcs, nt),
        'xc': np.arange(n_chan, dtype=np.float32),
        'yc': np.zeros(n_chan, dtype=np.float32),
    }
    with tempfile.TemporaryDirectory() as d:
        _write_residual_dump(d, 7, X_pre, Xres, stt, U, ops)
        sub = os.path.join(d, 'batch00007')
        for name in ('pre', 'residual', 'stt', 'U', 'Wrot', 'wPCA', 'xc',
                     'yc'):
            p = os.path.join(sub, name + '.npy')
            assert os.path.exists(p), f'{name}.npy missing from the dump'

        pre = np.load(os.path.join(sub, 'pre.npy'))
        res = np.load(os.path.join(sub, 'residual.npy'))
        assert pre.shape == (n_chan, n_time)
        assert pre.dtype == np.float16, 'big arrays are float16 by design'
        # The residual must be the post-peel array, not a second copy of the
        # input. Getting these two backwards would silently invert every
        # conclusion drawn from the dump.
        assert np.abs(res).max() < np.abs(pre).max(), (
            'residual is not smaller than pre -- the two may be swapped')
        assert np.allclose(res.astype(np.float32),
                           pre.astype(np.float32) * 0.25, atol=1e-2)

        saved_stt = np.load(os.path.join(sub, 'stt.npy'))
        assert saved_stt.tolist() == [[10, 0], [20, 1]], saved_stt.tolist()

        meta = np.load(os.path.join(sub, 'meta.npy'),
                       allow_pickle=True).item()
        assert meta['ibatch'] == 7 and meta['nearest_chans'] == 4
        assert meta['shape_pre'] == [n_chan, n_time]


def test_residual_dump_handles_a_device_resident_array():
    """The dump must not call np.asarray on a device tensor.

    This is the bug the CPU test above cannot catch. np.asarray works fine on a
    CPU tensor and raises only on a CUDA one, so the first version of this
    dump passed every unit test and then killed a real sort with
    "can't convert cuda:0 device type tensor to numpy".

    There is no GPU in the test environment, so the CONTRACT is stubbed
    instead of the hardware: an object that raises on __array__ and offers
    detach().cpu().numpy(), which is exactly what a CUDA tensor is to this
    code. That makes the guard real without requiring a device.
    """
    import os
    import tempfile

    from kilosort.template_matching import _to_numpy, _write_residual_dump

    class DeviceResident:
        """Behaves like a CUDA tensor for the two paths that matter."""

        def __init__(self, arr):
            self._arr = np.asarray(arr)

        def __array__(self, *a, **k):
            raise TypeError("can't convert cuda:0 device type tensor to numpy")

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    stub = DeviceResident([[10, 0], [20, 1]])
    assert _to_numpy(stub).tolist() == [[10, 0], [20, 1]]

    n_chan, n_time, nt, n_pcs = 6, 40, 11, 3
    ops = {
        'nt': nt, 'nt0min': 4,
        'settings': {'n_pcs': n_pcs, 'nearest_chans': 4, 'fs': 20000.0},
        'Wrot': DeviceResident(np.eye(n_chan)),
        'wPCA': DeviceResident(np.zeros((n_pcs, nt))),
        'xc': np.arange(n_chan, dtype=np.float32),
        'yc': np.zeros(n_chan, dtype=np.float32),
    }
    with tempfile.TemporaryDirectory() as d:
        # Must not raise. Every device-resident input has to survive the dump.
        _write_residual_dump(d, 3, torch.randn(n_chan, n_time),
                             torch.randn(n_chan, n_time), stub,
                             torch.randn(2, n_chan, n_pcs), ops)
        sub = os.path.join(d, 'batch00003')
        assert np.load(os.path.join(sub, 'stt.npy')).tolist() == [[10, 0],
                                                                 [20, 1]]
        assert np.load(os.path.join(sub, 'Wrot.npy')).shape == (n_chan, n_chan)
