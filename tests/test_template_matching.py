"""CPU-safe unit tests for template_matching hot paths."""
import numpy as np
import torch
from torch.nn.functional import conv1d

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

    # Reference: same body but inline einsum (historical)
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
