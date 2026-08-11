"""Unit tests for kilosort.postprocessing (CPU-safe)."""
import numpy as np
import pytest
import torch

from kilosort.clustering_qr import get_data_cpu, xy_templates
from kilosort.postprocessing import (
    compute_spike_positions,
    make_pc_features,
    remove_duplicates,
)


def reference_remove_duplicates(spike_times, spike_clusters, dt=15):
    """Independent implementation of the historical keep rule.

    First spike per cluster is kept. Later same-cluster spikes are kept only
    when at least `dt` samples after the previous *kept* spike of that cluster.
    """
    spike_times = np.asarray(spike_times, dtype=np.int64)
    spike_clusters = np.asarray(spike_clusters, dtype=np.int32)
    keep = np.zeros(spike_times.size, dtype=bool)
    last_kept = {}
    for i in range(spike_times.size):
        t = int(spike_times[i])
        c = int(spike_clusters[i])
        t0 = last_kept[c] if c in last_kept else t - dt
        if t >= t0 + dt:
            last_kept[c] = t
            keep[i] = True
    return spike_times[keep], spike_clusters[keep], keep


def test_remove_duplicates_matches_reference_rule():
    rng = np.random.default_rng(0)
    n = 5_000
    # Interleaved multi-unit stream with many near-duplicates.
    spike_times = np.sort(rng.integers(0, 50_000, size=n)).astype(np.int64)
    spike_clusters = rng.integers(0, 40, size=n, dtype=np.int32)
    # Inject exact same-cluster collisions within dt.
    for _ in range(200):
        i = int(rng.integers(1, n))
        spike_times[i] = spike_times[i - 1] + int(rng.integers(0, 10))
        spike_clusters[i] = spike_clusters[i - 1]
    order = np.argsort(spike_times, kind='stable')
    spike_times = spike_times[order]
    spike_clusters = spike_clusters[order]

    exp_t, exp_c, exp_keep = reference_remove_duplicates(
        spike_times, spike_clusters, dt=15
    )
    got_t, got_c, got_keep = remove_duplicates(
        spike_times.copy(), spike_clusters.copy(), dt=15
    )

    np.testing.assert_array_equal(got_keep, exp_keep)
    np.testing.assert_array_equal(got_t, exp_t)
    np.testing.assert_array_equal(got_c, exp_c)
    # Must actually drop some spikes on this synthetic collision stream.
    assert got_keep.sum() < n
    assert got_keep.sum() == exp_keep.sum()


def test_remove_duplicates_empty_input():
    times = np.zeros(0, dtype=np.int64)
    clusters = np.zeros(0, dtype=np.int32)
    out_t, out_c, keep = remove_duplicates(times, clusters, dt=15)
    assert out_t.size == 0
    assert out_c.size == 0
    assert keep.size == 0
    assert keep.dtype == np.bool_ or keep.dtype == bool


def test_remove_duplicates_independent_clusters_do_not_suppress_each_other():
    # Same sample times, different clusters: both kept.
    spike_times = np.array([100, 100, 100], dtype=np.int64)
    spike_clusters = np.array([0, 1, 2], dtype=np.int32)
    out_t, out_c, keep = remove_duplicates(spike_times, spike_clusters, dt=15)
    assert keep.all()
    np.testing.assert_array_equal(out_t, spike_times)
    np.testing.assert_array_equal(out_c, spike_clusters)


def test_remove_duplicates_suppresses_within_window_same_cluster():
    spike_times = np.array([0, 5, 14, 15, 30], dtype=np.int64)
    spike_clusters = np.array([0, 0, 0, 0, 0], dtype=np.int32)
    out_t, out_c, keep = remove_duplicates(spike_times, spike_clusters, dt=15)
    # Keep 0; drop 5 and 14 (within 15 of 0); keep 15 (exactly t0+dt); keep 30.
    np.testing.assert_array_equal(keep, np.array([True, False, False, True, True]))
    np.testing.assert_array_equal(out_t, np.array([0, 15, 30], dtype=np.int64))


def test_remove_duplicates_gapped_cluster_ids():
    # Dense table must still work when labels are not 0..K-1 contiguous usage.
    spike_times = np.array([0, 1, 20, 21], dtype=np.int64)
    spike_clusters = np.array([0, 7, 0, 7], dtype=np.int32)
    out_t, out_c, keep = remove_duplicates(spike_times, spike_clusters, dt=15)
    np.testing.assert_array_equal(keep, np.array([True, True, True, True]))
    np.testing.assert_array_equal(out_c, spike_clusters)


# ---------------------------------------------------------------------------
# make_pc_features: group-by path vs historical per-cluster mask loop
# ---------------------------------------------------------------------------


def reference_make_pc_features(ops, spike_templates, spike_clusters, tF):
    """Independent historical loop using `spike_clusters == i` masks.

    Same get_data_cpu + mean-norm channel ranking as production; only the
    cluster iteration / template unique gather differs from the optimized path.
    Mutates tF in-place (clone before calling).
    """
    xy, iC = xy_templates(ops)
    n_templates = iC.shape[1]
    n_clusters = np.unique(spike_clusters).size
    n_chans = ops['nearest_chans']
    feature_ind = np.zeros((n_clusters, n_chans), dtype=np.uint32)

    for i in np.unique(spike_clusters):
        iunq = np.unique(spike_templates[spike_clusters == i]).astype(int)
        ix = torch.from_numpy(np.zeros(n_templates, bool))
        ix[iunq] = True
        Xd, igood, ichan = get_data_cpu(
            ops, xy, iC, spike_templates, tF, None, None,
            dmin=ops['dmin'], dminx=ops['dminx'], ix=ix, merge_dim=False,
        )
        spike_mean = Xd.mean(0)
        chan_norm = torch.linalg.norm(spike_mean, dim=1)
        _, ind = torch.sort(chan_norm, descending=True)
        tF[igood, :] = Xd[:, ind[:n_chans], :]
        feature_ind[i, :] = ichan[ind[:n_chans]].cpu().numpy()

    tF = torch.permute(tF, (0, 2, 1))
    return tF, feature_ind


def _synthetic_ops_pc_features(
    n_channels=40, n_templates=12, nearest_chans=8, seed=0,
):
    """Minimal ops dict for xy_templates / make_pc_features (CPU, no MEA)."""
    rng = np.random.default_rng(seed)
    xc = np.zeros(n_channels, dtype=np.float64)
    yc = np.arange(n_channels, dtype=np.float64) * 20.0

    # iCC: nearest_chans unique neighbors per channel (by |y| distance)
    iCC = np.empty((nearest_chans, n_channels), dtype=np.int64)
    for c in range(n_channels):
        iCC[:, c] = np.argsort(np.abs(yc - yc[c]))[:nearest_chans]

    # One best channel per template, spread along the probe
    iU = np.linspace(0, n_channels - 1, n_templates).round().astype(np.int64)
    # Ensure uniqueness when n_templates <= n_channels
    if n_templates <= n_channels:
        # stable unique-ify while keeping spread
        seen = set()
        for k in range(n_templates):
            v = int(iU[k])
            while v in seen:
                v = (v + 1) % n_channels
            seen.add(v)
            iU[k] = v

    return {
        'xc': xc,
        'yc': yc,
        'iCC': torch.from_numpy(iCC),
        'iU': torch.from_numpy(iU),
        'nearest_chans': nearest_chans,
        'dmin': 20.0,
        'dminx': 32.0,
        # not read by make_pc_features but keeps ops realistic
        '_rng_seed': seed,
        '_unused': rng,
    }


def _synthetic_spikes_pc_features(
    n_templates, nearest_chans, n_pcs, n_spikes, n_clusters, seed=1,
    multi_template_clusters=True,
):
    """Dense cluster labels 0..K-1 with optional multi-template merges."""
    rng = np.random.default_rng(seed)
    spike_templates = rng.integers(0, n_templates, size=n_spikes).astype(np.int64)

    if multi_template_clusters:
        # Map templates -> clusters so some clusters own several templates
        template_to_cluster = np.zeros(n_templates, dtype=np.int64)
        # First n_clusters templates get identity; rest fold into existing
        for t in range(n_templates):
            template_to_cluster[t] = t if t < n_clusters else (t % n_clusters)
        spike_clusters = template_to_cluster[spike_templates].astype(np.int32)
    else:
        # 1:1 when n_templates == n_clusters; else mod
        spike_clusters = (spike_templates % n_clusters).astype(np.int32)

    # Ensure every cluster appears at least once
    for c in range(n_clusters):
        if not np.any(spike_clusters == c):
            spike_clusters[c % n_spikes] = c
            spike_templates[c % n_spikes] = c % n_templates

    tF = torch.from_numpy(
        rng.standard_normal((n_spikes, nearest_chans, n_pcs)).astype(np.float32)
    )
    return spike_templates, spike_clusters, tF


def test_make_pc_features_identity_vs_historical_mask_loop():
    n_templates, nearest_chans, n_pcs = 12, 8, 6
    n_spikes, n_clusters = 800, 7
    ops = _synthetic_ops_pc_features(
        n_channels=40, n_templates=n_templates, nearest_chans=nearest_chans, seed=11,
    )
    spike_templates, spike_clusters, tF = _synthetic_spikes_pc_features(
        n_templates, nearest_chans, n_pcs, n_spikes, n_clusters, seed=12,
        multi_template_clusters=True,
    )

    # make_pc_features mutates tF in-place — independent clones for each path
    got_tF, got_ind = make_pc_features(
        ops, spike_templates, spike_clusters, tF.clone()
    )
    ref_tF, ref_ind = reference_make_pc_features(
        ops, spike_templates, spike_clusters, tF.clone()
    )

    assert got_tF.shape == (n_spikes, n_pcs, nearest_chans)
    assert got_ind.shape == (n_clusters, nearest_chans)
    assert torch.equal(got_tF, ref_tF)
    np.testing.assert_array_equal(got_ind, ref_ind)


def test_make_pc_features_identity_one_to_one_clusters():
    n_templates = n_clusters = 10
    nearest_chans, n_pcs, n_spikes = 6, 3, 400
    ops = _synthetic_ops_pc_features(
        n_channels=32, n_templates=n_templates, nearest_chans=nearest_chans, seed=21,
    )
    spike_templates, spike_clusters, tF = _synthetic_spikes_pc_features(
        n_templates, nearest_chans, n_pcs, n_spikes, n_clusters, seed=22,
        multi_template_clusters=False,
    )

    got_tF, got_ind = make_pc_features(
        ops, spike_templates, spike_clusters, tF.clone()
    )
    ref_tF, ref_ind = reference_make_pc_features(
        ops, spike_templates, spike_clusters, tF.clone()
    )

    assert torch.equal(got_tF, ref_tF)
    np.testing.assert_array_equal(got_ind, ref_ind)


def test_make_pc_features_permutes_dims_for_phy():
    n_templates, nearest_chans, n_pcs, n_spikes = 4, 5, 3, 50
    ops = _synthetic_ops_pc_features(
        n_channels=20, n_templates=n_templates, nearest_chans=nearest_chans, seed=31,
    )
    spike_templates, spike_clusters, tF = _synthetic_spikes_pc_features(
        n_templates, nearest_chans, n_pcs, n_spikes, n_clusters=4, seed=32,
        multi_template_clusters=False,
    )
    out, ind = make_pc_features(
        ops, spike_templates, spike_clusters, tF.clone()
    )
    assert out.shape == (n_spikes, n_pcs, nearest_chans)
    assert ind.dtype == np.uint32
    assert ind.shape == (4, nearest_chans)


def test_compute_spike_positions_finite_when_weights_zero():
    """All-zero feature norms / masks must not emit NaN positions."""
    n_spikes, n_near, n_pcs = 5, 4, 3
    n_templates, n_chan = 3, 12
    tF = torch.zeros(n_spikes, n_near, n_pcs)
    st = np.zeros((n_spikes, 3), dtype=np.int64)
    st[:, 1] = np.arange(n_spikes) % n_templates
    ops = {
        'iCC_mask': torch.ones(n_near, n_templates),
        'iU': torch.arange(n_templates),
        'iCC': torch.randint(0, n_chan, (n_near, n_templates)),
        'xc': np.linspace(0, 100, n_chan).astype(np.float32),
        'yc': np.linspace(0, 50, n_chan).astype(np.float32),
    }
    xs, ys = compute_spike_positions(st, tF, ops)
    assert np.isfinite(xs).all()
    assert np.isfinite(ys).all()


def test_compute_spike_positions_identity_on_normal_weights():
    """clamp_min must not change positions when weight sums are positive."""
    n_spikes, n_near, n_pcs = 20, 5, 3
    n_templates, n_chan = 4, 16
    rng = np.random.default_rng(9)
    tF = torch.from_numpy(rng.standard_normal((n_spikes, n_near, n_pcs)).astype(np.float32))
    st = np.zeros((n_spikes, 3), dtype=np.int64)
    st[:, 1] = rng.integers(0, n_templates, size=n_spikes)
    ops = {
        'iCC_mask': torch.ones(n_near, n_templates),
        'iU': torch.arange(n_templates),
        'iCC': torch.from_numpy(rng.integers(0, n_chan, size=(n_near, n_templates))),
        'xc': np.linspace(0, 100, n_chan).astype(np.float32),
        'yc': np.linspace(0, 50, n_chan).astype(np.float32),
    }
    xs, ys = compute_spike_positions(st, tF, ops)
    # Manual historical formula (sums > 0 for random normal features)
    cpu = torch.device('cpu')
    tmass = torch.norm(tF, 2, dim=-1)
    tmask = ops['iCC_mask'][:, ops['iU'][st[:, 1]]].T
    tmass = tmass * tmask
    tmass = tmass / tmass.sum(1, keepdim=True)
    chs = ops['iCC'][:, ops['iU'][st[:, 1]]]
    xc0 = torch.from_numpy(ops['xc'])[chs.T]
    yc0 = torch.from_numpy(ops['yc'])[chs.T]
    exp_x = (xc0 * tmass).sum(1).numpy()
    exp_y = (yc0 * tmass).sum(1).numpy()
    np.testing.assert_allclose(xs, exp_x, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ys, exp_y, rtol=1e-5, atol=1e-5)
