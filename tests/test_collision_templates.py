"""kilosort.collision_templates: drop a learned template that is the sum of two neighbours."""
import numpy as np
import pytest

from kilosort import collision_templates as ct

NT, NC = 81, 16


def _spike(channel, width, amp=-100.0, centre=30):
    t = np.arange(NT)
    w = np.zeros((NT, NC))
    shape = amp * np.exp(-0.5 * ((t - centre) / width) ** 2)
    shape += -0.3 * amp * np.exp(-0.5 * ((t - centre - 3 * width) / (2 * width)) ** 2)
    for dc, gain in ((-1, 0.4), (0, 1.0), (1, 0.4)):
        if 0 <= channel + dc < NC:
            w[:, channel + dc] = gain * shape
    return w


def _shift(w, s):
    out = np.zeros_like(w)
    if s > 0:
        out[s:] = w[:-s]
    elif s < 0:
        out[:s] = w[-s:]
    else:
        out[:] = w
    return out


def _train(rng, n):
    return np.sort(rng.choice(np.arange(1000, 20_000 * 600), n, replace=False)).astype(np.int64)


@pytest.fixture
def scene():
    rng = np.random.default_rng(0)
    positions = np.stack([60.0 * np.arange(NC), np.zeros(NC)], axis=1)
    a, b, far = _spike(4, 3.0), _spike(5, 5.0, amp=-80.0), _spike(13, 4.0)
    ta, tb, tfar = _train(rng, 4000), _train(rng, 5000), _train(rng, 3000)
    tcol = _train(rng, 300)                      # the pair's collisions: neither A nor B has a spike there
    return positions, a, b, far, ta, tb, tfar, tcol, rng


def test_sum_of_two_neighbours_is_found_with_its_shifts(scene):
    positions, a, b, far, ta, tb, tfar, tcol, _ = scene
    col = _shift(a, 2) + _shift(b, -3)
    found = ct.find_collision_templates(np.stack([a, b, far, col]), positions, [ta, tb, tfar, tcol])
    assert [f['unit'] for f in found] == [3]
    f = found[0]
    assert {f['a'], f['b']} == {0, 1}
    by_unit = {f['a']: f['shift_a'], f['b']: f['shift_b']}
    assert by_unit == {0: 2, 1: -3}
    assert f['r2'] > 0.99 and f['r2'] - f['r2_single'] >= ct.GAIN


def test_real_neurons_are_kept(scene):
    positions, a, b, far, ta, tb, tfar, _, rng = scene
    third = _spike(4, 8.0, amp=-60.0, centre=34)          # its own shape, same place
    found = ct.find_collision_templates(np.stack([a, b, far, third]), positions,
                                        [ta, tb, tfar, _train(rng, 300)])
    assert found == []


def test_a_large_unit_that_looks_like_a_sum_is_kept(scene):
    """A pair cannot collide more often than its rarer member fires."""
    positions, a, b, far, ta, tb, tfar, _, rng = scene
    col = a + b
    found = ct.find_collision_templates(np.stack([a, b, far, col]), positions,
                                        [ta, tb, tfar, _train(rng, 3000)])
    assert found == []


def test_a_sum_whose_members_do_fire_there_is_kept(scene):
    """If A already has its spike at C's events, C is not a missed collision."""
    positions, a, b, far, ta, tb, tfar, tcol, _ = scene
    col = a + b
    ta_with = np.sort(np.concatenate([ta, tcol]))
    found = ct.find_collision_templates(np.stack([a, b, far, col]), positions, [ta_with, tb, tfar, tcol])
    assert found == []


def test_distant_templates_cannot_explain_a_unit(scene):
    positions, a, b, far, ta, tb, tfar, tcol, _ = scene
    far2 = _spike(11, 5.0, amp=-80.0)
    col = a + b                                           # A and B are not in the set; only distant units are
    found = ct.find_collision_templates(np.stack([far, far2, col]), positions, [tfar, tb, tcol])
    assert found == []


def _sort_state(scene, with_collision=True):
    import torch
    positions, a, b, far, ta, tb, tfar, tcol, _ = scene
    col = _shift(a, 2) + _shift(b, -3)
    wf = np.stack([a, b, far, col] if with_collision else [a, b, far])       # (n, nt, nc)
    # identity "PCA" (n_pcs = nt): Wall[i, c, p] = waveform[i, p, c]
    ops = {'wPCA': torch.eye(NT), 'xc': positions[:, 0], 'yc': positions[:, 1]}
    Wall = torch.as_tensor(wf, dtype=torch.float32).transpose(1, 2).contiguous()
    trains = [ta, tb, tfar, tcol] if with_collision else [ta, tb, tfar]
    t = np.concatenate(trains)
    clu = np.concatenate([np.full(x.size, k) for k, x in enumerate(trains)]).astype(np.int32)
    order = np.argsort(t, kind='stable')
    st = np.zeros((t.size, 3)); st[:, 0] = t[order]; st[:, 2] = np.arange(t.size)   # column 2 tags each row
    tF = torch.arange(t.size, dtype=torch.float32).reshape(-1, 1, 1)
    return ops, Wall, clu[order], st, tF, trains


def test_resolve_gives_the_collision_spikes_to_both_neurons(scene):
    ops, Wall, clu, st, tF, (ta, tb, tfar, tcol) = _sort_state(scene)
    W2, clu2, st2, tF2 = ct.resolve_collision_units(ops, Wall, clu, st, tF)
    assert W2.shape[0] == 3 and int(clu2.max()) == 2
    assert [f['unit'] for f in ops['collision_units']] == [3]
    t2 = st2[:, 0].astype(np.int64)
    assert np.all(np.diff(t2) >= 0), 'rows must be sorted by time'
    # A few collision events have a chance spike of A or B within TOL; those are not added twice.
    for unit, own, given in ((0, ta, tcol + 2), (1, tb, tcol - 3)):
        got = np.sort(t2[clu2 == unit])
        assert np.isin(got, np.concatenate([own, given])).all()
        assert np.isin(own, got).all()
        assert given.size - 6 <= np.isin(given, got).sum() <= given.size
    assert np.array_equal(np.sort(t2[clu2 == 2]), tfar)
    # features travel with their rows: column 2 of st and tF were the same tag
    assert np.array_equal(st2[:, 2], tF2[:, 0, 0].numpy())
    assert len(st) + tcol.size - 12 <= len(st2) <= len(st) + tcol.size


def test_resolve_leaves_a_sort_without_collision_units_alone(scene, monkeypatch):
    ops, Wall, clu, st, tF, _ = _sort_state(scene, with_collision=False)
    W2, clu2, st2, tF2 = ct.resolve_collision_units(ops, Wall, clu, st, tF)
    assert W2 is Wall and clu2 is clu and st2 is st and tF2 is tF
    ops, Wall, clu, st, tF, _ = _sort_state(scene)
    monkeypatch.setenv('KILOSORT_NO_COLLISION_UNITS', '1')
    W2, clu2, st2, tF2 = ct.resolve_collision_units(ops, Wall, clu, st, tF)
    assert W2 is Wall and st2 is st and ops['collision_units'] == []


def test_resolve_does_not_add_a_spike_the_neuron_already_has(scene):
    import torch
    ops, Wall, clu, st, tF, (ta, tb, tfar, tcol) = _sort_state(scene)
    # give A its own spike at 20 % of the collision events (below PRESENT, so C is still a collision unit)
    extra = tcol[::5] + 2
    st_x = np.zeros((extra.size, 3)); st_x[:, 0] = extra; st_x[:, 2] = -1
    st = np.concatenate([st, st_x]); clu = np.concatenate([clu, np.zeros(extra.size, np.int32)])
    tF = torch.cat([tF, torch.full((extra.size, 1, 1), -1.0)])
    o = np.argsort(st[:, 0], kind='stable'); st, clu, tF = st[o], clu[o], tF[torch.as_tensor(o)]
    _, clu2, st2, _ = ct.resolve_collision_units(ops, Wall, clu, st, tF)
    a_times = np.sort(st2[clu2 == 0, 0].astype(np.int64))
    assert np.isin(extra, a_times).all()
    assert np.isin(a_times, np.concatenate([ta, tcol + 2])).all()
    assert (a_times == extra[0]).sum() == 1                                      # no doubles
    assert a_times.size <= ta.size + tcol.size
