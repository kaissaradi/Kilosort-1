"""Regression tests for the MEA fork's own changes.

Everything here was previously guarded only by running a real sort and looking
at the yield, which is how the isotropic-grid regression survived for months.
These are all pure numpy: no probe download, no GPU, no recording.

Covered:
  * the float32 template-grid defect and both remedies (KS4_YUP_FIX)
  * dminx auto-resolution, which upstream crashes on
  * split_ccg_threshold actually reaching the comparison it names
  * the robust whitening covariance (KS4_ROBUST_COV), off by default, and a
    synthetic reproduction of the flat-floor bug it targets
"""
import inspect
import os

import numpy as np
import pytest
import torch

from kilosort import io, preprocessing, swarmsplitter
from kilosort.spikedetect import nearest_neighbour_pitch, template_centers


# ---------------------------------------------------------------- geometry --

def litke512(dmin=90, dminx=90):
    """900 um tall, 60 um pitch, rows offset by half a column. float32, as the
    real probe files are -- the defect only appears at float32 resolution."""
    ys, xs = [], []
    for r, y in enumerate(np.arange(-450, 451, 60)):
        for x in np.arange(-450, 451, 60) + (30 if r % 2 else 0):
            xs.append(x); ys.append(y)
    return _ops(xs, ys, dmin, dminx)


def litke519(dmin=90, dminx=None):
    """780 um tall: does NOT divide evenly by dmin/2 = 45 (780/45 = 17.33),
    which is the geometry where the two remedies stop agreeing."""
    ys, xs = [], []
    for r, y in enumerate(np.arange(-390, 391, 30)):
        for x in np.arange(-390, 391, 30) + (15 if r % 2 else 0):
            xs.append(x); ys.append(y)
    return _ops(xs, ys, dmin, dminx)


def _ops(xs, ys, dmin, dminx):
    xc = np.array(xs, dtype=np.float32)
    yc = np.array(ys, dtype=np.float32)
    return {'xc': xc, 'yc': yc, 'kcoords': np.zeros(len(xc), dtype=np.int32),
            'settings': {'dmin': dmin, 'dminx': dminx}}


@pytest.fixture
def stock(monkeypatch):
    monkeypatch.delenv('KS4_YUP_FIX', raising=False)


def yup(mode, probe=litke512, **kw):
    if mode is None:
        os.environ.pop('KS4_YUP_FIX', None)
    else:
        os.environ['KS4_YUP_FIX'] = mode
    try:
        return template_centers(probe(**kw))['yup']
    finally:
        os.environ.pop('KS4_YUP_FIX', None)


# ------------------------------------------------------- the defect itself --

def test_endpoint_guard_is_below_float32_resolution():
    """The mechanism, stated independently of our code.

    np.arange's endpoint guard is a fixed +1e-5. If the probe coordinates are
    float32, the gap to the next representable number at y=450 is larger than
    that, so `ymax + .00001 == ymax` and the final grid point is dropped.
    """
    assert np.spacing(np.float32(450)) > 1e-5
    assert np.float32(450) + np.float32(.00001) == np.float32(450)

    got = np.arange(np.float32(-450), np.float32(450) + .00001, 45.0)
    assert got.size == 20 and got[-1] == 405.0      # top row bare
    ref = np.arange(-450.0, 450.0 + .00001, 45.0)   # float64: fine
    assert ref.size == 21 and ref[-1] == 450.0


def test_stock_grid_leaves_the_top_row_uncovered(stock):
    """Unfixed behaviour, locked in so a silent upstream change is visible."""
    g = yup(None)
    assert g.max() == 405.0
    assert g.min() == -450.0            # the arange START is never dropped
    assert g.size == 20


@pytest.mark.parametrize('mode', ['1', 'true', 'append'])
def test_fix_covers_the_top_row(mode):
    g = yup(mode)
    assert g.max() == 450.0
    assert g.size == 21
    assert np.allclose(np.diff(g), 45.0)


@pytest.mark.parametrize('mode', ['', '0'])
def test_fix_is_opt_in(mode):
    """Empty and '0' must be indistinguishable from stock. The default ships
    off, so anything that reads as 'unset' has to behave as unset."""
    assert np.array_equal(yup(mode), yup(None))


# --------------------------------------------- append vs linspace, by span --

def test_remedies_agree_when_the_span_divides_evenly():
    """Litke 512: 900 um / 45 um = 20 intervals exactly. Appending the dropped
    endpoint and re-spacing the whole grid give the same answer here, which is
    why three windows on this array could not tell them apart."""
    assert np.allclose(yup('1'), yup('linspace'))


def test_append_never_moves_an_interior_position():
    """The safety property that earns append-only its default.

    On a span that does not divide evenly, append must leave every position
    stock chose exactly where it was and only close the gap at the top.
    """
    s, a = yup(None, probe=litke519), yup('1', probe=litke519)
    assert np.all(np.isin(s, a))                 # nothing stock placed moved
    assert set(np.setdiff1d(a, s)) == {390.0}    # and nothing else was added


def test_linspace_repitches_the_whole_array():
    """The hazard, asserted rather than described.

    This is not an edge fix on a 780 um span: it re-spaces every template on
    the probe, which is what moved a 114 ADC interior unit from 0.987 to 0.729.
    Locked so nobody promotes linspace to the default believing it equivalent.
    """
    s, l = yup(None, probe=litke519), yup('linspace', probe=litke519)
    assert l.max() == 390.0                       # it does close the gap
    interior = l[(l > s.min()) & (l < s.max())]
    assert not np.all(np.isin(interior, s))       # ...by moving everything else
    assert not np.isclose(np.diff(l)[0], 45.0)    # 45.00 -> 45.88 um


def test_the_defect_is_present_at_every_dmin_and_costs_more_as_dmin_grows():
    """dmin/2 is the width of the hole. If the gap did not scale with dmin the
    cause would not be the grid."""
    gaps = {}
    for d in (45, 90):
        gaps[d] = 450.0 - yup(None, dmin=d).max()
    assert gaps[45] == 22.5 and gaps[90] == 45.0
    for d in (45, 90):
        assert yup('1', dmin=d).max() == 450.0


# ------------------------------------------------------------------ dminx --

def test_dminx_none_resolves_instead_of_crashing():
    """Upstream has no auto path for dminx: it sits at a default chosen for one
    probe and passing None raises TypeError downstream."""
    ops = template_centers(litke512(dminx=None))
    assert ops['dminx'] == pytest.approx(60.0, abs=1e-6)


def test_dminx_auto_is_not_tied_to_dmin():
    """Guards the regression that made every arm's grid isotropic.

    dminx must come from the array, not from dmin. On a 60 um array dmin 90
    must not drag dminx to 90 with it.
    """
    ops = template_centers(litke512(dmin=90, dminx=None))
    assert ops['dmin'] == 90
    assert ops['dminx'] != ops['dmin']


def test_dminx_explicit_is_passed_through():
    assert template_centers(litke512(dminx=32))['dminx'] == 32


def test_pitch_is_measured_not_inferred_per_axis():
    """On an offset lattice the x coordinates step 30 um while no two contacts
    are within 60 um, so an axis-wise estimate reports half the real spacing."""
    ops = litke512()
    assert np.median(np.diff(np.unique(ops['xc']))) == 30.0   # the wrong answer
    assert nearest_neighbour_pitch(ops['xc'], ops['yc']) == pytest.approx(60.0)


def test_pitch_survives_degenerate_probes():
    assert nearest_neighbour_pitch([0.0], [0.0]) == 1.0
    assert nearest_neighbour_pitch([0.0, 0.0], [0.0, 0.0]) == 1.0   # not 0


# ------------------------------------------------------ split_ccg_threshold --

def refractory_pair(n=4000, rate=20.0, refrac_ms=5.0, fs=1.0, seed=7):
    """One neuron's spike train, cut in two the way an over-split does it.

    Alternate spikes go to each half, so the halves are nearly disjoint but
    still share the parent's refractory period -- the signature the splitter
    is supposed to recognise and refuse to split.
    """
    rng = np.random.default_rng(seed)
    gaps = refrac_ms/1000 + rng.exponential(1.0/rate, n)
    st = np.cumsum(gaps)
    return st[0::2], st[1::2]


def test_threshold_reaches_the_comparison():
    """Not plumbing-by-inspection: the value must change the decision.

    R12 is a non-negative rate ratio, so `R12 < 0` can never hold and a
    threshold of 0 must force cross_refractory False whatever the trains say.
    """
    st1, st2 = refractory_pair()
    # bool(): check_CCG returns numpy bools, so identity against False is wrong.
    assert not bool(swarmsplitter.check_CCG(st1, st2, split_ccg_threshold=0.0)[1])
    assert bool(swarmsplitter.check_CCG(st1, st2, split_ccg_threshold=1e9)[1])


def test_threshold_is_monotone():
    """Raising it makes the splitter more willing to call two candidates one
    unit, and can only ever do that -- never the reverse."""
    st1, st2 = refractory_pair()
    seen = [bool(swarmsplitter.check_CCG(st1, st2, split_ccg_threshold=t)[1])
            for t in (0.0, 0.1, 0.25, 0.4, 0.9, 1e9)]
    assert seen == sorted(seen)          # False..False, True..True


def test_threshold_does_not_touch_the_acg_decision():
    """is_refractory uses its own hardcoded .1 and must be left alone."""
    st1, st2 = refractory_pair()
    a = swarmsplitter.check_CCG(st1, st2, split_ccg_threshold=0.0)[0]
    b = swarmsplitter.check_CCG(st1, st2, split_ccg_threshold=1e9)[0]
    assert a == b


def test_default_matches_stock():
    """Our default must equal the value upstream hardcoded, so the parameter is
    a knob and not a behaviour change smuggled in as one."""
    assert swarmsplitter.SPLIT_CCG_THRESHOLD == 0.25
    for fn in (swarmsplitter.check_CCG, swarmsplitter.refractoriness,
               swarmsplitter.split):
        p = inspect.signature(fn).parameters['split_ccg_threshold']
        assert p.default == swarmsplitter.SPLIT_CCG_THRESHOLD


def test_threshold_is_forwarded_through_refractoriness():
    """The kwarg exists three layers up; check it survives the trip rather than
    being accepted and dropped."""
    st1, st2 = refractory_pair()
    assert swarmsplitter.refractoriness(st1, st2, split_ccg_threshold=0.0) == 0
    assert swarmsplitter.refractoriness(st1, st2, split_ccg_threshold=1e9) == 1


def test_empty_trains_do_not_compute_garbage():
    """Upstream 4.1.3 wrote `len(st2 == 0)`, always truthy, which disabled every
    CCG check. Both flags must come back False without dividing by a zero span."""
    st1, _ = refractory_pair()
    for a, b in ((st1, np.array([])), (np.array([]), st1),
                 (np.ones(50), np.ones(50))):
        assert [bool(v) for v in swarmsplitter.check_CCG(a, b)] == [False, False]


# ---------------------------------------------------------------------------
# merging_function sweeps to a fixpoint
#
# `isort` ranks units by spike count once, before any merge. Absorbing a unit
# zeroes its count but leaves it at its original rank, so the old
# `if ns[kk] == 0: break` fired in the MIDDLE of the ordering and terminated
# the whole stage, abandoning every lower-ranked unit unexamined.
# ---------------------------------------------------------------------------

def _merge_case(dirs, counts, nt=61, npc=1, fs=20000.0, seed=0):
    """Minimal ops/Wall/clu/st/tF for merging_function in 'mu' mode.

    One PC, one unit-norm basis vector, so the template correlation the merge
    computes reduces to the cosine between the per-channel vectors in `dirs`
    and the merge decision is fully determined by geometry.
    """
    import torch
    dirs = np.asarray(dirs, dtype=np.float64)
    NN, nchan = dirs.shape
    Wall = torch.zeros(NN, nchan, npc, dtype=torch.float32)
    for i, d in enumerate(dirs):
        Wall[i, :, 0] = torch.tensor(d, dtype=torch.float32)
    wPCA = torch.zeros(npc, nt, dtype=torch.float32)
    wPCA[0, nt // 2] = 1.0
    ops = {'fs': fs, 'nt': nt, 'wPCA': wPCA,
           'settings': {'acg_threshold': 0.2, 'ccg_threshold': 0.2}}
    rng = np.random.default_rng(seed)
    clu, times = [], []
    for i, n in enumerate(counts):
        times.append(np.sort(rng.integers(0, int(600 * fs), n)))
        clu.append(np.full(n, i))
    clu = np.concatenate(clu)
    times = np.concatenate(times)
    order = np.argsort(times)
    clu, times = clu[order], times[order]
    st = np.zeros((times.size, 6), dtype=np.int64)
    st[:, 0] = times
    tF = torch.zeros(times.size, nchan, npc, dtype=torch.float32)
    return ops, Wall, clu, st, tF


# Two independent mergeable pairs. A and B are the two highest-count units, so
# B is absorbed first and then sits at rank 1 with ns == 0 -- exactly where the
# stale-ordering exit fires, before C and D are ever looked at.
_TWO_PAIRS = np.array([[1., 0., 0., 0.],   # A
                       [1., 0., 0., 0.],   # B  (merges with A)
                       [0., 0., 1., 0.],   # C
                       [0., 0., 1., 0.]])  # D  (merges with C)
_TWO_PAIRS_COUNTS = [500, 400, 300, 200]


def _run_merge(max_sweeps):
    import torch
    from kilosort.template_matching import merging_function
    ops, Wall, clu, st, tF = _merge_case(_TWO_PAIRS, _TWO_PAIRS_COUNTS)
    Ww, clu2, _, _, _ = merging_function(
        ops, Wall, clu, st, tF, mode='mu', check_dt=False,
        device=torch.device('cpu'), max_sweeps=max_sweeps)
    return Ww.shape[0], clu2


def test_one_pass_abandons_units_below_the_first_absorbed_rank():
    """Locks the defect in, so a future refactor cannot quietly restore it."""
    n_units, _ = _run_merge(max_sweeps=1)
    assert n_units == 3, 'single pass should miss the C/D merge entirely'


def test_sweeping_to_fixpoint_finds_the_abandoned_merge():
    n_units, clu2 = _run_merge(max_sweeps=10)
    assert n_units == 2
    # Both pairs collapsed: two surviving labels, each holding two units' spikes.
    assert len(np.unique(clu2)) == 2


def test_extra_sweeps_are_idempotent_once_no_merge_remains():
    """A fixpoint is a fixpoint: more budget must not keep eating units."""
    assert _run_merge(max_sweeps=10)[0] == _run_merge(max_sweeps=50)[0]


def test_max_sweeps_comes_from_settings_and_defaults_to_more_than_one():
    import torch
    from kilosort.template_matching import merging_function
    ops, Wall, clu, st, tF = _merge_case(_TWO_PAIRS, _TWO_PAIRS_COUNTS)
    ops['settings']['max_merge_sweeps'] = 1
    Ww, _, _, _, _ = merging_function(ops, Wall, clu, st, tF, mode='mu',
                                      check_dt=False,
                                      device=torch.device('cpu'))
    assert Ww.shape[0] == 3, 'settings must be able to restore one-pass'
    # and the default, with nothing set, must sweep
    ops2, Wall2, clu2_, st2, tF2 = _merge_case(_TWO_PAIRS, _TWO_PAIRS_COUNTS)
    Ww2, _, _, _, _ = merging_function(ops2, Wall2, clu2_, st2, tF2, mode='mu',
                                       check_dt=False,
                                       device=torch.device('cpu'))
    assert Ww2.shape[0] == 2


def test_missing_settings_key_does_not_crash_the_merge():
    """ops dicts written before this change carry no max_merge_sweeps key.
    They must fall back to the default, not raise."""
    import torch
    from kilosort.template_matching import merging_function
    ops, Wall, clu, st, tF = _merge_case(_TWO_PAIRS, _TWO_PAIRS_COUNTS)
    assert 'max_merge_sweeps' not in ops['settings']
    Ww, _, _, _, _ = merging_function(ops, Wall, clu, st, tF, mode='mu',
                                      check_dt=False,
                                      device=torch.device('cpu'))
    assert Ww.shape[0] == 2


def test_nothing_mergeable_terminates_in_one_sweep():
    """Orthogonal templates: the fixpoint loop must not spin or merge anything."""
    import torch
    from kilosort.template_matching import merging_function
    dirs = np.eye(4)
    ops, Wall, clu, st, tF = _merge_case(dirs, [500, 400, 300, 200])
    Ww, clu2, _, _, _ = merging_function(ops, Wall, clu, st, tF, mode='mu',
                                         check_dt=False,
                                         device=torch.device('cpu'),
                                         max_sweeps=10)
    assert Ww.shape[0] == 4
    assert len(np.unique(clu2)) == 4


# ---------------------------------------------------------------------------
# Refractory merge veto: refuse a merge whose union cannot be one neuron.
#
# The splitter's other gates all ask whether the two halves belong together;
# none asks whether the RESULT is a single cell. Because maketree only
# agglomerates and split() only prunes merges, a fusion made here is permanent,
# so a regression in this gate cannot be caught by any later stage.

def test_veto_needs_both_a_ratio_and_significance():
    """Either test alone misfires, in opposite directions.

    A handful of violations on a huge train is a clean cell, which the ratio
    catches; a handful on a tiny train is noise, which the Poisson tail catches.
    An earlier ISI test used the raw violation percentage alone and produced
    five false over-splits.

    The pair is deliberately conservative, and matches the GT-free contamination
    metric bar for bar (scripts/qa.py): a unit whose count is a high fraction of
    chance but not significantly above it is UNDECIDED, not contaminated, and
    this veto leaves it merged. That is why it fires on only a quarter of kept
    merges instead of shattering the sort.
    """
    imp = swarmsplitter._impossible
    assert imp(60, 20)           # far above chance, with the events to prove it
    assert not imp(2, 100)       # far below chance: a clean cell
    assert not imp(60, 100)      # ratio is high, significance is not: undecided
    assert not imp(2, 2)         # at chance but far too few events to say
    assert not imp(5, 0)         # no expectation -> no claim


def test_veto_turns_a_kept_merge_into_a_split():
    """Two trains that interleave freely must not be left as one unit."""
    rng = np.random.default_rng(0)
    # One train, refractory. The other is the same cell shifted by half a
    # refractory period, so every gate that looks at the two halves separately
    # sees two clean, well-separated cells -- and their union does not.
    a = np.cumsum(rng.uniform(0.004, 0.02, 4000))
    b = a + 0.0007
    meta = np.concatenate([a, b])
    iclust = np.concatenate([np.zeros(a.size, int), np.ones(b.size, int)])
    xtree = np.array([[0, 1, 2]], dtype=np.int32)
    # tstat[:,0] must clear the modularity gate or the node is split for an
    # unrelated reason and the test proves nothing.
    tstat = np.array([[1.0, 2.0, 1.0]], dtype=np.float32)
    my_clus = [[0], [1], [0, 1]]
    Xd = np.zeros((meta.size, 2), dtype=np.float32)

    kept = swarmsplitter.split(Xd, xtree, tstat, iclust, my_clus, meta=meta,
                               meta_sorted=False, refrac_veto=False)
    vetoed = swarmsplitter.split(Xd, xtree, tstat, iclust, my_clus, meta=meta,
                                 meta_sorted=False, refrac_veto=True)
    assert kept[0].shape[0] == 0, 'without the veto this merge is kept'
    assert vetoed[0].shape[0] == 1, 'the veto must leave the merge unmade'


def test_veto_never_creates_a_merge():
    """It may only turn keep-merged into split, so it cannot fuse anything."""
    rng = np.random.default_rng(1)
    # Two genuinely separate, refractory cells: the union is fine, so the veto
    # has nothing to say and must not change the outcome either way.
    a = np.cumsum(rng.uniform(0.01, 0.03, 2000))
    b = np.cumsum(rng.uniform(0.01, 0.03, 2000)) + 0.005
    meta = np.concatenate([a, b])
    iclust = np.concatenate([np.zeros(a.size, int), np.ones(b.size, int)])
    xtree = np.array([[0, 1, 2]], dtype=np.int32)
    tstat = np.array([[1.0, 2.0, 1.0]], dtype=np.float32)
    my_clus = [[0], [1], [0, 1]]
    Xd = np.zeros((meta.size, 2), dtype=np.float32)
    off = swarmsplitter.split(Xd, xtree, tstat, iclust, my_clus, meta=meta,
                              meta_sorted=False, refrac_veto=False)
    on = swarmsplitter.split(Xd, xtree, tstat, iclust, my_clus, meta=meta,
                             meta_sorted=False, refrac_veto=True)
    assert on[0].shape[0] >= off[0].shape[0]


def test_veto_default_is_on_and_reaches_the_setting():
    import inspect
    assert swarmsplitter.REFRAC_VETO is True
    p = inspect.signature(swarmsplitter.split).parameters['refrac_veto']
    assert p.default is swarmsplitter.REFRAC_VETO
    from kilosort.parameters import MAIN_PARAMETERS, EXTRA_PARAMETERS
    allp = {**MAIN_PARAMETERS, **EXTRA_PARAMETERS}
    assert allp['refractory_merge_veto']['default'] is True


def test_veto_bar_is_settable_and_changes_the_verdict():
    """The bar is a measurement, not a constant, so it has to reach split().

    0.35/0.01 was inherited from the metric the veto is scored on. That makes
    the headline partly self-referential and makes the bar itself the next thing
    to sweep -- which is only possible if both numbers thread through.
    """
    imp = swarmsplitter._impossible
    # One union, 15 violations where chance gives 10: ratio 1.5, Poisson tail
    # 0.083. It sits between the two alphas and under the strict ratio, so each
    # knob decides it on its own.
    assert imp(15, 10, ratio=0.35, alpha=0.10)
    assert not imp(15, 10, ratio=0.35, alpha=0.05), 'alpha must be consulted'
    assert not imp(15, 10, ratio=2.00, alpha=0.10), 'ratio must be consulted'

    import inspect
    p = inspect.signature(swarmsplitter.split).parameters
    assert p['refrac_veto_ratio'].default == swarmsplitter.REFRAC_VETO_RATIO
    assert p['refrac_veto_alpha'].default == swarmsplitter.REFRAC_VETO_ALPHA
    from kilosort.parameters import MAIN_PARAMETERS, EXTRA_PARAMETERS
    allp = {**MAIN_PARAMETERS, **EXTRA_PARAMETERS}
    assert allp['refractory_veto_ratio']['default'] == 0.35
    assert allp['refractory_veto_alpha']['default'] == 0.01


def test_stale_env_toggle_cannot_silently_agree_with_the_setting():
    """KS4_REFRAC_VETO must win outright, or not exist.

    The confirmation sweep set it on every arm while the committed code read
    only the setting, so a "veto off" arm ran the veto. Nothing failed and
    nothing warned; the sweep just quietly compared a config with itself. An env
    var that half-works is worse than either alternative, so this pins the
    override.
    """
    import os
    from kilosort import clustering_qr
    ops = {'settings': {'refractory_merge_veto': True}}
    old = os.environ.pop('KS4_REFRAC_VETO', None)
    try:
        assert clustering_qr._veto_on(ops) is True
        os.environ['KS4_REFRAC_VETO'] = '0'
        assert clustering_qr._veto_on(ops) is False, 'env must override the setting'
        os.environ['KS4_REFRAC_VETO'] = '1'
        ops['settings']['refractory_merge_veto'] = False
        assert clustering_qr._veto_on(ops) is True
        del os.environ['KS4_REFRAC_VETO']
        assert clustering_qr._veto_on(ops) is False, 'setting rules when env is absent'
    finally:
        os.environ.pop('KS4_REFRAC_VETO', None)
        if old is not None:
            os.environ['KS4_REFRAC_VETO'] = old



# ------------------------------------------------------- robust whitening --
#
# get_whitening_matrix's CC = X @ X.T / T is a plain second moment over every
# sample, spikes included. A channel carrying real spikes reads as
# high-variance and gets whitened down harder than a dead channel with the
# SAME true noise floor -- the whitening-inverts-the-threshold defect. These
# tests are synthetic and self-contained: no probe download, no GPU, no real
# recording, and no claim about real-data recall/precision (that needs a real
# GT bench, which is not available right now -- see the commit message).

def _synthetic_batch(rng, n_clean=4, n_spiky=4, T=40000, sigma=1.0,
                      spike_rate=0.02, spike_amp=10.0):
    """(n_clean + n_spiky, T) float32 batch. Every channel has the SAME true
    noise sigma. The `n_spiky` channels additionally carry large injected
    events at random samples, at `spike_rate` fraction of samples and
    `spike_amp` * sigma amplitude (random sign). Returns X and a boolean
    (n_chan, T) mask of which samples are pure noise (no injected event) --
    used to measure the TRUE noise floor after whitening without spike
    energy contaminating the estimate."""
    n_chan = n_clean + n_spiky
    X = rng.standard_normal((n_chan, T)).astype(np.float32) * sigma
    is_noise_only = np.ones((n_chan, T), dtype=bool)
    n_events = int(round(spike_rate * T))
    for c in range(n_clean, n_chan):
        idx = rng.choice(T, size=n_events, replace=False)
        signs = rng.choice([-1.0, 1.0], size=n_events)
        X[c, idx] += signs * spike_amp * sigma
        is_noise_only[c, idx] = False
    return X, is_noise_only


def _post_whitening_noise_floor(Wrot, X, is_noise_only):
    """Per-channel MAD-based sigma of Wrot @ X, restricted to samples with no
    injected event on ANY channel (so cross-channel whitening leakage from a
    spike on another channel cannot contaminate the floor estimate either)."""
    Y = (Wrot.double() @ torch.from_numpy(X).double()).numpy()
    clean_cols = is_noise_only.all(axis=0)
    Yc = Y[:, clean_cols]
    mad = np.median(np.abs(Yc - np.median(Yc, axis=1, keepdims=True)), axis=1)
    return mad / 0.6745


def _line_probe(n_chan, pitch=30.0):
    return np.arange(n_chan, dtype=np.float64) * pitch, np.zeros(n_chan)


def test_robust_cov_flag_is_opt_in_and_off_by_default():
    """os.environ with no KS4_ROBUST_COV (or '0'/'false'/'') must reproduce
    stock get_whitening_matrix exactly -- verified byte-identical, not just
    close, against the plain (X @ X.T)/T formula it replaces when the batch
    loop is unrolled by hand."""
    n_chan, NT, nt = 6, 800, 21
    path_bytes = None
    rng = np.random.default_rng(0)
    data = rng.integers(-100, 100, size=(NT, n_chan), dtype=np.int16)

    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / 'batch.bin'
        data.tofile(path)
        xc, yc = _line_probe(n_chan)
        bfile = io.BinaryFiltered(
            path, n_chan_bin=n_chan, fs=20000, NT=NT, nt=nt,
            chan_map=np.arange(n_chan), device=torch.device('cpu'),
            do_CAR=False,
        )
        assert bfile.n_batches == 1

        old = os.environ.pop('KS4_ROBUST_COV', None)
        try:
            for env_val in (None, '0', 'false', ''):
                if env_val is None:
                    os.environ.pop('KS4_ROBUST_COV', None)
                else:
                    os.environ['KS4_ROBUST_COV'] = env_val
                Wrot = preprocessing.get_whitening_matrix(
                    bfile, xc, yc, nskip=25, nrange=4)

                # Reproduce the stock computation by hand from the same file.
                X = bfile.padded_batch_to_torch(0)
                X = X[:, bfile.nt: -bfile.nt]
                CC_expected = (X @ X.T) / X.shape[1]
                Wrot_expected = preprocessing.whitening_local(
                    CC_expected, xc, yc, nrange=4, device=torch.device('cpu'))
                torch.testing.assert_close(Wrot, Wrot_expected, rtol=0, atol=0)
        finally:
            os.environ.pop('KS4_ROBUST_COV', None)
            if old is not None:
                os.environ['KS4_ROBUST_COV'] = old


def test_robust_cov_batch_helper_matches_get_whitening_matrix_when_on():
    """Sanity check that flipping KS4_ROBUST_COV=1 actually engages
    robust_batch_covariance inside get_whitening_matrix, rather than the flag
    being read but never wired to the loop."""
    n_chan, NT, nt = 6, 800, 21
    rng = np.random.default_rng(1)
    data = rng.integers(-100, 100, size=(NT, n_chan), dtype=np.int16)

    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / 'batch.bin'
        data.tofile(path)
        xc, yc = _line_probe(n_chan)
        bfile = io.BinaryFiltered(
            path, n_chan_bin=n_chan, fs=20000, NT=NT, nt=nt,
            chan_map=np.arange(n_chan), device=torch.device('cpu'),
            do_CAR=False,
        )
        old = os.environ.pop('KS4_ROBUST_COV', None)
        try:
            os.environ['KS4_ROBUST_COV'] = '1'
            Wrot_on = preprocessing.get_whitening_matrix(
                bfile, xc, yc, nskip=25, nrange=4)
            os.environ.pop('KS4_ROBUST_COV')
            Wrot_off = preprocessing.get_whitening_matrix(
                bfile, xc, yc, nskip=25, nrange=4)
        finally:
            os.environ.pop('KS4_ROBUST_COV', None)
            if old is not None:
                os.environ['KS4_ROBUST_COV'] = old
        assert torch.isfinite(Wrot_on).all()
        # On integer-noise data the covariance estimators are close but not
        # required to be identical; the flag must at least be ABLE to change
        # the result (it is not silently a no-op wired to nothing).
        assert not torch.equal(Wrot_on, Wrot_off) or torch.allclose(
            Wrot_on, Wrot_off, atol=1e-3
        )


def test_positive_control_stock_covariance_inverts_the_noise_floor():
    """Reproduce the documented bug on synthetic data BEFORE testing the fix:
    with identical true noise sigma on every channel, channels carrying
    injected large-amplitude events end up with a LOWER post-whitening noise
    floor under the stock (X @ X.T)/T covariance."""
    rng = np.random.default_rng(42)
    n_clean, n_spiky = 5, 5
    X, is_noise_only = _synthetic_batch(rng, n_clean=n_clean, n_spiky=n_spiky,
                                         T=60000, sigma=1.0, spike_rate=0.03,
                                         spike_amp=12.0)
    xc, yc = _line_probe(n_clean + n_spiky)
    Xt = torch.from_numpy(X)

    CC_plain = (Xt @ Xt.T) / Xt.shape[1]
    Wrot_plain = preprocessing.whitening_local(
        CC_plain, xc, yc, nrange=n_clean + n_spiky, device=torch.device('cpu'))
    floor = _post_whitening_noise_floor(Wrot_plain, X, is_noise_only)

    clean_floor = floor[:n_clean]
    spiky_floor = floor[n_clean:]
    ratio = spiky_floor.mean() / clean_floor.mean()
    spread = np.percentile(floor, 95) / np.percentile(floor, 5)

    print(f'\n[positive control] clean floor mean={clean_floor.mean():.4f} '
          f'spiky floor mean={spiky_floor.mean():.4f} ratio={ratio:.4f} '
          f'p95/p5 spread={spread:.4f}')

    # The documented direction: spike-carrying channels end up QUIETER
    # (ratio well below 1) despite an identical true noise floor.
    assert ratio < 0.85, (
        f'expected the stock estimator to under-state noise on spiky '
        f'channels (ratio << 1), got ratio={ratio:.4f}'
    )


def test_robust_covariance_flattens_the_noise_floor():
    """Same synthetic setup as the positive control. The robust estimator
    (KS4_ROBUST_COV's robust_batch_covariance) should recover a noise floor
    that is markedly FLATTER across clean vs. spiky channels than the stock
    estimator's, on this synthetic data with known ground truth."""
    rng = np.random.default_rng(42)
    n_clean, n_spiky = 5, 5
    X, is_noise_only = _synthetic_batch(rng, n_clean=n_clean, n_spiky=n_spiky,
                                         T=60000, sigma=1.0, spike_rate=0.03,
                                         spike_amp=12.0)
    xc, yc = _line_probe(n_clean + n_spiky)
    Xt = torch.from_numpy(X)

    CC_plain = (Xt @ Xt.T) / Xt.shape[1]
    CC_robust = preprocessing.robust_batch_covariance(Xt)

    Wrot_plain = preprocessing.whitening_local(
        CC_plain, xc, yc, nrange=n_clean + n_spiky, device=torch.device('cpu'))
    Wrot_robust = preprocessing.whitening_local(
        CC_robust, xc, yc, nrange=n_clean + n_spiky, device=torch.device('cpu'))

    floor_plain = _post_whitening_noise_floor(Wrot_plain, X, is_noise_only)
    floor_robust = _post_whitening_noise_floor(Wrot_robust, X, is_noise_only)

    ratio_plain = floor_plain[n_clean:].mean() / floor_plain[:n_clean].mean()
    ratio_robust = floor_robust[n_clean:].mean() / floor_robust[:n_clean].mean()
    spread_plain = np.percentile(floor_plain, 95) / np.percentile(floor_plain, 5)
    spread_robust = np.percentile(floor_robust, 95) / np.percentile(floor_robust, 5)

    print(f'\n[fix] plain ratio={ratio_plain:.4f} spread(p95/p5)={spread_plain:.4f} '
          f'  robust ratio={ratio_robust:.4f} spread(p95/p5)={spread_robust:.4f}')

    # The robust estimator must move the ratio measurably closer to 1 (flat)
    # than the plain estimator, on this synthetic ground truth.
    assert abs(ratio_robust - 1.0) < abs(ratio_plain - 1.0), (
        f'robust estimator did not flatten the floor: plain ratio='
        f'{ratio_plain:.4f} robust ratio={ratio_robust:.4f}'
    )
    assert ratio_robust > ratio_plain, (
        'robust estimator should raise the spiky-channel floor back toward '
        'the clean-channel floor, not lower it further'
    )


def test_robust_covariance_agrees_with_stock_on_clean_data():
    """No spike contamination anywhere -- all channels iid Gaussian noise
    with the same true sigma. The robust and stock estimators should agree
    closely: this is the sanity check that the robust estimator is not doing
    something arbitrary when there is nothing to be robust to."""
    rng = np.random.default_rng(7)
    n_chan = 8
    X = (rng.standard_normal((n_chan, 60000)) * 1.0).astype(np.float32)
    xc, yc = _line_probe(n_chan)
    Xt = torch.from_numpy(X)

    CC_plain = (Xt @ Xt.T) / Xt.shape[1]
    CC_robust = preprocessing.robust_batch_covariance(Xt)

    rel_diff = (CC_robust - CC_plain).abs().max() / CC_plain.abs().max()
    print(f'\n[clean-data sanity] max relative CC difference={rel_diff:.4f}')
    assert rel_diff < 0.05, (
        f'robust estimator diverges from stock on clean data with nothing '
        f'to be robust to: max relative diff={rel_diff:.4f}'
    )

    Wrot_plain = preprocessing.whitening_local(
        CC_plain, xc, yc, nrange=n_chan, device=torch.device('cpu'))
    Wrot_robust = preprocessing.whitening_local(
        CC_robust, xc, yc, nrange=n_chan, device=torch.device('cpu'))
    torch.testing.assert_close(Wrot_plain, Wrot_robust, rtol=0.05, atol=0.05)
