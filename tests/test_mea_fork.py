"""Regression tests for the MEA fork's own changes.

Everything here was previously guarded only by running a real sort and looking
at the yield, which is how the isotropic-grid regression survived for months.
These are all pure numpy: no probe download, no GPU, no recording.

Covered:
  * the float32 template-grid defect and both remedies (KS4_YUP_FIX)
  * dminx auto-resolution, which upstream crashes on
  * split_ccg_threshold actually reaching the comparison it names
"""
import inspect
import os

import numpy as np
import pytest

from kilosort import swarmsplitter
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
