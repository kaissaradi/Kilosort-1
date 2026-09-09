"""The fused peel subtract must equal the stock one, bit for bit.

fused_peel replaces run_matching's two `-=` scatter statements with one Triton
kernel each. It is exact only under two conditions, and both are easy to break
by a well-meaning edit, so both are pinned here:

  * the phase's windows must be DISJOINT, because advanced-index `-=` drops
    rather than accumulates duplicate contributions. A census of real data
    found overlap in 1 phase of 29,114 -- rare enough to survive any smoke
    test and still corrupt a sort.
  * the multiply and the subtract must stay two separately rounded fp32 steps.
    Contracted into one FFMA they differ; the launch passes
    enable_fp_fusion=False to prevent that.

Comparisons here are on RAW BIT PATTERNS. torch.equal is value equality and
reports +0.0 == -0.0, which is precisely the difference a rounding barrier can
introduce -- an earlier version of this kernel passed torch.equal while
flipping the sign of every zero.

Skipped without CUDA or Triton, which is the fallback case anyway.
"""
import pytest
import torch

from kilosort import fused_peel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_peel._HAVE_TRITON,
    reason='fused peel needs CUDA and Triton')

NT_LEN, NCHAN, NUNITS, NT = 4096, 64, 48, 61
dev = 'cuda'


def bit_eq(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


def problem(n_spk, seed, gap=None):
    """Spike positions spread far enough apart to be window-disjoint unless
    `gap` forces otherwise."""
    g = torch.Generator().manual_seed(seed)
    Xres = (torch.randn(NCHAN, NT_LEN, generator=g) * 10).to(dev)
    B = (torch.randn(NUNITS, NT_LEN, generator=g) * 10).to(dev)
    U_time = torch.randn(NUNITS, NCHAN, NT, generator=g).to(dev)
    ctc = torch.randn(NUNITS, NUNITS, 2 * NT + 1, generator=g).to(dev)
    if gap is None:
        step = (NT_LEN - 2 * NT - 2) // max(n_spk, 1)
        assert step > 2 * NT + 2
        pos = torch.arange(n_spk) * step + NT + 1
    else:
        pos = torch.arange(n_spk) * gap + NT + 1
    iX = pos.to(dev).unsqueeze(1)
    iY = torch.randint(0, NUNITS, (n_spk, 1), generator=g).to(dev)
    amp = torch.randn(n_spk, 1, generator=g).to(dev)
    tiwave = torch.arange(-(NT // 2), NT // 2 + 1, device=dev)
    trange = torch.arange(-NT, NT + 1, device=dev)
    return Xres, B, U_time, ctc, iX, iY, amp, tiwave, trange


def stock_all(Xres, B, iX, iY, amp, U_time, ctc, tiwave, trange, n=2):
    for j in range(n):
        fused_peel._stock_phase(Xres, B, iX, iY, amp, U_time, ctc,
                                tiwave, trange, j, n)


@pytest.fixture(autouse=True)
def reset_choice():
    saved, saved_lut = fused_peel._CHOICE, fused_peel._LUT_CHOICE
    fused_peel._CHOICE = None
    fused_peel._LUT_CHOICE = None
    fused_peel._LUT_CACHE.clear()
    yield
    fused_peel._CHOICE, fused_peel._LUT_CHOICE = saved, saved_lut
    fused_peel._LUT_CACHE.clear()


@pytest.mark.parametrize('n_spk', [1, 2, 5, 16])
def test_matches_stock_bit_for_bit(n_spk):
    Xres, B, U_time, ctc, iX, iY, amp, tiwave, trange = problem(n_spk, n_spk)
    Xs, Bs = Xres.clone(), B.clone()
    stock_all(Xs, Bs, iX, iY, amp, U_time, ctc, tiwave, trange)

    fused_peel.peel_subtract(Xres, B, iX, iY, amp, U_time, ctc,
                             ctc.permute(1, 0, 2), tiwave, trange, NT)
    assert bit_eq(Xres, Xs)
    assert bit_eq(B, Bs)


def test_still_matches_after_the_gate_has_decided():
    """The first call validates and latches; later calls run fused unchecked,
    so they are the ones that can go wrong unnoticed."""
    args = problem(8, 3)
    Xres, B, U_time, ctc, iX, iY, amp, tiwave, trange = args
    fused_peel.peel_subtract(Xres, B, iX, iY, amp, U_time, ctc,
                             ctc.permute(1, 0, 2), tiwave, trange, NT)
    if fused_peel._CHOICE is not True:
        pytest.skip('fused peel not bit-identical on this device')

    Xres, B, U_time, ctc, iX, iY, amp, tiwave, trange = problem(11, 4)
    Xs, Bs = Xres.clone(), B.clone()
    stock_all(Xs, Bs, iX, iY, amp, U_time, ctc, tiwave, trange)
    fused_peel.peel_subtract(Xres, B, iX, iY, amp, U_time, ctc,
                             ctc.permute(1, 0, 2), tiwave, trange, NT)
    assert bit_eq(Xres, Xs), 'diverged on a phase the gate never validated'
    assert bit_eq(B, Bs)


def test_overlapping_windows_take_the_stock_path(monkeypatch):
    """The 1-in-29k case: positions 3 apart make the (2nt+1) windows overlap
    within a phase, where the fused kernel is not equivalent.

    The assertion is on the code PATH, not on the values, because stock has no
    single answer here to compare against: with duplicate indices its scatter
    is last-write-wins, and running it twice on identical inputs was measured
    to differ on 152-838 elements (up to 4.96 in magnitude). Falling back
    preserves stock's behaviour, nondeterminism included -- it does not, and
    cannot, make that phase reproducible.
    """
    Xres, B, U_time, ctc, iX, iY, amp, tiwave, trange = problem(6, 5, gap=3)
    assert not bool(fused_peel.phases_disjoint(iX[:, 0], NT))

    def boom(*a, **k):
        raise AssertionError('fused kernel ran on an overlapping phase')

    monkeypatch.setattr(fused_peel, '_fused_phase', boom)
    fused_peel._CHOICE = True          # gate already satisfied elsewhere
    fused_peel.peel_subtract(Xres, B, iX, iY, amp, U_time, ctc,
                             ctc.permute(1, 0, 2), tiwave, trange, NT)


def test_gate_stays_undecided_on_an_overlapping_first_phase():
    """The gate must not validate against a phase the fused path would never
    take -- it should leave the verdict open and try the next peel."""
    Xres, B, U_time, ctc, iX, iY, amp, tiwave, trange = problem(6, 5, gap=3)
    assert fused_peel._CHOICE is None
    fused_peel.peel_subtract(Xres, B, iX, iY, amp, U_time, ctc,
                             ctc.permute(1, 0, 2), tiwave, trange, NT)
    assert fused_peel._CHOICE is None


def test_phases_disjoint_boundary():
    """Windows are (2nt+1) wide, so within a phase (stride 2) the pairwise sum
    must exceed 2nt."""
    p = torch.tensor([100, 150, 100 + 2 * NT, 400], device=dev)
    assert not bool(fused_peel.phases_disjoint(p, NT))
    p = torch.tensor([100, 150, 100 + 2 * NT + 1, 400], device=dev)
    assert bool(fused_peel.phases_disjoint(p, NT))
    assert bool(fused_peel.phases_disjoint(torch.tensor([5], device=dev), NT))


def test_env_switch_forces_stock():
    import os
    Xres, B, U_time, ctc, iX, iY, amp, tiwave, trange = problem(8, 6)
    Xs, Bs = Xres.clone(), B.clone()
    stock_all(Xs, Bs, iX, iY, amp, U_time, ctc, tiwave, trange)
    os.environ['KILOSORT_NO_FUSED_PEEL'] = '1'
    try:
        fused_peel.peel_subtract(Xres, B, iX, iY, amp, U_time, ctc,
                                 ctc.permute(1, 0, 2), tiwave, trange, NT)
    finally:
        del os.environ['KILOSORT_NO_FUSED_PEEL']
    assert fused_peel._CHOICE is False
    assert bit_eq(Xres, Xs)
    assert bit_eq(B, Bs)


# ---------------------------------------------------------------------------
# Live-tile LUT. Dead tiles are dropped BEFORE the launch, which is exact only
# when a dropped tile is bitwise all +0.0 and amp > 0. Both guards are easy to
# "simplify" into something that looks equivalent and is not:
#   * `src == 0` instead of a bit test admits -0.0, and `o - (a * -0.0)` is
#     `o + 0.0`, which maps a stored -0.0 to +0.0.
#   * `amp >= 0` admits a = -0.0, which has the same effect on a +0.0 tile.
# The two tests named for those cases fail if either guard is loosened.

def _dead_mask(n_unit, n_row, zero_frac, g):
    """Dead-row mask where each unit's LIVE rows form one contiguous band.

    Deliberately not uniform-random. Liveness has to be spatially clustered or
    no BLOCK_R-wide tile is ever entirely dead: with 85% of rows zeroed at
    random, a 16-row tile survives with probability 1 - 0.85**16 = 93%, so a
    random mask tests the skip on data where it cannot fire. Real U is zero
    off a cluster's own channels, which are contiguous, and the measured ctc
    has max 19 live tiles of 65 for exactly that reason.
    """
    band = max(1, int(round(n_row * (1.0 - zero_frac))))
    start = torch.randint(0, max(1, n_row - band + 1), (n_unit,), generator=g)
    rows = torch.arange(n_row)
    live = ((rows[None, :] >= start[:, None])
            & (rows[None, :] < start[:, None] + band))
    return ~live


def sparse_problem(n_spk, seed, zero_frac=0.85, neg_zero=False,
                   positive_amp=True):
    """problem(), but with most (unit, row) blocks of ctc/U_time exactly zero,
    which is what real templates look like -- U is written into zeros and only
    a cluster's own channels are touched."""
    g = torch.Generator().manual_seed(seed)
    Xres = (torch.randn(NCHAN, NT_LEN, generator=g) * 10).to(dev)
    B = (torch.randn(NUNITS, NT_LEN, generator=g) * 10).to(dev)
    U_time = torch.randn(NUNITS, NCHAN, NT, generator=g).to(dev)
    ctc = torch.randn(NUNITS, NUNITS, 2 * NT + 1, generator=g).to(dev)
    # U_time is [y, chan, t] and the LUT tiles over chan, so the band is
    # already on the right axis. ctc is [i, j, t] but the kernel sees the
    # permuted ctc_p[j, i, t] and tiles over i -- so ctc's band must be
    # contiguous in i FOR EACH j, i.e. built as (j, i) and transposed.
    dead_u = _dead_mask(NUNITS, NCHAN, zero_frac, g).to(dev)
    dead_c = _dead_mask(NUNITS, NUNITS, zero_frac, g).T.contiguous().to(dev)
    fill = -0.0 if neg_zero else 0.0
    U_time = torch.where(dead_u[:, :, None], torch.full_like(U_time, fill),
                         U_time)
    ctc = torch.where(dead_c[:, :, None], torch.full_like(ctc, fill), ctc)
    step = (NT_LEN - 2 * NT - 2) // max(n_spk, 1)
    pos = torch.arange(n_spk) * step + NT + 1
    iX = pos.to(dev).unsqueeze(1)
    iY = torch.randint(0, NUNITS, (n_spk, 1), generator=g).to(dev)
    amp = torch.rand(n_spk, 1, generator=g).to(dev) + 0.1
    if not positive_amp:
        amp[0] = -0.0            # not caught by `amp >= 0`
        if n_spk > 1:
            amp[1] = -amp[1]
    return (Xres, B, U_time.to(dev), ctc.to(dev), iX, iY, amp.to(dev),
            torch.arange(-(NT // 2), NT // 2 + 1, device=dev),
            torch.arange(-NT, NT + 1, device=dev))


def latch_fused():
    """Run one peel so the FUSED gate decides, without deciding the LUT gate.

    peel_subtract validates fused-vs-stock on its first eligible phase and
    returns immediately, so the LUT gate is not reached until the second call.
    That is correct for a real sort (thousands of peels) but means a
    single-call test never exercises the LUT at all.
    """
    Xres, B, U_time, ctc, iX, iY, amp, tiwave, trange = sparse_problem(4, 999)
    fused_peel.peel_subtract(Xres, B, iX, iY, amp, U_time, ctc,
                             ctc.permute(1, 0, 2), tiwave, trange, NT)
    assert fused_peel._CHOICE is True, 'fused peel not exact on this device'
    assert fused_peel._LUT_CHOICE is None


@pytest.mark.parametrize('n_spk', [1, 2, 5, 16])
def test_lut_matches_stock_bit_for_bit(n_spk):
    latch_fused()
    Xres, B, U_time, ctc, iX, iY, amp, tiwave, trange = sparse_problem(
        n_spk, 100 + n_spk)
    Xs, Bs = Xres.clone(), B.clone()
    stock_all(Xs, Bs, iX, iY, amp, U_time, ctc, tiwave, trange)
    ctc_p = ctc.permute(1, 0, 2)
    fused_peel.peel_subtract(Xres, B, iX, iY, amp, U_time, ctc, ctc_p,
                             tiwave, trange, NT)
    assert fused_peel._LUT_CHOICE is True
    assert bit_eq(Xres, Xs) and bit_eq(B, Bs)


def test_lut_is_actually_used_and_shrinks_the_grid():
    """Guard against the LUT silently degenerating to the full kernel."""
    _, _, U_time, ctc, _, _, _, _, _ = sparse_problem(4, 7)
    lut, n_live, maxt = fused_peel._get_tile_lut(ctc.permute(1, 0, 2))
    n_tiles = (NUNITS + fused_peel.BLOCK_R - 1) // fused_peel.BLOCK_R
    assert 0 < maxt <= n_tiles
    assert int(n_live.float().median()) < n_tiles
    assert lut.shape == (NUNITS, maxt)


def test_lut_skips_only_POSITIVE_zero_tiles():
    latch_fused()
    """Dead tiles made of -0.0 must NOT be skipped.

    Stock computes o - (amp * -0.0) = o + 0.0, which turns a stored -0.0 into
    +0.0. Skipping leaves the -0.0 in place. A `src == 0.0` liveness test
    would wrongly call these tiles dead; only a bit test gets this right.
    """
    Xres, B, U_time, ctc, iX, iY, amp, tiwave, trange = sparse_problem(
        8, 21, neg_zero=True)
    B = torch.where(torch.rand_like(B) < 0.5, torch.full_like(B, -0.0), B)
    Xres = torch.where(torch.rand_like(Xres) < 0.5,
                       torch.full_like(Xres, -0.0), Xres)
    Xs, Bs = Xres.clone(), B.clone()
    stock_all(Xs, Bs, iX, iY, amp, U_time, ctc, tiwave, trange)
    fused_peel.peel_subtract(Xres, B, iX, iY, amp, U_time, ctc,
                             ctc.permute(1, 0, 2), tiwave, trange, NT)
    assert bit_eq(Xres, Xs) and bit_eq(B, Bs)


def test_lut_falls_back_when_any_amp_is_not_positive():
    latch_fused()
    """amp = -0.0 gives -0.0 * +0.0 = -0.0, flipping a stored -0.0 to +0.0.

    `amp >= 0` is true for -0.0, so the guard has to be strict. The phase must
    take the full path, and the result must still equal stock.
    """
    Xres, B, U_time, ctc, iX, iY, amp, tiwave, trange = sparse_problem(
        8, 31, positive_amp=False)
    B = torch.where(torch.rand_like(B) < 0.5, torch.full_like(B, -0.0), B)
    Xs, Bs = Xres.clone(), B.clone()
    stock_all(Xs, Bs, iX, iY, amp, U_time, ctc, tiwave, trange)
    fused_peel.peel_subtract(Xres, B, iX, iY, amp, U_time, ctc,
                             ctc.permute(1, 0, 2), tiwave, trange, NT)
    assert bit_eq(Xres, Xs) and bit_eq(B, Bs)
    assert fused_peel._LUT_CHOICE is None      # never got an eligible phase


def test_lut_env_switch_forces_the_full_fused_path():
    import os
    latch_fused()
    os.environ['KILOSORT_NO_PEEL_LUT'] = '1'
    try:
        Xres, B, U_time, ctc, iX, iY, amp, tiwave, trange = sparse_problem(
            8, 41)
        Xs, Bs = Xres.clone(), B.clone()
        stock_all(Xs, Bs, iX, iY, amp, U_time, ctc, tiwave, trange)
        fused_peel.peel_subtract(Xres, B, iX, iY, amp, U_time, ctc,
                                 ctc.permute(1, 0, 2), tiwave, trange, NT)
        assert fused_peel._LUT_CHOICE is False
        assert bit_eq(Xres, Xs) and bit_eq(B, Bs)
    finally:
        del os.environ['KILOSORT_NO_PEEL_LUT']


def test_lut_cache_survives_a_fresh_permuted_view_each_call():
    """ctc_p is rebuilt by run_matching on every batch; the cache must key on
    the batch-invariant base, or it misses ~3300 times per production sort."""
    _, _, _, ctc, _, _, _, _, _ = sparse_problem(2, 51)
    first = fused_peel._get_tile_lut(ctc.permute(1, 0, 2))
    second = fused_peel._get_tile_lut(ctc.permute(1, 0, 2))
    assert first[0].data_ptr() == second[0].data_ptr()


def test_lut_rebuilds_when_the_source_tensor_changes():
    _, _, _, ctc, _, _, _, _, _ = sparse_problem(2, 61)
    first = fused_peel._get_tile_lut(ctc.permute(1, 0, 2))
    ctc[0, 0, :] = 1.0                       # bumps ._version
    second = fused_peel._get_tile_lut(ctc.permute(1, 0, 2))
    assert first[0].data_ptr() != second[0].data_ptr()


def test_lut_key_separates_two_views_with_the_same_shape_and_stride():
    """id(base) plus shape plus stride is not a key. Two slices of one tensor
    can agree on all three and still cover different rows, so the cache handed
    back a table for the wrong tiles -- not a stale answer, a wrong one."""
    _, _, _, ctc, _, _, _, _, _ = sparse_problem(2, 81)
    src = ctc.permute(1, 0, 2)
    half = src.shape[0] // 2
    assert half >= 1
    lo, hi = src[:half], src[half:2 * half]
    assert lo.shape == hi.shape and lo.stride() == hi.stride()
    assert lo.storage_offset() != hi.storage_offset()

    # Make the two halves genuinely different, so a wrong table is visible.
    ctc[:, :half] = 0.0
    ctc[:, half:2 * half] = 1.0

    lut_lo = fused_peel._get_tile_lut(src[:half])
    lut_hi = fused_peel._get_tile_lut(src[half:2 * half])
    # the all-zero half has no live tile; the all-ones half has one per row
    assert int(lut_lo[1].max()) == 0
    assert int(lut_hi[1].min()) > 0


def test_lut_key_separates_two_tile_sizes():
    """A table built for one block_r describes a different tile set than a
    table built for another, so block_r has to be part of the key."""
    _, _, _, ctc, _, _, _, _, _ = sparse_problem(2, 82)
    src = ctc.permute(1, 0, 2)
    coarse = fused_peel._get_tile_lut(src, block_r=src.shape[1])
    fine = fused_peel._get_tile_lut(src, block_r=1)
    assert coarse[2] != fine[2], 'MAXT must differ between tile sizes'


def test_lut_cache_does_not_outlive_its_source():
    """The entry holds the table STRONGLY and only a weakref to the base, so
    without an eviction hook every table ever built stays for the life of the
    process. On CUDA that is retained device memory."""
    fused_peel._LUT_CACHE.clear()
    for seed in range(8):
        _, _, _, ctc, _, _, _, _, _ = sparse_problem(2, 90 + seed)
        fused_peel._get_tile_lut(ctc.permute(1, 0, 2))
        del ctc
    import gc
    gc.collect()
    assert len(fused_peel._LUT_CACHE) == 0, (
        '{0} tables outlived their source tensors'.format(
            len(fused_peel._LUT_CACHE)))


def test_lut_handles_an_all_zero_source():
    latch_fused()
    Xres, B, U_time, ctc, iX, iY, amp, tiwave, trange = sparse_problem(4, 71)
    ctc = torch.zeros_like(ctc)
    Xs, Bs = Xres.clone(), B.clone()
    stock_all(Xs, Bs, iX, iY, amp, U_time, ctc, tiwave, trange)
    fused_peel.peel_subtract(Xres, B, iX, iY, amp, U_time, ctc,
                             ctc.permute(1, 0, 2), tiwave, trange, NT)
    assert bit_eq(Xres, Xs) and bit_eq(B, Bs)
