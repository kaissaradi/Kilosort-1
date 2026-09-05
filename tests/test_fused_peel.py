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
    saved = fused_peel._CHOICE
    fused_peel._CHOICE = None
    yield
    fused_peel._CHOICE = saved


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
