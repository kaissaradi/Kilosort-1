"""The fused condition tail must equal the stock statements, bit for bit.

fused_peel_cond replaces relu -> square -> edge zero -> max_pool1d -> two
comparisons -> and with one kernel. Unlike fused_peaks this tail is NOT
arithmetic-free -- `cnd2` does a real fp32 subtract -- so the pins here are:

  * `cmax` is compared on RAW BIT PATTERNS, not with torch.equal. It feeds
    `th_amps = cmax[iX]**.5`, so a signed-zero or last-bit difference would
    reach an output file.
  * max_pool1d pads with -inf, so a window clipped by the array end reduces
    over its in-range part only. Getting that wrong shows up only at the two
    edges, which is where the edge-zeroing also lives.
  * relu-then-square must stay in that order: relu(x)**2 != x**2 for x < 0.

Skipped without CUDA or Triton, which is the fallback case anyway.
"""
import pytest
import torch
from torch.nn.functional import max_pool1d

from kilosort import fused_peel_cond

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_peel_cond._HAVE_TRITON,
    reason='fused peel cond needs CUDA and Triton')

NT_LEN, NT = 10122, 61
dev = 'cuda'


def bit_eq(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


@pytest.fixture(autouse=True)
def reset_choice():
    saved = fused_peel_cond._CHOICE
    fused_peel_cond._CHOICE = None
    yield
    fused_peel_cond._CHOICE = saved


def raw_signal(seed, n=NT_LEN, scale=10.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, generator=g) * scale).to(dev)


def stock(raw, nt, Th2):
    c = torch.relu(raw)
    c = c * c
    c[:nt] = 0
    c[-nt:] = 0
    cmax = max_pool1d(c.view(1, 1, -1), (2 * nt + 1), stride=1,
                      padding=nt)[0, 0]
    return cmax, (cmax > Th2) & (torch.abs(cmax - c) < 1e-9)


@pytest.mark.parametrize('seed,Th2', [(1, 64.0), (2, 9.0), (3, 400.0)])
def test_matches_stock_bit_for_bit(seed, Th2):
    raw = raw_signal(seed)
    ref_cmax, ref_mask = stock(raw.clone(), NT, Th2)
    cmax, mask = fused_peel_cond.peak_condition(raw, NT, Th2)
    assert fused_peel_cond._CHOICE is not False
    assert bit_eq(cmax, ref_cmax)
    assert torch.equal(mask, ref_mask)


def test_still_matches_after_the_gate_has_decided():
    """The first call validates and latches; later calls run unchecked."""
    fused_peel_cond.peak_condition(raw_signal(11), NT, 64.0)
    if fused_peel_cond._CHOICE is False:
        pytest.skip('not bit-identical on this device')
    raw = raw_signal(12)
    ref_cmax, ref_mask = stock(raw.clone(), NT, 64.0)
    cmax, mask = fused_peel_cond.peak_condition(raw, NT, 64.0)
    assert bit_eq(cmax, ref_cmax) and torch.equal(mask, ref_mask)


def test_does_not_modify_raw():
    """Stock squares Cfmax in place; the fused path folds the transform into
    the kernel instead. If it ever wrote back, the caller's `imax`-aligned
    buffer would be silently corrupted."""
    raw = raw_signal(13)
    before = raw.clone()
    fused_peel_cond.peak_condition(raw, NT, 64.0)
    assert bit_eq(raw, before)


def test_negative_input_is_relu_then_squared_not_squared():
    """relu(x)**2 == 0 for x < 0, but x**2 > 0. An all-negative input must
    therefore produce an all-zero cmax and an empty mask."""
    raw = -raw_signal(14).abs() - 1.0
    ref_cmax, ref_mask = stock(raw.clone(), NT, 1e-30)
    cmax, mask = fused_peel_cond.peak_condition(raw, NT, 1e-30)
    assert bit_eq(cmax, ref_cmax) and torch.equal(mask, ref_mask)
    assert not mask.any()


def test_edges_and_pooling_boundary():
    """The two nt-wide zeroed bands and the -inf pool padding both live at the
    array ends, so a spike planted just inside each edge is the case that
    separates a correct boundary from a plausible one."""
    raw = torch.zeros(NT_LEN, device=dev)
    for i in (0, NT - 1, NT, NT + 1, NT_LEN // 2,
              NT_LEN - NT - 2, NT_LEN - NT, NT_LEN - 1):
        raw[i] = 50.0
    ref_cmax, ref_mask = stock(raw.clone(), NT, 64.0)
    cmax, mask = fused_peel_cond.peak_condition(raw, NT, 64.0)
    assert bit_eq(cmax, ref_cmax) and torch.equal(mask, ref_mask)


def test_exact_ties_are_handled_like_stock():
    """cnd2 admits exact ties (that is how the 1-in-29k overlapping peel phase
    arises), so a plateau must select the same positions stock does."""
    raw = torch.zeros(NT_LEN, device=dev)
    raw[1000:1010] = 30.0            # flat plateau -> several exact ties
    raw[5000:5003] = 30.0
    ref_cmax, ref_mask = stock(raw.clone(), NT, 64.0)
    cmax, mask = fused_peel_cond.peak_condition(raw, NT, 64.0)
    assert bit_eq(cmax, ref_cmax) and torch.equal(mask, ref_mask)
    assert torch.equal(torch.nonzero(mask), torch.nonzero(ref_mask))


def test_signed_zero_input():
    """relu may map -0.0 to either zero, but the square makes both +0.0. If a
    future edit drops the square or reorders it, this is what catches it."""
    raw = torch.full((NT_LEN,), -0.0, device=dev)
    raw[500] = 40.0
    ref_cmax, ref_mask = stock(raw.clone(), NT, 64.0)
    cmax, mask = fused_peel_cond.peak_condition(raw, NT, 64.0)
    assert bit_eq(cmax, ref_cmax) and torch.equal(mask, ref_mask)


def test_env_switch_forces_stock():
    import os
    raw = raw_signal(15)
    ref_cmax, ref_mask = stock(raw.clone(), NT, 64.0)
    os.environ['KILOSORT_NO_PEEL_COND'] = '1'
    try:
        cmax, mask = fused_peel_cond.peak_condition(raw, NT, 64.0)
    finally:
        del os.environ['KILOSORT_NO_PEEL_COND']
    assert fused_peel_cond._CHOICE is False
    assert bit_eq(cmax, ref_cmax) and torch.equal(mask, ref_mask)


def test_short_input_falls_back():
    """The kernel folds the edge zeroing into the window transform, which
    assumes the two nt-wide bands do not overlap."""
    raw = raw_signal(16, n=2 * NT - 4)
    ref_cmax, ref_mask = stock(raw.clone(), NT, 64.0)
    cmax, mask = fused_peel_cond.peak_condition(raw, NT, 64.0)
    assert bit_eq(cmax, ref_cmax) and torch.equal(mask, ref_mask)
