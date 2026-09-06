"""The fused spike store must equal the stock statements, bit for bit.

The interesting failure here is not a race or an index bug -- it is that
`tl.sqrt` LOOKS exact and is not. It lowers to the approximate hardware
instruction and differs from torch's `x**.5` by 1 ULP on a few values in
every real peel. `tl.math.sqrt_rn` is the correctly-rounded one.
`test_sqrt_is_correctly_rounded` fails if anyone swaps it back.

The other pin is ORDER: st/amps/th_amps rows are written in `nonzero` order
and indexed positionally downstream, so a compaction or atomic-counter
implementation would silently permute them while still passing a set-equality
check. One program per spike is what makes that impossible.
"""
import pytest
import torch

from kilosort import fused_peel_store

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_peel_store._HAVE_TRITON,
    reason='fused peel store needs CUDA and Triton')

NT, NU, CAP = 10122, 1031, 4096
dev = 'cuda'


def bit_eq(a, b):
    return torch.equal(a.reshape(-1).view(torch.int32),
                       b.reshape(-1).view(torch.int32))


@pytest.fixture(autouse=True)
def reset_choice():
    saved = fused_peel_store._CHOICE
    fused_peel_store._CHOICE = None
    yield
    fused_peel_store._CHOICE = saved


def problem(nsp, seed, k=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    B = torch.randn(NU, NT, generator=g, device=dev)
    imax = torch.randint(0, NU, (NT,), generator=g, device=dev)
    s = torch.rand(NU, generator=g, device=dev) + 0.1
    cmax = torch.rand(NT, generator=g, device=dev) * 100
    st = torch.zeros((CAP, 2), dtype=torch.int64, device=dev)
    amps = torch.zeros((CAP, 1), device=dev)
    th = torch.zeros((CAP, 1), device=dev)
    iX = (torch.sort(torch.randperm(NT - 200, generator=g,
                                    device=dev)[:nsp]).values + 100).unsqueeze(1)
    return st, amps, th, k, iX, imax, B, s, cmax


def run_both(args):
    st, amps, th, k, iX, imax, B, s, cmax = args
    nsp = iX.shape[0]
    ref_st = torch.zeros_like(st)
    ref_amps = torch.zeros_like(amps)
    ref_th = torch.zeros_like(th)
    ref_iY = fused_peel_store._stock(ref_st, ref_amps, ref_th, k, iX, imax,
                                     B, s, cmax)
    iY = fused_peel_store.store_spikes(st, amps, th, k, iX, imax, B, s, cmax)
    return (iY, ref_iY, st[k:k + nsp], ref_st[k:k + nsp],
            amps[k:k + nsp], ref_amps[k:k + nsp],
            th[k:k + nsp], ref_th[k:k + nsp])


@pytest.mark.parametrize('nsp,k', [(1, 0), (26, 0), (82, 7), (200, 1000)])
def test_matches_stock_bit_for_bit(nsp, k):
    iY, r_iY, st, r_st, amps, r_amps, th, r_th = run_both(problem(nsp, nsp, k))
    assert fused_peel_store._CHOICE is True
    assert torch.equal(iY, r_iY)
    assert torch.equal(st, r_st)
    assert bit_eq(amps, r_amps)
    assert bit_eq(th, r_th)


def test_sqrt_is_correctly_rounded():
    """tl.sqrt is the approximate instruction and is NOT bit-identical to
    torch's `x**.5`; only tl.math.sqrt_rn is. This is the guard on that.

    Asserted over enough values that a 1-ULP-per-few-dozen error cannot slip
    through: the original bug showed 7 differences in 26 spikes.
    """
    args = problem(2000, 99)
    _, _, _, _, _, _, th, r_th = run_both(args)
    assert bit_eq(th, r_th), 'th_amps differs -- tl.sqrt instead of sqrt_rn?'


def test_row_order_follows_nonzero_not_sorted_order():
    """st rows must land in the order iX was given, positionally."""
    st, amps, th, k, iX, imax, B, s, cmax = problem(32, 5)
    # deliberately un-sorted positions, as nonzero would never produce
    iX = iX[torch.randperm(iX.shape[0], device=dev)]
    fused_peel_store.store_spikes(st, amps, th, k, iX, imax, B, s, cmax)
    assert torch.equal(st[k:k + iX.shape[0], 0], iX[:, 0])


def test_writes_only_the_k_slice():
    """A stray write outside [k, k+nsp) would corrupt earlier peels' spikes."""
    nsp, k = 26, 500
    st, amps, th, _, iX, imax, B, s, cmax = problem(nsp, 6)
    st.fill_(-7); amps.fill_(-7.0); th.fill_(-7.0)
    fused_peel_store.store_spikes(st, amps, th, k, iX, imax, B, s, cmax)
    assert (st[:k] == -7).all() and (st[k + nsp:] == -7).all()
    assert (amps[:k] == -7.0).all() and (amps[k + nsp:] == -7.0).all()
    assert (th[:k] == -7.0).all() and (th[k + nsp:] == -7.0).all()


def test_env_switch_forces_stock():
    import os
    args = problem(26, 8)
    os.environ['KILOSORT_NO_PEEL_STORE'] = '1'
    try:
        iY, r_iY, st, r_st, amps, r_amps, th, r_th = run_both(args)
    finally:
        del os.environ['KILOSORT_NO_PEEL_STORE']
    assert fused_peel_store._CHOICE is False
    assert torch.equal(st, r_st) and bit_eq(amps, r_amps) and bit_eq(th, r_th)


def test_gate_rejects_a_wrong_kernel_and_falls_back(monkeypatch):
    """If the kernel ever stops matching, the caller must still get stock
    values -- not a silently wrong output."""
    args = problem(26, 9)
    st, amps, th, k, iX, imax, B, s, cmax = args

    def broken(*a, **kw):
        _st, _amps, _th = a[0], a[1], a[2]
        _th.fill_(123.0)                      # obviously wrong th_amps

    monkeypatch.setattr(fused_peel_store, '_run', broken)
    ref_st = torch.zeros_like(st); ref_amps = torch.zeros_like(amps)
    ref_th = torch.zeros_like(th)
    fused_peel_store._stock(ref_st, ref_amps, ref_th, k, iX, imax, B, s, cmax)
    fused_peel_store.store_spikes(st, amps, th, k, iX, imax, B, s, cmax)
    assert fused_peel_store._CHOICE is False
    assert bit_eq(th[k:k + 26], ref_th[k:k + 26])
