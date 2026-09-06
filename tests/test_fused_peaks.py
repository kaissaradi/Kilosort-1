"""The fused peak-selection tail must never be used unless it is identical.

fused_peaks replaces the tail of spikedetect.template_match -- edge zeroing,
max_pool1d, the two comparisons and the and -- with one Triton kernel that is
7.2x faster at production shapes.

Unlike fused_detect, this kernel does no floating-point arithmetic: `max` is a
selection (order cannot change the result), the maxima's *indices* are never
used so ties are harmless, and `==`, `>` and `&` are exact. So identity is a
much stronger property here than it is there, and these tests assert it
directly rather than only pinning a gate.

What is genuinely delicate, and what these tests are really for:

  * **The short-circuit.** The kernel skips the whole window max for any block
    where no lane has `As > Th`. That is only sound because `(m == a) & (a >
    Th)` is False wherever `a > Th` is False. `test_short_circuit_boundary`
    builds a peak sitting exactly on a block boundary, so the window that
    proves it a local maximum straddles two blocks and one of those blocks is
    a skip candidate.
  * **Edge semantics.** max_pool1d pads with -inf, and the stock code zeroes
    the first and last nt columns *before* pooling. Those two interact within
    nt0 of the ends. `test_edges` puts the only peaks there.
  * **Ties.** Equal maxima inside one window are the case where an
    argmax-based implementation would diverge. `test_ties` makes them dense.

Skipped without CUDA or without Triton, which is the fallback case anyway.
"""
import pytest
import torch

from kilosort import fused_peaks

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_peaks._HAVE_TRITON,
    reason='fused peaks needs CUDA and Triton')

NFILT, NT, NT_, NT0 = 96, 1024, 61, 20


def stock(As, Amaxs, nt, nt0, Th):
    """The statements this kernel replaces, verbatim, on a copy."""
    from torch.nn.functional import max_pool1d
    Am = Amaxs.clone()
    Am[:, :nt] = 0
    Am[:, -nt:] = 0
    Am = max_pool1d(Am.unsqueeze(0), (2 * nt0 + 1), stride=1,
                    padding=nt0).squeeze(0)
    return torch.logical_and(Am == As, As > Th)


def check(As, Amaxs, Th, nt=NT_, nt0=NT0, cfgs=None):
    ref = stock(As, Amaxs, nt, nt0, Th)
    for cfg in (cfgs or fused_peaks._CONFIGS):
        out = torch.empty(As.shape, dtype=torch.bool, device=As.device)
        fused_peaks._run(As, Amaxs, out, nt, nt0, Th, cfg)
        assert torch.equal(out, ref), (
            f'cfg {cfg}: {(out != ref).sum().item()} of {ref.numel()} differ')
        # the caller uses nonzero(), and its ORDER matters downstream
        assert torch.equal(out.nonzero(), ref.nonzero()), f'cfg {cfg}: order'
    return ref


def rand(seed, shape=(NFILT, NT)):
    g = torch.Generator(device='cpu').manual_seed(seed)
    return torch.randn(*shape, generator=g).cuda()


def test_random_dense():
    """Th low enough that a large fraction survives -- exercises the non-skip
    path on essentially every block."""
    As = rand(0).abs()
    check(As, rand(1).abs(), Th=0.1)


def test_random_sparse():
    """Th high enough that almost every block short-circuits, which is the
    production regime (~1.4k survivors of 41 M)."""
    As = rand(2).abs()
    check(As, rand(3).abs(), Th=3.5)


def test_no_candidates_at_all():
    """Every block skips. Result must be all-False, not stale memory."""
    As = rand(4).abs()
    ref = check(As, rand(5).abs(), Th=1e9)
    assert not ref.any()


def test_real_peaks_are_found():
    """A constructed local maximum must actually survive both conditions, so
    the test cannot pass by trivially returning all-False."""
    As = torch.zeros(NFILT, NT, device='cuda')
    Amaxs = torch.zeros(NFILT, NT, device='cuda')
    cols = [200, 300, 512, 513, 700]
    for r, c in enumerate(cols):
        As[r % NFILT, c] = 5.0
        Amaxs[r % NFILT, c] = 5.0
    ref = check(As, Amaxs, Th=1.0)
    assert ref.sum().item() == len(cols), ref.sum().item()


def test_short_circuit_boundary():
    """A peak whose +/-nt0 window straddles a block boundary, for every block
    size. The neighbouring block has no candidate of its own and is a skip
    candidate; the peak's own block must still see the whole window."""
    for cfg in fused_peaks._CONFIGS:
        BLOCK_C = cfg[0]
        As = torch.zeros(NFILT, NT, device='cuda')
        Amaxs = torch.zeros(NFILT, NT, device='cuda')
        for d in (-2, -1, 0, 1, 2):
            c = BLOCK_C + d
            As[0, c] = 5.0
            Amaxs[0, c] = 5.0
            # a taller neighbour just across the boundary must SUPPRESS it
            Amaxs[1, c] = 5.0
            As[1, c] = 5.0
            Amaxs[1, c + 3] = 9.0
        check(As, Amaxs, Th=1.0, cfgs=[cfg])


def test_edges():
    """Peaks inside the zeroed margins and within nt0 of the array ends, where
    the -inf padding and the explicit zeroing interact."""
    As = torch.zeros(NFILT, NT, device='cuda')
    Amaxs = torch.zeros(NFILT, NT, device='cuda')
    for c in (0, 1, NT_ - 1, NT_, NT_ + 1, NT - NT_ - 1, NT - NT_, NT - 2, NT - 1):
        As[0, c] = 5.0
        Amaxs[0, c] = 5.0
    check(As, Amaxs, Th=1.0)


def test_ties():
    """Dense exact ties, including whole constant rows: equal maxima in one
    window are where an argmax-based rewrite would diverge from a value
    comparison."""
    As = torch.full((NFILT, NT), 2.0, device='cuda')
    Amaxs = torch.full((NFILT, NT), 2.0, device='cuda')
    Amaxs[::2] = 2.0
    As[1::2, ::7] = 2.0
    check(As, Amaxs, Th=1.0)


def test_negative_and_zero():
    """As <= 0 must never pass `> Th` for Th >= 0, and negative Amaxs must not
    be confused with the zeroed margins."""
    As = -rand(6).abs()
    Amaxs = -rand(7).abs()
    ref = check(As, Amaxs, Th=0.0)
    assert not ref.any()
    check(rand(8), rand(9), Th=-1.0)


def test_non_square_and_odd_widths():
    """NT not a multiple of any block size, and a single row."""
    for shape in ((7, 1000), (1, 333), (33, 129), (2, 41)):
        check(rand(10, shape).abs(), rand(11, shape).abs(), Th=0.5,
              nt=5, nt0=3)


def test_gate_rejects_ineligible_inputs():
    """try_mask must return None (caller runs stock) for anything the kernel
    does not handle, and must not leave _CHOICE poisoned for real inputs."""
    saved = fused_peaks._CHOICE
    try:
        fused_peaks._CHOICE = None
        cpu = torch.randn(8, 64)
        assert fused_peaks.try_mask(cpu, cpu.clone(), NT_, NT0, 1.0) is None

        fused_peaks._CHOICE = None
        half = rand(12, (8, 64)).half()
        assert fused_peaks.try_mask(half, half.clone(), NT_, NT0, 1.0) is None

        fused_peaks._CHOICE = None
        a = rand(13, (8, 64))
        assert fused_peaks.try_mask(a, a[:, :32].contiguous(), NT_, NT0, 1.0) is None
    finally:
        fused_peaks._CHOICE = saved


def test_try_mask_matches_stock_and_persists():
    """The gate's own output, on the batch it validates against AND on a later
    batch it does not -- the failure mode a smoke test would miss."""
    saved = fused_peaks._CHOICE
    try:
        fused_peaks._CHOICE = None
        for seed in (20, 21, 22):
            As, Amaxs = rand(seed).abs(), rand(seed + 100).abs()
            got = fused_peaks.try_mask(As, Amaxs, NT_, NT0, 0.8)
            assert got is not None
            assert torch.equal(got, stock(As, Amaxs, NT_, NT0, 0.8)), seed
    finally:
        fused_peaks._CHOICE = saved


def test_env_switch_disables():
    """KILOSORT_NO_FUSED_PEAKS=1 must fall back, not silently stay fused."""
    import os
    saved, prev = fused_peaks._CHOICE, os.environ.get('KILOSORT_NO_FUSED_PEAKS')
    try:
        fused_peaks._CHOICE = None
        os.environ['KILOSORT_NO_FUSED_PEAKS'] = '1'
        a = rand(30, (8, 64)).abs()
        assert fused_peaks.try_mask(a, a.clone(), NT_, NT0, 0.5) is None
        assert fused_peaks._CHOICE is False
    finally:
        fused_peaks._CHOICE = saved
        if prev is None:
            os.environ.pop('KILOSORT_NO_FUSED_PEAKS', None)
        else:
            os.environ['KILOSORT_NO_FUSED_PEAKS'] = prev
