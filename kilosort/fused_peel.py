"""Fused scatter-subtract for template_matching.run_matching's peel loop.

The learned-template pass is 506.6 s of a 1602 s production sort (31.6%), and
65.9% of it is the two subtract lines at the bottom of each peel:

    waves = U_time[iY[j::2, 0]].permute(1, 0, 2)
    Xres[:, iX[j::2] + tiwave] -= amp[j::2] * waves
    B[:, iX[j::2] + trange]    -= amp[j::2] * ctc[:, iY[j::2, 0], :]

Each line builds an int64 index tensor, gathers, broadcasts a multiply,
subtracts and scatters -- five materialised (row, n_sel, window) tensors and
about six kernel launches, to touch windows only 61 and 123 columns wide with
a median of ~26 spikes. It runs ~29k times per 300-batch sort, so it is
launch- and allocation-bound, not bandwidth-bound.

Two facts make a fused kernel exactly equal to that, and both were measured:

1. DISJOINTNESS. `-=` through advanced indexing is gather / subtract /
   scatter, so duplicate indices DROP contributions instead of accumulating
   them -- which is what the stock `n = 2` comment means by the stride being
   "load-bearing for identity". Fusing reproduces stock exactly when the
   windows within a phase are disjoint, because then every element is touched
   once and order cannot matter.

   A census over 29,114 real phases (776,894 spikes) of 20260724A found the
   Xres windows ALWAYS disjoint, and the B windows overlapping in exactly ONE
   phase. Rare, but not never: detections in a peel are (2nt+1) max-pool
   maxima so they normally sit > nt apart, yet the `abs(cmax - Cfmax) < 1e-9`
   test admits exact ties, and a tie can put two of them adjacent. So the
   fused path is taken only for phases that are checked disjoint, and that
   one-in-29k phase falls back to stock. Skipping the check would be
   bit-identical 29,113 times out of 29,114 and silently wrong once.

2. NO FMA CONTRACTION. Stock rounds twice, once for `amp * src` and once for
   the subtract. Written plainly, a kernel contracts those into one FFMA and
   differs from stock on 1,092 of 4.19M elements. `enable_fp_fusion=False` at
   the launch keeps them separate and is exact, including on signed zeros --
   note the hand-rolled barrier `v = a*src; v = v + 0.0` fixes the bulk but
   maps -0.0 to +0.0 and then differs on every signed zero. (torch.equal and
   np.array_equal will NOT show you that: both report +0.0 == -0.0. Compare
   raw bit patterns.)

Measured at production shapes (Nchan=519, n_units=600, nt=61, 26 spikes):

    stock  178.1 us/phase
    fused   27.2 us/phase          6.56x
    guard   17.0 us/peel           (at a sync the peel loop already performs)
    net     178.1 -> 35.6 us       5.00x

As with fused_detect, none of this is trusted at runtime: the first phase of
every sort is computed both ways and compared bit-for-bit, and the fused path
is used only if they agree. Set KILOSORT_NO_FUSED_PEEL=1 to skip it.
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception as _e:            # pragma: no cover - depends on install
    _HAVE_TRITON = False
    _TRITON_ERR = _e

# None = not yet checked this process, True = verified, False = disabled.
_CHOICE = None
BLOCK_R = 16


if _HAVE_TRITON:

    @triton.jit
    def _scatter_sub_kernel(
            OUT_ptr, SRC_ptr, POS_ptr, YI_ptr, AMP_ptr,
            n_row, n_t, off0, NT,
            sO0, sO1,
            sS_y, sS_r, sS_t,
            sP, sY, sA,
            BLOCK_R: tl.constexpr, BLOCK_T: tl.constexpr):
        """OUT[r, POS[s] + off0 + t] -= AMP[s] * SRC[YI[s], r, t]

        One program per (spike, row tile). SRC's three strides are passed
        separately so one kernel serves both stock lines: U_time is indexed
        (spike, row, t) while ctc is (row, spike, t), which is only a swap of
        sS_y and sS_r -- the caller passes a permuted view, never a copy.
        POS/YI/AMP are strided because the caller slices them [j::2].
        """
        s = tl.program_id(0)
        offs_r = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_t = tl.arange(0, BLOCK_T)
        mask = (offs_r[:, None] < n_row) & (offs_t[None, :] < n_t)

        p = tl.load(POS_ptr + s * sP)
        y = tl.load(YI_ptr + s * sY)
        a = tl.load(AMP_ptr + s * sA)

        src = tl.load(SRC_ptr + y * sS_y + offs_r[:, None] * sS_r
                      + offs_t[None, :] * sS_t, mask=mask, other=0.0)

        col = p + off0 + offs_t
        cmask = mask & ((col[None, :] >= 0) & (col[None, :] < NT))
        out_ptr = OUT_ptr + offs_r[:, None] * sO0 + col[None, :] * sO1

        # Two rounded fp32 steps, as stock. Kept that way by
        # enable_fp_fusion=False at the launch -- do not fold these together.
        v = a * src
        o = tl.load(out_ptr, mask=cmask, other=0.0)
        tl.store(out_ptr, o - v, mask=cmask)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _scatter_sub(out, src, pos, yi, amp, off0):
    """out[r, pos[s]+off0+t] -= amp[s] * src[yi[s], r, t]."""
    n_spk = pos.shape[0]
    if n_spk == 0:
        return
    n_row, n_t = src.shape[1], src.shape[2]
    grid = (n_spk, triton.cdiv(n_row, BLOCK_R))
    _scatter_sub_kernel[grid](
        out, src, pos, yi, amp,
        n_row, n_t, off0, out.shape[1],
        out.stride(0), out.stride(1),
        src.stride(0), src.stride(1), src.stride(2),
        pos.stride(0), yi.stride(0), amp.stride(0),
        BLOCK_R=BLOCK_R, BLOCK_T=_next_pow2(n_t),
        num_warps=4, num_stages=1,
        # Load-bearing: without it the mul and sub contract into one FFMA and
        # the kernel stops matching stock.
        enable_fp_fusion=False)


def phases_disjoint(pos_all, nt):
    """Are BOTH n=2 phases' (2nt+1)-wide windows non-overlapping?

    Consecutive members of pos_all[j::2] are pos[i+2] - pos[i] = d[i] + d[i+1],
    so one reduction over the pairwise sums answers it for both phases at
    once. Returns a 0-dim device bool; the caller decides when to sync.
    """
    if pos_all.numel() < 3:
        return torch.ones((), dtype=torch.bool, device=pos_all.device)
    d = torch.diff(pos_all)
    return (d[:-1] + d[1:]).min() > 2 * nt


def _stock_phase(Xres, B, iX, iY, amp, U_time, ctc, tiwave, trange, j, n):
    waves = U_time[iY[j::n, 0]].permute(1, 0, 2)
    Xres[:, iX[j::n] + tiwave] -= amp[j::n] * waves
    B[:, iX[j::n] + trange] -= amp[j::n] * ctc[:, iY[j::n, 0], :]


def _fused_phase(Xres, B, iX, iY, amp, U_time, ctc_p, nt, j, n):
    pos = iX[j::n, 0]
    yi = iY[j::n, 0]
    a = amp[j::n, 0]
    _scatter_sub(Xres, U_time, pos, yi, a, -(nt // 2))
    _scatter_sub(B, ctc_p, pos, yi, a, -nt)


def _eligible(Xres, B, U_time, ctc, iX):
    if not _HAVE_TRITON or os.environ.get('KILOSORT_NO_FUSED_PEEL'):
        return False
    if not (Xres.is_cuda and Xres.dtype == torch.float32):
        return False
    if B.dtype != torch.float32 or U_time.dtype != torch.float32:
        return False
    if ctc.dtype != torch.float32 or iX.dtype != torch.int64:
        return False
    return True


def peel_subtract(Xres, B, iX, iY, amp, U_time, ctc, ctc_p, tiwave, trange,
                  nt, n=2):
    """Apply one peel's subtractions to Xres and B, fused where provably safe.

    Replaces the stock `for j in range(n): ...` block one-for-one. Falls back
    to the stock statements for any phase whose windows are not disjoint, and
    for the whole run if the first phase does not come out bit-identical.
    """
    global _CHOICE

    if _CHOICE is False or not _eligible(Xres, B, U_time, ctc, iX):
        if _CHOICE is None:
            _CHOICE = False
            if not _HAVE_TRITON:
                logger.info(f'fused peel unavailable (no triton: {_TRITON_ERR})')
        for j in range(n):
            _stock_phase(Xres, B, iX, iY, amp, U_time, ctc, tiwave, trange, j, n)
        return

    disjoint = bool(phases_disjoint(iX[:, 0], nt))

    if _CHOICE is None:
        if not disjoint:
            # Cannot validate against a phase the fused path is not allowed to
            # take; stay undecided and try again on the next peel.
            for j in range(n):
                _stock_phase(Xres, B, iX, iY, amp, U_time, ctc, tiwave,
                             trange, j, n)
            return
        Xs, Bs = Xres.clone(), B.clone()
        for j in range(n):
            _stock_phase(Xs, Bs, iX, iY, amp, U_time, ctc, tiwave, trange, j, n)
        for j in range(n):
            _fused_phase(Xres, B, iX, iY, amp, U_time, ctc_p, nt, j, n)
        # Raw bit patterns: torch.equal reports +0.0 == -0.0, which is exactly
        # the class of difference a rounding barrier can introduce.
        ok = (torch.equal(Xres.view(torch.int32), Xs.view(torch.int32))
              and torch.equal(B.view(torch.int32), Bs.view(torch.int32)))
        if ok:
            _CHOICE = True
            logger.info(
                f'fused peel enabled: bit-identical to the stock subtract on '
                f'the first phase ({Xres.numel() + B.numel():,} elements)')
        else:
            _CHOICE = False
            Xres.copy_(Xs)
            B.copy_(Bs)
            logger.info('fused peel disabled: not bit-identical on this '
                        'device; using the stock subtract')
        del Xs, Bs
        return

    for j in range(n):
        if disjoint:
            _fused_phase(Xres, B, iX, iY, amp, U_time, ctc_p, nt, j, n)
        else:
            _stock_phase(Xres, B, iX, iY, amp, U_time, ctc, tiwave, trange,
                         j, n)
