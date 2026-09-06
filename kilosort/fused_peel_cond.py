"""Fused condition tail for template_matching.run_matching's peel loop.

After `Cfmax, imax = torch.max(B, 0)` each peel iteration runs

    Cfmax = torch.relu(Cfmax)
    Cfmax.mul_(Cfmax)
    Cfmax[:nt] = 0
    Cfmax[-nt:] = 0
    cmax = max_pool1d(Cfmax.view(1,1,-1), 2*nt+1, stride=1, padding=nt)[0,0]
    cnd1 = cmax > Th2
    cnd2 = torch.abs(cmax - Cfmax) < 1e-9
    xs   = torch.nonzero(cnd1 & cnd2)

which is ~10 kernels on a (NT,) array of 10,122 float32 -- 40 KB. Nothing here
is bandwidth-bound; it is pure launch overhead, repeated ~48 times per batch.
Statement profile of the learned pass after the peel LUT landed put this block
at 17.5%, second only to peel_subtract's 30.8%.

Measured at 60 um production shapes (NT=10122, nt=61), per call:

    relu+square+edges  17.7 us      cnd2 (abs<1e-9)  10.2 us
    max_pool1d          9.5 us      and              17.1 us
    cnd1 (>Th2)         3.6 us      nonzero          41.0 us
    ------------------------------------------------------------
    whole tail         87.5 us,  of which nonzero 23.8 us (27%)

`nonzero` is NOT fusable -- its output length is data-dependent, and the row
ORDER it produces is load-bearing downstream (st/amps rows are written in it).
So it stays, and this kernel targets the other 73%.

BIT-IDENTITY
------------
Unlike fused_peaks, this tail is not arithmetic-free: `cnd2` performs a real
fp32 subtract. That is still exactly reproducible, because it is a SINGLE
IEEE-754 operation -- correctly rounded, no accumulation order, and nothing for
the compiler to contract into an FMA (`enable_fp_fusion` is irrelevant to a
lone subtract). The rest is:

  * `max` is a selection, so the window max is bit-exact regardless of order,
    and the maxima's indices are never used here, so ties are harmless.
  * relu-then-square is two rounded steps in stock and two here. The square
    also removes the one signed-zero hazard for free: relu may return -0.0 or
    +0.0 for a -0.0 input depending on the max convention, but (-0.0)**2 and
    (+0.0)**2 are both exactly +0.0.
  * `>`, `<`, `&` and `abs` are exact.

`max_pool1d(..., padding=nt)` pads with -inf, so a window clipped by the array
end reduces over its in-range part only; the kernel reproduces that by loading
out-of-range positions as -inf.

NOT covered: a NaN in B. torch.relu propagates NaN, and tl.maximum's NaN
convention is not guaranteed to agree. B is a convolution of real data and
prepare_matching already nan_to_num's U, so this should not arise -- and if it
does, the runtime gate below catches it on the first peel and falls back.

As everywhere in this series, none of this is trusted: the first eligible peel
of every sort is computed both ways and compared on raw bit patterns, and the
fused path is used only if they agree. KILOSORT_NO_PEEL_COND=1 skips it.
"""
import logging
import os

import torch
from torch.nn.functional import max_pool1d

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
    _TRITON_ERR = ''
except Exception as e:                 # pragma: no cover - depends on install
    _HAVE_TRITON = False
    _TRITON_ERR = repr(e)

# None = undecided, False = stock for the rest of the process, else the config.
_CHOICE = None
_CONFIGS = [(256, 4), (512, 4), (128, 4), (1024, 8)]


if _HAVE_TRITON:

    @triton.jit
    def _cond_kernel(RAW_ptr, CMAX_ptr, MASK_ptr, NT, nt, Th2, EPS,
                     BLOCK: tl.constexpr, WIN: tl.constexpr):
        """cmax = pool(f(raw)); mask = (cmax > Th2) & (|cmax - f(raw)| < EPS).

        f(x) = relu(x)**2, then zeroed on the first and last nt positions --
        applied inline instead of writing the transformed array back, which is
        what lets the four stock kernels collapse into one.
        """
        cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = cols < NT
        neg_inf = float('-inf')

        x = tl.load(RAW_ptr + cols, mask=valid, other=0.0)
        a = tl.maximum(x, 0.0)
        a = a * a
        a = tl.where((cols < nt) | (cols >= NT - nt), 0.0, a)

        m = tl.full((BLOCK,), neg_inf, tl.float32)
        for k in tl.static_range(WIN):
            kc = cols + k - nt
            inb = (kc >= 0) & (kc < NT)
            v = tl.load(RAW_ptr + kc, mask=inb, other=0.0)
            v = tl.maximum(v, 0.0)
            v = v * v
            v = tl.where((kc < nt) | (kc >= NT - nt), 0.0, v)
            m = tl.maximum(m, tl.where(inb, v, neg_inf))

        tl.store(CMAX_ptr + cols, m, mask=valid)
        res = (m > Th2) & (tl.abs(m - a) < EPS)
        tl.store(MASK_ptr + cols, res.to(tl.int8), mask=valid)


def _stock(raw, nt, Th2):
    """Exactly the stock statements, on a copy (stock squares in place)."""
    c = torch.relu(raw)
    c = c * c
    c[:nt] = 0
    c[-nt:] = 0
    cmax = max_pool1d(c.view(1, 1, -1), (2 * nt + 1), stride=1,
                      padding=nt)[0, 0]
    return cmax, (cmax > Th2) & (torch.abs(cmax - c) < 1e-9)


def _eligible(raw, nt):
    if not _HAVE_TRITON or os.environ.get('KILOSORT_NO_PEEL_COND'):
        return False
    if raw.device.type != 'cuda' or raw.dtype != torch.float32:
        return False
    if raw.dim() != 1 or raw.stride(0) != 1:
        return False
    # The kernel folds the edge zeroing into the window transform, which
    # assumes the two zeroed bands do not overlap.
    if raw.numel() < 2 * nt + 1:
        return False
    return True


def _run(raw, cmax, mask, nt, Th2, cfg):
    BLOCK, num_warps = cfg
    NT = raw.numel()
    _cond_kernel[(triton.cdiv(NT, BLOCK),)](
        raw, cmax, mask, NT, nt, float(Th2), 1e-9,
        BLOCK=BLOCK, WIN=2 * nt + 1, num_warps=num_warps)


def peak_condition(raw, nt, Th2):
    """Return (cmax, mask) for one peel, fused where provably safe.

    `raw` is the untransformed `torch.max(B, 0)` result; the relu/square/edge
    zeroing that stock applies in place is folded into the kernel, so `raw` is
    NOT modified. Falls back to the stock statements for the whole process if
    the first peel does not come out bit-identical.
    """
    global _CHOICE

    if _CHOICE is False or not _eligible(raw, nt):
        if _CHOICE is None:
            _CHOICE = False
            if not _HAVE_TRITON:
                logger.info(f'fused peel cond unavailable '
                            f'(no triton: {_TRITON_ERR})')
        return _stock(raw, nt, Th2)

    if _CHOICE is not None:
        cmax = torch.empty_like(raw)
        mask = torch.empty(raw.shape, dtype=torch.bool, device=raw.device)
        _run(raw, cmax, mask, nt, Th2, _CHOICE)
        return cmax, mask

    ref_cmax, ref_mask = _stock(raw, nt, Th2)
    cmax = torch.empty_like(raw)
    mask = torch.empty(raw.shape, dtype=torch.bool, device=raw.device)
    for cfg in _CONFIGS:
        try:
            _run(raw, cmax, mask, nt, Th2, cfg)
            # Raw bit patterns on cmax: it feeds th_amps = cmax[iX]**.5, so a
            # signed-zero or last-bit difference would propagate to an output.
            ok = (torch.equal(cmax.view(torch.int32),
                              ref_cmax.view(torch.int32))
                  and torch.equal(mask, ref_mask))
        except Exception as e:
            logger.debug(f'fused peel cond config {cfg} failed to run: {e}')
            continue
        if not ok:
            logger.debug(f'fused peel cond config {cfg} is not identical')
            continue
        _CHOICE = cfg
        logger.info(
            f'fused peel cond enabled: bit-identical to the stock condition '
            f'tail on the first peel ({raw.numel():,} positions), config {cfg}')
        return cmax, mask

    _CHOICE = False
    logger.info('fused peel cond disabled: no identical config on this '
                'device; using the stock condition tail')
    return ref_cmax, ref_mask
