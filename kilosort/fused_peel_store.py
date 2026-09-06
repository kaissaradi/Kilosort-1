"""Fused spike-store tail for template_matching.run_matching's peel loop.

Once a peel has its surviving positions, stock writes them out with

    iX = xs[:, :1]
    iY = imax[iX]
    st[k:k+nsp, 0] = iX[:, 0]
    st[k:k+nsp, 1] = iY[:, 0]
    amps[k:k+nsp]  = B[iY, iX] * s[iY]
    th_amps[k:k+nsp] = cmax[iX[:, 0], None]**.5

That is four gathers, a multiply, a sqrt and four scatters -- roughly eight
launches to move about 26 spikes. Measured at 60 um production shapes
(NT=10122, n_units=1031, nsp=26):

    iY = imax[iX]        5.4 us      amps = B[iY,iX]*s   23.6 us
    st column writes    12.5 us      th_amps = cmax**.5  10.5 us
    ---------------------------------------------------------------
    whole store tail    49.9 us

comparable to the condition tail this sits next to. One program per spike does
all of it.

CALIBRATE THE EXPECTATION. The synced statement profile calls this 15.6% of
the learned pass, but that measurement inflates launch-bound blocks -- a sync
barrier forbids exactly the overlap those launches normally get. The condition
tail was 17.5% by the same measure and returned 1.03x end to end after a 5.1x
kernel. Expect the same order here, not 15%.

BIT-IDENTITY
------------
  * `B[iY, iX] * s[iY]` is a single fp32 multiply -- correctly rounded, no
    accumulation, nothing to contract.
  * `**.5` is exactly `sqrt` ON THE TORCH SIDE. Verified rather than assumed,
    because `pow` is not required to be correctly rounded: torch's `x**.5` was
    compared to `torch.sqrt(x)` on 4,194,304 random positives and on
    0.0/-0.0/1.0/4.0/denormal/3.4e38/inf, and every bit pattern agreed.
    ON THE TRITON SIDE THE OBVIOUS CHOICE IS WRONG. `tl.sqrt` lowers to the
    approximate hardware instruction and differed from torch on 7 of 26 real
    values by 1 ULP -- the gate rejected the kernel on its first run. Use
    `tl.math.sqrt_rn`, which is round-to-nearest and therefore correctly
    rounded, hence bit-identical. This is the only place in the series where
    an IEEE-exact-looking operation was not exact by default.
  * The index writes are integer copies.

The row ORDER of `xs` is load-bearing: st/amps/th_amps rows are written in the
order `nonzero` produced, and downstream code indexes them positionally. One
program per spike preserves that by construction -- there is no compaction or
atomic counter here, which is the mistake that would silently permute them.

KILOSORT_NO_PEEL_STORE=1 skips it. As everywhere in this series the first peel
of a sort is computed both ways and compared on raw bit patterns, and the
fused path is used only if they agree.
"""
import logging
import os

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
    _TRITON_ERR = ''
except Exception as e:                 # pragma: no cover - depends on install
    _HAVE_TRITON = False
    _TRITON_ERR = repr(e)

_CHOICE = None                         # None undecided / False stock / True on
BLOCK = 128


if _HAVE_TRITON:

    @triton.jit
    def _store_kernel(IX_ptr, IMAX_ptr, B_ptr, S_ptr, CMAX_ptr,
                      ST_ptr, AMPS_ptr, TH_ptr, IY_ptr,
                      k, nsp, sB0, sST0,
                      BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = off < nsp
        ix = tl.load(IX_ptr + off, mask=m, other=0)
        iy = tl.load(IMAX_ptr + ix, mask=m, other=0)
        tl.store(IY_ptr + off, iy, mask=m)

        row = k + off
        tl.store(ST_ptr + row * sST0, ix, mask=m)
        tl.store(ST_ptr + row * sST0 + 1, iy, mask=m)

        b = tl.load(B_ptr + iy * sB0 + ix, mask=m, other=0.0)
        sv = tl.load(S_ptr + iy, mask=m, other=0.0)
        tl.store(AMPS_ptr + row, b * sv, mask=m)

        c = tl.load(CMAX_ptr + ix, mask=m, other=0.0)
        # sqrt_rn, NOT tl.sqrt. tl.sqrt lowers to the approximate hardware
        # instruction and was measured to differ from torch on 7 of 26 real
        # values by 1 ULP -- caught by the gate on the first run. torch's
        # `x**.5` is correctly-rounded sqrt (verified against torch.sqrt on
        # 4.19M values), so only the round-to-nearest variant matches.
        tl.store(TH_ptr + row, tl.math.sqrt_rn(c), mask=m)


def _stock(st, amps, th_amps, k, iX, imax, B, s, cmax):
    nsp = iX.shape[0]
    iY = imax[iX]
    st[k:k + nsp, 0] = iX[:, 0]
    st[k:k + nsp, 1] = iY[:, 0]
    amps[k:k + nsp] = B[iY, iX] * s[iY]
    th_amps[k:k + nsp] = cmax[iX[:, 0], None]**.5
    return iY


def _eligible(st, amps, th_amps, iX, imax, B, s, cmax):
    if not _HAVE_TRITON or os.environ.get('KILOSORT_NO_PEEL_STORE'):
        return False
    if not (st.is_cuda and st.dtype == torch.int64):
        return False
    if amps.dtype != torch.float32 or th_amps.dtype != torch.float32:
        return False
    if B.dtype != torch.float32 or s.dtype != torch.float32:
        return False
    if cmax.dtype != torch.float32 or iX.dtype != torch.int64:
        return False
    if imax.dtype != torch.int64:
        return False
    # Flat (k+off) addressing on amps/th_amps and (row, col) on st.
    if amps.stride(0) != 1 or th_amps.stride(0) != 1 or st.stride(1) != 1:
        return False
    if iX.dim() != 2 or iX.shape[1] != 1 or iX.stride(0) != 1:
        return False
    if B.stride(1) != 1 or s.stride(0) != 1 or cmax.stride(0) != 1:
        return False
    return True


def _run(st, amps, th_amps, k, iX, imax, B, s, cmax, iY):
    nsp = iX.shape[0]
    _store_kernel[(triton.cdiv(nsp, BLOCK),)](
        iX, imax, B, s, cmax, st, amps, th_amps, iY,
        k, nsp, B.stride(0), st.stride(0),
        BLOCK=BLOCK, num_warps=4)


def store_spikes(st, amps, th_amps, k, iX, imax, B, s, cmax):
    """Write one peel's spikes into st/amps/th_amps and return iY.

    Replaces the stock statements one-for-one. Falls back to them for the
    whole process if the first peel does not come out bit-identical.
    """
    global _CHOICE

    if _CHOICE is False or not _eligible(st, amps, th_amps, iX, imax, B, s,
                                         cmax):
        if _CHOICE is None:
            _CHOICE = False
            if not _HAVE_TRITON:
                logger.info(f'fused peel store unavailable '
                            f'(no triton: {_TRITON_ERR})')
        return _stock(st, amps, th_amps, k, iX, imax, B, s, cmax)

    nsp = iX.shape[0]
    iY = torch.empty_like(iX)

    if _CHOICE is not None:
        _run(st, amps, th_amps, k, iX, imax, B, s, cmax, iY)
        return iY

    ref_iY = imax[iX]
    ref_st = torch.stack((iX[:, 0], ref_iY[:, 0]), dim=1)
    ref_amps = B[ref_iY, iX] * s[ref_iY]
    ref_th = cmax[iX[:, 0], None]**.5
    try:
        _run(st, amps, th_amps, k, iX, imax, B, s, cmax, iY)
        ok = (torch.equal(iY, ref_iY)
              and torch.equal(st[k:k + nsp], ref_st)
              # raw bit patterns: amps feeds the peel subtract and th_amps is
              # written straight to st[:, 2] in the output file.
              and torch.equal(amps[k:k + nsp].view(torch.int32),
                              ref_amps.view(torch.int32))
              and torch.equal(th_amps[k:k + nsp].view(torch.int32),
                              ref_th.view(torch.int32)))
    except Exception as e:
        logger.debug(f'fused peel store failed to run: {e}')
        ok = False

    if ok:
        _CHOICE = True
        logger.info('fused peel store enabled: bit-identical to the stock '
                    f'spike store on the first peel ({nsp} spikes)')
        return iY

    _CHOICE = False
    logger.info('fused peel store disabled: not bit-identical on this device; '
                'using the stock spike store')
    return _stock(st, amps, th_amps, k, iX, imax, B, s, cmax)
