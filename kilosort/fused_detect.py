"""Fused Triton kernel for the inner loop of spikedetect.template_match.

Universal detection is the dominant stage of an MEA sort (927 s of 1602 s on
20260724A/chunk12_9-11, 57.9%), and 97% of it is template_match, of which 96%
is the per-chunk loop body. That body was measured at the memory roofline: its
statements run at 35-103% of this card's 279 GB/s device-to-device copy rate,
so no amount of retiling can help it. It moves 8.5 GB per chunk to produce
65 MB of output -- 130x the irreducible traffic -- because

    A    = einsum('ijk, jklm-> iklm', weigh, Bsl[iC])      # (ns, Nfilt, nk, M)
    Aa   = max(|A|, 0);  imax = argmax;  sign via gather
    Amax = max(Aa[iC2_flat].view(nC2, Nfilt, -1), 0)

materialises both gathers in full. `Bsl[iC]` alone expands 1.3 M floats into
103 M (78x). Every output column depends only on its own column of B, so a
kernel that keeps those gathers in registers and L2 does the same arithmetic
with ~1/130 of the traffic.

BIT-IDENTITY
------------
This is not a numerically-equivalent rewrite, it is the same arithmetic:

  * einsum lowers to a plain torch.bmm here (verified bitwise), so the
    reference is cuBLAS's sgemm with K = nC.
  * Modelling candidate fp32 instruction sequences in float64 showed cuBLAS
    reproduces, on 20000/20000 sampled outputs, exactly SPLIT-K 2: two
    contiguous halves of K accumulated separately with FMA, then added. A
    single ascending FMA chain matches only 54%. The j loop below emits the
    split.
  * max/min are exact in floating point, so the reductions' order does not
    matter; torch.max(dim=0)'s first-index tie-break is reproduced by taking
    the minimum index among the maximal entries.

That gets it bit-identical *on this GPU, this cuBLAS, this Triton and these
shapes* -- and only at some block sizes. At production shapes BLOCK_M=128 with
4 warps matches and 64 and 32 do not; at the small shapes in
tests/test_fused_detect.py it is the other way round and the gate settles on
(64, 2). cuBLAS picks a different kernel per shape and Triton schedules
differently per block size, so which config agrees is not predictable and is
not a constant of the code.

So none of it is trusted. `try_fill` runs the stock path and the fused path on
the first real batch of every sort and compares all three output buffers
element by element; the fused path is used only for configs that come out
exactly equal, and a sort where nothing matches logs why and runs stock
throughout, one batch slower.

Measured on 40 real batches of 20260724A (Nfilt=4048, nC=10, nk=10, ns=5,
NT=10122), 4.92 billion elements compared, all equal:

    stock tiled dispatch   229.85 ms/batch
    fused single launch     33.33 ms/batch     6.90x

Set KILOSORT_NO_FUSED_DETECT=1 to skip the fused path entirely.
"""
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception as _e:            # pragma: no cover - depends on install
    _HAVE_TRITON = False
    _TRITON_ERR = _e

# Config candidates, best-first. Only ones that pass the runtime bit-identity
# check are eligible; the fastest survivor is used. BLOCK_M=128/warps=4 is the
# one verified at production shapes on the RTX 4000 Ada. The rest are not
# padding: the unit tests' small shapes reject it and land on (64, 2), so the
# ladder is exercised in practice, not only on hypothetical other cards.
_CONFIGS = ((128, 4), (128, 2), (64, 2), (64, 4), (256, 4), (32, 2))

# None = not yet tested this process; False = disabled; else (block_m, warps).
_CHOICE = None


if _HAVE_TRITON:

    @triton.jit
    def _tm_fused_kernel(
            B_ptr, W_ptr, iC_ptr, Aa_ptr, Im_ptr,
            NT,
            sB0, sB1, sB2,
            sW0, sW1, sW2,
            siC0, siC1,
            sA0, sA1,
            NK: tl.constexpr, NC: tl.constexpr,
            NIL: tl.constexpr, NIL_P: tl.constexpr, BLOCK_M: tl.constexpr):
        """One program per (template k, time tile).

            A[i,l,m] = sum_j weigh[i,j,k] * B[iC[j,k], l, m]
            Aa[k,m]  = max_{i,l} |A[i,l,m]|
            Im[k,m]  = (1 + argmax_{i,l} |A|) * sign(A there)

        The (i,l) axis is flattened as il = i*NK + l, which is what
        A.transpose(1,2).reshape(-1, Nfilt, M) produces, so the argmax index
        means the same thing as the stock body's. NIL_P is NIL rounded up to a
        power of two (Triton tiles must be); pad rows are forced to -1 before
        the max so they can never win.
        """
        k = tl.program_id(0)
        offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < NT

        offs_il = tl.arange(0, NIL_P)
        il_mask = offs_il < NIL
        i_idx = offs_il // NK
        l_idx = offs_il % NK

        b_off = l_idx[:, None] * sB1 + offs_m[None, :] * sB2
        b_mask = il_mask[:, None] & mask_m[None, :]
        w_off = i_idx * sW0 + k * sW2

        # Two contiguous half-K accumulators summed at the end: this is
        # cuBLAS's arithmetic for this gemm, established by modelling. A
        # single ascending chain is a different answer (54% match).
        acc0 = tl.zeros((NIL_P, BLOCK_M), dtype=tl.float32)
        acc1 = tl.zeros((NIL_P, BLOCK_M), dtype=tl.float32)
        half = (NC + 1) // 2
        for j in range(NC):
            c = tl.load(iC_ptr + j * siC0 + k * siC1)
            b = tl.load(B_ptr + c * sB0 + b_off, mask=b_mask, other=0.0)
            w = tl.load(W_ptr + w_off + j * sW1, mask=il_mask, other=0.0)
            if j < half:
                acc0 = acc0 + w[:, None] * b
            else:
                acc1 = acc1 + w[:, None] * b
        acc = acc0 + acc1

        aabs = tl.where(il_mask[:, None], tl.abs(acc), -1.0)
        best = tl.max(aabs, axis=0)
        # First maximal index, matching torch.max(dim=0). max/min are exact,
        # so the reduction order here cannot change the answer.
        bidx = tl.min(tl.where(aabs == best[None, :], offs_il[:, None], NIL_P),
                      axis=0)
        sgn = tl.where(acc > 0, 1, tl.where(acc < 0, -1, 0)).to(tl.int32)
        bsgn = tl.sum(tl.where(offs_il[:, None] == bidx[None, :], sgn, 0),
                      axis=0)

        tl.store(Aa_ptr + k * sA0 + offs_m * sA1, best, mask=mask_m)
        tl.store(Im_ptr + k * sA0 + offs_m * sA1,
                 (1 + bidx).to(tl.int32) * bsgn, mask=mask_m)

    @triton.jit
    def _amax_kernel(Aa_ptr, iC2_ptr, Am_ptr, NT, Nfilt, sA0, sA1,
                     NC2: tl.constexpr, BLOCK_M: tl.constexpr):
        """Am[k,m] = max_p Aa[iC2[p,k], m].

        iC2_flat is (NC2*Nfilt,) laid out p-major, so entry (p,k) sits at
        p*Nfilt + k. Replaces the index_select that materialised a
        (NC2, Nfilt, M) copy -- 1.5 GB per chunk at production shapes.
        """
        k = tl.program_id(0)
        offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < NT
        acc = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
        for p in range(NC2):
            c = tl.load(iC2_ptr + p * Nfilt + k)
            acc = tl.maximum(acc, tl.load(Aa_ptr + c * sA0 + offs_m * sA1,
                                          mask=mask_m, other=float('-inf')))
        tl.store(Am_ptr + k * sA0 + offs_m * sA1, acc, mask=mask_m)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _run(B, weigh, iC, iC2_flat, nC2, Nfilt, As, imaxs, Amaxs, cfg):
    """Fill As / imaxs / Amaxs in place from the full-width B. One launch each
    -- the fused body needs no column tiling, since output column m reads only
    B[:, :, m]."""
    block_m, num_warps = cfg
    NT = B.shape[-1]
    ns, nC, _ = weigh.shape
    nk = B.shape[1]
    nil = ns * nk
    grid = (Nfilt, triton.cdiv(NT, block_m))
    _tm_fused_kernel[grid](
        B, weigh, iC, As, imaxs, NT,
        B.stride(0), B.stride(1), B.stride(2),
        weigh.stride(0), weigh.stride(1), weigh.stride(2),
        iC.stride(0), iC.stride(1),
        As.stride(0), As.stride(1),
        NK=nk, NC=nC, NIL=nil, NIL_P=_next_pow2(nil), BLOCK_M=block_m,
        num_warps=num_warps, num_stages=1)
    _amax_kernel[grid](
        As, iC2_flat, Amaxs, NT, Nfilt, As.stride(0), As.stride(1),
        NC2=nC2, BLOCK_M=block_m, num_warps=num_warps, num_stages=1)


def _eligible(B, weigh, iC, As, imaxs, Amaxs):
    """Cheap structural preconditions. Anything unusual falls back to stock
    rather than being handled -- the fused path is an optimisation, not a
    second implementation to keep in sync."""
    if not _HAVE_TRITON or os.environ.get('KILOSORT_NO_FUSED_DETECT'):
        return False
    if not (B.is_cuda and B.dtype == torch.float32):
        return False
    if weigh.dtype != torch.float32 or As.dtype != torch.float32:
        return False
    if Amaxs.dtype != torch.float32 or imaxs.dtype != torch.int32:
        return False
    if iC.dtype not in (torch.int32, torch.int64):
        return False
    ns, nC, _ = weigh.shape
    nk = B.shape[1]
    # Register budget: the kernel holds two (NIL_P, BLOCK_M) accumulators.
    if ns * nk > 128 or nC > 32:
        return False
    return True


def try_fill(B, weigh, iC, iC2_flat, nC2, Nfilt, As, imaxs, Amaxs, stock_fill):
    """Fill the three peak buffers, fused if it is provably safe to.

    Returns True if the buffers are filled and the caller should do nothing
    else. On the first call of a process this runs `stock_fill` (which fills
    the buffers) and then every candidate config into scratch buffers,
    comparing all three element by element; the fastest config that is exactly
    equal is kept for the rest of the sort. It returns True in that case too --
    the stock result is already in place and is the one used for this batch.
    """
    global _CHOICE

    if _CHOICE is False:
        return False
    if _CHOICE is not None:
        _run(B, weigh, iC, iC2_flat, nC2, Nfilt, As, imaxs, Amaxs, _CHOICE)
        return True

    if not _eligible(B, weigh, iC, As, imaxs, Amaxs):
        _CHOICE = False
        if not _HAVE_TRITON:
            logger.info(f'fused detection unavailable (no triton: {_TRITON_ERR})')
        return False

    # First batch: stock result is the reference, and is what this batch uses.
    stock_fill()
    fA = torch.empty_like(As)
    fI = torch.empty_like(imaxs)
    fM = torch.empty_like(Amaxs)
    passed = []
    for cfg in _CONFIGS:
        try:
            _run(B, weigh, iC, iC2_flat, nC2, Nfilt, fA, fI, fM, cfg)
            ok = (torch.equal(fA, As) and torch.equal(fI, imaxs)
                  and torch.equal(fM, Amaxs))
        except Exception as e:
            logger.debug(f'fused detect config {cfg} failed to run: {e}')
            continue
        if not ok:
            logger.debug(f'fused detect config {cfg} is not bit-identical')
            continue
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            _run(B, weigh, iC, iC2_flat, nC2, Nfilt, fA, fI, fM, cfg)
        torch.cuda.synchronize()
        passed.append(((time.perf_counter() - t0) / 3, cfg))
        # First candidate is the one validated in development; if it passes
        # and is fast there is nothing to gain from compiling the rest.
        if cfg == _CONFIGS[0]:
            break
    del fA, fI, fM

    if not passed:
        _CHOICE = False
        logger.info('fused detection disabled: no bit-identical config on this '
                    'device; using the stock template_match loop')
        return True

    dt, cfg = min(passed)
    _CHOICE = cfg
    logger.info(f'fused detection enabled: BLOCK_M={cfg[0]} warps={cfg[1]}, '
                f'bit-identical to the stock loop on batch 0 '
                f'({As.numel()*3:,} elements), {dt*1e3:.1f} ms/batch')
    return True
