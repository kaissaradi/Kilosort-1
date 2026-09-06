"""Fused Triton kernel for the peak-selection tail of spikedetect.template_match.

After the (already fused, see fused_detect.py) per-chunk loop fills As / Amaxs /
imaxs, the tail selects peaks with

    Amaxs[:, :nt] = 0 ; Amaxs[:, -nt:] = 0
    Amaxs = max_pool1d(Amaxs, 2*nt0+1, stride=1, padding=nt0)
    xy    = logical_and(Amaxs == As, As > Th).nonzero()

At production shapes (Nfilt=4048, NT=10122) each of As/Amaxs is 41 M float32 =
164 MB, and that sequence moves ~1 GB per batch to produce one bool array:
the pool reads and writes a full copy, the two comparisons read another three,
and logical_and reads and writes two more. Every output element depends only on
its own row, on a +/-nt0 window, so one kernel reading As and Amaxs once does
the same work with ~1/3 of the traffic.

BIT-IDENTITY
------------
This tail contains **no floating-point arithmetic at all** -- only a max
reduction and two comparisons:

  * `max` is a selection, not an accumulation. The maximum of a set of float32
    values is the same bit pattern regardless of the order in which the
    comparisons are performed, so the fused window max is exactly the value
    max_pool1d produces. There is no rounding, no FMA contraction, and no
    accumulation order to reproduce.
  * The *indices* of maxima are never used here (unlike fused_detect, where the
    argmax tie-break had to be matched), so exact ties are harmless: only the
    value is compared.
  * `==`, `>` and `&` are exact.

That makes this the safest fusion in this series -- but "safest" is not
"trusted". The one thing that is a genuine semantic question is what
`max_pool1d` does at the array ends: it pads with -inf, so a window clipped by
the boundary is the max over the in-range part only. The kernel reproduces that
by loading out-of-range positions as -inf, and `try_mask` checks the whole
result against the stock statements on the first real batch of every sort.

Set KILOSORT_NO_FUSED_PEAKS=1 to skip the fused path entirely.
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
except Exception as e:  # pragma: no cover - depends on the install
    _HAVE_TRITON = False
    _TRITON_ERR = repr(e)

# None = undecided (validate on the first eligible batch), False = use stock
# for the rest of the process, otherwise the winning config.
_CHOICE = None

# Ordered by measured speed at production shapes (Nfilt=4048, NT=10122): the
# short-circuit makes small blocks win, because a block is skipped whole and
# smaller blocks are skipped more often. try_mask keeps the first config that
# is identical, so this order is the preference order.
_CONFIGS = [(128, 4), (256, 4), (512, 4)]


if _HAVE_TRITON:

    @triton.jit
    def _peak_mask_kernel(As_ptr, Am_ptr, out_ptr, s_row, NT, nt, nt0, Th,
                          BLOCK_C: tl.constexpr, WIN: tl.constexpr):
        r = tl.program_id(0)
        cb = tl.program_id(1)
        cols = cb * BLOCK_C + tl.arange(0, BLOCK_C)
        valid = cols < NT
        base = r.to(tl.int64) * s_row

        # `As > Th` is the cheap half of the test and is extremely sparse --
        # at production shapes ~1.4k of 41 M positions survive the pair. Test
        # it first and skip the whole WIN-iteration window max for any block
        # with no candidate, which is nearly all of them. This is a pure
        # short-circuit: `(m == a) & (a > Th)` is False wherever `a > Th` is
        # False, whatever m turns out to be, so the skipped blocks are storing
        # exactly the value the full path would have stored.
        a = tl.load(As_ptr + base + cols, mask=valid, other=0.0)
        hot = (a > Th) & valid
        if tl.sum(hot.to(tl.int32)) == 0:
            tl.store(out_ptr + base + cols, tl.zeros((BLOCK_C,), tl.int8),
                     mask=valid)
        else:
            neg_inf = float('-inf')
            m = tl.full((BLOCK_C,), neg_inf, tl.float32)
            # max_pool1d(..., stride=1, padding=nt0) pads with -inf, so a
            # window clipped by the array end reduces over its in-range part
            # only.
            for k in tl.static_range(WIN):
                kc = cols + k - nt0
                inb = (kc >= 0) & (kc < NT)
                v = tl.load(Am_ptr + base + kc, mask=inb, other=neg_inf)
                # Amaxs[:, :nt] = 0 and Amaxs[:, -nt:] = 0, folded in rather
                # than written back to the buffer.
                edge = (kc < nt) | (kc >= NT - nt)
                v = tl.where(inb & edge, 0.0, v)
                m = tl.maximum(m, v)

            res = (m == a) & hot
            tl.store(out_ptr + base + cols, res.to(tl.int8), mask=valid)


def _run(As, Amaxs, out, nt, nt0, Th, cfg):
    BLOCK_C, num_warps = cfg
    Nfilt, NT = As.shape
    grid = (Nfilt, triton.cdiv(NT, BLOCK_C))
    _peak_mask_kernel[grid](
        As, Amaxs, out, As.stride(0), NT, nt, nt0, float(Th),
        BLOCK_C=BLOCK_C, WIN=2 * nt0 + 1, num_warps=num_warps,
    )


def _eligible(As, Amaxs):
    if not _HAVE_TRITON or os.environ.get('KILOSORT_NO_FUSED_PEAKS'):
        return False
    if As.device.type != 'cuda':
        return False
    if As.dtype != torch.float32 or Amaxs.dtype != torch.float32:
        return False
    if As.shape != Amaxs.shape or As.dim() != 2:
        return False
    # Row-major with contiguous columns: the kernel indexes base + col.
    if As.stride(1) != 1 or Amaxs.stride(1) != 1:
        return False
    if As.stride(0) != Amaxs.stride(0):
        return False
    return True


def _stock_mask(As, Amaxs, nt, nt0, Th):
    """Exactly the stock statements, on a copy (Amaxs is written in place)."""
    from torch.nn.functional import max_pool1d
    Am = Amaxs.clone()
    Am[:, :nt] = 0
    Am[:, -nt:] = 0
    Am = max_pool1d(Am.unsqueeze(0), (2 * nt0 + 1), stride=1,
                    padding=nt0).squeeze(0)
    return torch.logical_and(Am == As, As > Th)


def try_mask(As, Amaxs, nt, nt0, Th):
    """Return the peak mask, fused if it is provably safe to, else None.

    Returning None means the caller must run the stock statements itself.
    On the first eligible batch of a process this computes the stock mask and
    every candidate config, and keeps the first config that is exactly equal.
    """
    global _CHOICE

    if _CHOICE is False:
        return None
    if not _eligible(As, Amaxs):
        if _CHOICE is None:
            _CHOICE = False
            if not _HAVE_TRITON:
                logger.info(f'fused peaks unavailable (no triton: {_TRITON_ERR})')
        return None

    if _CHOICE is not None:
        out = torch.empty(As.shape, dtype=torch.bool, device=As.device)
        _run(As, Amaxs, out, nt, nt0, Th, _CHOICE)
        return out

    ref = _stock_mask(As, Amaxs, nt, nt0, Th)
    out = torch.empty(As.shape, dtype=torch.bool, device=As.device)
    for cfg in _CONFIGS:
        try:
            _run(As, Amaxs, out, nt, nt0, Th, cfg)
            ok = torch.equal(out, ref)
        except Exception as e:
            logger.debug(f'fused peaks config {cfg} failed to run: {e}')
            continue
        if not ok:
            logger.debug(f'fused peaks config {cfg} is not identical')
            continue
        _CHOICE = cfg
        logger.info(
            f'fused peaks enabled: identical to the stock peak selection on '
            f'the first batch ({ref.numel():,} elements), config {cfg}')
        del ref
        return out

    _CHOICE = False
    logger.info('fused peaks disabled: no identical config on this device; '
                'using the stock peak selection')
    del out
    return ref
