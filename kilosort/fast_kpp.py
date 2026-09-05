"""kmeans_plusplus without the per-iteration device->host reads.

clustering_qr.kmeans_plusplus is where clustering spends its time. Profiled on
slice300 (a 300-batch sort with production settings) it is 78% of the
'template' clustering pass and 87% of the 'spikes' pass -- 23.2 s of the 28.2 s
the two passes take together, and a third of the whole sort:

    mode='template'   run() total 14.58 s      mode='spikes'   run() 13.63 s
      cluster TOTAL           12.36  84.8%       cluster TOTAL      12.69 93.1%
      kmeans_plusplus         11.44  78.5%       kmeans_plusplus    11.80 86.6%
      split                    1.80  12.3%       split               0.50  3.7%
      neigh_mat                0.32   2.2%       neigh_mat           0.27  2.0%
      get_data_cpu             0.14   0.9%       get_data_cpu        0.15  1.1%
      maketree                 0.11   0.8%       maketree            0.12  0.9%

and none of that is arithmetic. The loop runs 200 iterations over tensors of at
most 13k x 75 floats; a 1,162-spike centre takes 53.2 ms and a 3,263-spike
centre takes 53.0 ms -- the same, because the cost is per-iteration overhead,
not per-element work. Statement-level timing of the stock loop, synchronizing
after each statement (12,941 spikes; us per iteration):

    multinomial               104.3      dexp relu                  23.2
    vexp0[ix] = vexp[ix,imax]  71.2      dexp.sum(0)                19.4
    vexp matmul                57.0      mu[j] = Xc[imax]           17.9
    int((weights>0).sum())     34.0      float(weights.sum())       17.4
    ix = dexp[:,imax] > 0      26.6      relu(vtot - vexp0)         15.6
                                         iclust[ix] = j             12.1

Three of those block the host. Two are the loop's guard:

    if float(weights.sum()) <= 0: break
    n_pos = int((weights > 0).sum().item())
    if n_pos <= 0: break
    n_draw = min(ntry, n_pos)

and one is hidden inside `vexp0[ix] = vexp[ix, imax]`, where the boolean
advanced index has to run nonzero() to size its output.

WHAT THIS MODULE CHANGES, AND WHY EACH IS THE SAME ANSWER

1. The guard collapses to one question. weights = relu(...) >= 0, so
   `sum <= 0` <=> every entry is zero <=> `n_pos == 0`; and `n_pos == 0`
   implies `n_pos < ntry`. So the only branch that can change what stock
   computes is `n_pos < ntry`, and it is enough to know the SMALLEST n_pos the
   loop ever saw.

   That minimum is the last one. vexp0 is written only through
   `vexp0[ix] = vexp[ix, imax]` with `ix = dexp[:, imax] > 0` and
   `dexp = relu(vexp - vexp0[:, None])`, so ix is true exactly where
   vexp[:, imax] > vexp0 -- every write raises an entry and none lowers one.
   (NaN cannot slip through: relu(NaN) is NaN and NaN > 0 is false.) vexp0
   non-decreasing makes relu(vtot - vexp0) non-increasing elementwise, so
   n_pos is non-increasing, so n_pos after the last iteration bounds every
   n_pos the loop saw. One count_nonzero and one host read per CALL replace
   two per ITERATION -- 400 reads become 1.

   If that final count is below ntry the fast result is thrown away and the
   caller runs stock; nothing is guessed. Measured over a whole slice300 sort
   (393 calls): every call ran all 200 iterations and the smallest n_pos seen
   anywhere was 877, versus ntry=100. The fallback exists for the case that
   isn't in this recording, not for one that is.

2. `vexp0[ix] = vexp[ix, imax]` becomes `torch.where(ix, vexp[:, imax], vexp0)`
   and `iclust[ix] = j` becomes `iclust.masked_fill_(ix, j)`. Both are pure
   selection -- the same bits are chosen, only without materializing an index
   list, so neither blocks the host.

3. `2 * Xg @ Xc.T` is hoisted. `*` and `@` share precedence and associate
   left, so that expression is `(2 * Xg) @ Xc.T`: stock rebuilds the whole
   scaled feature matrix on all 200 iterations. Hoisting feeds cuBLAS
   identical bytes.

4. `mu` is not built. It is written every iteration and then never read --
   kmeans_plusplus returns iclust alone, and the only code that used mu is the
   commented-out block at the end of the stock function. Dropping it removes a
   (niter, n_features) allocation and one index+copy per iteration. If that
   block is ever revived, this shortcut has to go with it.

Nothing else moves: the multinomial draw, the gemm, the relu, the sum and the
argmax are the same calls on the same values in the same order.

RNG. The loop consumes the global torch generator through torch.multinomial,
and either path leaves it in the same state: the fast path draws the same 200
times, and the fallback re-seeds (stock's own `torch.manual_seed(seed)`) before
drawing. So downstream code cannot tell which ran.

Measured on 32 real Xd matrices dumped from a slice300 sort, 1,002 to 12,941
spikes: iclust identical in all 32, generator state identical in all 32,
1,759.7 ms -> 1,125.6 ms (1.56x).

Set KILOSORT_NO_FAST_KPP=1 to skip this entirely.
"""
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Candidate centroids tested per iteration; stock's local `ntry`. It is the
# threshold the guard is checked against, so it must match the caller's.
NTRY = 100

# None = not yet checked in this process, True = verified against stock,
# False = disabled (env switch, ineligible input, or failed verification).
_CHOICE = None


def _fast_loop(Xg, niter, seed, device):
    """The stock loop with the four changes in the module docstring.

    Returns (iclust, n_pos_final). n_pos_final < NTRY means stock would have
    drawn fewer than NTRY candidates somewhere, or broken out early, and the
    result must be discarded.
    """
    vtot = torch.norm(Xg, 2, dim=1)**2

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_spikes = Xg.shape[0]
    vexp0 = torch.zeros(n_spikes, device=device)
    iclust = torch.zeros((n_spikes,), dtype=torch.int, device=device)

    # (2 * Xg) once instead of 200 times; see note 3.
    Xg2 = 2 * Xg
    # j as a device scalar so masked_fill_ needs no host->device copy.
    jvals = torch.arange(niter, dtype=torch.int32, device=device)

    for j in range(niter):
        v2 = torch.relu(vtot - vexp0)
        isamp = torch.multinomial(v2, NTRY, replacement=False)
        Xc = Xg[isamp]
        vexp = Xg2 @ Xc.T - (Xc**2).sum(1)
        dexp = torch.relu(vexp - vexp0.unsqueeze(1))
        vsum = dexp.sum(0)
        imax = torch.argmax(vsum)
        ix = dexp[:, imax] > 0
        iclust.masked_fill_(ix, jvals[j])
        vexp0 = torch.where(ix, vexp[:, imax], vexp0)
        del vexp, dexp

    # The whole guard, once. See note 1 for why the final count bounds every
    # count the loop passed through.
    n_pos_final = torch.count_nonzero(torch.relu(vtot - vexp0))
    return iclust, n_pos_final


def _eligible(Xg, niter, device):
    if os.environ.get('KILOSORT_NO_FAST_KPP'):
        return False
    if not (Xg.is_cuda and Xg.dtype == torch.float32):
        return False
    if getattr(device, 'type', None) != 'cuda':
        return False
    # Stock subsamples the candidate pool above 2**24 spikes because
    # torch.multinomial cannot address more; that branch is not reproduced.
    if Xg.shape[0] > 2**24 or Xg.shape[0] < NTRY:
        return False
    if niter <= 0:
        return False
    return True


def try_run(Xg, niter, seed, device, stock_loop):
    """Run the fast loop if it can be shown to give stock's answer.

    Returns iclust, or None to tell the caller to run its own stock body.
    `stock_loop` is called only once per process, to validate the first result;
    it must be a zero-argument callable returning stock's iclust for this same
    input.
    """
    global _CHOICE

    if _CHOICE is False or not _eligible(Xg, niter, device):
        if _CHOICE is None and os.environ.get('KILOSORT_NO_FAST_KPP'):
            _CHOICE = False
            logger.info('fast kmeans_plusplus disabled by KILOSORT_NO_FAST_KPP')
        return None

    try:
        iclust, n_pos_final = _fast_loop(Xg, niter, seed, device)
    except RuntimeError as e:
        # Drawing NTRY candidates without replacement needs NTRY positive
        # weights; if there were fewer, stock would have drawn fewer too, so
        # this is the same fallback as n_pos_final < NTRY.
        logger.debug(f'fast kmeans_plusplus fell back: {e}')
        return None

    if int(n_pos_final) < NTRY:
        logger.debug(
            f'fast kmeans_plusplus fell back: n_pos {int(n_pos_final)} < {NTRY}, '
            f'so stock would not have drawn a full candidate set')
        return None

    if _CHOICE is None:
        ref = stock_loop()
        if torch.equal(ref, iclust):
            _CHOICE = True
            logger.info(
                f'fast kmeans_plusplus enabled: identical to the stock loop on '
                f'{iclust.numel():,} labels')
        else:
            _CHOICE = False
            logger.info('fast kmeans_plusplus disabled: labels differ from the '
                        'stock loop on this device')
            return None

    return iclust
