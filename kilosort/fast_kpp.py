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


# ---------------------------------------------------------------------------
# CUDA graph path
#
# _fast_loop still issues ~20 kernel launches per iteration on tensors small
# enough that the launch costs more than the work: 200 iterations take 33.6 ms
# at 1,162 spikes and 44.3 ms at 12,941, so ~168 us/iteration is overhead and
# only ~54 us/iteration is data. A CUDA graph collapses those 20 launches into
# one. Two obstacles, both measured rather than assumed:
#
# RNG. Capturing torch's generator makes every replay draw different numbers
# from what eager would, which changes the answer. But
# torch.multinomial(w, k, replacement=False) is exactly
# topk(w / empty_like(w).exponential_(), k) -- verified bitwise on this install
# over 50 weight vectors and all 32 real matrices -- i.e. one exponential_ over
# an (n_spikes,) tensor per iteration and nothing else. Drawing those niter
# vectors up front, ONE ROW AT A TIME, consumes the generator in exactly the
# stock order (also verified bitwise). One exponential_ over the whole
# (niter, n_spikes) buffer does NOT: the offset advance depends on the numel of
# each call, so that produces a different stream. It was checked, and it
# differs. Do not "simplify" _draw_noise into a single call.
#
# SHAPE. Every centre has a different n_spikes, so a graph cannot be reused
# between centres and capture is paid ~393 times per sort. That is affordable
# only because capture itself costs 0.18 ms -- but torch.cuda.graph() costs
# 75 ms, and all of it is the gc.collect() its __enter__ runs (measured: 79 ms
# for the gc, 0.01 ms for the empty_cache, 0.02 ms for the synchronize). Hence
# capture_begin/capture_end by hand.
#
# Padding to a bucket size would let graphs be reused, but it is not available:
# dexp.sum(0) over (n + pad, NTRY) is a different reduction tree from
# (n, NTRY), so the sum comes out with different bits even though the padded
# rows are exactly zero.
#
# MEMORY. A graph's intermediates live in a private pool, and destroying the
# graph does NOT return that pool to the allocator: with a fresh pool per
# graph, reserved memory grew 23 MB per capture and reached 9.9 GB after 400.
# Sharing one pool handle fixes it (reserved plateaus at 790 MB and stays
# flat), but a shared handle whose graphs have all been destroyed trips
# `it->second->use_count > 0 INTERNAL ASSERT FAILED` in the caching allocator.
# So the previous graph is held until the next one has been captured, which
# keeps the handle's use count above zero without pinning a graph forever.
# ---------------------------------------------------------------------------

# None until the first capture; then a pool handle kept for the process.
_POOL = None
_CAPTURE_STREAM = None
# The graph captured last, held only so _POOL keeps a live user; see MEMORY.
_PREV_GRAPH = None
# None = untested, True = verified against stock, False = unavailable.
_GRAPH_CHOICE = None


def _draw_noise(niter, n_spikes, device):
    """The niter exponential vectors torch.multinomial would have drawn.

    One row at a time. See the RNG note above: this is not the same stream as
    a single exponential_ over the whole buffer.
    """
    Q = torch.empty(niter, n_spikes, device=device)
    for j in range(niter):
        Q[j].exponential_()
    return Q


def _graph_loop(Xg, niter, seed, device):
    """_fast_loop with the iteration body captured in a CUDA graph.

    Same arithmetic and same random draws; see the block comment above for why
    each departure from _fast_loop is the same computation. Returns
    (iclust, n_pos_final) exactly as _fast_loop does.
    """
    global _POOL, _CAPTURE_STREAM, _PREV_GRAPH

    n_spikes = Xg.shape[0]
    vtot = torch.norm(Xg, 2, dim=1)**2

    torch.manual_seed(seed)
    np.random.seed(seed)
    Q = _draw_noise(niter, n_spikes, device)

    Xg2 = 2 * Xg
    vexp0 = torch.zeros(n_spikes, device=device)
    iclust = torch.zeros((n_spikes,), dtype=torch.int, device=device)
    # j lives on the device and is advanced inside the graph, so the body has
    # no host-side state and the same recording serves every iteration.
    jt = torch.zeros(1, dtype=torch.int64, device=device)
    jt32 = torch.zeros((), dtype=torch.int32, device=device)

    def body():
        v2 = torch.relu(vtot - vexp0)
        q = Q.index_select(0, jt).view(-1)
        isamp = torch.topk(v2 / q, NTRY).indices
        Xc = Xg.index_select(0, isamp)
        vexp = Xg2 @ Xc.T - (Xc**2).sum(1)
        dexp = torch.relu(vexp - vexp0.unsqueeze(1))
        imax = torch.argmax(dexp.sum(0), dim=0, keepdim=True)
        ix = dexp.index_select(1, imax).squeeze(1) > 0
        # out=iclust / out=vexp0: the loop state must be written back to the
        # SAME addresses every replay, because the graph has those baked in.
        torch.where(ix, jt32, iclust, out=iclust)
        torch.where(ix, vexp.index_select(1, imax).squeeze(1), vexp0, out=vexp0)
        jt.add_(1)
        jt32.add_(1)

    def reset():
        vexp0.zero_()
        iclust.zero_()
        jt.zero_()
        jt32.zero_()

    # Warm up on a side stream: capture records without executing, so anything
    # that lazily initializes (cuBLAS handles, topk workspaces) has to have run
    # already or capture fails.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            body()
    torch.cuda.current_stream().wait_stream(s)
    reset()

    if _CAPTURE_STREAM is None:
        _CAPTURE_STREAM = torch.cuda.Stream()
    if _POOL is None:
        _POOL = torch.cuda.graph_pool_handle()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(_CAPTURE_STREAM):
        g.capture_begin(_POOL)
        body()
        g.capture_end()
    # Only now is it safe to drop the previous graph: _POOL needs a live user
    # at capture_begin. See MEMORY above.
    _PREV_GRAPH = g

    reset()                    # capture leaves the buffers undefined
    for _ in range(niter):
        g.replay()

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
    global _CHOICE, _GRAPH_CHOICE

    if _CHOICE is False or not _eligible(Xg, niter, device):
        if _CHOICE is None and os.environ.get('KILOSORT_NO_FAST_KPP'):
            _CHOICE = False
            logger.info('fast kmeans_plusplus disabled by KILOSORT_NO_FAST_KPP')
        return None

    if _GRAPH_CHOICE is None and os.environ.get('KILOSORT_NO_KPP_GRAPH'):
        # Latch it, or the validation below would re-run stock on every call.
        _GRAPH_CHOICE = False
        logger.info('kmeans_plusplus CUDA graph disabled by '
                    'KILOSORT_NO_KPP_GRAPH; using the ungraphed loop')
    use_graph = _GRAPH_CHOICE is not False
    try:
        if use_graph:
            iclust, n_pos_final = _graph_loop(Xg, niter, seed, device)
        else:
            iclust, n_pos_final = _fast_loop(Xg, niter, seed, device)
    except RuntimeError as e:
        if use_graph:
            # Capture can fail for reasons that have nothing to do with this
            # centre (another stream capturing, a driver that will not graph
            # one of these kernels). Drop to the ungraphed loop for the rest of
            # the process rather than retrying 393 times.
            _GRAPH_CHOICE = False
            logger.info(f'kmeans_plusplus CUDA graph unavailable, using the '
                        f'ungraphed loop: {e}')
            try:
                iclust, n_pos_final = _fast_loop(Xg, niter, seed, device)
            except RuntimeError as e2:
                logger.debug(f'fast kmeans_plusplus fell back: {e2}')
                return None
        else:
            # Drawing NTRY candidates without replacement needs NTRY positive
            # weights; if there were fewer, stock would have drawn fewer too,
            # so this is the same fallback as n_pos_final < NTRY.
            logger.debug(f'fast kmeans_plusplus fell back: {e}')
            return None

    if int(n_pos_final) < NTRY:
        logger.debug(
            f'fast kmeans_plusplus fell back: n_pos {int(n_pos_final)} < {NTRY}, '
            f'so stock would not have drawn a full candidate set')
        return None

    if _CHOICE is None or _GRAPH_CHOICE is None:
        # Whichever path just ran gets checked against stock before anything is
        # trusted. The graph path swaps torch.multinomial for the topk/
        # exponential identity it is built on, so it needs its own check even
        # once the ungraphed loop has been cleared.
        ref = stock_loop()
        ok = torch.equal(ref, iclust)
        if use_graph:
            _GRAPH_CHOICE = ok
        if ok:
            _CHOICE = True
            logger.info(
                f'fast kmeans_plusplus enabled ({"graph" if use_graph else "eager"}): '
                f'identical to the stock loop on {iclust.numel():,} labels')
        else:
            if not use_graph:
                _CHOICE = False
            logger.info(
                f'fast kmeans_plusplus '
                f'{"CUDA graph" if use_graph else "loop"} disabled: labels '
                f'differ from the stock loop on this device')
            return None

    return iclust
