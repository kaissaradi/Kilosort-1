"""The sync-free kmeans_plusplus must give stock's labels, or refuse to answer.

fast_kpp removes the two device->host reads at the top of the stock loop and
replaces them with a single check after it. That check is the whole safety
argument, so what is pinned here is the check, not the kernel:

  * a normal centre gets stock's labels exactly, and leaves the global torch
    generator in the same state (the loop consumes it, and later code must not
    be able to tell which path ran);
  * a centre whose residual variance collapses -- fewer than NTRY spikes left
    with any unexplained variance -- must be REFUSED, because there stock
    draws a smaller candidate set or breaks out early, and the fast loop does
    neither;
  * the refusal has to keep working after the gate has latched True, since
    that is when nothing is checking the result any more.

Labels are int32, so `torch.equal` is exact equality here -- there are no
signed zeros to hide behind it, unlike the float comparisons in
test_fused_peel. The labels are still decided by float comparisons upstream,
so any changed bit in the loop shows up as a changed label.

Skipped without CUDA: the fast path declines on CPU and there is nothing left
to test.
"""
import numpy as np
import pytest
import torch

from kilosort import clustering_qr, fast_kpp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason='fast kmeans_plusplus is CUDA-only')

dev = torch.device('cuda')
NITER = 25          # enough to exercise the loop; 200 is the production value


@pytest.fixture(autouse=True)
def reset_choice():
    saved = (fast_kpp._CHOICE, fast_kpp._GRAPH_CHOICE)
    fast_kpp._CHOICE = None
    fast_kpp._GRAPH_CHOICE = None
    yield
    fast_kpp._CHOICE, fast_kpp._GRAPH_CHOICE = saved


def blobs(n_spikes, n_features, seed, n_blobs=40):
    """Well-separated clusters, so residual variance stays spread out and the
    loop behaves the way it does on a real centre."""
    g = torch.Generator().manual_seed(seed)
    centres = torch.randn(n_blobs, n_features, generator=g) * 20
    which = torch.randint(0, n_blobs, (n_spikes,), generator=g)
    return (centres[which] + torch.randn(n_spikes, n_features, generator=g)
            ).to(dev)


def mostly_zero_rows(n_spikes, n_features, seed, n_active=50):
    """All but n_active rows are exactly zero, so at most n_active spikes ever
    have unexplained variance and n_pos is below NTRY from the first iteration.
    Stock then draws n_active candidates instead of NTRY, which is a different
    computation, so the fast path must refuse.

    Repeating a handful of distinct rows does NOT work for this: a spike that
    is exactly its own centroid still gets vtot from torch.norm and vexp from a
    gemm, and those disagree in the last bits, so its residual stays positive.
    An exactly-zero row is zero by both routes.
    """
    g = torch.Generator().manual_seed(seed)
    Xg = torch.zeros(n_spikes, n_features)
    Xg[:n_active] = torch.randn(n_active, n_features, generator=g) * 20
    return Xg.to(dev)


def rng_state():
    return torch.cuda.get_rng_state(), torch.random.get_rng_state()


def stock_of(Xg, niter, seed):
    return clustering_qr._kmeans_plusplus_stock(Xg, niter=niter, seed=seed,
                                                device=dev)


def test_matches_stock_labels_and_rng():
    Xg = blobs(2000, 12, seed=1)
    ref = stock_of(Xg, NITER, 7)
    ref_rng = rng_state()

    got = fast_kpp.try_run(Xg, NITER, 7, dev, lambda: stock_of(Xg, NITER, 7))
    assert got is not None, 'fast path declined a well-conditioned centre'
    assert torch.equal(ref, got)
    for a, b in zip(ref_rng, rng_state()):
        assert torch.equal(a, b), 'generator left in a different state'


def test_public_entry_point_agrees_with_the_stock_body():
    """kmeans_plusplus must return the same thing whichever path it takes."""
    Xg = blobs(1500, 9, seed=2)
    ref = stock_of(Xg, NITER, 3)
    got = clustering_qr.kmeans_plusplus(Xg, niter=NITER, seed=3, device=dev)
    assert torch.equal(ref, got)


def test_still_matches_after_the_gate_has_latched():
    """Once latched, later calls run unchecked -- those are the ones that can
    go wrong unnoticed."""
    Xg = blobs(1200, 8, seed=4)
    fast_kpp.try_run(Xg, NITER, 5, dev, lambda: stock_of(Xg, NITER, 5))
    if fast_kpp._CHOICE is not True:
        pytest.skip('fast kmeans_plusplus not identical on this device')

    Xg = blobs(3000, 16, seed=6)
    ref = stock_of(Xg, NITER, 9)
    got = fast_kpp.try_run(Xg, NITER, 9, dev, lambda: pytest.fail(
        'stock loop must not be re-run once the gate has latched'))
    assert got is not None
    assert torch.equal(ref, got), 'diverged on a centre the gate never saw'


def test_refuses_when_the_candidate_pool_collapses():
    """The case the guard exists for: stock would draw fewer than NTRY
    candidates (or break), so the fast answer is not stock's answer."""
    Xg = mostly_zero_rows(600, 10, seed=11, n_active=50)
    got = fast_kpp.try_run(Xg, 200, 1, dev, lambda: pytest.fail(
        'must refuse before validating against stock'))
    assert got is None


def test_refuses_after_latching_too():
    """A latched gate must not turn the guard off."""
    Xg = blobs(1200, 8, seed=12)
    fast_kpp.try_run(Xg, NITER, 5, dev, lambda: stock_of(Xg, NITER, 5))
    if fast_kpp._CHOICE is not True:
        pytest.skip('fast kmeans_plusplus not identical on this device')
    assert fast_kpp.try_run(mostly_zero_rows(600, 10, seed=13, n_active=50),
                            200, 1, dev, lambda: None) is None


def test_the_refused_case_really_does_break_stock_early():
    """Guards the test above: if `mostly_zero_rows` ever stopped collapsing the
    pool, test_refuses_* would pass for the wrong reason."""
    Xg = mostly_zero_rows(600, 10, seed=11, n_active=50)
    vtot = torch.norm(Xg, 2, dim=1)**2
    torch.manual_seed(1)
    np.random.seed(1)
    n_pos_seen = []
    vexp0 = torch.zeros(Xg.shape[0], device=dev)
    for _ in range(200):
        v2 = torch.relu(vtot - vexp0)
        n_pos_seen.append(int((v2 > 0).sum()))
        if n_pos_seen[-1] < fast_kpp.NTRY:
            break
        isamp = torch.multinomial(v2, fast_kpp.NTRY, replacement=False)
        Xc = Xg[isamp]
        vexp = 2 * Xg @ Xc.T - (Xc**2).sum(1)
        dexp = torch.relu(vexp - vexp0.unsqueeze(1))
        imax = torch.argmax(dexp.sum(0))
        ix = dexp[:, imax] > 0
        vexp0[ix] = vexp[ix, imax]
    assert min(n_pos_seen) < fast_kpp.NTRY


def test_env_switch_and_ineligible_inputs_decline():
    import os
    Xg = blobs(1200, 8, seed=14)
    os.environ['KILOSORT_NO_FAST_KPP'] = '1'
    try:
        assert fast_kpp.try_run(Xg, NITER, 1, dev, lambda: None) is None
        assert fast_kpp._CHOICE is False
    finally:
        del os.environ['KILOSORT_NO_FAST_KPP']

    fast_kpp._CHOICE = None
    assert not fast_kpp._eligible(Xg.cpu(), NITER, torch.device('cpu'))
    assert not fast_kpp._eligible(Xg.double(), NITER, dev)
    assert not fast_kpp._eligible(Xg[:fast_kpp.NTRY - 1], NITER, dev)
    assert not fast_kpp._eligible(Xg, 0, dev)
    assert fast_kpp._eligible(Xg, NITER, dev)


# --- CUDA graph path -------------------------------------------------------
#
# The graph replaces torch.multinomial with the identity it is built on and
# pre-draws its randomness, so it needs its own checks even though the labels
# it produces are compared against stock by the same gate.


def test_graph_matches_the_ungraphed_loop():
    Xg = blobs(2500, 14, seed=21)
    ref, ref_n = fast_kpp._fast_loop(Xg, NITER, 5, dev)
    got, got_n = fast_kpp._graph_loop(Xg, NITER, 5, dev)
    assert torch.equal(ref, got)
    assert int(ref_n) == int(got_n)


def test_pre_drawn_noise_is_the_stream_multinomial_would_have_used():
    """Load-bearing for the graph: the body cannot contain the RNG, so the
    draws are made up front. Two things have to hold, and both have bitten:

      * torch.multinomial(w, k, replacement=False) IS
        topk(w / empty_like(w).exponential_(), k);
      * drawing niter vectors one row at a time gives the same stream as niter
        separate calls -- one exponential_ over the whole buffer does NOT,
        because the generator's offset advance depends on each call's numel.
    """
    n, k = 900, 100
    torch.manual_seed(31)
    sep = [torch.empty(n, device=dev).exponential_() for _ in range(4)]
    torch.manual_seed(31)
    Q = fast_kpp._draw_noise(4, n, dev)
    for j in range(4):
        assert torch.equal(sep[j].view(torch.int32), Q[j].view(torch.int32))

    torch.manual_seed(32)
    Qbig = torch.empty(4, n, device=dev).exponential_()
    assert not torch.equal(Qbig.view(torch.int32), Q.view(torch.int32)), \
        'one big exponential_ must NOT match; if it does, the trap is gone'

    w = torch.rand(n, device=dev)
    state = torch.cuda.get_rng_state()
    a = torch.multinomial(w, k, replacement=False)
    torch.cuda.set_rng_state(state)
    q = torch.empty_like(w).exponential_()
    assert torch.equal(a, torch.topk(w / q, k).indices)


def test_graph_env_switch_falls_back_to_the_eager_loop():
    import os
    Xg = blobs(1400, 10, seed=22)
    ref = stock_of(Xg, NITER, 8)
    os.environ['KILOSORT_NO_KPP_GRAPH'] = '1'
    try:
        got = fast_kpp.try_run(Xg, NITER, 8, dev, lambda: stock_of(Xg, NITER, 8))
    finally:
        del os.environ['KILOSORT_NO_KPP_GRAPH']
    assert fast_kpp._GRAPH_CHOICE is False
    assert got is not None and torch.equal(ref, got)


def test_graph_choice_latches_so_stock_is_not_re_run():
    """Both gates must clear on the first call; otherwise every later call
    would pay for a stock loop it does not need."""
    Xg = blobs(1300, 11, seed=23)
    fast_kpp.try_run(Xg, NITER, 4, dev, lambda: stock_of(Xg, NITER, 4))
    if fast_kpp._GRAPH_CHOICE is not True:
        pytest.skip('CUDA graph path unavailable on this device')
    assert fast_kpp._CHOICE is True
    Xg = blobs(1700, 11, seed=24)
    ref = stock_of(Xg, NITER, 6)
    got = fast_kpp.try_run(Xg, NITER, 6, dev, lambda: pytest.fail(
        'stock must not be re-run once both gates have latched'))
    assert got is not None and torch.equal(ref, got)
