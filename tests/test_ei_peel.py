"""ei_peel: the EI peel is the stock peel when the EIs are the stock templates' time-domain waveforms."""
import numpy as np
import pytest
import torch

from kilosort import ei_peel, template_matching as tm


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs cuda')
def test_ei_peel_equals_stock_on_pc_templates():
    torch.manual_seed(0); rng = np.random.default_rng(0); dev = torch.device('cuda')
    nt, P, C, n = 61, 6, 40, 30
    W = torch.linalg.qr(torch.randn(nt, P))[0].T.contiguous().to(dev)
    ops = dict(nt=nt, wPCA=W, Th_learned=8., max_peels=100, nt0min=20)
    U = torch.zeros(n, P, C, device=dev)
    for i in range(n):
        c = int(rng.integers(0, C - 5)); U[i, :, c:c + 5] = torch.randn(P, 5, device=dev) * 3
    ctc0, cache0 = tm.prepare_matching(ops, U, return_cache=True)
    E = cache0['U_time'].clone()
    ctc1, cache1 = ei_peel.ei_matching(ops, E)
    assert torch.allclose(ctc0, ctc1, rtol=1e-3, atol=1e-2)
    assert torch.allclose(ei_peel.ei_pcs(ops, E), U, atol=1e-4)
    X = torch.randn(C, 5000, device=dev)
    for _ in range(60):
        i = int(rng.integers(n)); t = int(rng.integers(100, 4800)); X[:, t - nt // 2:t + nt // 2 + 1] += 2 * E[i]
    X1 = X.clone()
    st0, a0, _, _ = tm.run_matching(ops, X, U, ctc0, device=dev, unit_cache=cache0)
    st1, a1, _, _ = tm.run_matching(ops, X1, U, ctc1, device=dev, unit_cache=cache1)
    assert torch.equal(st0, st1)
    assert torch.allclose(a0, a1, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs cuda')
def test_merge_eis_joins_shifted_copy_only():
    dev = torch.device('cuda'); torch.manual_seed(1)
    env = torch.exp(-((torch.arange(61, device=dev) - 25.) / 6) ** 2)          # waveforms that end inside the window
    a = torch.randn(1, 20, 61, device=dev) * env; b = torch.roll(a, 7, -1); c = torch.randn(1, 20, 61, device=dev) * env
    E = torch.cat((a, b, c)); cnt = torch.tensor([300., 100., 300.], device=dev)
    keep, target = ei_peel.merge_eis(E, cnt)
    assert keep.tolist() == [True, False, True] and target.tolist() == [0, 0, 2]

