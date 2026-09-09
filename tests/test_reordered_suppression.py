"""Exact-mask properties and integration/gate tests for the opt-in experiment."""
import numpy as np
import pytest
import torch

from kilosort import fused_detect, fused_peaks, reordered_suppression, spikedetect


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_masks_and_output_order(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(13)
    for repeat in range(60):
        nf, length = int(rng.integers(1,30)), int(rng.integers(10,130))
        nc = int(rng.integers(1,nf+1))
        nt, radius = int(rng.integers(1,length//2+3)), int(rng.integers(0,12))
        scores = torch.tensor(rng.integers(0,10,(nf,length)), dtype=torch.float32, device=device)
        neighbors = torch.tensor(rng.integers(0,nf,(nc,nf)), device=device)
        if repeat % 2:
            neighbors[0] = torch.arange(nf, device=device)
        threshold = float(rng.choice([0,1,5,9]))
        spatial = scores[neighbors].max(0).values
        ref = fused_peaks._stock_mask(scores, spatial, nt, radius, threshold)
        before = scores.clone()
        got = reordered_suppression.mask(scores, neighbors, nt, radius, threshold,
                                          workspace=torch.empty_like(scores), chunk_size=17)
        assert torch.equal(ref, got), repeat
        assert torch.equal(ref.nonzero(), got.nonzero()), repeat
        assert torch.equal(before, scores)


def test_edges_ties_and_empty_candidates():
    # No self-neighbors: a candidate in the zeroed edge can still match a
    # neighbor's interior maximum. Do not discard edge candidates wholesale.
    scores = torch.zeros(2,20)
    scores[0,2] = scores[1,4] = 5
    neighbors = torch.tensor([[1,0]])
    ref = fused_peaks._stock_mask(scores, scores[neighbors].max(0).values, 4, 2, 1)
    got = reordered_suppression.mask(scores, neighbors, 4, 2, 1)
    assert got[0,2] and torch.equal(ref, got)
    assert not reordered_suppression.mask(scores, neighbors, 4, 2, 100).any()


def test_workspace_alias_and_chunk_validation():
    scores = torch.ones(2,20)
    neighbors = torch.tensor([[0,1]])
    with pytest.raises(ValueError, match='share storage'):
        reordered_suppression.mask(scores, neighbors, 4, 2, 1, workspace=scores.view_as(scores))
    with pytest.raises(ValueError, match='chunk_size'):
        reordered_suppression.mask(scores, neighbors, 4, 2, 1, chunk_size=0)


def test_nonfinite_scores_and_signed_zeros():
    scores = torch.zeros(3,30)
    scores[0,8] = float('nan')
    scores[1,14] = float('inf')
    scores[2,10] = -0.0
    neighbors = torch.tensor([[0,1,2], [1,2,0]])
    ref = fused_peaks._stock_mask(scores, scores[neighbors].max(0).values, 4, 2, 0)
    got = reordered_suppression.mask(scores, neighbors, 4, 2, 0)
    assert torch.equal(ref, got)


def test_score_only_dispatch_omits_spatial_kernel(monkeypatch):
    # Tests launch selection without pretending to execute CUDA arithmetic.
    from types import SimpleNamespace
    launches = []

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            return lambda *a, **k: launches.append(self.name)

    monkeypatch.setattr(fused_detect, 'triton', SimpleNamespace(cdiv=lambda a,b: (a+b-1)//b), raising=False)
    monkeypatch.setattr(fused_detect, '_tm_fused_kernel', Kernel('scores'), raising=False)
    monkeypatch.setattr(fused_detect, '_amax_kernel', Kernel('spatial'), raising=False)
    B, weights, iC = torch.ones(4,2,20), torch.ones(2,3,5), torch.zeros(3,5,dtype=torch.long)
    A = torch.empty(5,20)
    common = (B, weights, iC, torch.zeros(10,dtype=torch.long), 2, 5, A, A, A, (32,2))
    fused_detect._run(*common)
    assert launches == ['scores','spatial']
    launches.clear()
    fused_detect._run(*common, compute_amax=False)
    assert launches == ['scores']


def problem(device='cpu'):
    gen = torch.Generator().manual_seed(42)
    X = torch.randn(16,513,generator=gen).to(device)
    ops = {'nt':21, 'settings':{'nt0min':10,'n_templates':4},
           'wTEMP':torch.randn(4,21,generator=gen).to(device), 'Th_universal':3.}
    iC = torch.randint(16,(5,20),generator=gen).to(device)
    iC2 = torch.randint(20,(6,20),generator=gen).to(device)
    weights = torch.randn(3,5,20,generator=gen).to(device)
    return X, ops, iC, iC2, weights


def assert_bytes(a, b):
    for x,y in zip(a,b):
        assert x.shape == y.shape
        assert x.contiguous().cpu().numpy().tobytes() == y.contiguous().cpu().numpy().tobytes()


@pytest.mark.parametrize('mode', ['1','check'])
def test_full_function_and_validation_lifecycle(monkeypatch, mode):
    args = problem()
    calls = []

    def fill(*args, compute_amax=True):
        calls.append(compute_amax)
        # Emulate the validated fused fill on CPU. Poison the omitted output
        # so accidentally consuming it is caught by full-output comparisons.
        args[-1]()
        if not compute_amax:
            args[-2].fill_(float('nan'))
        return True

    monkeypatch.setattr(fused_detect, 'try_fill', fill)
    monkeypatch.setattr(fused_peaks, 'try_mask', lambda *a: None)
    monkeypatch.delenv('KILOSORT_REORDERED_SUPPRESSION', raising=False)
    ref = spikedetect.template_match(*args, device=torch.device('cpu'))
    monkeypatch.setenv('KILOSORT_REORDERED_SUPPRESSION', mode)
    scratch = {}
    calls.clear()
    for _ in range(2):
        got = spikedetect.template_match(*args, device=torch.device('cpu'), scratch=scratch)
        assert_bytes(ref, got)
    assert calls == ([True,False] if mode == '1' else [True,True])
    # Reusing scratch with an in-place neighborhood edit must revalidate.
    args[3][0,0] = (args[3][0,0] + 1) % 20
    spikedetect.template_match(*args, device=torch.device('cpu'), scratch=scratch)
    assert calls[-1] is True
    fresh = {}
    spikedetect.template_match(*args, device=torch.device('cpu'), scratch=fresh)
    assert calls[-1] is True


def test_mismatch_falls_back_and_check_mode_raises(monkeypatch):
    args = problem()
    monkeypatch.setattr(fused_detect, 'try_fill', lambda *a, **k: False)
    monkeypatch.setattr(fused_peaks, 'try_mask', lambda *a: None)
    monkeypatch.delenv('KILOSORT_REORDERED_SUPPRESSION', raising=False)
    ref = spikedetect.template_match(*args, device=torch.device('cpu'))
    real = reordered_suppression.mask
    count = []

    def wrong(*a, **k):
        count.append(1)
        return ~real(*a, **k)

    monkeypatch.setattr(reordered_suppression, 'mask', wrong)
    monkeypatch.setenv('KILOSORT_REORDERED_SUPPRESSION', '1')
    scratch = {}
    for _ in range(2):
        assert_bytes(ref, spikedetect.template_match(*args, device=torch.device('cpu'), scratch=scratch))
    assert len(count) == 1
    monkeypatch.setenv('KILOSORT_REORDERED_SUPPRESSION', 'check')
    with pytest.raises(RuntimeError, match='differs'):
        spikedetect.template_match(*args, device=torch.device('cpu'), scratch=scratch)


def test_cuda_fused_integration(monkeypatch):
    if not torch.cuda.is_available() or not fused_detect._HAVE_TRITON:
        pytest.skip('CUDA and Triton required')
    monkeypatch.setattr(fused_detect, '_CHOICE', None)
    monkeypatch.setattr(fused_peaks, '_CHOICE', None)
    monkeypatch.delenv('KILOSORT_REORDERED_SUPPRESSION', raising=False)
    args = problem('cuda')
    ref = spikedetect.template_match(*args, device=torch.device('cuda'))
    if not fused_detect._CHOICE:
        pytest.skip('No validated fused score kernel on this GPU')
    monkeypatch.setenv('KILOSORT_REORDERED_SUPPRESSION', '1')
    scratch = {}
    for _ in range(2):
        assert_bytes(ref, spikedetect.template_match(*args, device=torch.device('cuda'), scratch=scratch))
