"""The fused detection kernel must never be used unless it is bit-identical.

fused_detect replaces template_match's inner loop with a Triton kernel that is
6.9x faster and, on the development machine, exactly equal to the stock loop.
"Exactly equal" is a property of cuBLAS's K-split, Triton's codegen and the
block size -- BLOCK_M=128 matched on an RTX 4000 Ada while 64 and 32 did not --
so it cannot be assumed anywhere else. The whole design rests on the runtime
gate in try_fill discarding configs that disagree.

These tests pin the gate, not the kernel: whatever try_fill decides, the
buffers it leaves behind must equal the stock loop's, on the batch it
validated against AND on a later batch it did not. A gate that enabled a
config which only happened to agree on batch 0 would pass a smoke test and
silently corrupt a sort, so the second batch is the point of this file.

The small shapes below are not a scaled-down version of the same situation:
they reject BLOCK_M=128 -- the config verified at production shapes -- and the
gate falls through to (64, 2). That is the intended behaviour and the reason
the fallback ladder exists, so do not "simplify" these shapes to match
production and do not assert a particular config.

Skipped without CUDA or without Triton, which is the fallback case anyway.
"""
import numpy as np
import pytest
import torch

from kilosort import fused_detect, spikedetect

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_detect._HAVE_TRITON,
    reason='fused detection needs CUDA and Triton')


NCHAN, NK, NT, NFILT, NC, NC2, NS = 64, 10, 512, 128, 10, 20, 5


def make_problem(seed):
    """Production-shaped but small. Real detection state is not needed -- the
    kernel only cares about shapes, dtypes and the index tables."""
    dev = torch.device('cuda')
    g = torch.Generator(device='cpu').manual_seed(seed)
    B = torch.randn(NCHAN, NK, NT, generator=g).to(dev)
    weigh = torch.randn(NS, NC, NFILT, generator=g).to(dev)
    iC = torch.randint(0, NCHAN, (NC, NFILT), generator=g).to(dev)
    iC2 = torch.randint(0, NFILT, (NC2, NFILT), generator=g).to(dev)
    return B, weigh, iC, iC2.reshape(-1).contiguous()


def buffers():
    dev = torch.device('cuda')
    return (torch.empty((NFILT, NT), device=dev),
            torch.empty((NFILT, NT), dtype=torch.int32, device=dev),
            torch.empty((NFILT, NT), device=dev))


def stock(B, weigh, iC, iC2_flat, As, imaxs, Amaxs):
    """The stock body, untiled -- tiling is irrelevant here and the fused path
    is untiled by construction."""
    Aa, imax, Amax = spikedetect._template_match_body(
        B, weigh, iC, iC2_flat, NC2, NFILT)
    As.copy_(Aa)
    imaxs.copy_(imax.to(torch.int32))
    Amaxs.copy_(Amax)


@pytest.fixture(autouse=True)
def reset_choice():
    """_CHOICE is process-global and sticky by design; tests must not inherit
    each other's verdict."""
    saved = fused_detect._CHOICE
    fused_detect._CHOICE = None
    yield
    fused_detect._CHOICE = saved


def test_gate_result_matches_stock_on_the_validated_batch():
    B, weigh, iC, iC2_flat = make_problem(0)
    As, imaxs, Amaxs = buffers()
    filled = fused_detect.try_fill(
        B, weigh, iC, iC2_flat, NC2, NFILT, As, imaxs, Amaxs,
        lambda: stock(B, weigh, iC, iC2_flat, As, imaxs, Amaxs))
    assert filled, 'first call must leave the buffers filled'

    rA, rI, rM = buffers()
    stock(B, weigh, iC, iC2_flat, rA, rI, rM)
    assert torch.equal(As, rA)
    assert torch.equal(imaxs, rI)
    assert torch.equal(Amaxs, rM)


def test_enabled_config_still_matches_on_a_later_batch():
    B, weigh, iC, iC2_flat = make_problem(0)
    As, imaxs, Amaxs = buffers()
    fused_detect.try_fill(
        B, weigh, iC, iC2_flat, NC2, NFILT, As, imaxs, Amaxs,
        lambda: stock(B, weigh, iC, iC2_flat, As, imaxs, Amaxs))
    if not fused_detect._CHOICE:
        pytest.skip('no bit-identical config on this device; stock is used')

    # Fresh data the gate never saw. Same weigh/iC so the kernel keeps the
    # shapes it validated at, which is what a real sort does batch to batch.
    B2 = torch.randn(NCHAN, NK, NT,
                     generator=torch.Generator().manual_seed(7)).cuda()
    filled = fused_detect.try_fill(
        B2, weigh, iC, iC2_flat, NC2, NFILT, As, imaxs, Amaxs,
        lambda: pytest.fail('stock_fill must not be called after the gate '
                            'has chosen a config'))
    assert filled

    rA, rI, rM = buffers()
    stock(B2, weigh, iC, iC2_flat, rA, rI, rM)
    assert torch.equal(As, rA), 'enabled config diverged on an unvalidated batch'
    assert torch.equal(imaxs, rI)
    assert torch.equal(Amaxs, rM)


def test_env_switch_forces_stock(monkeypatch):
    monkeypatch.setenv('KILOSORT_NO_FUSED_DETECT', '1')
    B, weigh, iC, iC2_flat = make_problem(1)
    As, imaxs, Amaxs = buffers()
    called = []
    filled = fused_detect.try_fill(
        B, weigh, iC, iC2_flat, NC2, NFILT, As, imaxs, Amaxs,
        lambda: called.append(1))
    assert filled is False, 'caller must be told to run the stock loop itself'
    assert not called, 'try_fill must not run stock_fill when disabled'
    assert fused_detect._CHOICE is False


def test_ineligible_dtypes_fall_back():
    B, weigh, iC, iC2_flat = make_problem(2)
    As, _, Amaxs = buffers()
    imaxs64 = torch.empty((NFILT, NT), dtype=torch.int64, device='cuda')
    filled = fused_detect.try_fill(
        B, weigh, iC, iC2_flat, NC2, NFILT, As, imaxs64, Amaxs,
        lambda: pytest.fail('stock_fill must not run during eligibility check'))
    assert filled is False


def test_next_pow2():
    assert [fused_detect._next_pow2(n) for n in (1, 2, 3, 50, 64, 65)] == \
        [1, 2, 4, 64, 64, 128]
