"""Regression tests for the fork optimizations that had no on/off equality test.

- KILOSORT_NO_COMPILE: the torch.compile'd template_match body against eager
  (fused detect off, else the body runs only in fused_detect's first-batch check),
  bit for bit, at MEA shapes, including the exact magnitude ties of a zeroed
  batch edge. Bit identity is a property of the generated kernels, so this test
  must pass on each new GPU or torch build before the compiled path is trusted.
- KILOSORT_TRACE_DIR: _trace is a no-op when unset and saves exact copies.
- End to end (slow, needs data): one short sort with every optimization on and
  one with every KILOSORT_NO_* switch set give identical spike tables. Each sort
  runs in its own process, because the gates latch per process.
  Set KS_REGRESSION_BIN (flat int16) and KS_REGRESSION_OPS (an ops .npy with
  'settings' and 'probe'); the test skips without them.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import kilosort.spikedetect as sd
from kilosort.run_kilosort import _trace

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')

SWITCHES = ['KILOSORT_NO_FUSED_DETECT', 'KILOSORT_NO_FUSED_PEEL', 'KILOSORT_NO_FUSED_PEAKS',
            'KILOSORT_NO_FAST_KPP', 'KILOSORT_NO_KPP_GRAPH', 'KILOSORT_NO_PEEL_COND',
            'KILOSORT_NO_PEEL_STORE', 'KILOSORT_NO_PEEL_LUT', 'KILOSORT_NO_COMPILE']


def _mea_inputs(device, seed=0):
    """template_match inputs at 512-channel MEA scale (batch_size 10000, nt 61)."""
    g = torch.Generator().manual_seed(seed)
    n_chan, NT, nt, n_temp, nC, nC2, nsize, Nfilt = 512, 10122, 61, 6, 10, 7, 3, 2048
    X = torch.randn(n_chan, NT, generator=g)
    X[:, :200] = 0.0                       # zero-padded batch edge: exact |A| ties
    wTEMP = torch.randn(n_temp, nt, generator=g)
    wTEMP = wTEMP / wTEMP.norm(dim=1, keepdim=True)
    ops = {'nt': nt, 'settings': {'nt0min': 20, 'n_templates': n_temp},
           'wTEMP': wTEMP.to(device), 'Th_universal': 9.0}
    iC = torch.randint(0, n_chan, (nC, Nfilt), generator=g)
    iC2 = torch.randint(0, Nfilt, (nC2, Nfilt), generator=g)
    weigh = torch.randn(nsize, nC, Nfilt, generator=g)
    weigh = weigh / weigh.norm(dim=1, keepdim=True)
    return X.to(device), ops, iC.to(device), iC2.to(device), weigh.to(device)


@cuda
def test_compiled_template_match_is_bit_identical_to_eager(monkeypatch):
    device = torch.device('cuda')
    X, ops, iC, iC2, weigh = _mea_inputs(device)
    monkeypatch.delenv('KILOSORT_NO_COMPILE', raising=False)
    # With fused detect on, template_match uses the Triton kernel and the body
    # runs only inside its first-batch check; turn it off to reach the body.
    monkeypatch.setenv('KILOSORT_NO_FUSED_DETECT', '1')
    saved = sd._TM_BODY
    try:
        sd._TM_BODY = sd._template_match_body
        eager = sd.template_match(X, ops, iC, iC2, weigh, device=device)
        sd._TM_BODY = None
        compiled = sd.template_match(X, ops, iC, iC2, weigh, device=device)
        assert sd._TM_BODY is not sd._template_match_body, 'compile fell back to eager; nothing tested'
        # the direct body outputs too, before any downstream thresholding
        A = sd._template_match_body(X[:, None, :64].expand(-1, 6, -1).contiguous(), weigh, iC,
                                    iC2.reshape(-1), iC2.shape[0], iC.shape[1])
        B = sd._TM_BODY(X[:, None, :64].expand(-1, 6, -1).contiguous(), weigh, iC,
                        iC2.reshape(-1), iC2.shape[0], iC.shape[1])
    finally:
        sd._TM_BODY = saved
    for a, b in zip(eager, compiled):
        assert torch.equal(a, b)
    for a, b in zip(A, B):
        assert torch.equal(a, b)


def test_no_compile_switch_forces_eager_on_gpu(monkeypatch):
    monkeypatch.setenv('KILOSORT_NO_COMPILE', '1')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    saved = sd._TM_BODY
    try:
        sd._TM_BODY = None
        X, ops, iC, iC2, weigh = _mea_inputs(torch.device('cpu'))
        Bsl = X[:, None, :32].expand(-1, 6, -1).contiguous()
        sd._template_match_body_dispatch(Bsl, weigh, iC, iC2.reshape(-1), iC2.shape[0], iC.shape[1])
        assert sd._TM_BODY is sd._template_match_body
    finally:
        sd._TM_BODY = saved


def test_trace_is_a_noop_when_unset(monkeypatch, tmp_path):
    monkeypatch.delenv('KILOSORT_TRACE_DIR', raising=False)
    monkeypatch.chdir(tmp_path)
    _trace('stage_x', st=np.arange(5))
    assert list(tmp_path.iterdir()) == []


def test_trace_saves_exact_copies(monkeypatch, tmp_path):
    out = tmp_path / 'trace'
    monkeypatch.setenv('KILOSORT_TRACE_DIR', str(out))
    st = np.random.default_rng(0).random((100, 6))
    clu = torch.arange(100, dtype=torch.int32)
    _trace('stage_x', st=st, clu=clu, fs=20000.0)
    z = np.load(out / 'stage_x.npz')
    assert np.array_equal(z['st'], st) and z['st'].dtype == st.dtype
    assert np.array_equal(z['clu'], clu.numpy()) and z['clu'].dtype == np.int32
    assert float(z['fs']) == 20000.0


_RUNNER = r'''
import json, sys, numpy as np, torch
from kilosort.run_kilosort import run_kilosort
cfg = json.loads(sys.argv[1])
ops = np.load(cfg['ops'], allow_pickle=True).item()
s = dict(ops['settings']); s['filename'] = cfg['bin']; s['results_dir'] = cfg['out']
s.pop('data_dir', None)
torch.use_deterministic_algorithms(True, warn_only=True)
run_kilosort(settings=s, probe=ops['probe'], filename=cfg['bin'], results_dir=cfg['out'],
             data_dtype='int16', invert_sign=bool(ops.get('invert_sign', True)),
             do_CAR=bool(ops.get('do_CAR', False)), save_extra_vars=False, save_plots=False)
'''


def _sort(tmp, name, env_extra):
    out = tmp / name
    env = {k: v for k, v in os.environ.items() if k not in SWITCHES and k != 'KILOSORT_TRACE_DIR'}
    env.update(env_extra, CUBLAS_WORKSPACE_CONFIG=':4096:8', KILOSORT_NO_PLOTS='1')
    cfg = json.dumps(dict(ops=os.environ['KS_REGRESSION_OPS'], bin=str(tmp / 'slice.bin'), out=str(out)))
    r = subprocess.run([sys.executable, '-c', _RUNNER, cfg], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return out


@pytest.mark.slow
@cuda
@pytest.mark.skipif(not (os.environ.get('KS_REGRESSION_BIN') and os.environ.get('KS_REGRESSION_OPS')),
                    reason='set KS_REGRESSION_BIN and KS_REGRESSION_OPS')
def test_all_optimizations_on_vs_off_give_identical_sorts(tmp_path):
    ops = np.load(os.environ['KS_REGRESSION_OPS'], allow_pickle=True).item()
    n_chan = int(ops['settings']['n_chan_bin'])
    seconds = float(os.environ.get('KS_REGRESSION_SECONDS', 60))
    n = int(seconds * float(ops['settings'].get('fs', 20000)))
    src = np.memmap(os.environ['KS_REGRESSION_BIN'], dtype=np.int16, mode='r')
    src = src[:src.size // n_chan * n_chan].reshape(-1, n_chan)
    np.ascontiguousarray(src[:n]).tofile(tmp_path / 'slice.bin')
    on = _sort(tmp_path, 'on', {})
    off = _sort(tmp_path, 'off', {s: '1' for s in SWITCHES})
    for f in ('spike_times.npy', 'spike_clusters.npy', 'templates.npy', 'amplitudes.npy'):
        a, b = np.load(on / f), np.load(off / f)
        assert a.shape == b.shape and np.array_equal(a, b), f
    assert (on / 'cluster_KSLabel.tsv').read_text() == (off / 'cluster_KSLabel.tsv').read_text()
