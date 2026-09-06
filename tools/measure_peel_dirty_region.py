"""Measure how much of the learned pass's per-peel full-width work is redundant.

Every peel iteration recomputes torch.max(B,0), max_pool1d, the comparisons and
nonzero over all NT columns. But peel_subtract only writes B[:, iX +/- nt], so
the detection condition can only change within +/- 2nt of the previous
iteration's spikes. This script measures the size of that dirty set against NT,
which is the ceiling on what a dirty-region rewrite can save.

Read-only: it records shapes, it does not alter any value.
"""
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, '/home/localadmin/Downloads/Kilosort-1')
from kilosort import template_matching, fused_peel  # noqa: E402
from kilosort.run_kilosort import run_kilosort      # noqa: E402

STATS = []
_state = {'iter': 0, 'nt': None, 'NT': None}

_orig_run_matching = template_matching.run_matching
_orig_peel_subtract = fused_peel.peel_subtract


def run_matching_instr(ops, X, U, ctc, device=torch.device('cuda'), unit_cache=None):
    _state['iter'] = 0
    _state['nt'] = ops['nt']
    _state['NT'] = int(X.shape[-1])
    return _orig_run_matching(ops, X, U, ctc, device=device, unit_cache=unit_cache)


def peel_subtract_instr(Xres, B, iX, iY, amp, U_time, ctc, ctc_p, tiwave,
                        trange, nt, n=2):
    NT = int(B.shape[-1])
    pos = iX[:, 0].detach().to(torch.int64)
    nsp = int(pos.numel())
    # Union of [p-2nt, p+2nt] clamped to [0, NT). Exact via a boolean mask.
    if nsp:
        mask = torch.zeros(NT, dtype=torch.bool, device=B.device)
        lo = torch.clamp(pos - 2 * nt, min=0)
        hi = torch.clamp(pos + 2 * nt + 1, max=NT)
        # scatter the interval ends, then cumulative-sum to fill
        delta = torch.zeros(NT + 1, dtype=torch.int32, device=B.device)
        delta.scatter_add_(0, lo, torch.ones_like(lo, dtype=torch.int32))
        delta.scatter_add_(0, hi, -torch.ones_like(hi, dtype=torch.int32))
        mask = torch.cumsum(delta[:NT], 0) > 0
        dirty = int(mask.sum().item())
    else:
        dirty = 0
    STATS.append((_state['iter'], nsp, dirty, NT, int(B.shape[0])))
    _state['iter'] += 1
    return _orig_peel_subtract(Xres, B, iX, iY, amp, U_time, ctc, ctc_p,
                               tiwave, trange, nt, n)


template_matching.run_matching = run_matching_instr
fused_peel.peel_subtract = peel_subtract_instr

prod = np.load(sys.argv[1], allow_pickle=True).item()
settings = dict(prod['settings'])
data = sys.argv[2]
settings['filename'] = data
settings['data_dir'] = None
settings.pop('probe', None)

t0 = time.time()
run_kilosort(settings=settings, probe=prod['probe'], filename=data,
             results_dir=sys.argv[3], do_CAR=prod['do_CAR'],
             device=torch.device('cuda'), save_preprocessed_copy=False,
             clear_cache=False)
print(f'\nTOTAL_WALL_SECONDS {time.time() - t0:.2f}')

arr = np.array(STATS, dtype=np.int64)
np.save(sys.argv[4], arr)
it, nsp, dirty, NT, nunits = arr.T

print(f'\n=== peel-loop dirty-region census: {len(arr):,} peel iterations ===')
print(f'n_units (median)         : {np.median(nunits):.0f}')
print(f'NT (median)              : {np.median(NT):.0f}')

# Work model: iteration 0 of each batch must be full width (no prior state).
# Every later iteration only needs its dirty columns.
first = it == 0
cols_stock = NT.sum()
cols_opt = np.where(first, NT, dirty).sum()
print(f'\ncolumns scanned, stock   : {cols_stock:,}')
print(f'columns scanned, dirty   : {cols_opt:,}')
print(f'REDUCTION                : {cols_stock / max(cols_opt, 1):.2f}x '
      f'({100 * (1 - cols_opt / cols_stock):.1f}% of the work is provably redundant)')

print('\nby iteration index (how the dirty set collapses as the peel converges):')
print(f'{"iter":>5} {"calls":>7} {"med spikes":>11} {"med dirty":>10} {"med NT":>8} {"dirty %":>8}')
for i in range(0, min(50, it.max() + 1)):
    m = it == i
    if not m.any():
        continue
    if i % 5 == 0 or i < 6:
        print(f'{i:>5} {m.sum():>7} {np.median(nsp[m]):>11.0f} '
              f'{np.median(dirty[m]):>10.0f} {np.median(NT[m]):>8.0f} '
              f'{100 * np.median(dirty[m]) / np.median(NT[m]):>7.1f}%')
