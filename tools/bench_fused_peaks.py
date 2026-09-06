"""Bench + identity-check the fused peak-selection tail on REAL As/Amaxs.

Hooks template_match, grabs the real buffers from the first few batches, then
for each: computes the stock mask and the fused mask, compares them element by
element, and times both. Exits before the sort finishes -- this only needs a
handful of real batches.
"""
import sys
import time

import numpy as np
import torch
from torch.nn.functional import conv1d, max_pool1d

sys.path.insert(0, '/home/localadmin/Downloads/Kilosort-1')
from kilosort import spikedetect, fused_peaks              # noqa: E402
from kilosort.run_kilosort import run_kilosort             # noqa: E402

NBATCH = 8
seen = []


class Done(Exception):
    pass


_orig = spikedetect.template_match


def tm(X, ops, iC, iC2, weigh, device=torch.device('cuda'), scratch=None):
    out = _orig(X, ops, iC, iC2, weigh, device=device, scratch=scratch)
    if scratch is not None and scratch.get('As') is not None:
        seen.append((scratch['As'], scratch['Amaxs'],
                     ops['nt'], ops['settings']['nt0min'],
                     ops['Th_universal']))
        if len(seen) >= NBATCH:
            raise Done()
    return out


spikedetect.template_match = tm

prod = np.load(sys.argv[1], allow_pickle=True).item()
settings = dict(prod['settings'])
data = sys.argv[2]
settings['filename'] = data
settings['data_dir'] = None
settings.pop('probe', None)

try:
    run_kilosort(settings=settings, probe=prod['probe'], filename=data,
                 results_dir=sys.argv[3], do_CAR=prod['do_CAR'],
                 device=torch.device('cuda'), save_preprocessed_copy=False,
                 clear_cache=False)
except Done:
    pass
except Exception as e:
    print('sort stopped:', type(e).__name__, e)

print(f'\ncaptured {len(seen)} real batches')
if not seen:
    sys.exit('no batches captured')


def stock(As, Amaxs, nt, nt0, Th):
    # NO clone: this mirrors the real call site, which mutates Amaxs in place.
    # Edge zeroing is idempotent and max_pool1d returns a new tensor, so this
    # is safe to time repeatedly and is what the sort actually pays.
    Amaxs[:, :nt] = 0
    Amaxs[:, -nt:] = 0
    Am = max_pool1d(Amaxs.unsqueeze(0), (2 * nt0 + 1), stride=1, padding=nt0).squeeze(0)
    return torch.logical_and(Am == As, As > Th)


def timeit(fn, n=10):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


As, Amaxs, nt, nt0, Th = seen[0]
print(f'shapes: As {tuple(As.shape)} {As.dtype}  nt={nt} nt0={nt0} '
      f'Th={Th}  ({As.numel()/1e6:.1f} M elements, {As.numel()*4/1e6:.0f} MB each)')

# ---- identity over every captured batch, every config -------------------
print('\n=== identity check (all captured batches, all configs) ===')
allok = True
for cfg in fused_peaks._CONFIGS:
    ok_all = True
    ndiff_tot = 0
    for (A, M, _nt, _nt0, _Th) in seen:
        ref = stock(A, M, _nt, _nt0, _Th)
        out = torch.empty(A.shape, dtype=torch.bool, device=A.device)
        fused_peaks._run(A, M, out, _nt, _nt0, _Th, cfg)
        if not torch.equal(out, ref):
            ok_all = False
            ndiff_tot += int((out != ref).sum().item())
    n = sum(a.numel() for a, *_ in seen)
    print(f'  cfg {cfg}: {"IDENTICAL" if ok_all else f"DIFFERS ({ndiff_tot} elements)"} '
          f'over {n:,} elements / {len(seen)} batches')
    allok = allok and ok_all

# also confirm the resulting xy (what the caller actually uses) matches
ref = stock(As, Amaxs, nt, nt0, Th)
out = torch.empty(As.shape, dtype=torch.bool, device=As.device)
fused_peaks._run(As, Amaxs, out, nt, nt0, Th, fused_peaks._CONFIGS[0])
xr, xf = ref.nonzero(), out.nonzero()
print(f'  nonzero(): stock {tuple(xr.shape)} vs fused {tuple(xf.shape)} -> '
      f'{"IDENTICAL" if torch.equal(xr, xf) else "DIFFERS"}')

# ---- timing -------------------------------------------------------------
print('\n=== timing (ms/batch, mean of 10) ===')
ts = timeit(lambda: stock(As, Amaxs, nt, nt0, Th))
print(f'  stock tail                 {ts:8.3f}')
best = None
for cfg in fused_peaks._CONFIGS:
    o = torch.empty(As.shape, dtype=torch.bool, device=As.device)
    tf = timeit(lambda: fused_peaks._run(As, Amaxs, o, nt, nt0, Th, cfg))
    print(f'  fused cfg {str(cfg):<10}       {tf:8.3f}   {ts/tf:5.2f}x')
    if best is None or tf < best[0]:
        best = (tf, cfg)
print(f'\nBEST {best[1]}: {ts:.3f} -> {best[0]:.3f} ms  ({ts/best[0]:.2f}x, '
      f'saves {ts-best[0]:.3f} ms/batch)')

# what that is worth on the production sort
print(f'\nproduction has 4054 batches -> saves '
      f'{(ts-best[0])*4054/1000:.1f} s of the 637 s sort '
      f'({100*(ts-best[0])*4054/1000/637:.1f}%)')
print(f'identity: {"ALL CONFIGS IDENTICAL" if allok else "SOME CONFIGS DIFFER (gate handles)"}')
