"""Count the TRUE candidate set for Astra #1, not the lower bound.

KS4 keeps a peak only where BOTH tests pass:

    mask = (Amaxs == As) & (As > Th_universal)

The previous measurement counted `mask.nonzero()` -- peaks -- and reported
381x as an explicit UPPER bound, because peaks are a lower bound on
candidates. The number that actually decides whether a candidate-only kernel
is worth writing is the left operand's own population:

    candidates = (As > Th_universal).sum()

because that, not the peak count, is how many columns a deferred spatial
reduction would still have to visit. `As` does not depend on `Amax`, so the
reduction can be deferred to those columns -- but only if there are few of
them.

This patches fused_peaks.try_mask, which sees As and Th and nothing else it
would perturb: it counts, then delegates to the real implementation, so the
sort it runs is byte-identical to an unpatched one.
"""
import os, sys
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
sys.path.insert(0, '/home/localadmin/Downloads/Kilosort-1')
import numpy as np, torch
from kilosort import spikedetect
from kilosort import fused_peaks

rows = []
_orig_try = fused_peaks.try_mask


def counting_try_mask(As, Amaxs, nt, nt0, Th):
    # As is (Nfilt, NT). The edge zeroing that the reference path applies to
    # Amaxs never touches As, so this count is the same in both paths.
    cand = int((As > Th).sum().item())
    out = _orig_try(As, Amaxs, nt, nt0, Th)
    peaks = None if out is None else int(out.sum().item())
    rows.append((As.shape[0], As.shape[1], cand, peaks))
    return out


fused_peaks.try_mask = counting_try_mask

from kilosort.run_kilosort import run_kilosort

ops = np.load(sys.argv[1], allow_pickle=True).item()
KEYS = ('n_chan_bin', 'fs', 'nt', 'Th_universal', 'Th_learned', 'dmin', 'dminx',
        'nearest_chans', 'nearest_templates', 'batch_size', 'nblocks', 'n_pcs')
settings = {k: ops[k] for k in KEYS if k in ops}
settings['n_chan_bin'] = int(ops['n_chan_bin'])

run_kilosort(settings=settings, probe=ops['probe'], filename=sys.argv[2],
             data_dtype='int16', device=torch.device('cuda'), invert_sign=True,
             do_CAR=False, results_dir=sys.argv[3], save_plots=False)

a = np.array([(r[0], r[1], r[2], -1 if r[3] is None else r[3]) for r in rows],
             dtype=np.int64)
Nfilt, NT, cand, peaks = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
dense_cols = (Nfilt.astype(float) * NT)          # columns the dense max reduces
cand_f = cand.astype(float)
print(f"\ncalls probed          : {len(a)}")
print(f"Nfilt x NT            : {Nfilt[0]} x {NT[0]}")
print(f"dense columns / call  : {dense_cols.mean():.3e}")
print(f"candidates (median)   : {np.median(cand_f):.0f}")
print(f"candidates (mean)     : {cand_f.mean():.1f}")
print(f"candidates (max)      : {cand_f.max():.0f}")
print(f"peaks kept (median)   : {np.median(peaks[peaks >= 0]) if (peaks >= 0).any() else 'n/a'}")
print(f"candidate fraction    : {cand_f.sum() / dense_cols.sum():.3e}")
print(f"TRUE reduction ratio  : {dense_cols.sum() / max(cand_f.sum(), 1):.1f}x")
np.save(sys.argv[3] + '/candidate_counts.npy', a)
