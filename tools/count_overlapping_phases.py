#!/usr/bin/env python
"""Count peel phases whose windows OVERLAP, per sort.

Why this exists. fused_peel documents that stock's `-=` through advanced
indexing is last-write-wins on duplicate indices, and that running it twice on
identical inputs differed on 152-838 elements. That makes the overlapping-window
phase a NONDETERMINISTIC operation inside kilosort4 itself -- and fused_peel
deliberately falls back to it, because reproducing stock is the requirement.

That is a candidate root cause for the long-standing run-to-run wobble, which
was seen on 20260724A at production scale and NOT on 20260514A's full run. If
overlapping phases occur on the first recording and not the second, the two
facts line up.

This does not prove causation on its own -- it tests whether the necessary
condition is present where the wobble is and absent where it is not.

    python tools/count_overlapping_phases.py --data X.bin --ops ops.npy
"""
import argparse
import sys
import time

sys.path.insert(0, '/home/localadmin/Downloads/Kilosort-1')
import numpy as np
import torch

from kilosort import fused_peel
from kilosort.run_kilosort import run_kilosort

STATS = {'peels': 0, 'overlap': 0, 'amp_nonpos': 0}
_orig_disjoint = fused_peel.phases_disjoint


def counting_disjoint(pos_all, nt):
    r = _orig_disjoint(pos_all, nt)
    STATS['peels'] += 1
    if not bool(r):                    # extra sync; instrumented run only
        STATS['overlap'] += 1
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ops', required=True)
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--label', default='')
    a = ap.parse_args()

    fused_peel.phases_disjoint = counting_disjoint
    prod = np.load(a.ops, allow_pickle=True).item()
    settings = dict(prod['settings'])
    settings['filename'] = a.data
    settings['data_dir'] = None
    settings.pop('probe', None)

    t0 = time.time()
    run_kilosort(settings=settings, probe=prod['probe'], filename=a.data,
                 results_dir=a.results_dir, do_CAR=prod['do_CAR'],
                 device=torch.device('cuda'), save_preprocessed_copy=False,
                 clear_cache=False)
    wall = time.time() - t0

    n, o = STATS['peels'], STATS['overlap']
    print(f'\n=== overlapping-phase census {a.label} ({wall:.1f}s) ===')
    print(f'  peels with a disjointness check : {n:,}')
    print(f'  peels with OVERLAPPING windows  : {o:,}'
          f'   ({100 * o / max(n, 1):.4f}%)')
    print('  -> stock scatter is last-write-wins on those, i.e. '
          'nondeterministic' if o else
          '  -> every phase disjoint: the nondeterministic path never ran')


if __name__ == '__main__':
    main()
