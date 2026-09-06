#!/usr/bin/env python
"""Run one full kilosort4 sort on a flat .bin and print its wall time.

This is the harness both arms of every end-to-end A/B in KS4_VALIDATION_NOTES
were run through. A change is only accepted once two runs of it -- one with the
optimization on, one with `KILOSORT_NO_*=1` -- produce byte-identical result
directories under `tools/compare_sorts.py`, so this tool exists to make those
two runs differ in nothing but the environment variable.

Settings come from one of two places:

  --ops OPS.npy       replay a previous sort's saved ops: its `settings`,
                      `probe` and `do_CAR` verbatim. This is the way to
                      reproduce an existing production sort exactly, because
                      it cannot drift from what that sort actually used.

  --probe P.mat       start from a probe file plus `--settings` JSON. Use this
    --settings S.json for a recording that has no prior kilosort4 sort to
                      replay -- a new array geometry, say.

Exactly one of the two is required; there is deliberately no default settings
profile here, because a silent default is how an A/B ends up comparing two
different sorts.

    python tools/run_full_sort.py --ops prod_ops.npy \
        --data slice300.bin --results-dir /tmp/run_a

    KILOSORT_NO_FUSED_PEAKS=1 python tools/run_full_sort.py ... --results-dir /tmp/run_b
    python tools/compare_sorts.py /tmp/run_a /tmp/run_b
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# The lab pipeline's run_kilosort4.py sets this before importing torch, so the
# harness has to as well: without it an A/B can OOM where production does not,
# and the fragmentation behaviour is part of what is being timed.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kilosort.run_kilosort import run_kilosort   # noqa: E402

SWITCHES = ['KILOSORT_NO_FUSED_DETECT', 'KILOSORT_NO_FUSED_PEEL',
            'KILOSORT_NO_FUSED_PEAKS', 'KILOSORT_NO_FAST_KPP',
            'KILOSORT_NO_KPP_GRAPH']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', required=True, help='flat int16 .bin')
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--ops', help='a previous sort\'s ops.npy to replay')
    ap.add_argument('--probe', help='probe .mat (with --settings)')
    ap.add_argument('--settings', help='settings JSON (with --probe)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--invert-sign', action='store_true',
                    help='the lab pipeline passes invert_sign=True for Litke')
    ap.add_argument('--do-car', action='store_true')
    args = ap.parse_args()

    if bool(args.ops) == bool(args.probe):
        ap.error('give exactly one of --ops or --probe')
    if args.probe and not args.settings:
        ap.error('--probe requires --settings')

    if args.ops:
        prod = np.load(args.ops, allow_pickle=True).item()
        settings = dict(prod['settings'])
        probe = prod['probe']
        do_CAR = prod['do_CAR']
        invert_sign = prod.get('invert_sign', args.invert_sign)
    else:
        from kilosort import io
        probe = io.load_probe(args.probe)
        settings = json.loads(Path(args.settings).read_text())
        do_CAR = args.do_car
        invert_sign = args.invert_sign

    settings['filename'] = args.data
    settings['data_dir'] = None
    settings.pop('probe', None)

    # Print the switch state so the log itself records which arm this was; an
    # A/B whose two logs do not differ here compared the same thing twice.
    on = {k: os.environ[k] for k in SWITCHES if os.environ.get(k)}
    print(f'off-switches set: {on if on else "none (all optimizations active)"}')
    print(f'n_chan={settings.get("n_chan_bin")} fs={settings.get("fs")} '
          f'batch_size={settings.get("batch_size")}')

    t0 = time.time()
    run_kilosort(settings=settings, probe=probe, filename=args.data,
                 results_dir=args.results_dir, data_dtype='int16',
                 do_CAR=do_CAR, invert_sign=invert_sign,
                 device=torch.device(args.device), save_extra_vars=False,
                 save_preprocessed_copy=False, clear_cache=False)
    print(f'\nTOTAL_WALL_SECONDS {time.time() - t0:.2f}')


if __name__ == '__main__':
    main()
