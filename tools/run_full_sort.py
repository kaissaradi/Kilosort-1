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
            'KILOSORT_NO_KPP_GRAPH', 'KILOSORT_REORDERED_SUPPRESSION']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', required=True, help='flat int16 .bin')
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--ops', help='a previous sort\'s ops.npy to replay')
    ap.add_argument('--probe', help='probe .mat (with --settings)')
    ap.add_argument('--settings', help='settings JSON (with --probe)')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='override one settings key after --ops/--settings is '
                         'loaded, e.g. --set batch_size=60000. Values parse as '
                         'JSON, falling back to str. Repeatable. The override '
                         'is printed and written to settings_override.json in '
                         'the results dir -- a settings sweep that does not '
                         'record its own arm is indistinguishable from a '
                         'wobble. Only keys already present are accepted, so a '
                         'typo fails loudly instead of being silently ignored.')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--invert-sign', action='store_true',
                    help='the lab pipeline passes invert_sign=True for Litke')
    ap.add_argument('--do-car', action='store_true')
    ap.add_argument('--no-pc-features', action='store_true',
                    help='skip make_pc_features and the pc_features.npy / '
                         'pc_feature_ind.npy writes. Only Phy feature views '
                         'read them, and the MEA pipeline deletes both right '
                         'after the sort. tF is consumed by spike positions '
                         'and amplitudes BEFORE this block and never after, '
                         'so no other output file can change.')
    ap.add_argument('--deterministic', action='store_true',
                    help='force deterministic CUDA algorithms. Diagnostic for '
                         'the run-to-run wobble: if two runs with this flag '
                         'agree where two runs without it do not, the wobble '
                         'is a nondeterministic reduction (index_add / '
                         'scatter_add / atomics), not anything shape- or '
                         'data-dependent. warn_only=True so ops with no '
                         'deterministic implementation degrade instead of '
                         'raising -- which means a CLEAN result here is '
                         'informative and a dirty one is not conclusive.')
    args = ap.parse_args()

    if args.deterministic:
        # cuBLAS needs this set before the first handle is created, or its
        # own reductions stay nondeterministic regardless of the torch flag.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False

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

    overrides = {}
    for item in args.set:
        if '=' not in item:
            ap.error(f'--set expects KEY=VALUE, got {item!r}')
        key, raw = item.split('=', 1)
        if key not in settings:
            ap.error(f'--set {key}: not a key of the loaded settings '
                     f'(typo?). Known keys: {sorted(settings)}')
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        overrides[key] = val
        settings[key] = val

    settings['filename'] = args.data
    settings['data_dir'] = None
    settings.pop('probe', None)

    # Print the switch state so the log itself records which arm this was; an
    # A/B whose two logs do not differ here compared the same thing twice.
    on = {k: os.environ[k] for k in SWITCHES if os.environ.get(k)}
    print(f'off-switches set: {on if on else "none (all optimizations active)"}')
    print(f'n_chan={settings.get("n_chan_bin")} fs={settings.get("fs")} '
          f'batch_size={settings.get("batch_size")}')
    print(f'settings overrides: {overrides if overrides else "none"}')
    os.makedirs(args.results_dir, exist_ok=True)
    Path(args.results_dir, 'settings_override.json').write_text(
        json.dumps(overrides, indent=1))

    t0 = time.time()
    run_kilosort(settings=settings, probe=probe, filename=args.data,
                 results_dir=args.results_dir, data_dtype='int16',
                 do_CAR=do_CAR, invert_sign=invert_sign,
                 device=torch.device(args.device), save_extra_vars=False,
                 save_preprocessed_copy=False, clear_cache=False,
                 save_pc_features=not args.no_pc_features)
    print(f'\nTOTAL_WALL_SECONDS {time.time() - t0:.2f}')


if __name__ == '__main__':
    main()
