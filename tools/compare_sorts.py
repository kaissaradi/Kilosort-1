#!/usr/bin/env python
"""Byte-compare two kilosort4 result directories.

RAW BYTES, deliberately. `np.array_equal` reports `+0.0 == -0.0` and
`NaN != NaN`, so it can call two different files identical and two identical
files different. Arrays are compared through `.view(np.uint8)` after checking
dtype and shape, which has neither failure mode.

`ops.npy` is telemetry (timers, peak memory) and is reported separately rather
than counted as a mismatch. `diagnostics.png` / `spike_positions.png` are
matplotlib output carrying a creation timestamp, and the log has wall-clock
lines in it, so all three are skipped.

    python tools/compare_sorts.py DIR_A DIR_B

Exit status 0 iff every compared file is byte-identical.

Remember what this can and cannot prove. Kilosort4 is **not** bit-reproducible
run to run: two runs of identical code on identical input have been seen to
disagree on ~13 float32 of 33 M in the tF-derived outputs (amplitudes,
spike_positions, templates, pc_features), at different spikes each time, with
spike times and cluster assignments never moving. So a single matching pair is
not proof, and a single mismatching pair is not a verdict. The technique that
settles attribution is the three-way comparison: run the new code twice and
compare run1 vs baseline, run2 vs baseline, and run1 vs run2. If the two
same-code runs disagree with each other at different places than either
disagrees with the baseline, the difference belongs to the program, not to the
change under test.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SKIP = {'diagnostics.png', 'spike_positions.png', 'kilosort4.log'}
TELEMETRY = {'ops.npy'}


def arr_bytes_equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False, f'dtype/shape {a.dtype}{a.shape} vs {b.dtype}{b.shape}'
    if a.dtype == object:
        return bool((a == b).all()), 'object array (value compare)'
    ab = np.ascontiguousarray(a).view(np.uint8)
    bb = np.ascontiguousarray(b).view(np.uint8)
    if ab.shape != bb.shape:
        return False, 'byte length differs'
    d = int((ab != bb).sum())
    return d == 0, ('' if d == 0 else f'{d:,} of {ab.size:,} bytes differ')


def compare(d1, d2, quiet=False):
    p1, p2 = Path(d1), Path(d2)
    names = sorted({f.name for f in p1.iterdir()} | {f.name for f in p2.iterdir()})
    same = diff = 0
    if not quiet:
        print(f'A = {p1}\nB = {p2}\n')
    for n in names:
        if n in SKIP:
            continue
        f1, f2 = p1 / n, p2 / n
        if f1.is_dir() or f2.is_dir():
            continue
        if not f1.exists() or not f2.exists():
            print(f'  MISSING  {n:<32} (only in {"A" if f1.exists() else "B"})')
            diff += 1
            continue
        if n.endswith('.npy'):
            a = np.load(f1, allow_pickle=True)
            b = np.load(f2, allow_pickle=True)
            if n in TELEMETRY:
                if not quiet:
                    print(f'  telemetry {n:<31} (not compared: timers/peak mem)')
                continue
            ok, why = arr_bytes_equal(a, b)
        else:
            ok = f1.read_bytes() == f2.read_bytes()
            why = '' if ok else 'text differs'
        if ok:
            same += 1
        else:
            diff += 1
            print(f'  DIFFERS  {n:<32} {why}')
    print(f'\n  {same} files byte-identical, {diff} differ')
    return same, diff


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('dir_a')
    ap.add_argument('dir_b')
    ap.add_argument('-q', '--quiet', action='store_true')
    args = ap.parse_args()
    _, diff = compare(args.dir_a, args.dir_b, args.quiet)
    return 0 if diff == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
