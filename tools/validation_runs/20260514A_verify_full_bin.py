#!/usr/bin/env python
"""Verify the full flat .bin is a faithful materialization of data000.

Run BEFORE spending an hour of GPU on it. The failure this is built around is
the file-order bug that silently corrupted run E: a concatenation that reads
the right bytes in the wrong order produces a file of exactly the right size
that sorts to garbage. Size alone proves nothing.

Checks:
  a) part files sort numerically, not just lexicographically
  b) size == n_samples * n_chan * 2, and n_samples == sum of per-file bodies
  c) (done by cmp outside) first 3.07 GB identical to slice300_514a.bin
  d) every inter-file boundary lands where sample_edges says it does
  e) the tail is real signal, not zero padding
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/localadmin/Downloads/Kilosort-1')
from kilosort.litke import LitkeRecording

SRC = '/home/localadmin/Documents/Development/data/raw/20260514A/data000'
FULL = ('/tmp/claude-1001/-home-localadmin-Documents-Development-MEA-fieldlab'
        '-src-utilities/5436d4cb-2b4c-4887-a914-4bb3f3826c88/scratchpad/'
        '20260514A_validation/full/full_514a_data000.bin')

fails = []


def check(name, ok, detail=''):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}' + (f'  {detail}' if detail else ''))
    if not ok:
        fails.append(name)


rec = LitkeRecording(SRC)
n_total, n_chan = rec.shape
itemsize = 2
print(f'recording: {n_total} samples x {n_chan} ch, fs={rec.fs}, '
      f'array {rec.array_id}, {len(rec.paths)} part files')

# (a) file ordering -----------------------------------------------------------
names = [p.name for p in rec.paths]
nums = [int(re.search(r'(\d+)\.bin$', n).group(1)) for n in names]
check('a. part files in numeric order', nums == sorted(nums),
      f'{names[0]} .. {names[-1]}, indices {nums[0]}..{nums[-1]}')
check('a. no gaps in part indices', nums == list(range(len(nums))))

# (b) size accounting ---------------------------------------------------------
size = Path(FULL).stat().st_size
expect = n_total * n_chan * itemsize
check('b. flat size == n_samples * n_chan * 2', size == expect,
      f'{size} vs {expect}')
check('b. sample_edges sum == n_samples', rec.sample_edges[-1] == n_total,
      f'edges {rec.sample_edges[:3]} ... {rec.sample_edges[-2:]}')

flat = np.memmap(FULL, dtype=np.int16, mode='r', shape=(n_total, n_chan))

# (d) boundary straddles ------------------------------------------------------
# Read 64 samples centred on every inter-file edge straight from the Litke
# reader and compare against the same rows of the flat file. If the parts were
# concatenated in the wrong order, or an edge is off by even one sample, the
# rows on one side of the seam will not line up.
row = 32
bad_edges = []
for e in rec.sample_edges[1:-1]:
    lo, hi = e - row, min(e + row, n_total)
    src = np.ascontiguousarray(rec[lo:hi], dtype=np.int16)
    if not np.array_equal(src, np.asarray(flat[lo:hi])):
        bad_edges.append(e)
check('d. all inter-file boundaries match the reader', not bad_edges,
      f'{len(rec.sample_edges) - 2} seams checked, {len(bad_edges)} bad')

# Also spot-check interior blocks well away from any seam.
rng = np.random.default_rng(0)
bad_spots = []
for start in rng.integers(0, n_total - 128, size=12):
    start = int(start)
    src = np.ascontiguousarray(rec[start:start + 128], dtype=np.int16)
    if not np.array_equal(src, np.asarray(flat[start:start + 128])):
        bad_spots.append(start)
check('d. random interior blocks match the reader', not bad_spots,
      f'12 spots checked, {len(bad_spots)} bad')

# (e) tail is real signal -----------------------------------------------------
tail = np.asarray(flat[-10000:])
head = np.asarray(flat[:10000])
check('e. tail is not zero padding', tail.any() and tail.std() > 1.0,
      f'std {tail.std():.1f}, min {tail.min()}, max {tail.max()}')
check('e. head std comparable to tail std',
      0.2 < (head.std() / tail.std()) < 5.0,
      f'head {head.std():.1f} / tail {tail.std():.1f}')

rec.close()
print()
if fails:
    print(f'{len(fails)} CHECK(S) FAILED: {fails}')
    sys.exit(1)
print('all checks passed')
