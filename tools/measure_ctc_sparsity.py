#!/usr/bin/env python
"""Census of exact zeros in `ctc`, the peel loop's cross-template filter bank.

Why this exists. `fused_peel._scatter_sub` applies

    B[r, pos[s] + t] -= amp[s] * ctc[r, iY[s], t]

over EVERY unit row r, for every spike. But U is built by
`clustering_qr.mean_cluster_templates`, which writes into `torch.zeros(...)`
and only ever touches the channels of one cluster centre -- so U is exactly
zero off a template's local channel set, and

    ctc[i, j, t] = sum_{k,m} sum_l U[i,k,l] * U[j,m,l] * WtW[k,m,t]

contracts l over channels. If units i and j have disjoint channel support then
every product in that sum is exactly 0, so the whole (i, j, :) block is zero
and the subtract for that row is a no-op. On a 519-channel array a template
spans a median of 13 channels, so most pairs should be disjoint.

Skipping a zero row is bit-identical ONLY under two conditions, and this tool
checks both rather than assuming them:

  1. The block must be +0.0, not -0.0. `o - (+0.0) == o` for every float o
     including -0.0, but `o - (-0.0)` maps -0.0 to +0.0. Products of the form
     0.0 * x are -0.0 when x < 0, and -0.0 + -0.0 stays -0.0, so a negative
     zero in ctc is possible in principle and would break identity on exactly
     the values `torch.equal` refuses to show you. We compare raw bit
     patterns, never `== 0`.

  2. amp must be > 0, so that amp * 0.0 cannot be -0.0. That should follow
     from detection (a peak needs relu(max B)**2 > Th**2 > 0, so B[iY,iX] > 0
     and s > 0), but it is checked against the real amps rather than argued.

Output is the achievable saving, not just the sparsity: the kernel skips at
BLOCK_R granularity, so what matters is the fraction of (unit, row-tile)
pairs that are entirely zero, which is lower than the fraction of zero rows.

    python tools/measure_ctc_sparsity.py --data slice300.bin --ops prod_ops.npy
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kilosort import template_matching  # noqa: E402
from kilosort.run_kilosort import run_kilosort  # noqa: E402

BLOCK_R = 16   # must match fused_peel.BLOCK_R

_stats = []
_dumped = []


def _census(ctc, block_r):
    """Zero structure of ctc, judged on raw bit patterns."""
    n_i, n_j, n_t = ctc.shape
    bits = ctc.view(torch.int32)

    # A row (i, j, :) is skippable iff every element is +0.0 == bits 0x00000000.
    pos_zero_row = (bits == 0).all(dim=2)
    # Distinguish blocks that are zero-valued but carry a -0.0 somewhere: those
    # are NOT skippable and their existence would change the design.
    any_zero_row = (ctc == 0).all(dim=2)
    neg_zero_rows = int((any_zero_row & ~pos_zero_row).sum())

    # The kernel skips a (spiking unit j, row tile) pair only if all BLOCK_R
    # rows in that tile are skippable. Pad the row axis the way the grid does.
    n_tiles = (n_i + block_r - 1) // block_r
    pad = n_tiles * block_r - n_i
    tiles = pos_zero_row
    if pad:
        tiles = torch.cat([tiles, torch.ones((pad, n_j), dtype=torch.bool,
                                             device=tiles.device)], dim=0)
    tiles = tiles.view(n_tiles, block_r, n_j).all(dim=1)

    return {
        'shape': (n_i, n_j, n_t),
        'row_zero_frac': float(pos_zero_row.float().mean()),
        'neg_zero_rows': neg_zero_rows,
        'tile_zero_frac': float(tiles.float().mean()),
        'n_tiles_per_spike': n_tiles,
        'live_rows_per_unit_med': float(
            (~pos_zero_row).sum(0).float().median()),
        'live_tiles_per_unit_med': float((~tiles).sum(0).float().median()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', required=True)
    ap.add_argument('--ops', required=True, help='ops.npy to replay')
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--invert-sign', action='store_true')
    ap.add_argument('--block-r', type=int, default=BLOCK_R)
    ap.add_argument('--dump-ctc', metavar='PATH',
                    help='save the first real ctc (and U_time) to PATH.pt, so '
                         'kernel work can be benchmarked against the true '
                         'liveness structure instead of a synthetic one. '
                         'Uniform-random sparsity is NOT representative: real '
                         'dead rows cluster into whole dead tiles, which is '
                         'the entire reason a tile skip pays.')
    args = ap.parse_args()

    real_prepare = template_matching.prepare_matching

    def wrapped(ops, U, return_cache=False):
        out = real_prepare(ops, U, return_cache=return_cache)
        ctc = out[0] if return_cache else out
        if args.dump_ctc and not _dumped:
            _dumped.append(1)
            cache = out[1] if return_cache else None
            torch.save({'ctc': ctc.cpu(),
                        'U': U.cpu(),
                        'U_time': (cache['U_time'].cpu()
                                   if cache and 'U_time' in cache else None)},
                       args.dump_ctc)
            print(f'\n[dumped real ctc {tuple(ctc.shape)} to {args.dump_ctc}]\n',
                  flush=True)
        st = _census(ctc, args.block_r)
        st['U_shape'] = tuple(U.shape)
        # Channel support of U itself, for the "why" half of the story.
        # U is (n_units, n_pc, n_chan); a channel is live if any PC is nonzero.
        live_ch = (U != 0).any(dim=1)
        st['chan_per_unit_med'] = float(live_ch.sum(1).float().median())
        st['n_chan'] = int(U.shape[2])
        _stats.append(st)
        print(f'\n[ctc census #{len(_stats)}] {st}\n', flush=True)
        return out

    template_matching.prepare_matching = wrapped

    # Also verify the amp > 0 precondition on real detections.
    amp_min = [float('inf')]
    real_peel = template_matching.fused_peel.peel_subtract

    def peel_wrapped(Xres, B, iX, iY, amp, *a, **kw):
        if amp.numel():
            amp_min[0] = min(amp_min[0], float(amp.min()))
        return real_peel(Xres, B, iX, iY, amp, *a, **kw)

    template_matching.fused_peel.peel_subtract = peel_wrapped

    prod = np.load(args.ops, allow_pickle=True).item()
    settings = dict(prod['settings'])
    settings['filename'] = args.data
    settings['data_dir'] = None
    settings.pop('probe', None)

    t0 = time.time()
    run_kilosort(settings=settings, probe=prod['probe'], filename=args.data,
                 results_dir=args.results_dir, data_dtype='int16',
                 do_CAR=prod['do_CAR'],
                 invert_sign=prod.get('invert_sign', args.invert_sign),
                 device=torch.device('cuda'), save_extra_vars=False,
                 save_preprocessed_copy=False, clear_cache=False)
    wall = time.time() - t0

    print('\n' + '=' * 70)
    print(f'ctc census over {len(_stats)} prepare_matching call(s), {wall:.1f}s')
    print('=' * 70)
    for i, st in enumerate(_stats):
        n_i, n_j, n_t = st['shape']
        print(f'\ncall {i}: ctc {n_i} x {n_j} x {n_t}   U {st["U_shape"]}')
        print(f'  channels per template (median) : {st["chan_per_unit_med"]:.0f}'
              f' of {st["n_chan"]}')
        print(f'  rows that are exactly +0.0     : {st["row_zero_frac"]*100:.2f}%')
        print(f'  zero rows carrying a -0.0      : {st["neg_zero_rows"]}'
              f'   (must be 0 to skip safely)')
        print(f'  live rows per spiking unit     : '
              f'{st["live_rows_per_unit_med"]:.0f} (median)')
        print(f'  --- at BLOCK_R={args.block_r} ---')
        print(f'  row-tiles per spike            : {st["n_tiles_per_spike"]}')
        print(f'  tiles skippable                : {st["tile_zero_frac"]*100:.2f}%')
        print(f'  live tiles per spiking unit    : '
              f'{st["live_tiles_per_unit_med"]:.0f} (median)')
    print(f'\nsmallest amp seen in any peel    : {amp_min[0]:.6g}'
          f'   (must be > 0 for the skip to be exact)')


if __name__ == '__main__':
    main()
