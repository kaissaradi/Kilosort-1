#!/usr/bin/env python
"""Write a flat int16 .bin slice of a real Litke recording.

The output matches what the lab pipeline's `prepare_data.py` produces:
sample-major, TTL channel dropped, no filtering (`params.py` from the
production sort records `hp_filtered = False`). That means a slice made here
can be fed straight to `tools/run_full_sort.py` and the result compared to a
production sort of the same recording.

Slices exist so a byte-identity A/B finishes in a minute instead of ten. They
are a development convenience and are written to the scratchpad, not into the
data tree; delete them when the comparison they were made for is done.

    python tools/make_litke_slice.py RAW_DIR OUT.bin --batches 300

`--batches` is in kilosort batches at `--batch-size` (default 10000), which is
the unit everything else in this repo is quoted in. `--samples` takes a raw
sample count instead if you want one.

The channel count is read from the recording rather than assumed, and printed,
because it is the number that decides whether a slice is comparable to another
one: a 519-channel 30 um slice and a 512-channel 60 um slice exercise
different template geometry, different channel neighbourhoods and therefore
different Triton block shapes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kilosort.litke import LitkeRecording   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('src', help='Litke recording directory (holds dataNNNNNN.bin)')
    ap.add_argument('out', help='output flat int16 .bin')
    ap.add_argument('--batches', type=int, default=300,
                    help='kilosort batches to write (default 300)')
    ap.add_argument('--samples', type=int, default=None,
                    help='raw samples to write; overrides --batches')
    ap.add_argument('--batch-size', type=int, default=10000)
    ap.add_argument('--block', type=int, default=200_000,
                    help='samples read per I/O block')
    ap.add_argument('--expect-channels', type=int, default=None,
                    help='fail unless the recording has this many channels')
    args = ap.parse_args()

    rec = LitkeRecording(args.src)
    n_total, n_chan = rec.shape
    if args.expect_channels is not None and n_chan != args.expect_channels:
        rec.close()
        raise SystemExit(f'{args.src}: {n_chan} channels, expected '
                         f'{args.expect_channels}')

    want = args.samples if args.samples is not None else args.batches * args.batch_size
    n = min(want, n_total)
    if n < want:
        print(f'note: recording holds {n_total} samples, writing all of them '
              f'({n // args.batch_size} batches, not {args.batches})')

    written = 0
    with open(args.out, 'wb') as f:
        for start in range(0, n, args.block):
            stop = min(start + args.block, n)
            block = np.ascontiguousarray(rec[start:stop], dtype=np.int16)
            assert block.shape == (stop - start, n_chan), block.shape
            f.write(block.tobytes())
            written += block.shape[0]
    rec.close()

    print(f'wrote {written} samples x {n_chan} ch int16 -> {args.out}')
    print(f'  {written * n_chan * 2 / 1e9:.2f} GB, {written / 20000:.1f} s of data, '
          f'{written // args.batch_size} batches at batch_size={args.batch_size}')


if __name__ == '__main__':
    main()
