#!/usr/bin/env python3
"""Export one completed Kilosort run to a Vision-native bundle."""

import argparse
from pathlib import Path

from kilosort.vision_export import EI_MODES, VisionExportError, export_vision, load_kilosort_spikes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sort_dir", type=Path)
    parser.add_argument("raw_path", type=Path, help="raw Litke dataXXX directory or .bin")
    parser.add_argument("-o", "--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--good-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--left", type=int, default=67)
    parser.add_argument("--right", type=int, default=133)
    parser.add_argument("--time-constant", type=float, default=0.01)
    parser.add_argument("--mode", choices=EI_MODES, default="exact")
    parser.add_argument("--chunk", type=int, default=1 << 19)
    parser.add_argument("--cell-block", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.dry_run:
            ids, times, _ = load_kilosort_spikes(args.sort_dir, args.good_only)
            print(f"Vision export dry-run: {len(ids)} cells, {len(times)} spikes")
            return 0
        manifest = export_vision(
            args.sort_dir, args.raw_path, output_dir=args.output_dir,
            dataset_name=args.dataset_name, good_only=args.good_only,
            overwrite=args.overwrite, left=args.left, right=args.right,
            time_constant=args.time_constant, mode=args.mode,
            chunk=args.chunk, cell_block=args.cell_block)
        print(f"Exported {manifest['cell_count']} cells and "
              f"{manifest['spike_count']} spikes to {args.output_dir or args.sort_dir}")
        return 0
    except VisionExportError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
