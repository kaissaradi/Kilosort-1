#!/usr/bin/env python
"""Profile every sort this fork has ever produced, from the logs it already wrote.

Why this exists. Optimization work here kept being steered by `slice300` --
one small recording, profiled under instrumentation, in one sitting. But every
real sort already writes two things to disk that are strictly better evidence:

  * `ops.npy`        -- per-stage timers (`runtime_st0`, `runtime_st`, ...)
  * `kilosort4.log`  -- per-stage `Max alloc` lines, and timestamps around the
                        post-sort plotting that the timers do NOT cover

Together that is a retrospective profiling database over every production sort,
across every array geometry, at zero cost and with no instrumentation to
perturb what it measures. It is also the only source that sees the difference
between `ops['runtime']` and actual wall clock.

Three questions it answers that a slice cannot:

  1. WHICH STAGE IS THE BOTTLENECK NOW, weighted by real GPU-hours rather than
     by one recording. The ranking moves after every optimization, so this
     should be re-read after each one.
  2. DOES MEMORY EVER BIND. `Max alloc` is logged per stage and reset between
     stages, so the max over the file is the run's peak device allocation.
  3. WHAT IS OUTSIDE `runtime`. `save_sorting` finishes, and only *then* does
     `plot_spike_positions` run. That time is in nobody's budget.

    python tools/log_census.py /path/to/data/sorted
    python tools/log_census.py /path/to/data/sorted --per-sort
"""
import argparse
import datetime as dt
import glob
import os
import re
from collections import defaultdict

import numpy as np

STAGES = ['preproc', 'drift', 'st0', 'clu0', 'st', 'clu', 'merge', 'postproc']

_TS = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+)')
# The logger name matters: `Max alloc` is emitted by several modules, and which
# one carries the peak says which stage owns it.
_MEM = re.compile(
    r'kilosort\.(\S+)\s+\S+\s+Max alloc:\s+[\d.]+ %\s+\|\s+([\d.]+)\s+/\s+([\d.]+) GB'
)
_PHY = re.compile(r'Exporting to Phy took: ([\d.]+)s')

# Post-sort work that `ops['runtime']` does not account for. Each is measured as
# the gap to the next timestamped line, which is exact here because these are
# blocking calls that log immediately before and after.
PLOT_MARKS = ['Generating spike position plot', 'Generating diagnostic plots']


def _stamp(s):
    return dt.datetime.strptime(s, '%Y-%m-%d %H:%M:%S,%f')


def read_sort(sort_dir):
    """Return one record per sort, or None if it predates the timers."""
    log = os.path.join(sort_dir, 'kilosort4.log')
    ops_p = os.path.join(sort_dir, 'ops.npy')
    if not (os.path.exists(log) and os.path.exists(ops_p)):
        return None
    try:
        ops = np.load(ops_p, allow_pickle=True).item()
    except Exception:
        return None
    if 'runtime' not in ops:
        return None

    rec = {
        'tag': sort_dir,
        'n_chan': int(ops.get('Nchan', 0)),
        'n_batches': int(ops.get('Nbatches', 0)),
        'n_units': int(ops.get('n_units_total', 0)),
        'dmin': float(ops.get('dmin', 0) or 0),
        'runtime': float(ops['runtime']),
    }
    for s in STAGES:
        rec[s] = float(ops.get('runtime_' + s, 0.0))

    lines = [ln for ln in open(log, errors='ignore') if _TS.match(ln)]
    if not lines:
        return rec

    rec['wall'] = (_stamp(_TS.match(lines[-1]).group(1))
                   - _stamp(_TS.match(lines[0]).group(1))).total_seconds()

    peak, owner, card = 0.0, '', 0.0
    for ln in lines:
        m = _MEM.search(ln)
        if m and float(m.group(2)) > peak:
            peak, owner, card = float(m.group(2)), m.group(1), float(m.group(3))
        m = _PHY.search(ln)
        if m:
            rec['phy_export'] = float(m.group(1))
    rec.update(peak_gb=peak, peak_stage=owner, card_gb=card)

    plot = 0.0
    for mark in PLOT_MARKS:
        for i, ln in enumerate(lines):
            if mark in ln and i + 1 < len(lines):
                plot += (_stamp(_TS.match(lines[i + 1]).group(1))
                         - _stamp(_TS.match(ln).group(1))).total_seconds()
                break
    rec['plotting'] = plot
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='data/sorted -- scanned for */*/kilosort4/')
    ap.add_argument('--per-sort', action='store_true')
    args = ap.parse_args()

    recs = [r for r in (read_sort(d) for d in
                        sorted(glob.glob(os.path.join(args.root, '*', '*', 'kilosort4'))))
            if r]
    if not recs:
        print(f'no sorts with per-stage timers under {args.root}')
        return

    if args.per_sort:
        print(f"{'sort':34s} {'ch':>4s} {'wall':>6s} {'sort':>6s} {'plot':>5s} "
              f"{'peakGB':>7s} {'peak stage':<18s}")
        for r in sorted(recs, key=lambda r: -r.get('peak_gb', 0)):
            tag = r['tag'].split('/sorted/')[-1].replace('/kilosort4', '')
            print(f"{tag:34s} {r['n_chan']:4d} {r.get('wall', 0):6.0f} "
                  f"{r['runtime']:6.0f} {r.get('plotting', 0):5.0f} "
                  f"{r.get('peak_gb', 0):7.2f} {r.get('peak_stage', ''):<18s}")
        print()

    # --- where the time goes, weighted by real GPU-hours ---------------------
    # Only sorts that actually carry per-stage timers may enter the denominator.
    # Sorts predating them have runtime but all-zero stages, and including them
    # silently deflates every share -- 41.7% for st0 became 12.8% that way.
    timed = [r for r in recs if r['st0'] > 0]
    if not timed:
        print(f'{len(recs)} sorts found, none with per-stage timers')
        return
    by_geom = defaultdict(lambda: [0.0, defaultdict(float)])
    tot = 0.0
    agg = defaultdict(float)
    for r in timed:
        tot += r['runtime']
        by_geom[r['n_chan']][0] += r['runtime']
        for s in STAGES:
            agg[s] += r[s]
            by_geom[r['n_chan']][1][s] += r[s]

    live = [s for s in STAGES if agg[s] / tot > 0.005]
    print(f"{len(timed)} timed sorts of {len(recs)} found, "
          f"{tot/3600:.2f} GPU-hours of sorting")
    print(f"{'':10s}" + ' '.join(f'{s:>9s}' for s in live))
    print(f"{'ALL':10s}" + ' '.join(f'{100*agg[s]/tot:8.1f}%' for s in live))
    for ch, (t, a) in sorted(by_geom.items()):
        print(f"{str(ch)+' ch':10s}" + ' '.join(f'{100*a[s]/t:8.1f}%' for s in live)
              + f'   ({t/3600:.2f} h)')

    # --- does memory ever bind? ---------------------------------------------
    mem = [r for r in recs if r.get('peak_gb')]
    if mem:
        worst = max(mem, key=lambda r: r['peak_gb'])
        print(f"\npeak device allocation over {len(mem)} sorts: "
              f"{worst['peak_gb']:.2f} GB of {worst['card_gb']:.2f} GB "
              f"({100*worst['peak_gb']/worst['card_gb']:.0f}% of the card) "
              f"on {worst['tag'].split('/sorted/')[-1].replace('/kilosort4','')}, "
              f"in {worst['peak_stage']}")

    # --- what is outside runtime? -------------------------------------------
    plotted = [r for r in recs if r.get('plotting', 0) > 0 and r.get('wall')]
    if plotted:
        w = sum(r['wall'] for r in plotted)
        p = sum(r['plotting'] for r in plotted)
        e = sum(r.get('phy_export', 0.0) for r in plotted)
        print(f"\npost-sort work NOT in ops['runtime'], over {len(plotted)} sorts:")
        print(f"  plotting     {p:7.0f} s  ({100*p/w:.1f}% of wall)")
        print(f"  phy export   {e:7.0f} s  ({100*e/w:.1f}% of wall, this one IS in postproc)")
        print(f"  wall total   {w:7.0f} s")


if __name__ == '__main__':
    main()
