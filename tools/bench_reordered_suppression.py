"""Compare suppression paths on live batches while run_full_sort sorts normally.

Pass --report FILE.json and optionally --measure-batches N / --repeats N;
all remaining arguments are forwarded to tools/run_full_sort.py. This adds
profiling overhead, so use separate ordinary runs for end-to-end timing.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from kilosort import fused_detect, fused_peaks, reordered_suppression, spikedetect


def measure(fn, repeats):
    fn()
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wall, events = [], []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter() - t0) * 1000)
        events.append(start.elapsed_time(end))
        del result
    return {'host_ms': statistics.median(wall), 'event_ms': statistics.median(events),
            'extra_peak_allocated_bytes': torch.cuda.max_memory_allocated() - allocated}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--measure-batches', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=5)
    args, forwarded = parser.parse_known_args()
    if args.measure_batches < 1 or args.repeats < 1:
        parser.error('batch/repeat counts must be positive')
    if not torch.cuda.is_available() or not fused_detect._HAVE_TRITON:
        parser.error('This benchmark requires CUDA and Triton')
    if os.environ.get('KILOSORT_REORDERED_SUPPRESSION', '') not in ('', '0'):
        parser.error('Unset KILOSORT_REORDERED_SUPPRESSION for this paired benchmark')
    report = {'commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
              'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(),
              'runner_args': forwarded, 'batches': []}
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def save():
        destination.write_text(json.dumps(report, indent=2) + '\n')

    original = spikedetect.template_match

    def hooked(X, ops, iC, iC2, weigh, device=torch.device('cuda'), scratch=None):
        result = original(X, ops, iC, iC2, weigh, device=device, scratch=scratch)
        if len(report['batches']) >= args.measure_batches:
            return result
        if scratch is None or not fused_detect._CHOICE:
            raise RuntimeError('Benchmark requires a validated fused detect path and scratch buffers')
        with torch.cuda.device(X.device):
            scores, workspace = scratch['As'], scratch['Amaxs']
            nf, length = scores.shape
            block, warps = fused_detect._CHOICE
            flat = iC2.reshape(-1)
            nt, radius, threshold = ops['nt'], ops['settings']['nt0min'], ops['Th_universal']

            def baseline():
                fused_detect._amax_kernel[(nf, (length + block - 1)//block)](
                    scores, flat, workspace, length, nf, scores.stride(0), scores.stride(1),
                    NC2=iC2.shape[0], BLOCK_M=block, num_warps=warps, num_stages=1)
                mask = fused_peaks.try_mask(scores, workspace, nt, radius, threshold)
                if mask is None:
                    # Match the actual fallback without charging for a clone.
                    workspace[:, :nt] = 0
                    workspace[:, -nt:] = 0
                    pooled = torch.nn.functional.max_pool1d(workspace[None], 2*radius+1,
                                                            stride=1, padding=radius)[0]
                    mask = (pooled == scores) & (scores > threshold)
                return mask

            def candidate():
                return reordered_suppression.mask(scores, iC2, nt, radius, threshold,
                                                   workspace=workspace)

            ref, new = baseline(), candidate()
            if not torch.equal(ref, new) or not torch.equal(new.nonzero(), result[0]):
                raise RuntimeError('Mask or ordered detection indices differ on a real batch')
            del ref, new
            # Alternate order across batches; both methods get their own warmup.
            functions = [('baseline',baseline), ('reordered',candidate)]
            if len(report['batches']) % 2:
                functions.reverse()
            entry = {'shape': list(scores.shape), 'neighbors': iC2.shape[0],
                     'candidate_count': int((scores > threshold).sum()),
                     'survivors': int(result[0].shape[0]), 'identical': True}
            for name, fn in functions:
                entry[name] = measure(fn, args.repeats)
            report['batches'].append(entry)
            save()
            print('SUPPRESSION_BENCH ' + json.dumps(entry), flush=True)
        return result

    old_argv = sys.argv
    try:
        sys.argv = [str(REPO/'tools/run_full_sort.py')] + forwarded
        with patch.object(spikedetect, 'template_match', hooked):
            runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        sys.argv = old_argv
        save()


if __name__ == '__main__':
    main()
