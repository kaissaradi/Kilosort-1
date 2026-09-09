# Experimental universal suppression reorder

Default behavior is unchanged. This branch tests a different evaluation order
for the same suppression mask. It does not change template scores, their signed
argmax, detection thresholds, learned peeling, or clustering.

For a score matrix A, a fixed template-neighbor graph N, and a temporal window W:

    max(t in W, max(p in N(k), A[p,t]))
      = max(p in N(k), max(t in W, A[p,t]))

Both sides must zero the batch-edge samples before pooling. The candidate test
still uses the ORIGINAL A[k,t] > threshold, including at the edges. Equality
ties and filter-major/time-major output ordering are preserved. No self-neighbor
assumption is needed.

Pool each source row once, then gather spatial neighbors only at above-threshold
positions. This avoids the candidate dilation discussed in validation notes
section 12a: temporal pooling is shared between candidate neighborhoods instead
of computing spatial maxima throughout each candidate's time halo.

## Scope and limits

- This targets `_amax_kernel` and the suppression tail, not the dense score fill.
  Section 12a attributed 45% of the fill to `_amax_kernel`, about 70 seconds of
  the old production budget. That is an upper bound on removable work, not a
  speedup prediction. Pooling, compaction, gathers, and launches still cost time.
- The prototype is eager PyTorch. It can lose on dense candidates or small
  inputs. CUDA measurement must decide whether a fused implementation is worth
  building. The old fused score kernel and its validation remain in use.
- An additional full-size float32 temporal buffer costs about 156 MiB at
  4,048 x 10,122. Candidate indices use 16 bytes per candidate; neighbor gathers
  are chunked at 16,384 candidates. The benchmark reports incremental peak CUDA
  allocation. This is not a memory-saving optimization.
- A CPU fallback still computes the old spatial maximum before this prototype;
  it is for correctness checks, not CPU acceleration.
- CPU properties and full-function comparisons passed. CUDA tests are included
  but were skipped on the development machine, which has no GPU. No production
  speedup or complete-sort byte identity has been measured for this branch.

## Try it

Run the same settings/input on both arms, in separate processes and result dirs.
Use the complete production file after trying a slice. Keep all other switches,
including plots and PC-feature export, identical.

    python tools/run_full_sort.py --ops /path/ops.npy --data /path/data.bin --results-dir /tmp/ks_baseline --deterministic
    KILOSORT_REORDERED_SUPPRESSION=check python tools/run_full_sort.py --ops /path/ops.npy --data /path/data.bin --results-dir /tmp/ks_check --deterministic
    KILOSORT_REORDERED_SUPPRESSION=1 python tools/run_full_sort.py --ops /path/ops.npy --data /path/data.bin --results-dir /tmp/ks_candidate --deterministic
    python tools/compare_sorts.py /tmp/ks_baseline /tmp/ks_check
    python tools/compare_sorts.py /tmp/ks_baseline /tmp/ks_candidate

`check` compares both masks on EVERY batch and raises on disagreement. `1`
validates on the first batch for each pass/signature, uses the reference on that
batch, then skips the dense spatial kernel on later fused batches. Failed first
validation disables the experiment for that signature. Changing the neighborhood,
its mutation version, shape, or mask settings revalidates. Calls without reusable
scratch validate every time. The runner records the environment switch.

For paired timings on real buffers, leave the switch unset and use:

    python tools/bench_reordered_suppression.py --report /tmp/ks_suppression.json --measure-batches 8 --ops /path/ops.npy --data /path/data.bin --results-dir /tmp/ks_profile --deterministic

This runs a normal sort and profiles the first eight template-match calls. It
compares the complete suppression blocks (spatial max plus mask versus temporal
max plus sparse gathers), checks ordered detection indices, and reports host
time, CUDA-event elapsed time, and extra peak allocated memory. It deliberately
does not report GPU occupancy. Profiling changes total runtime; use ordinary,
warmed runs without the hook for end-to-end performance, and separate timing
runs from deterministic correctness comparisons.

## Clustering investigation, not changed here

Section 12e corrects the earlier denominator mismatch: k-means++ is estimated at
57.4 seconds of the 111.4-second clustering stage, and swarmsplitter at 22.2
seconds. They are not 82.9 seconds plus work inside `cluster()`.

A separate CPU prototype checked sparse assignment candidates: labels receiving
neighbor votes plus the best baseline-penalty label suffice for the argmax.
It preserved the current repeated-addition arithmetic and matched winning labels
and score bits on 180 synthetic cases / 6,195 rows. This has not been integrated
or benchmarked on GPU and targets only the alternating-assignment portion.

Larger algorithm experiments should preserve the 200 seeds initially while
varying k-means++'s 100 candidate trials, then assess rare-cell recall, collisions,
false merges/splits, contamination, and held-out waveform reconstruction. A
different initialization is not guaranteed to reach the same graph partition.
The merge tree also cannot subdivide a contaminated leaf; repairing that needs
an explicit within-leaf split proposal and biological QA, not faster tree traversal.
