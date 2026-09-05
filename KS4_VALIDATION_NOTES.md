# Validation notes: universal spike detection

What was measured, on what, and what it proves. Referenced from comments in
`kilosort/spikedetect.py` and `kilosort/fused_detect.py`.

Everything here is on **20260724A/chunk12_9-11**: 519 channels, fs = 20000,
`batch_size` = 10000 (NT = 10122), nt = 61, nblocks = 0, dmin = 15.0,
dminx = 32.0, `nearest_chans` = 10, `nearest_templates` = 100,
Th_universal = 9, Th_learned = 8, `max_peels` = 50. 4784 universal templates
on a 46x104 grid, 4048 surviving `max_channel_distance`. Hardware: RTX 4000
Ada (19.54 GB), Threadripper PRO 7975WX, torch 2.5.1, Triton 3.1.0.

Two benchmarks are used below:

* **production sort** — the full 4054-batch run, 1601.58 s total.
* **slice300** — a 300-batch (150 s) flat int16 slice of the same recording,
  sorted end to end with the production settings. 600 units, 481 good,
  776,125 spikes. Used for A/B byte comparison because a full sort per
  variant is not affordable.

---

## Where the time actually goes

Production runtime by stage:

| stage | s | % |
|---|---:|---:|
| preprocessing | 1.4 | 0.09 |
| drift | 0.0 | 0.00 |
| **spike detection (universal)** | **927.4** | **57.90** |
| clustering (templates) | 71.1 | 4.44 |
| spike detection (learned) | 506.6 | 31.63 |
| clustering (final) | 58.1 | 3.63 |
| cluster merge | 8.6 | 0.54 |
| postprocessing | 21.5 | 1.34 |

A serial decomposition of universal detection on real batches, using the
production `ops.npy` so the detection state is exact:

| step | ms/batch | share |
|---|---:|---:|
| disk read | 0.34 | 0.2% |
| H2D + CAR + highpass + whiten | 5.33 | 2.4% |
| **`template_match`** | **212.53** | **97.3%** |
| tail | 0.31 | 0.1% |
| total | 218.51 | (production: 228.7) |

**There is no I/O problem.** The "45.4 MB/s vs a 4.3 GB/s NVMe" figure that
prompted this work is data volume divided by a compute-bound runtime, not a
transfer rate. Median 1468 spikes/batch confirms detection is really running.

Inside `template_match`: conv1d 2.80 ms, the column-tiled loop 203.95 ms
(95.9%), tail 5.94 ms.

---

## 1. Column tiling: `niter = 40` -> `min(40, ceil(NT/1024))`

`niter` was a flat 40, sized for the stock `batch_size` = 60000
(NT ~ 60122, ~1500-column chunks). MEA sorts run `batch_size` = 10000, where
40 iterations means 254-column chunks: identical work split into 6x more
kernel launches, each too small to fill the GPU.

Tiling is arithmetically inert. In
`A = einsum('ijk, jklm-> iklm', weigh, Bsl[iC])` the only summed index is the
neighbour axis `j`; time-within-chunk `m` is a pure batch dimension, and every
reduction afterwards (max over templates, max over nC2 neighbours,
`max_pool1d`, threshold) runs on the reassembled full-width `(Nfilt, NT)`
buffers.

Measured (Nfilt = 4048, nC = 10, nk = 10, NT = 10122), one process per setting
because `_template_match_body_dispatch` permanently downgrades a process to
eager after a single OOM and would otherwise contaminate a sweep:

| niter | chunk | ms/batch | peak MiB | bit-identical to niter=40 |
|---:|---:|---:|---:|---|
| 40 | 254 | 212.8 | 2832 | (reference) |
| 10 | 1013 | 196.3 | 4621 | yes |
| 8 | 1266 | 298.8 | 6217 | yes |

There is a hard cliff just past ~1024 columns. `min(40, ...)` means the rule
can only ever *reduce* over-tiling on short batches: any NT >= 40*1024 keeps
the stock tiling exactly, so the default-`batch_size` path is untouched.

Verified per element on `(xy, imax, amp, adist)` over 30 batches / ~40k
detections for niter in {40, 20, 16, 13, 10, 8, 6, 5, 4, 3}, and end to end on
slice300: all 18 output `.npy` arrays and all 4 `cluster_*.tsv` byte-identical.
118.24 s -> 113.15 s whole sort; 68.8 s -> 63.8 s universal detection.

Note the peak-memory cost: universal detection's peak device allocation goes
1.29 GiB -> 3.04 GiB, because the chunks are 4x wider. Production already
peaks at 3.02 GB of 19.54 GB, so this is inside the existing envelope.

Commit `9380c96`.

---

## 2. Fusing the loop body (Triton), 6.9x

### Why fusion, and not more tiling

The body is at the memory roofline. Measured against this card's own
device-to-device copy rate (279 GB/s for a 512 MiB copy):

| statement | ms | GB moved | GB/s | % of copy peak |
|---|---:|---:|---:|---:|
| gather `Bsl[iC]` | 5.87 | 1.547 | 264 | 94% |
| einsum -> A | 7.97 | 2.291 | 288 | 103% |
| abs + max over 50 | 8.11 | 0.794 | 98 | 35% |
| index_select + max (Amax) | 12.39 | 1.558 | 126 | 45% |

Eager total 35.4 ms/chunk; the compiled (Inductor) path used in production is
17.5 ms/chunk, so fusion was already worth 2x. The body moves ~8.5 GB per
chunk to produce 65 MB of output — **130x the irreducible traffic** — because
`Bsl[iC]` (1.3 M floats -> 103 M, 78x) and `Aa[iC2_flat]` are both
materialised. Reordering cannot fix that; only moving less data can.

### Establishing what "identical" means here

1. `einsum('ijk, jklm-> iklm', weigh, Bsl[iC])` is **bitwise equal to a plain
   `torch.bmm`** for these shapes. So the reference is cuBLAS's sgemm with
   K = nC = 10. (`mulsum`, ascending/descending sequential, and pairwise
   groupings all differ from it, median 1 ULP.)
2. Candidate fp32 instruction sequences were then modelled in float64 with an
   explicit float32 rounding at every step and compared bit-for-bit against
   cuBLAS on 20,000 sampled outputs, excluding the zero-padded edge columns
   where every ordering trivially agrees:

   | candidate | exact match |
   |---|---:|
   | sequential FMA, ascending | 54.25% |
   | sequential FMA, descending | 40.58% |
   | sequential mul+add, ascending | 43.71% |
   | **split-K 2 (FMA)** | **100.0000%** |
   | split-K 4 / 5 (FMA) | 47.86% / 43.72% |
   | interleaved 2 / 4 / 8 accumulators | 40.98% / 39.85% / 41.61% |

   cuBLAS splits K into two contiguous halves, accumulates each sequentially
   with FMA, and adds the partials. That is what the kernel emits.
3. `max`/`min` are exact in floating point, so those reductions' order is
   free. `torch.max(dim=0)`'s first-index tie-break is reproduced by taking
   the minimum index among the maximal entries.

### Why it is still gated at runtime

Bit-identity turned out to depend on the Triton block size *and the problem
shape*, not just on the arithmetic above: at production shapes BLOCK_M=128 /
4 warps matches and 64 and 32 do not, while at the small shapes in
`tests/test_fused_detect.py` 128 fails and the gate settles on (64, 2). So
`fused_detect.try_fill` runs the stock loop and each candidate config on the
first real batch of every sort (122,921,568 elements at production shapes) and
uses only configs that come out exactly equal. If none do, it logs that and
runs stock for the whole sort. `KILOSORT_NO_FUSED_DETECT=1` skips it.

### Verification

* 40 real batches, fused vs the stock tiled compiled path, **4,916,862,720
  elements, all equal** — both tiled the same way as stock and as a single
  full-width launch (the fused kernel needs no tiling: output column `m`
  reads only `B[:, :, m]`).
* slice300 end to end against the stock-`niter=40` baseline: all 22 output
  `.npy` arrays byte-identical (`spike_times`, `spike_clusters`, `amplitudes`,
  `pc_features` (776125, 3, 10), `templates`, ...), all 4 `cluster_*.tsv`
  identical, `params.py` identical. Only `ops.npy` differs, in `runtime_*`,
  `usage_*`, `cuda_*` and `results_dir`.
* `KILOSORT_NO_FUSED_DETECT=1` on the same slice: 113.35 s, all 23 files
  byte-identical to the baseline — the fallback is exercised, not assumed.

### Result

| | before | after |
|---|---:|---:|
| loop body | 229.9 ms/batch | 33.3 ms/batch (6.9x) |
| universal detection | 68.8 s | 18.5 s (3.7x) |
| whole slice300 sort | 118.2 s | 67.6 s (1.75x) |
| cumulative device allocation, universal detection | 8.53 TiB | 0.53 TiB |

Commit `0dba0ec`.

---

## 3. Fusing the peel subtract (Triton), 5x

The learned-template pass is 506.6 s (31.6%), and 65.9% of it is the two
subtract lines at the bottom of each peel. Each builds an int64 index tensor,
gathers, broadcasts a multiply, subtracts and scatters — five materialised
`(row, n_sel, window)` tensors and about six launches, to touch windows 61 and
123 columns wide with a median of 26 spikes. It runs ~29k times per 300-batch
sort, so it is launch- and allocation-bound, not bandwidth-bound.

### The disjointness census

Advanced-index `-=` is gather / subtract / scatter, so duplicate indices
**drop** contributions rather than accumulating them — this is what the stock
`n = 2` comment means by the stride being "load-bearing for identity". A fused
kernel equals stock exactly when a phase's windows are disjoint.

Measured over a full 300-batch sort — 14,557 peels, 29,114 phases, 776,894
spikes, nt = 61 (trange window 123, tiwave window 62):

| | |
|---|---:|
| phases whose `Xres` windows overlap | 0 |
| phases whose `B` windows overlap | **1** |
| smallest gap between two detections in one peel | 1 |
| per-phase min gap: min / 1st pct / median | 81 / 127 / 156 |

Detections are `(2nt+1)` max-pool maxima so they normally sit more than nt
apart, but `abs(cmax - Cfmax) < 1e-9` admits exact ties and a tie can put two
adjacent. **Skipping the check would have been bit-identical 29,113 times out
of 29,114 and silently wrong once**, then propagated through the peel. So
every phase is checked — one reduction per peel at a sync the loop already
performs, using `d[i] + d[i+1]` so both phases are covered at once — and the
odd one falls back to stock.

### Stock is nondeterministic on an overlapping phase

Worth knowing, because it bounds what "byte-identical" can mean here. Running
the *stock* subtract twice on identical inputs with overlapping windows gives
different answers: over 7 repeats, 152–838 elements differed, up to 4.96 in
magnitude. `index_put_` without accumulate is documented as nondeterministic
and this is that. Falling back preserves stock's behaviour, nondeterminism
included; it cannot make that phase reproducible. In practice the four
end-to-end runs below all agreed byte-for-byte, so the scatter is stable at
these shapes on this machine — but that is an observation, not a guarantee.

### FMA contraction, and a comparator that lied

Stock rounds twice: once for `amp * src`, once for the subtract. Written
plainly the kernel contracts them into one FFMA and differs on 1,092 of 4.19M
elements. Three barriers were compared:

| form | bulk diffs | signed-zero diffs |
|---|---:|---:|
| plain `out - amp*src` | 1,092 / 4,194,304 | 0 / 1,024 |
| `v = amp*src; v = v + 0.0` | 0 | **1,024 / 1,024** |
| plain, `enable_fp_fusion=False` | **0** | **0** |

The hand-rolled `+ 0.0` barrier works only because `x + 0.0` is *not* an
identity — it maps `-0.0` to `+0.0`, which is exactly why the compiler cannot
fold it, and exactly why it then differs on every signed zero. The launch flag
is the correct fix.

**`torch.equal` and `np.array_equal` did not catch that**: both are value
equality and report `+0.0 == -0.0`. An earlier version of this kernel passed
`torch.equal` while flipping the sign of every zero it touched. Every
comparison in these notes is on raw bit patterns
(`t.view(torch.int32)` / `a.view(np.uint8)`). Re-checking §2's end-to-end
result that way: still byte-identical, and the baseline outputs contain zero
negative zeros, so nothing was hiding there either.

### Verification

* Bit-exact against stock at production shapes (Nchan=519, n_units=600,
  nt=61) for 1, 8, 26 and 64 spikes, on both `Xres` and `B`, plus the forced
  `-0.0` corner.
* Full sort of the 300-batch slice: all 25 output files byte-identical to the
  stock baseline (raw buffers). 600 units / 481 good / 776,125 spikes.
* `KILOSORT_NO_FUSED_PEEL=1` on the same slice: 25 files byte-identical.

### Result

| | before | after |
|---|---:|---:|
| one phase | 178.1 us | 27.2 us (6.56x); 35.6 us with the guard (5.00x) |
| learned pass | 16.2 s | 11.6 s |
| whole slice300 sort | 67.6 s | 62.6 s |

Commit `c086c00`.

---

## 4. Clustering: k-means++ without the per-iteration host reads, 1.56x

With detection dealt with, clustering was the largest stage left — 29.1 s of a
70.7 s slice300 sort, 41%, and the only stage this series had not touched.

### Where clustering's time goes

`profile_cluster.py` wraps every function `clustering_qr.run` calls and
synchronizes CUDA before reading the clock, so GPU work is charged to the call
that launched it. Both passes, one slice300 sort:

| function | template | | spikes | |
|---|---:|---:|---:|---:|
| `run()` total | 14.58 s | | 13.63 s | |
| `cluster` | 12.36 s | 84.8% | 12.69 s | 93.1% |
| ↳ `kmeans_plusplus` | **11.44 s** | **78.5%** | **11.80 s** | **86.6%** |
| ↳ assign loop (rest) | 0.55 s | 3.8% | 0.57 s | 4.2% |
| ↳ `neigh_mat` | 0.32 s | 2.2% | 0.27 s | 2.0% |
| ↳ `Mstats` | 0.05 s | 0.3% | 0.05 s | 0.3% |
| `swarmsplitter.split` | 1.80 s | 12.3% | 0.50 s | 3.7% |
| `get_data_cpu` | 0.14 s | 0.9% | 0.15 s | 1.1% |
| `hierarchical.maketree` | 0.11 s | 0.8% | 0.12 s | 0.9% |
| `mean_cluster_templates` | 0.08 s | 0.5% | 0.08 s | 0.6% |
| `new_clusters` | 0.01 s | 0.1% | 0.01 s | 0.1% |
| loop body, unaccounted | 0.08 s | 0.6% | 0.07 s | 0.5% |

The standing guess — "235 cluster centres in a Python loop over small tensors"
— was pointing at the wrong loop. **The per-centre loop is 0.6% of the time.**
One function inside it is 78–87%, and the previous session's work on `cluster`
(`_counts_into`, the early exit) had already dealt with everything around it:
the alternating-assignment loop is 4%.

### It is not doing arithmetic

`kmeans_plusplus` runs 200 iterations on tensors of at most 13k × 75 floats.
A 1,162-spike centre takes 53.2 ms; a 3,263-spike centre takes 53.0 ms. Same
time, 2.8× the data — the cost is per-iteration overhead. Statement-level
timing, synchronizing after each statement (12,941 spikes, µs/iteration):

| statement | µs | statement | µs |
|---|---:|---|---:|
| `multinomial` | 104.3 | `dexp` relu | 23.2 |
| `vexp0[ix] = vexp[ix, imax]` | 71.2 | `dexp.sum(0)` | 19.4 |
| `vexp` matmul | 57.0 | `mu[j] = Xc[imax]` | 17.9 |
| `int((weights > 0).sum())` | 34.0 | `float(weights.sum())` | 17.4 |
| `ix = dexp[:, imax] > 0` | 26.6 | `relu(vtot - vexp0)` | 15.6 |
| | | `iclust[ix] = j` | 12.1 |

Three of those block the host: the loop's two guard reads, and a third hidden
inside `vexp0[ix] = vexp[ix, imax]`, where the boolean advanced index has to
run `nonzero()` to size its output. Together they are ~32% of the serialized
total, and they stall the pipeline rather than queueing behind it.

### The guard is one question, and its answer is known at the end

```python
if float(weights.sum()) <= 0: break
n_pos = int((weights > 0).sum().item())
if n_pos <= 0: break
n_draw = min(ntry, n_pos)
```

`weights = relu(...) >= 0`, so `sum <= 0` iff every entry is zero iff
`n_pos == 0`, and `n_pos == 0` implies `n_pos < ntry`. All three lines are the
single question *was `n_pos` ever below `ntry`* — only the smallest `n_pos` the
loop saw matters.

That smallest value is the last one. `vexp0` is written only through
`vexp0[ix] = vexp[ix, imax]` with `ix = dexp[:, imax] > 0` and
`dexp = relu(vexp - vexp0[:, None])`, so `ix` is true exactly where
`vexp[:, imax] > vexp0`: every write raises an entry and none lowers one. (NaN
cannot slip through — `relu(NaN)` is NaN and `NaN > 0` is false.) A
non-decreasing `vexp0` makes `relu(vtot - vexp0)` non-increasing elementwise,
so `n_pos` is non-increasing, so `n_pos` after the last iteration bounds every
`n_pos` the loop passed through. **One `count_nonzero` and one host read per
call replace two per iteration: 400 reads become 1.**

If that final count is below `ntry`, the fast result is discarded and the stock
body runs. Nothing is guessed.

Census over a whole slice300 sort (`kpp_census.py`, 393 calls):

| | |
|---|---:|
| calls that ran fewer than 200 iterations | 0 |
| calls that ever saw `n_pos < 100` | 0 |
| smallest `n_pos` anywhere | 877 |
| 10th percentile of per-call minimum | 1,231 |
| median of per-call minimum | 2,738 |

The margin is 8.8×. The fallback is there for the case that is not in this
recording, not for one that is.

### The other three changes

* `vexp0[ix] = vexp[ix, imax]` becomes `torch.where(ix, vexp[:, imax], vexp0)`,
  and `iclust[ix] = j` becomes `iclust.masked_fill_(ix, j)`. Both are pure
  selection: the same bits are chosen, without materializing an index list, so
  neither blocks the host.
* `2 * Xg @ Xc.T` is hoisted out of the loop. `*` and `@` share precedence and
  associate left, so that expression is `(2 * Xg) @ Xc.T` — stock rebuilds the
  entire scaled feature matrix on all 200 iterations. Hoisting hands cuBLAS
  identical bytes.
* `mu` is not built. It is written every iteration and never read:
  `kmeans_plusplus` returns `iclust` alone, and the only code that used `mu` is
  the commented-out block at the end of the stock function. If that block is
  ever revived, this shortcut has to go with it.

Nothing else moves — the multinomial draw, the gemm, the relu, the sum and the
argmax are the same calls on the same values in the same order.

### RNG

The loop consumes the global torch generator through `torch.multinomial`, and
either path leaves it in the same state: the fast path draws the same 200
times, and the fallback re-seeds (stock's own `torch.manual_seed(seed)`) before
drawing. Downstream code cannot tell which ran. The identity harness checks
this explicitly, not just the labels.

### Verification

* 32 real `Xd` matrices dumped from a slice300 sort (1,002–12,941 spikes,
  30–75 features): `iclust` identical in **all 32**, generator state identical
  in **all 32**. 1,759.7 ms → 1,125.6 ms, **1.56×**.
* Full sort: **23 files byte-identical** to the stock baseline (raw bytes, not
  `np.array_equal` — see §3 for why that distinction matters). `ops.npy` is
  timers and peak memory and is reported separately.
* `KILOSORT_NO_FAST_KPP=1`: also 23 files byte-identical, so the switch really
  does reproduce the baseline.
* `tests/test_fast_kpp.py` pins the guard rather than the kernel: a normal
  centre matches stock and leaves the same generator state; a centre whose
  candidate pool collapses is **refused**, both before and after the gate has
  latched; and a separate test checks that the collapsing fixture really does
  break stock early, so the refusal tests cannot pass for the wrong reason.

  That fixture took two tries. Repeating a handful of distinct rows does *not*
  collapse the pool: a spike that is exactly its own centroid still gets `vtot`
  from `torch.norm` and `vexp` from a gemm, and those disagree in the last
  bits, so its residual stays positive. Rows of exact zeros are zero by both
  routes and collapse deterministically.

### Result

Same-session A/B on slice300, `KILOSORT_NO_FAST_KPP=1` as the control:

| | stock | fast | |
|---|---:|---:|---|
| cluster (temp) | 14.6 s | 10.1 s | 1.45× |
| cluster (final) | 14.5 s | 10.3 s | 1.41× |
| whole sort | 70.7 s | 61.7 s | 1.15× |

---

## Cumulative

Whole slice300 sort across this series, all byte-identical to the stock
baseline at every step:

| | s |
|---|---:|
| stock | 118.2 |
| + adaptive niter (§1) | 113.2 |
| + fused detection (§2) | 67.6 |
| + fused peel subtract (§3) | 62.6 |
| + sync-free k-means++ (§4) | 61.7 |

These rows are **not all from one sitting, and the machine drifts**: re-running
the §3 build in the §4 session gave 70.7 s, not 62.6 s. Only same-session A/B
numbers are comparable — for §4 that is the table just above (70.7 → 61.7 s),
with both runs byte-identical to the same baseline.

Stage shares have moved a long way: universal detection was 57.9% of the
production sort, and detection and clustering are now roughly the same size.

---

## Open, not resolved

* **`max_peels`.** Raising it finds more spikes and the cost is sublinear
  (25 -> 494,873 peeled / 114.5 s; 50 -> 776,894 / 128.3 s; 100 -> 954,178 /
  134.0 s; 200 -> 1,074,095 / 138.9 s; 1000 -> 1,082,284 / 137.5 s; batches
  hitting the cap 300, 220, 56, 16, 0). Whether the extra spikes are real is
  **not** settled here — unit counts and ContamPct move a lot (50: 600 units /
  481 good / ContamPct median 0.75; 1000: 755 / 435 / 12.40) and answering it
  needs the QA pipeline, not this file. Flagged, not concluded.
* **The learned pass** (506.6 s, 31.6%): `peel_subtract` is 65.9% of it,
  the repeated full-width `torch.max(B, 0)` only 10.3%. The `n = 2` split in
  the scatter is load-bearing for identity — advanced-index `-=` is
  last-write-wins on overlapping `+/-nt` windows — so a faster scatter is
  unlikely to be byte-identical without care.
* **Clustering, what is left after §4.** `kmeans_plusplus` is still ~60% of
  clustering, and it is still launch-bound: ~20 kernel launches per iteration
  on tensors small enough that each launch costs more than the work it does.
  The next step is capturing the iteration body in a CUDA graph, which needs
  the random draw pulled out of the body — measured feasible, because
  `torch.multinomial(w, k, replacement=False)` is exactly
  `topk(w / empty_like(w).exponential_(), k)` (verified bitwise on this
  install), i.e. one `exponential_` over an `(n_spikes,)` tensor per iteration
  and nothing else. Pre-drawing those 200 noise vectors leaves a body with no
  RNG in it. Not attempted yet.
* **`swarmsplitter.split`** is now the second-largest item in the template pass
  (1.80 s, 12.3%) and is CPU-side.
