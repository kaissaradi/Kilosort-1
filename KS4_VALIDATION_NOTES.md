# Validation notes: universal spike detection

What was measured, on what, and what it proves. Referenced from comments in
`kilosort/spikedetect.py` and `kilosort/fused_detect.py`.

Unless a section says otherwise, everything here is on
**20260724A/chunk12_9-11**: 519 channels at a 30 um pitch (array 1551),
fs = 20000, `batch_size` = 10000 (NT = 10122), nt = 61, nblocks = 0,
dmin = 15.0, dminx = 32.0, `nearest_chans` = 10, `nearest_templates` = 100,
Th_universal = 9, Th_learned = 8, `max_peels` = 50. 4784 universal templates
on a 46x104 grid, 4048 surviving `max_channel_distance`. Hardware: RTX 4000
Ada (19.54 GB), Threadripper PRO 7975WX, torch 2.5.1, Triton 3.1.0.

**One recording is one shape.** Every Triton kernel here is gated on a runtime
bit-comparison, and for §2 *which* block size passes that gate is not a
constant of the code — it depends on the tensor shapes, which depend on the
array. A second geometry is therefore not a nicety; it is the only way to find
out whether the gates fall back on hardware and probes they have not seen. See
*Second array geometry* below for the first such test, on a 512-channel 60 um
array — validated at production scale, 23/23 byte-identical three ways, 2.24x.

Two benchmarks are used below:

* **production sort** — the full 4054-batch run, 1601.58 s total before this
  series and 637.40 s after it; see *Production A/B, measured end to end*.
  Every production figure quoted before that section is the **baseline** run.
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

## 5. Capturing the k-means++ body in a CUDA graph, 2.3x on top of §4

§4 removed the stalls but not the launches: the loop still issues ~20 kernels
per iteration on tensors small enough that the launch costs more than the work.
200 iterations take 33.6 ms at 1,162 spikes and 44.3 ms at 12,941 — ~168
µs/iteration is overhead and only ~54 µs/iteration is data. A CUDA graph
collapses those 20 launches into one.

### Getting the randomness out of the body

Capturing torch's generator makes every replay draw *different* numbers from
what eager would, so the answer changes. The way out is that the loop's only
randomness is one call:

    torch.multinomial(w, k, replacement=False)
      ==  topk(w / torch.empty_like(w).exponential_(), k).indices

verified bitwise on this install over 50 weight vectors and all 32 real Xd
matrices. So each iteration consumes exactly one `exponential_` over an
`(n_spikes,)` tensor and nothing else, and those `niter` vectors can be drawn
up front — leaving a body with no RNG in it, which a graph can capture.

**They must be drawn one row at a time.** The generator's offset advance
depends on each call's numel, so `niter` calls of `n_spikes` elements is a
different stream from one call of `niter * n_spikes`. Both were checked: row at
a time is bit-identical to `niter` separate `empty_like().exponential_()` calls,
and the single big call is not. `test_pre_drawn_noise_is_the_stream_
multinomial_would_have_used` pins both directions, including asserting that the
big call still *differs* — so the trap cannot quietly disappear.

`j` also has to leave the host: it lives in a device scalar that the graph
increments itself, which is what lets one recording serve all 200 iterations.

### Three things that made this affordable

* **`torch.cuda.graph()` costs 75 ms per capture, and all of it is the
  `gc.collect()` its `__enter__` runs.** Measured on this box: gc 79 ms,
  `empty_cache` 0.01 ms, `synchronize` 0.02 ms. Calling `capture_begin` /
  `capture_end` by hand costs **0.18 ms**. At 393 centres per sort that is the
  difference between +29 s and +0.07 s.
* **Every centre needs its own graph** (each has a different `n_spikes`), and
  that is fine at 0.18 ms. Padding to a bucket size so graphs could be reused
  is *not* available: `dexp.sum(0)` over `(n + pad, NTRY)` is a different
  reduction tree from `(n, NTRY)`, so the sum comes out with different bits
  even though the padded rows are exactly zero.
* **A graph's private memory pool is not returned when the graph is
  destroyed.** With a fresh pool per graph, reserved memory grew ~23 MB per
  capture and reached 9.9 GB after 400 — an OOM on a 19.5 GB card. Sharing one
  pool handle fixes it (reserved plateaus at 790 MB and stays flat over 400
  captures), but a shared handle whose graphs have *all* been destroyed trips
  `it->second->use_count > 0 INTERNAL ASSERT FAILED` in the caching allocator.
  So the previous graph is held until the next one has been captured — the
  handle always has a live user, and only one graph's pool is alive at a time.

### Verification

* All 32 real Xd matrices: labels identical to the ungraphed loop, and
  `n_pos_final` identical. 2.26–2.31× on top of §4.
* Full sort, **three separate runs**: 23 files byte-identical to the stock
  baseline every time (53.32 / 53.12 / 53.48 s).
* `KILOSORT_NO_KPP_GRAPH=1` falls back to the §4 loop; `KILOSORT_NO_FAST_KPP=1`
  still falls back to stock.
* Both gates latch on the first call, so the stock loop is run once per
  process, not once per centre.

### Result

| | stock | §4 | §5 | |
|---|---:|---:|---:|---|
| cluster (temp) | 14.6 s | 10.1 s | 5.9 s | 2.47× |
| cluster (final) | 14.5 s | 10.3 s | 6.1 s | 2.38× |
| whole sort | 70.7 s | 61.7 s | 53.3 s | 1.33× |

---

## 6. Fusing the peak-selection tail (Triton), 6.2x

The tail of `template_match`, after the §2 body has filled `As`/`Amaxs`/`imaxs`:

    Amaxs[:, :nt] = 0 ; Amaxs[:, -nt:] = 0
    Amaxs = max_pool1d(Amaxs, 2*nt0+1, stride=1, padding=nt0)
    xy    = logical_and(Amaxs == As, As > Th_universal).nonzero()

At production shapes each buffer is `Nfilt x NT = 4048 x 10122` = 41 M float32
= 164 MB, and that sequence moves roughly a gigabyte per batch to produce one
bool array: the pool reads and writes a full copy, the two comparisons read
three more, `logical_and` reads and writes two. Every output element depends
only on its own row and a +/-nt0 window, so one kernel does it in a single pass.

### The short-circuit is where the win actually comes from

`As > Th` is the cheap half of the test and is brutally sparse -- **1,437
survivors of 41 M** at production shapes. The kernel tests it first and skips
the entire `2*nt0+1 = 41`-iteration window max for any block where no lane has
a candidate, which is nearly every block. This is a pure short-circuit, not an
approximation: `(m == a) & (a > Th)` is False wherever `a > Th` is False,
whatever `m` turns out to be, so a skipped block stores exactly what the full
path would have stored. Without it the kernel is 2.3x; with it, 6.2x.

Smaller blocks win here, which is the opposite of the usual advice: a block is
skipped whole, and smaller blocks are skipped more often. `_CONFIGS` is ordered
by measured speed, `(128, 4)` first.

### Why identity is easy here, unlike §2

This tail does **no floating-point arithmetic at all**:

  * `max` is a *selection*, not an accumulation. The maximum of a set of float32
    values is the same bit pattern regardless of comparison order, so the fused
    window max is exactly what `max_pool1d` produces. No rounding, no FMA
    contraction, no accumulation order to reproduce.
  * The maxima's **indices are never used** here -- unlike §2, where the argmax
    tie-break had to be matched -- so exact ties are harmless: only the value is
    compared.
  * `==`, `>` and `&` are exact.

Consequently all three block sizes matched, where §2's identity was contingent
on the block size, cuBLAS's kernel choice and the shape. The one genuine
semantic question is the array ends, where `max_pool1d`'s -inf padding meets the
explicit zeroing of the first and last `nt` columns; the kernel reproduces it by
loading out-of-range positions as -inf, and `tests/test_fused_peaks.py` puts
peaks exactly there, on block boundaries, and in dense ties.

### Verification

8 real batches captured from a live sort, **327,790,848 elements compared, all
equal**, for every config -- including the `nonzero()` the caller actually
consumes, since its order is load-bearing downstream.

Three full slice300 sorts -- stock tail (`KILOSORT_NO_FUSED_PEAKS=1`), fused,
and fused again -- compared all three ways: **23/23 files byte-identical in
every pairing**.

### Result

Call site, mean of 10, production shapes:

| | ms/batch |
|---|---:|
| stock tail | 5.332 |
| fused `(128, 4)` | **0.859** |
| | **6.20x** |

**What this is NOT yet.** The three sorts above ran 52.87 / 52.33 / 52.55 s --
a 0.43 s delta where the call-site number predicts 1.34 s over 300 batches.
That is inside this machine's known run-to-run drift, so **slice300 cannot
confirm the end-to-end win** and the production total is unconfirmed. The
per-batch figure is the claim; a production run is needed to settle the rest.

A first version of the benchmark reported 7.24x. Its stock baseline did a
164 MB `clone()` that the real call site does not do (the call site mutates
`Amaxs` in place, and both the edge zeroing and `max_pool1d` are safe to time
repeatedly without one). Corrected to 6.20x. `tools/bench_fused_peaks.py`.

---

## Kilosort4 is not bit-reproducible run to run

Found while verifying §5, and it changes how every claim in this file should be
read. Four runs of the *same* build on the same input: three were byte-identical
to each other and to the baseline, and **one differed**.

| file | bytes differing |
|---|---:|
| `templates.npy` | 145 of 75,981,600 |
| `pc_features.npy` | 96 of 93,135,000 |
| `similar_templates.npy` | 29 of 1,440,000 |
| `spike_positions.npy` | 9 of 6,209,000 |
| `amplitudes.npy` | 6 of 3,104,500 |

`spike_times.npy`, `spike_clusters.npy`, `spike_templates.npy` and
`kept_spikes.npy` were identical in all four. Every file that moved derives
from `tF` (`amplitudes = norm(tF)`, `spike_positions = f(st, tF)`, and
`templates`/`similar_templates` from `Wall`, which is a mean of `tF` rows); no
file that moved is a spike time or a cluster assignment. So the wobble is a
handful of last-bit differences in the extracted PC features, and it is **not**
introduced by anything in this series — the outlier run used the §4 code path,
which two other runs reproduced exactly. The root cause is **not** identified
here; only its footprint is.

What this means for the method: a single matching sort is weaker evidence than
it looks. The per-call harnesses (32 real matrices, 4.9 billion elements, the
peel census) are the load-bearing checks; the end-to-end sort is corroboration,
and should be **repeated** before a byte-identity claim rests on it. §5's claim
rests on three matching runs, not one.

---

## Production A/B, measured end to end

Everything above is measured on slice300, because a full sort per variant is
not affordable during development. This section is the one full-scale check:
the same 42 GB production chunk, sorted before the series and again after it.

**Baseline.** `4.1.8.dev73+g7f96af8d6`, 2026-09-04 20:02:42. That commit
("Test the fork's own changes, which nothing was testing") is an ancestor of
HEAD and predates every change in this file. **1601.58 s.**

**Optimized.** HEAD = `468a778`, 2026-09-05 08:02:53, all five changes live and
no off-switches set. **637.40 s.**

Both figures are Kilosort's own `Total runtime`, which stops before the
diagnostic plots; both runs spent a further ~82 s plotting after it
(`run_full.py` measured 721.46 s of wall for the optimized run). Same machine,
same `ops.npy`, same NVMe.

| stage | baseline s | optimized s | | responsible |
|---|---:|---:|---:|---|
| preprocessing | 1.4 | 1.4 | 1.00× | — |
| drift | 0.0 | 0.0 | — | — |
| **spike detection (universal)** | **927.4** | **231.0** | **4.01×** | §1 + §2 |
| clustering (templates) | 71.1 | 64.1 | 1.11× | §4 + §5 |
| **spike detection (learned)** | **506.6** | **255.6** | **1.98×** | §3 |
| clustering (final) | 58.1 | 47.3 | 1.23× | §4 + §5 |
| cluster merge | 8.6 | 9.8 | 0.88× | — |
| postprocessing | 21.5 | 21.6 | 1.00× | — |
| **total** | **1601.6** | **637.4** | **2.51×** | 16m 04s saved |

### The clustering win does not scale the way slice300 said

§5's slice numbers were 2.47× and 2.38× on the two clustering passes. In
production they are **1.11× and 1.23×**. This is recorded because the slice
number would otherwise be misleading. §4 and §5 attack *launch overhead*, which
dominates only while the per-centre matrices are small: slice300's were
1002–12941 spikes, production's are far larger, so the loop sits closer to
bandwidth-bound and there is less overhead left to remove. §5's pre-drawn noise
buffer also grows as `200 × n_spikes` floats per centre, so its cost grows with
exactly the thing that shrinks its benefit.

The two fused kernels (§2, §3) carry the production win almost entirely: 947 s
of the 964 s saved.

### What the output looks like

Compared against the original run's `premerge_backup/` — the state before the
post-hoc merge tool rewrote the top-level arrays — over **11,050,850 spikes**,
which is the spike count *both* runs produced:

| file | result |
|---|---|
| `spike_times.npy` | byte-identical (88,406,800 B) |
| `spike_clusters.npy` | byte-identical |
| `spike_templates.npy` | byte-identical |
| `spike_detection_templates.npy` | byte-identical |
| `cluster_KSLabel.tsv`, `cluster_group.tsv` | byte-identical |
| `cluster_Amplitude.tsv`, `cluster_ContamPct.tsv` | byte-identical |
| `amplitudes.npy` | **1** value of 11,050,850 differs |
| `spike_positions.npy` | **6** values of 22,101,700 differ |

Seven float32 values out of 33.15 million, every one of them **1 ULP**
(relative 6.4e-08 to 1.2e-07), and all seven belonging to four adjacent spikes
(indices 2,903,739 / 2,903,741 / 2,903,749 / 2,903,750) — one small
neighbourhood, i.e. one batch. Stage counts match exactly throughout:
2133 → 1326 clusters, 1084 units, 893 with good refractory periods.

That is the `tF` wobble footprint from the section above, not a new one: only
`tF`-derived files moved, and no spike time or cluster assignment moved. Per
that same section, one matching sort is not proof — so a **second optimized run
was made on the identical config**, and it settles the attribution:

| pair | `amplitudes` | `spike_positions` | max ULP | spikes involved |
|---|---:|---:|---:|---|
| run 1 vs baseline | 1 | 6 | 1 | 2903739 / 41 / 49 / 50 |
| run 2 vs baseline | 2 | 4 | 3 | 2160796 / 2160801 |
| **run 2 vs run 1** | **3** | **10** | **3** | the union of both |

The two optimized runs disagree **with each other**, at *different* spikes than
either disagrees with the baseline, and by more. Had a change in this series
deterministically shifted a value, run 1 and run 2 would agree with each other
and both be offset from the baseline at the same index. Instead each run wobbles
independently, which is the signature of nondeterminism in the algorithm itself.
`pc_features.npy` moved too (87 of 1,326,102,000 bytes), consistent with `tF`
being the origin.

`spike_times`, `spike_clusters`, `spike_templates` and
`spike_detection_templates` are identical across **all three** pairings.

Run 2 took 633.64 s against run 1's 637.40 s (0.6% apart) with identical stage
counts, so the timing above is reproducible as well.

### Provenance of the input

The original `chunk12_9-11.bin` had been deleted, and was rebuilt from the raw
Litke files by the same converter. Two independent checks that the rebuild is
the same data:

* the converter's own sidecar `.csv` is identical to the original's — same
  source paths, same per-file sample counts;
* `cmp -i 2076000000:0 -n 3114000000` against `slice300.bin` (data009's first
  3M samples, which sit at that offset in the chunk) reports byte-identical
  over 3.1 GB.

42,080,520,000 bytes = 40,540,000 samples × 519 channels × int16.

---

## Second array geometry: 20260514A, 512 channels at 60 um

Everything above this section was measured on one recording, and therefore on
one set of tensor shapes. That is a real limitation, not a cosmetic one: §2's
bit-identity is pinned to reproducing cuBLAS's split-K-2 accumulation, and
**which Triton block size reproduces it is shape-dependent** — BLOCK_M=128
agrees at 20260724A's production shapes while 64 and 32 do not, and at the
small test shapes it is the other way round. Nothing in the code guarantees
that a config which matched at 519 channels still matches at 512. The gates
exist precisely because it might not.

`20260514A` is the first test of that. It is a macaque recording on a
**different array**: 512 electrodes at a uniform 60.0 um nearest-neighbour
pitch (array id 504), against 20260724A's 519 electrodes at 30.0 um
(array 1551).

| | 20260724A | 20260514A |
|---|---:|---:|
| array id | 1551 | 504 |
| channels | 519 | 512 |
| NN pitch | 30.0 um | **60.0 um** |
| extent | 720 x 780 um | **1890 x 900 um** |
| universal templates (`Nfilt`) | 4048 | **1920** |

Raw lives on the share at `/mnt/lab/Array-data/20260514A` — 39 recordings,
772.9 GB — and there is a complete Vision-format **kilosort2.5 sort** beside it
at `/mnt/lab/Array-data/Duke/sorted/20260514A/kilosort25/`, with `.ei`,
`.neurons`, `.sta`, `.params` and phy output under `ksfiles/`. That makes this
the better of the two datasets for the QA pipeline when it exists, because it
already has an independent sort to compare against; 20260724A does not.

`data000` (14 files, 25.5 GB, 27.6 min) was staged to
`~/Documents/Development/data/raw/20260514A/data000` and a 300-batch slice cut
from it with `tools/make_litke_slice.py` — 3.07 GB, 150 s, 512 channels — so
the A/B matches slice300's scale.

### The tuned profile's template grid is wrong for 60 um arrays

The first attempt **OOMed** on a 19.54 GB card before finishing detection. The
cause is not the optimizations; it is the settings. The lab pipeline's `tuned`
profile lists `VALIDATED_PITCHES = (30, 60)` and applies `dmin = 15.0`,
`dminx = 32.0` to both. But those two numbers *are* the 30 um array's own row
and column pitch (its rows are 15 um apart, its columns 30 um). On a 60 um
array they lay a candidate grid four times finer than the electrodes:

| probe | dmin / dminx | grid | templates kept |
|---|---|---|---:|
| 519 ch @ 30 um | 15 / 32 | 45 x 104 | 3983 |
| 512 ch @ 60 um | 15 / 32 | 119 x 120 | **14280** |
| 512 ch @ 60 um | 60 / 60 | 63 x 30 | 1890 |

Two things compound. The template count is 3.5x higher, and
`max_channel_distance = 66` — the cull that discards 15% of the grid on the
30 um array — discards **nothing** here, because on a 60 um array every grid
point is within 66 um of some electrode by construction. `B` is `Nfilt x NT`
and everything downstream of it scales the same way.

This run therefore used `dmin = dminx = 60`, i.e. candidate templates at half
the electrode pitch, which is the relationship kilosort's grid is designed
around. That yields **1920** universal templates. Recorded here as a finding
against the pipeline profile, not as a tuning recommendation: which dmin gives
the best *sort* on this array is a yield question and needs the QA pipeline,
not this file.

### All four gates passed on the new geometry

This is the result the section was run for. Every gate re-validated against the
stock path on the first real batch of this recording and enabled:

| gate | verdict at 512 ch / 60 um | validated on |
|---|---|---|
| `fused_detect` | enabled, **BLOCK_M=128 warps=4**, 15.6 ms/batch | 58,302,720 elements |
| `fused_peaks` | enabled, config (128, 4) | 19,434,240 elements |
| `fused_peel` | enabled | 15,618,246 elements |
| `fast_kpp` | enabled (graph) | 10,364 labels |

The `fused_detect` line is the informative one: the block size whose agreement
was known to be shape-contingent picked **the same BLOCK_M=128** at a geometry
it had never seen. That is evidence the config is not a coincidence of one
recording — it is not proof that it generalises further, and the gate stays.

### Timing and stage counts — slice scale (300 batches, 150 s)

Everything in this subsection and the next is **slice-scale**: 300 batches,
150 s, ~9% of one recording, and `data000` is 1 of 39. The production-scale run
on the whole recording is a separate subsection further down, and where the two
disagree the production numbers are the ones to quote.

Arm A (all optimizations active) sorted the slice in **57.94 s**: 1920
universal templates, 528,099 spikes and 1224 clusters from the universal pass,
817,489 spikes and 1030 clusters from the learned pass, 905 units of which 743
have good refractory periods. For scale, 20260724A's slice300 gives 600 units /
481 good — the wider array finds more, which is what you would expect from
~3x the tissue area.

Arm B (all five `KILOSORT_NO_*` switches set) reproduced **528,099 spikes and
1224 clusters exactly**, with universal detection at 35.35 s against arm A's
22.20 s (1.59x) and template clustering at 7.80 s against 4.79 s (1.63x).

| arm | switches | s |
|---|---|---:|
| A | none (all optimizations active) | **57.94** |
| B | all five `KILOSORT_NO_*` set | **84.26** |
| A2 | none (second run of A) | 45.69 |
| | A vs B | **1.45x** |

A2 ran the *same code as A* 12.25 s faster (45.69 vs 57.94), which is a 21%
swing between two identical runs in one sitting. Arm A paid first-run costs
this benchmark does not otherwise expose — Triton JIT compilation and autotune
for four kernels at shapes never seen before, plus a cold page cache on a
freshly written 3.07 GB slice. Quote the A-vs-B ratio, not A2-vs-B (1.84x),
which would be flattered by exactly that. It is also a reminder that the
same-session rule stated under *Cumulative* is not a formality.

**1.45x, not the 2.26x slice300 gives on 20260724A.** The explanation recorded
here originally was geometric — 1920 universal templates against 4048, so the
detect body that §2 attacks is a smaller share of the sort, and on an array with
half the templates detection is half the prize.

**That explanation was wrong, and the production run below disproves it.** Two
compounding measurement faults produced the 1.45x:

1. It is A-vs-B, and arm A paid the Triton JIT cost described just above. The
   warm comparison at the same scale is A2-vs-B = **2.01x**. The instinct to
   avoid quoting A2-vs-B was right in general — it *is* flattered by a warm
   cache — but here it made the fused arm look 39% slower than it is.
2. Slice scale under-weights exactly the stages the optimizations attack.
   Detection is 60% of the sort at slice scale and 78% at production scale, so
   the slice mix hides most of the prize.

At production scale this geometry gives **2.24x**, against 20260724A's 2.51x.
The geometric argument survives only as a much smaller residual, and no reader
should conclude anything broke.

### Byte-identical, three ways — slice scale

All three pairings, **23 of 23 files byte-identical, 0 differ**:

| pairing | result |
|---|---|
| A vs B (fused vs stock) | 23/23 identical |
| A vs A2 (same code twice) | 23/23 identical |
| A2 vs B | 23/23 identical |

So the optimization series is bit-identical on a second array geometry, at
half the template count, with every gate enabled rather than falling back.

Note A vs A2 came out clean here. That does **not** contradict the wobble
section above: the wobble was only ever observed at production scale (13
float32 of 33 M), and 20260724A's slice300 is likewise clean. A 300-batch slice
does not have the statistical reach to see it, which is exactly why one clean
slice comparison is not treated as proof anywhere in this file.

**This paragraph is why the production run below exists.** As written, the
slice's clean A-vs-A2 is a test that structurally cannot fail, so it is not
evidence of anything. The subsection after next runs the whole recording.

Logs are one per arm, deliberately, under
`scratchpad/20260514A_validation/logs/`:

| log | arm |
|---|---|
| `00_make_slice.log` | slice construction |
| `01_slice_A_fused.log` | all optimizations active |
| `02_slice_B_stock.log` | all `KILOSORT_NO_*` set |
| `03_slice_A2_fused.log` | second run of arm A (wobble control) |
| `04_compare_A_vs_B.log` | the A/B verdict |
| `05_compare_A_vs_A2.log` | same-code control |
| `06_compare_A2_vs_B.log` | third leg |

`run_slice_ab.sh` in that directory is the driver. Note it deliberately does
**not** `set -u`: conda's own `activate.d` hooks read unset variables and abort
the script before the first sort starts.

### Production scale: the whole of data000

Everything above on this geometry is slice-scale. This subsection runs the
entire recording: **33,140,000 samples, 1657.0 s, 3314 batches, 33.94 GB** —
11.05x the slice. Same `settings.json`, same probe, same `--invert-sign`; the
only thing that differs from the slice A/B is `--data`.

**The input was verified before any GPU time was spent**, against the
file-order bug class that silently corrupted run E in the A/B/C/D/E comparison
— a concatenation that reads the right bytes in the wrong order yields a file
of exactly the right size that sorts to garbage, so a size check proves nothing:

| check | result |
|---|---|
| a. 14 part files in numeric order, no gaps | pass |
| b. `33935360000 == 33140000 x 512 x 2` | pass |
| c. first 3.07 GB `cmp`-identical to `slice300_514a.bin` | pass |
| d. all 13 inter-file seams + 12 random interior blocks match the reader | pass |
| e. tail is signal (std 49.9, range -811..1270), not zero padding | pass |

Check (c) is the decisive one: it makes the 34 GB file a strict superset of the
input that produced the slice result, so an ordering defect would have to be
identical in both files to survive. Check (d) pins every seam independently.

#### Byte-identical at production scale, three ways

| pairing | result |
|---|---|
| A vs B (fused vs stock) | **23/23 identical, 0 differ** |
| A vs A2 (same code twice) | **23/23 identical, 0 differ** |
| A2 vs B | **23/23 identical, 0 differ** |

This is the first run on this geometry with the statistical reach to see the
run-to-run wobble, and it did not wobble. The compared set is substantive —
737 MB `pc_features.npy`, 278 MB `templates.npy`, 6,168,855 kept spikes,
amplitudes, spike positions, cluster assignments — and `compare_sorts.py`
compares through `.view(np.uint8)` behind a dtype/shape gate, so `+0.0`/`-0.0`
and `NaN` cannot produce a false match. Stage counts are identical in all three
arms at every stage (3,606,390 spikes -> 2837 clusters -> 6,168,727 spikes ->
2558 clusters -> 2223 units -> 1630 good), which corroborates independently of
the comparison tool.

One clean production run is still not proof that this geometry never wobbles;
it is the strongest evidence available, and the wobble on 20260724A was itself
intermittent.

#### Timing: 2.24x, and why the slice said 1.45x

Quoting kilosort's internal `Total runtime` for all three arms. Arm A's harness
wall-time line was lost to a spurious background-task kill (see below), but the
internal timer is measured identically inside the same code in every arm.

| arm | switches | `Total runtime` |
|---|---|---:|
| A | none (all optimizations active) | **383.60 s** |
| B | all five `KILOSORT_NO_*` set | **858.81 s** |
| A2 | none (second run of A) | **383.57 s** |
| | **A vs B** | **2.24x** |

Per stage, against the warm arm A2:

| stage | slice B/A2 | full B/A2 | share of A2 runtime, full |
|---|---:|---:|---:|
| universal detect | 3.24x | **3.80x** | 25% |
| learned detect (peel) | 1.62x | **1.96x** | 53% |
| universal cluster | 1.66x | 1.19x | 6% |
| learned cluster | 1.72x | 1.16x | 8% |
| merge | 1.03x | 1.00x | 1% |
| **total** | **2.01x** | **2.24x** | |

Three things to take from this table, recorded because two of them refute
predictions written into the driver *before* the run:

1. **Clustering gains fall with scale; detection gains rise.** Clustering
   1.7x -> 1.16x reproduces the 20260724A slice->production pattern (2.47x ->
   1.11x, 2.38x -> 1.23x). Detection went the other way.
2. **The total rose, and the prediction that it would fall was wrong.** The
   prediction reasoned from 20260724A's *total* without decomposing. Detection
   is 60% of the slice sort and 78% of the production sort, so the stages that
   speed up most gain weight at scale. Predict per stage and weight by share,
   or don't predict.
3. **Slice A-vs-A2 was Triton JIT, confirmed — but not by the mechanism
   claimed.** The 21% same-code gap collapsed to 383.60 vs 383.57 s, a **0.008%**
   gap. The prediction said fixed costs would amortize over 11x the work.
   They didn't: `~/.triton/cache` holds 36 kernel files stamped during the slice
   session and **zero** stamped during the production session, so the full run
   compiled nothing at all — the on-disk cache was already warm. The JIT
   attribution is confirmed; the amortization claim is untested.

#### The live-tile LUT at production scale

Validated on the same recording against the byte-checked baseline above.
Two LUT runs, so this carries its own wobble control:

| pairing | result |
|---|---|
| LUT run 1 vs `full_A` (no LUT) | **23/23 identical, 0 differ** |
| LUT run 1 vs `full_A2` (no LUT) | **23/23 identical, 0 differ** |
| LUT run 1 vs LUT run 2 | **23/23 identical, 0 differ** |

| stage | stock | fused, no LUT | fused + LUT | vs stock |
|---|---:|---:|---:|---:|
| universal detect | 366.80 s | 96.47 s | 95.18 s | 3.85x |
| **learned detect (peel)** | 397.61 s | 202.71 s | **138.71 s** | **2.87x** |
| universal cluster | 28.26 s | 23.68 s | 23.93 s | 1.18x |
| learned cluster | 37.65 s | 32.42 s | 32.68 s | 1.15x |
| merge | 5.30 s | 5.29 s | 5.25 s | 1.01x |
| **Total runtime** | **858.81 s** | 383.60 s | **319.10 s** | **2.69x** |

Second LUT run 319.84 s. The LUT is worth **1.20x** here against 1.11x on the
slice, the expected direction: the peel is 53% of a production sort and 32% of
a slice one, so the slice under-weights exactly what this change attacks. That
prediction is recorded as correct only because it was made per stage and
weighted by share -- the two predictions that failed earlier in this section
both came from reasoning about totals.

Peel share after the change: 43.5% of the sort, still the largest single
stage, so it remains the place to look.

#### Both peel tails fused, and one trap worth the whole exercise

After the LUT, the statement profile of the learned pass reordered: the two
tails around the peel became 33.1% together, ahead of `peel_subtract`'s 30.8%.
Both are now one kernel each (`fused_peel_cond.py`, `fused_peel_store.py`),
plus `BLOCK_R` 16 -> 8 swept on the real ctc.

| pairing | result |
|---|---|
| M1 vs `full_A` / `full_A2` (pre-LUT) | **23/23 identical** |
| M1 vs `full_L1` / `full_L2` (LUT only) | **23/23 identical** |
| M1 vs M2 (wobble control) | **23/23 identical** |

| stage | stock | pre-LUT | + LUT | + tails | vs stock |
|---|---:|---:|---:|---:|---:|
| universal detect | 366.80 s | 96.47 s | 95.18 s | 95.62 s | 3.84x |
| **peel** | 397.61 s | 202.71 s | 138.71 s | **128.84 s** | **3.09x** |
| universal cluster | 28.26 s | 23.68 s | 23.93 s | 23.58 s | 1.20x |
| learned cluster | 37.65 s | 32.42 s | 32.68 s | 32.22 s | 1.17x |
| **Total runtime** | **858.81 s** | 383.60 s | 319.10 s | **309.03 s** | **2.78x** |

Second run 311.10 s.

**`tl.sqrt` is not bit-identical to torch's `**.5`.** This is the single most
transferable finding in this section. `th_amps = cmax[iX]**.5` was flagged in
the roadmap as needing a check, and the two sides disagree:

* torch's `x**.5` IS correctly-rounded sqrt -- compared against `torch.sqrt`
  on 4,194,304 random positives and on 0.0/-0.0/1.0/4.0/denormal/3.4e38/inf,
  every bit pattern agreed. So the stock side is clean.
* Triton's `tl.sqrt` lowers to the APPROXIMATE hardware instruction and
  differed on 7 of 26 real values by 1 ULP. `tl.math.sqrt_rn` is
  round-to-nearest and matches exactly.

The gate rejected the kernel on its first peel and fell back to stock, so no
output was ever at risk -- which is the entire argument for the gates. Nothing
else in this series has had an operation that looks IEEE-exact and is not;
assume the same for any future kernel reaching for a transcendental,
a reciprocal, or a rsqrt.

**The synced statement profiler overstates launch-bound blocks.** The
condition tail measured 17.5% of the learned pass. Fusing 73% of it at 5.1x
(45.7 -> 8.9 us per call) returned **1.03x** end to end. A CUDA sync around
each statement forbids exactly the overlap those launches normally get, so it
inflates precisely the blocks it is used to find. Treat every share in
`tools/profile_peel_statements.py` as an upper bound on what fusing that
statement can return. The store tail behaved the same way: 15.6% by the
profile, 1.085x in practice.

#### And it holds on the 30 um array too

Everything above is the 60 um geometry, so the same one-recording gap this
whole section exists to close would reopen if it stopped there. Re-run on
`slice300.bin` (20260724A, 519 ch at 30 um, 801-unit ctc), replaying that
sort's own `ops.npy`:

| pairing | result |
|---|---|
| new build vs LUT+both tails OFF | **23/23 identical** |
| new build vs itself (wobble control) | **23/23 identical** |

All three gates enabled at the new shapes without falling back --
`live-tile LUT`, `peel cond` (config 256,4) and `peel store`.

| stage | tails+LUT off | on |
|---|---:|---:|
| peel | 10.49 s | **6.81 s** (1.54x) |
| total | 43.77 s | 40.00 s (1.09x) |

Same 610,758 / 773,338 spikes, 956 / 683 clusters, 590 units, 480 good.

#### 30 um at production scale: the wobble shows up, and is not ours

Run on a **2000-batch cut of 20260724A/chunk12_9-11** (20.76 GB, 20,000,000
samples, `cmp`-verified as an exact prefix of the 42 GB chunk) -- 6.7x the
slice, 5.15 M spikes in the learned pass, which is the reach that matters
because the wobble was only ever seen on THIS recording at THIS scale.

Four arms, so BOTH codes have a same-code control. That is the whole point:
with only one run of the old code the result is unattributable.

| pairing | files differing | total bytes |
|---|---:|---:|
| N1 vs N2 -- **new code vs itself** | 3 | 33 |
| **O1 vs O2 -- old code vs itself** | **5** | **423** |
| N1 vs O1 / N2 vs O1 -- cross | 5 | 249 / 221 |
| N1 vs O2 / N2 vs O2 -- cross | 5 | 235 / 268 |

**Every defining output is byte-identical in all six pairings**:
`spike_times`, `spike_clusters`, `spike_templates`,
`spike_detection_templates`, `kept_spikes`, `templates_ind`, `whitening_mat`,
`channel_map`. Which spikes exist, which cluster each belongs to and which
template matched never move. The differences are confined to the four
tF-derived float arrays -- exactly the set `compare_sorts.py` documents.

Magnitudes, and this is the attribution:

| pairing | file | n differ | max ULP | max rel |
|---|---|---:|---:|---:|
| **O1 vs O2 (same code)** | pc_features | 151 of 154 M | **4502** | **4.13e-04** |
| | templates | 185 of 28 M | 176 | 1.61e-05 |
| N1 vs O1 (cross) | pc_features | 96 of 154 M | 1354 | 9.51e-05 |
| N1 vs N2 (same code) | pc_features | 27 of 154 M | 120 | 1.03e-05 |

The LARGEST disagreement anywhere in the matrix is between two runs of
**identical, unmodified code**, and it exceeds every new-vs-old pairing. A
systematic difference introduced by the peel work would make cross > same-code;
it does not. By the three-way test stated under *Cumulative*, these belong to
the program.

**So do not claim byte identity for this recording at this scale -- it is not
available to claim.** kilosort4 is not byte-reproducible on 20260724A at
production scale, for stock and optimized code alike. The defensible statement
is: defining outputs identical, residual float wobble demonstrably the
program's own.

Note the contrast with 20260514A, where the full 3314-batch run came out 23/23
identical across five pairings including two same-code controls. The wobble is
recording-specific, not universal, which is a new fact -- it was previously
only known to appear on 20260724A, but never checked for absence elsewhere at
matching scale.

Speed at this scale, same session:

| arm | peel | total |
|---|---:|---:|
| LUT + both tails OFF | 108.94 s | 274.17 / 274.61 s |
| **on** | **66.51 s** | **234.32 / 235.46 s** |
| | **1.64x** | **1.17x** |

#### ROOT CAUSE of the wobble, and byte identity recovered

The wobble has been carried as "root cause unknown" since it was first seen.
It is a **nondeterministic CUDA reduction**, and the proof is a single flag.

Two runs of the same code on the same 2000-batch cut, with
`torch.use_deterministic_algorithms(True)` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` (`run_full_sort.py --deterministic`):

| pairing | mode | result |
|---|---|---|
| N1 vs N2 (new vs new) | default | 3 files differ |
| O1 vs O2 (old vs old) | default | 5 files differ, pc_features 4502 ULP |
| C1 vs C2 (new vs new) | cuBLAS workspace pinned only | 5 files differ |
| **D1 vs D2 (new vs new)** | **deterministic** | **23/23 identical** |
| **DO1 vs D1 and D2 (OLD vs NEW)** | **deterministic** | **23/23 identical** |

Three things follow, in order of importance:

1. **The optimizations ARE byte-identical on this recording at production
   scale.** The earlier "not claimable" conclusion was correct about the
   evidence available at the time and wrong as a permanent verdict: remove the
   program's own nondeterminism and old and new code agree on all 23 files.
   Both geometries are now proven at production scale.
2. **ROOT CAUSE: the overlapping-window scatter.** Found by bisect, not by
   argument. Dumping each stage's outputs from two runs of identical code puts
   the FIRST divergence in `template_matching.extract`:

   | stage output | differs |
   |---|---|
   | `A_univ_st` / `A_univ_tF` (universal detect) | **identical** |
   | `B1_clu` / `B1_Wall` (template clustering) | **identical** |
   | **`C_learn_st` / `C_learn_tF` (learned extract)** | **11 / 163 bytes** |
   | `B2_Wall` (final clustering) | 3 bytes, inherited |

   The one knowingly-nondeterministic operation in there is the peel's
   overlapping-window scatter: `Xres[:, iX + tiwave] -= ...` is advanced-index
   `index_put_`, which on duplicate indices is last-write-wins and
   nondeterministic, and `fused_peel` falls back to it for exactly those
   phases. Forcing every phase onto the fused kernel instead (deterministic by
   construction -- one program per element, no duplicate writes) makes two runs
   **byte-identical, 0 of 618,187,920 bytes**, against 11 / 163 with the normal
   fallback. That is the proof.

   **An earlier version of this section recorded this hypothesis as REFUTED.
   The refutation was wrong**, and the error is worth keeping: it compared the
   *presence* of overlapping phases across recordings (20260724A/2000 batches
   6 of 94,633; 20260514A/full 3 of 144,642) and concluded that since both have
   them and only one wobbles, they cannot be the cause. Presence is necessary,
   not sufficient -- whether an overlap actually diverges depends on the values
   at the duplicated indices. A population-level correlation cannot refute a
   mechanism that fires conditionally.

   This also explains why `use_deterministic_algorithms(True)` works: it
   selects a deterministic `index_put_`.

   **A "zero-cost alternative" looked available here and DOES NOT WORK.**
   The idea was to route overlapping phases to the fused kernel permanently,
   on the reasoning that it is deterministic by construction. It is not. The
   kernel does a NON-ATOMIC read-modify-write (`o = load(out); store(out,
   o - v)`), so when two spikes' windows overlap, their programs race on the
   same element. It is deterministic only when the windows are DISJOINT --
   which is exactly the condition it already tests for.

   This was implemented, and a unit test with severe overlap (positions 3
   apart, 123-wide windows) failed immediately on two identical runs. It was
   then reverted. The full-run diagnostic that showed 0 of 618,187,920 bytes
   differing is not contradicted but is not a guarantee either: with only 6
   overlapping phases the races are rare and happened to resolve the same way
   twice.

   So the diagnostic above still localises the ROOT CAUSE correctly -- forcing
   those phases off the stock path removed the observed divergence -- but
   "force the fused kernel" is swapping one nondeterministic operation for
   another, not a fix. A real zero-cost fix needs atomics or serialised
   handling of the overlapping spikes, and is not done.

   `--deterministic` remains the supported way to get reproducibility.

   **It is safe as a QA default: it pins the answer, it does not shift it.**
   On 20260514A -- where there is no wobble to remove, five pairings having
   come out 23/23 -- a deterministic run is **23/23 identical to the
   non-deterministic one**. So enabling it does not perturb a sort that was
   already reproducible; it only removes the freedom where freedom existed.
   Cost on that geometry is 348.12 s against 309.56 s, 12.5% (7% on
   20260724A). No op reported a missing deterministic implementation on either
   recording.

3. **Deterministic mode pins the answer, it does not reproduce a prior run.**
   `D1 vs N1` differs on 3 files. Turning the flag on changes results by the
   usual wobble magnitude relative to any particular earlier nondeterministic
   run; it does not reconstruct one.

Cost: 251.5-264.9 s against 234.3 s, roughly 7-13%.

Not yet pinned: WHICH reduction. Two candidates were checked and cleared --
`clustering_qr._counts_into`'s `scatter_add_` accumulates only `ones`, and
identical addends make any atomic ordering produce the same float64 sequence;
and `postprocessing`'s means are fixed-order reductions. Narrowing further
needs bisection and was not done.

**REJECTED HYPOTHESIS, recorded so it is not re-run.** The obvious suspect was
the overlapping-window scatter: stock's `-=` is last-write-wins on duplicate
indices, and `fused_peel` documents it differing on 152-838 elements between
runs. `tools/count_overlapping_phases.py` refutes it as the explanation:

| recording | peels | overlapping | wobble? |
|---|---:|---:|---|
| 20260724A slice300 | 14,491 | **0** | no |
| 20260724A 2000 batches | 94,633 | 6 (0.0063%) | **yes** |
| 20260514A full data000 | 144,642 | 3 (0.0021%) | **no** |

Overlapping phases occur in both recordings but only one wobbles, so their
presence does not distinguish the cases.

**What this changes for the QA pipeline.** Byte identity has been the only safe
currency in this work *because* kilosort4 was not reproducible, which forced the
four-arm attribution discipline and made "23/23" unavailable on some
recordings. With this flag the program is reproducible, so a QA pipeline can
diff two sorts directly instead of reasoning about attribution. Use
`--deterministic` for any comparison run; leave it off for timing.

#### The background-task monitor kills on MemFree, and that is a false positive

The first attempt was killed between arms by the harness reporting "system is
running low on memory". `/proc/meminfo` at that moment:

| field | GB |
|---|---:|
| MemTotal | 197.3 |
| **MemAvailable** | **190.6 (96.6%)** |
| MemFree | 7.5 |
| Cached | 177.1 |

**This recurred on 2026-09-06** and is not a one-off: a second run was killed
with MemAvailable at **190.7 of 197.3 GB (96.6%)** and MemFree at 25.2 GB, so
the trigger is not a fixed small-MemFree threshold. The reliable workaround is
to run production sorts in the FOREGROUND, where the background-task monitor
is not involved at all; a fused run is ~356 s wall and fits inside the tool's
600 s cap. Reclaiming the cache of every large file EXCEPT the input also
helps, and preserves A/B cache fairness because the input stays warm exactly
as the baseline arms had it.

kilosort's own peak was 9.30 GB. `MemFree` was low purely because streaming a
34 GB input (and the 25.5 GB raw it was built from) fills the page cache with
fully reclaimable pages — which is what Linux is supposed to do. Confirmed by
`drop_cache.py`, which calls `posix_fadvise(POSIX_FADV_DONTNEED)` (no root
needed, unlike `/proc/sys/vm/drop_caches`): 74.3 GB reclaimed across two calls,
MemFree 7 -> 68 GB, MemAvailable flat at 181 GB. Any long job here that streams
a large file is exposed to this; `MemAvailable` is the field that matters.

**Cache fairness when resuming.** Reclaiming cache and then restarting arm B
cold would have penalized B and *inflated* the reported speedup. The resume
script therefore re-warms the input with `cat $BIN > /dev/null` before B, to
restore arm A's conditions. The A -> B -> A2 order is kept for the same reason:
it lets B inherit a warm cache, which understates the A/B speedup rather than
overstating it. The bias is deliberate and conservative in both cases.

Logs under `scratchpad/20260514A_validation/logs/`, harness committed to
`tools/validation_runs/`:

| log | arm |
|---|---|
| `10_make_full.log` | full .bin construction |
| `11_verify_full.log` | the five pre-flight checks |
| `12_full_A_fused.log` | all optimizations active |
| `13_full_B_stock.log` | all `KILOSORT_NO_*` set |
| `14_full_A2_fused.log` | second run of arm A (wobble control) |
| `15_full_compare_A_vs_B.log` | the A/B verdict |
| `16_full_compare_A_vs_A2.log` | same-code control |
| `17_full_compare_A2_vs_B.log` | third leg |

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
| + graph-captured k-means++ (§5) | 53.3 |
| + fused peak tail (§6) | 52.3 |

The §6 row is same-session against 52.9 s for the §5 build and is **within
drift** -- see §6 for why slice300 cannot resolve it.

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
* **Where `tF`'s run-to-run wobble comes from.** The footprint is pinned and it
  is now confirmed at production scale — two runs of *identical* code on
  identical input disagree on 13 float32 values of 33 M, at different spikes
  each time — but the **cause is still not identified**. Worth finding: until
  it is, no end-to-end byte-identity claim can rest on a single run, and the
  sorter cannot be made reproducible for a paper.
* **`swarmsplitter.split`** is now the second-largest item in the template pass
  (1.80 s, 12.3%) and is CPU-side.
* **Clustering at production scale.** §4 and §5 give 1.11×/1.23× there against
  2.47×/2.38× on slice300 — the remaining clustering time is data movement, not
  launch overhead, so a different approach would be needed to move it. Whether
  it is worth moving at all is now questionable: clustering is 17.5% of the
  optimized production sort, against detection+peel at 76.3%.
* **Where the profiling harnesses live.** Everything that verifies the claims in
  this file is in a session-local scratchpad under `/tmp`, which does not
  survive a reboot. Not yet preserved in this repo — see below. `tools/` is the
  start of fixing this.

---

## 7. Measured and rejected: dirty-region peel

Recorded so nobody builds it twice. **Rejected on measurement, before any
kernel was written.**

The idea: `run_matching`'s peel loop recomputes `torch.max(B, 0)`, the
`max_pool1d`, the two comparisons and the `nonzero` over all `NT` columns on
every one of up to `max_peels` iterations — but `peel_subtract` writes only
`B[:, iX + trange]`, `trange = arange(-nt, nt+1)`. So B changes only within
±`nt` of a spike, `Cfmax`/`imax` only there, and `cmax` (a `2nt+1` sliding max)
only within ±`2nt`. A position that failed `cnd1 & cnd2` at iteration *t* and
whose inputs did not change still fails it at *t+1*, so every new candidate must
lie within ±`2nt` of one of iteration *t*'s spikes. Keep persistent full-width
`Cfmax`/`imax`/`cmax`, recompute only the dirty columns. Over-approximating the
dirty set is safe, so block granularity would have been fine. `max` and
`max_pool` are *selections*, not accumulations, so restricted values are exactly
equal — the only identity risk was `imax` tie-breaking.

**It does not pay.** `tools/measure_peel_dirty_region.py` measures the dirty set
against `NT` over a full slice300 sort (14,557 peel iterations, 815 units,
NT 10,122, nt 61):

| iter | median spikes | median dirty cols | dirty % |
|---:|---:|---:|---:|
| 0 | 82 | 10,075 | 99.5% |
| 5 | 69 | 9,780 | 96.6% |
| 20 | 62 | 9,380 | 92.7% |
| 30 | 53 | 8,354 | 82.5% |
| 40 | 27 | 4,447 | 43.9% |
| 45 | 24 | 3,888 | 38.4% |

**Total reduction 1.27× — 21.5% of the columns are provably redundant.** The
peel does not converge: ~61 spikes per iteration at the median, each spreading a
`4nt+1 = 245`-column dirty window, so ~60 spikes already saturate NT ≈ 10k. The
premise ("late iterations touch almost nothing") is simply false for this data.
1.27× on ~30% of one stage is ~4% of the sort, for a Triton kernel plus an
`imax` tie-break identity argument. Not worth it. **Note this is data-dependent**
— on a probe where the peel converges in a handful of iterations the arithmetic
would look completely different, so re-measure before rejecting it elsewhere.

## Where the learned pass actually spends its time

Statement-level, same slice300 sort, CUDA-synced per block
(`tools/profile_peel_statements.py`). Coarse blocks, because per-statement syncs
inflate exactly the small items; the two agree to 0.3 s of 10.3 s, so the
attribution is trustworthy.

| block | s | share | µs/call |
|---|---:|---:|---:|
| `peel_subtract` (already fused, §3) | 4.70 | 45.7% | 323.0 |
| condition tail: relu→square→edges→`max_pool1d`→`cnd1`/`cnd2`→`nonzero` | 1.63 | 15.8% | 111.1 |
| store tail: `imax[iX]`, 4 slice writes, `B[iY,iX]`, `s[iY]`, `**.5` | 1.46 | 14.2% | 50.2 |
| `torch.max(B, 0)` | 1.25 | 12.2% | 85.6 |
| `einsum` (once per batch) | 0.70 | 6.8% | 2329.3 |
| `conv1d` (once per batch) | 0.44 | 4.3% | 1469.0 |

Two things this corrects:

* **`torch.max` is not a problem.** 33 MB (815×10,122 float32) in 85.6 µs is
  384 GB/s — near peak. The notes' earlier "10.3%" reading invited optimising it;
  there is nothing there to win.
* **The two tails are 30% of the pass and are almost pure launch overhead.**
  Roughly 20 kernels per peel iteration, each on a 10,122-element (40 KB) array —
  40 KB at 31 µs is 1.3 GB/s, three orders off bandwidth. This, not the
  reductions, is the remaining target in the learned pass.

---

## 8. Measured, CONFIRMED, and now BUILT: `ctc` is 89% exact zeros

Unlike §7, this one survived its measurement. The census below is what the
build was based on; the build landed as the live-tile LUT
(`_scatter_sub_lut_kernel` in `kilosort/fused_peel.py`, disabled by
`KILOSORT_NO_PEEL_LUT=1`) together with the two fused peel tails
(`NO_PEEL_COND`, `NO_PEEL_STORE`).

**Remeasured at production scale, 2026-09-06** — same binary at `f513a07`,
both arms replaying 20260724A/chunk12_9-11's own `ops.npy` against the 40 GB
production `.bin`, back to back in one sitting:

| arm | env | wall |
|---|---|---:|
| NEW | all optimizations on | **624 s** |
| OLD | `NO_PEEL_LUT=1 NO_PEEL_COND=1 NO_PEEL_STORE=1` | **700 s** |

**1.12×**, 76 s saved, against 1.24× on 20260514A where the work was developed.
`compare_sorts.py`: 18 of 23 arrays byte-identical, including `spike_times`,
`spike_clusters`, `spike_templates`, `kept_spikes`, `whitening_mat` and
`channel_map`. The 5 that move are the tF-derived floats and carry the known
20260724A wobble signature (see "Open, not resolved"), which one run per arm
cannot attribute — that needs the 4-arm NEW/NEW2/OLD/OLD2 design.

**Do not A/B against the archived 637.4 s.** This machine drifts ~10% between
sittings; today's own OLD arm measured 700 s. Comparing today's NEW against the
archived figure gives 1.02× and the false conclusion that the LUT bought
nothing.

**The observation.** `fused_peel._scatter_sub` applies

    B[r, pos[s] + t] -= amp[s] * ctc[r, iY[s], t]

over *every* unit row `r`. But `U` is built by
`clustering_qr.mean_cluster_templates`, which writes into `torch.zeros(...)` and
only ever touches one cluster centre's channels — so `U` is **exactly** zero off
a template's local channel set. And

    ctc[i, j, t] = sum_{k,m} sum_l U[i,k,l] * U[j,m,l] * WtW[k,m,t]

contracts `l` over channels (`U` is `(n_units, n_pc, n_chan)`; `WtW` is indexed
by PC, not channel — easy to misread). If units `i` and `j` have disjoint
channel support, every product in that sum is exactly 0, so the entire
`(i, j, :)` block is zero and that row's subtract is a no-op.

**The census** (`tools/measure_ctc_sparsity.py`, slice300, real `ctc`):

| | |
|---|---:|
| `ctc` | 801 × 801 × 123 |
| channels per template (median) | **18 of 519** |
| rows that are exactly `+0.0` | **88.87%** |
| live rows per spiking unit (median) | 89 of 801 |
| row-tiles per spike at `BLOCK_R=16` | 51 |
| **tiles fully skippable** | **80.44%** |
| live tiles per spiking unit (median) | **10 of 51** |

`peel_subtract` is 45.7% of the learned pass (251.9 s of a 623.6 s production
sort), and the B subtract is ~76% of the fused kernel's element traffic
(801×123 against Xres's 519×61). So a 5.1x cut in B tiles is aimed at the
single largest line item in the sort.

**Why skipping is bit-identical, and the two ways it could not have been.**
This is a §6-class argument — no arithmetic is reproduced, only omitted:
`o - (+0.0) == o` for every float `o`, including `-0.0`. Two preconditions,
both **measured rather than assumed**, because both are the kind of thing that
is invisible to `torch.equal`:

1. **The zero blocks must be `+0.0`, not `-0.0`.** `o - (-0.0)` maps `-0.0` to
   `+0.0`. Products `0.0 * x` are `-0.0` for `x < 0` and `-0.0 + -0.0` stays
   `-0.0`, so a negative zero in `ctc` is possible in principle. Measured:
   **0 of 801² blocks** carry one. The gate must compare bit patterns
   (`.view(torch.int32)`), never `== 0`.
2. **`amp > 0`,** or `amp * 0.0` is `-0.0` and precondition 1 stops helping.
   This follows from detection — a peak needs `relu(max B)**2 > Th**2 > 0`, so
   `B[iY,iX] > 0`, and `s > 0` — but it was checked anyway: smallest `amp` over
   the whole sort was **0.0876**.

Note the interaction: precondition 2 is what makes precondition 1 sufficient.
Neither alone is enough, and a settings change that allowed `Th_learned = 0`
would break 2 without touching 1. Gate on both.

**What is NOT yet known.** Whether the saving is realisable. §3's docstring
measured the peel as *launch- and allocation-bound, not bandwidth-bound*, and
an in-kernel early exit keeps the grid at `(n_spk, 51)` while only cutting
traffic. Cutting the grid instead — a per-unit compacted list of live tiles,
built once since `ctc` is batch-invariant — is what would cut launches, at the
cost of a real rewrite. Bench the early exit first: it is a few lines and it
answers which of the two limits actually binds before anyone commits to the
rewrite.

**Confirmed on both geometries.** The prediction was that sparsity would be
*higher* on 20260514A, since template footprints stay local while the array
gets bigger. Recorded before the run, and it held:

| | 20260724A, 30 µm | 20260514A, 60 µm |
|---|---:|---:|
| `ctc` | 801 × 801 × 123 | 1031 × 1031 × 123 |
| channels per template (median) | 18 of 519 | 15 of 512 |
| blocks exactly `+0.0` | 88.87% | **89.58%** |
| blocks carrying a `-0.0` | 0 | **0** |
| tiles skippable (`BLOCK_R=16`) | 80.44% | **82.22%** |
| live tiles per spike (median) | 10 of 51 | 11 of 65 |
| smallest `amp` in the sort | 0.0876 | **0.0930** |

That both preconditions hold on two independent array geometries is the point:
it says the zeros come from the *structure* of how `mean_cluster_templates`
builds `U` — local channel support written into a zeros tensor — and not from
some property of one recording. Unit count does not track template count here
(1031 units from 1920 universal templates against 801 from 4048), so the
sparsity is genuinely geometric rather than a headcount artifact.

---

## 9. Where the time actually goes now — 31 production sorts, not a slice

Every sort writes its per-stage timers into `ops.npy` (`runtime_st0`,
`runtime_clu0`, `runtime_st`, `runtime_clu`, `runtime_merge`,
`runtime_postproc`). That is a free bottleneck census over every sort this fork
has ever produced — no instrumentation, no re-runs — and it is worth far more
than slice300, because slice300's stage mix is not production's.

Read off 31 fork-era sorts under `data/sorted/*/chunk*/kilosort4/ops.npy`,
**2.99 GPU-hours** of real sorting across both array geometries:

| | preproc | **st0** | clu0 | **st** | clu | merge | postproc |
|---|---:|---:|---:|---:|---:|---:|---:|
| all 31 | 0.3% | **41.7%** | 9.9% | **31.8%** | 11.0% | 1.1% | 3.3% |
| 519 ch @ 30 µm (2.16 h) | 0.3% | **46.8%** | 7.8% | 31.5% | 9.5% | 0.7% | 2.6% |
| 512 ch @ 60 µm (0.83 h) | 0.3% | 28.2% | 15.5% | **32.6%** | 14.9% | 2.2% | 5.0% |

**Universal detection (`st0`) is now the single largest line item at
production scale — 41.7%, ahead of the whole learned pass at 31.8%.** §2
already took it 3.7× on slice300; it is back on top because everything around
it got faster, not because that work regressed.

Two things this says that slice300 could not:

1. **The geometry splits the profile.** `st0` is 46.8% at 30 µm against 28.2%
   at 60 µm, because `st0` scales with the universal template count and the
   30 µm grid lays 4,048 of them against 1,920 at 60 µm. Any further `st0`
   work pays roughly 1.7× more on the 519-channel arrays.
2. **Clustering is not negligible at 60 µm.** `clu0 + clu` is 30.4% there
   against 17.3% at 30 µm — so the earlier "clustering is 17.5%, questionable
   whether it is worth moving" conclusion is a 30 µm conclusion and does not
   transfer.

Reproduce with the loop over `ops.npy` in this section's header; it costs
nothing and should be re-read after every optimization, since the whole point
is that the ranking moves.

---

## 10. Candidate-only universal suppression: measured, not yet built

Third-party review (Astra, reading `3f0483b`) proposed deferring the dense
spatial reduction in `spikedetect.template_match` to above-threshold
candidates. Two of that review's seven items are already built here -- the
learned-pass tails are `NO_PEEL_COND`/`NO_PEEL_STORE`, and the "measure dirty
unit x time coverage" item is section 8 -- but this one is live, and it lands
on exactly the stage the production census (section 9) puts on top.

**The statement, at `spikedetect.py:323`:**

    Amax = torch.max(Aa.index_select(0, iC2_flat).view(nC2, Nfilt, -1), 0)[0]

`Aa` is `(Nfilt, NT)`; `iC2_flat` is `(nC2 * Nfilt)`. So this reads `Aa` with
**nC2-fold amplification**, densely, for every filter at every sample.

**The dependency is breakable, and that is the whole idea.** Candidates are
`As > Th_universal`, and `As` comes from `torch.max(A.abs(), 0)` -- it does not
depend on `Amax` at all. So the neighbourhood maximum can be computed for
candidates only, then compared to `As` exactly as now.

**Measured on 20260903A/chunk2, 74 real batches, 519 ch at 30 um:**

| | |
|---|---:|
| `nC2 x Nfilt x NT` | 100 x 4038 x 10122 |
| dense spatial reads per batch | **4.087e9** |
| peaks kept per batch (median) | 2422 |
| sparse reads at that count | 1.073e7 |
| **upper bound on the ratio** | **381x** |

The 4.087e9 confirms the review's 4.1 billion estimate on this repo's own
data. Note what the 381x is and is not: peaks are what survive BOTH the
threshold and the neighbourhood-max test, so they are a **lower bound** on
candidates, and 381x is therefore an **upper** bound on the saving.

**The true candidate count, now measured.** `fused_peaks.try_mask` was wrapped
with a counter reading `(As > Th_universal).sum()` that then delegates to the
real implementation, so the sort it ran was byte-identical to an unpatched one.
300 batches of `slice300.bin`, same 519 ch at 30 um geometry:

| | |
|---|---:|
| dense columns per call (`Nfilt x NT`) | 4.087e7 |
| candidates per call, median | **482,392** |
| candidates per call, mean | 748,307 |
| candidates per call, max | 3,733,187 |
| peaks kept, median | 1,682 |
| candidate fraction of all columns | **1.83%** |
| **true reduction ratio** | **54.6x** |

The real headroom is **54.6x, not 381x**: candidates outnumber the peaks that
survive both tests by roughly 290 to 1, so the peak count was never a usable
proxy for them. 54.6x on this statement's traffic is still large, but it is a
ceiling for one statement inside a stage that is 42% of a sort (16.3 s of
38.7 s on this slice), and the compaction that builds the candidate list must
itself read `As` densely -- a real kernel starts from 4.087e7 unavoidable reads
before it gathers anything. Amdahl, not the ratio, decides this one.

**Why the element count still overstates the win.** `fused_detect` computes
`Aa`, `imax` and `Amax` in one fused launch that already shares its read of
`A`, and `fused_peaks.try_mask` already short-circuits the temporal window for
blocks containing no candidate. Part of the theoretical saving is therefore
already collected. Benchmark before writing a kernel.

**Two identity hazards, both specific to this rewrite.** The mask is
`Amaxs == As`, an equality on a *selection* rather than an arithmetic result,
so ties are order-independent -- this is safer than the max/gather rewrite that
cost +567 spikes and -25 good units (see `template_match`'s note). But:

1. The edge zeroing `Amaxs[:, :nt] = 0` happens **before** the pool, so it
   changes pooled values near the batch edges. A candidate-only path must
   reproduce that, not just skip the edges.
2. `mask.nonzero()` fixes the output order (filter-major, then time). Any
   candidate compaction has to sort back to it.

**Fixtures.** `20260903A/chunk2.bin` (768 MB, 74 batches, ~14 s) carried the
first measurement but has since been cleaned off local disk; re-stage it with
`prepare_data.sh` if a small case is wanted. The candidate count above used
`slice300.bin` (3.1 GB, 300 batches, 38.7 s of sort, 49.8 s wall), which is a
production slice and the better fixture of the two: at 74 batches the median
candidate count is noisier than the 300-batch median reported here.

---

## 11. Process-level GPU concurrency: measured, and it does NOT work

The obvious parallelism play, and the one worth ruling out before any kernel
work: run **two sorts at once on the one card**. It needs no code, and it is
byte-identical by construction -- two OS processes, two result directories, no
shared mutable state, and the GPU is not a numerical participant in what they
share. The census made it look promising: peak allocation is 9.60 GB of a
19.54 GB card, so two fit; and the peel loop is documented launch-bound, which
is exactly the shape of workload where a second process fills the SM-idle gaps
between launches. `nvidia-smi` reading 100% does not contradict that -- that
counter is the fraction of time *any* kernel was resident, not SM occupancy.

Interleaved solo / pair / solo on `slice300.bin`, one process per arm:

| | |
|---|---:|
| solo A | 49.8 s |
| solo B (interleaved baseline) | 50.1 s |
| machine drift between them | 0.5% |
| two sorts back to back | 99.9 s |
| two sorts concurrent | 93.8 s |
| **throughput speedup** | **1.06x** |

**1.06x is nothing**, and most of even that is the ~11 s of Python/CUDA import
in each process overlapping, not GPU work overlapping. `compare_sorts.py` says
`23 files byte-identical, 0 differ` between the solo run and a concurrent one,
so the identity argument held -- it simply bought no time.

What this rules out, and it is worth the two minutes it cost: **the GPU is
already saturated at the whole-sort level.** The launch-bound diagnosis for the
peel loop is a statement about one loop, not about the sort, and it does not
generalize into free throughput from co-scheduling. Every remaining speedup on
this machine has to come from *doing less work*, not from overlapping more of
it. CPU-side parallelism is the exception and is already collected elsewhere:
staging the next chunk during the current sort (`run_fork_sorts.sh` prefetch)
is off the GPU's critical path entirely, and so is the EI computation.

Do not re-open this without a materially different setup (a second card, or
MPS with a workload that is actually launch-bound end to end).

---

## How the claims here were verified

The inventory, so the method survives even if the scripts do not. All of these
live in the session scratchpad
(`/tmp/claude-1001/-home-localadmin-…/scratchpad/`) and are **not in this
repo**; treat this list as the spec to rebuild from if they are gone.

| script | what it does |
|---|---|
| `run_full.py` | drives a full sort from a saved `ops.npy` + a flat `.bin`, and prints `TOTAL_WALL_SECONDS`. Both arms of the production A/B used it. |
| `cmp_sorts.py` | raw-byte comparison of two result directories, with per-file element/byte diff counts. Skips `ops.npy` (it stores timers and peak memory, so it always differs). |
| `make_slice.py` / `slice300.bin` | builds the 300-batch development benchmark. |
| `profile_cluster.py` | wraps every callee of `clustering_qr.run`, CUDA-syncs around each, prints per-pass tables and input-size percentiles. This is what showed `kmeans_plusplus` was 78–87%, correcting the standing assumption. |
| `dump_xd.py` | saves the 32 real `Xd` matrices (1002–12941 spikes) that the k-means++ identity checks run on. |
| `probe_kpp.py` | statement-level µs breakdown inside the k-means++ loop. |
| `kpp_census.py` | counts host-read branch outcomes over a whole sort — the 393-call census behind the guard argument. |
| `check_kpp.py`, `check_graph_kpp.py` | identity + speed harnesses over all 32 matrices; both compare raw bit patterns and the RNG end-state, not values. |
| `run_kpp_tests.py` | runs `tests/test_fast_kpp.py` without pytest, which **is not installed in any conda env on this machine**. |

The repo's own `tests/` (`test_fast_kpp.py`, and the fused-detect/peel gate
tests) are real pytest files and are the durable half of this; they pin the
*gates and guards*, which is where the safety argument lives, rather than the
kernels.

## 12. Is 160 s reachable? Measured answer: not from this shortlist

The stated goal was 1601.6 s -> 637.4 s -> **160 s**, a further 3.94x. This
section tests that target against measurement rather than hope. No code was
changed to produce any number here; every harness lives in the session
scratchpad and each one calls the untouched original function after timing a
replica of it, so the sorts that produced these numbers are byte-identical
(`compare_sorts.py`: **23 files byte-identical, 0 differ**, against a matched
control run).

### The arithmetic before any measurement

Against the optimized production budget (630.8 s of stage timers, 637.4 s
recorded total):

| stage | s | share |
|---|---:|---:|
| spike det. (learned) `st` | 255.6 | 40.5% |
| spike det. (universal) `st0` | 231.0 | 36.6% |
| clustering `clu0` + `clu` | 111.4 | 17.7% |
| postprocessing | 21.6 | 3.4% |
| merge | 9.8 | 1.6% |
| preprocessing | 1.4 | 0.2% |

Both detection stages together are 486.6 s, 77.1% of the sort. Reaching 160 s
means removing 470.8 s. **Deleting 100% of all spike detection lands at
144.2 s** -- so the target is only 15.8 s below a budget in which detection is
free. Nothing about that is a rounding error: 96.8% of the entire detection
budget has to disappear.

### Where the detection time actually goes

CUPTI is broken on this machine (`CUPTI_ERROR_INVALID_DEVICE`), so
`torch.profiler` reports zero device time and kernel-level attribution is not
available. Two CUDA-event measurements replace it. Events do not sync the host
between statements, so this does not repeat the mistake recorded in the
run-to-run wobble section, where synced statement profilers overstated
launch-bound blocks.

**GPU-busy vs host stall**, every call, all 300 batches of slice300:

| function | calls | host s | gpu s | gap | gap % |
|---|---:|---:|---:|---:|---:|
| `spikedetect.template_match` | 300 | 10.98 | 10.24 | 0.74 | 6.8% |
| `template_matching.run_matching` | 300 | 6.09 | 5.27 | 0.82 | 13.4% |
| `clustering_qr.cluster` | 184 | 5.67 | 5.67 | 0.00 | **0.1%** |
| `clustering_qr.kmeans_plusplus` | 184 | 4.22 | 4.22 | 0.00 | 0.1% |

Clustering is fully GPU-bound. Whatever is left in `clu0`/`clu` is *work*, not
launch overhead -- which retires the "launch overhead" half of the §5 story at
this scale and matches the production result that §5's slice speedups collapsed
from 2.47x/2.38x to 1.11x/1.23x. Detection has only 7-13% host gap, so CUDA
graphs over `st0` could recover at most ~6% of that stage.

**Block split inside each function** (3 probed batches, replayed on the real
inputs, shares of GPU time):

| `st0` = `template_match` | share | | `st` = `run_matching` | share |
|---|---:|---|---|---:|
| `fused_detect` fill (As/Amaxs/imaxs) | **83.9%** | | peel loop | **83.2%** |
| `conv1d` B | 10.4% | | `conv1d` + `einsum` | 16.8% |
| `fused_peaks` mask | 3.6% | | *of loop:* `peel_subtract` | 50.9% |
| `nonzero` + gathers + `adist` | 2.0% | | *of loop:* `max(B,0)` | 12.4% |

The replica runs ~18% faster than the in-situ calls (warm cache, second pass),
so the *shares* are the reliable output here, not the absolute extrapolation;
they are rescaled onto measured GPU-busy time below.

Two incidental facts worth recording. `max_peels` is 50 and the peel loop ran
**exactly 50 iterations on every probed batch** -- it is truncated by its cap,
never by the `len(xs)==0` break, so no proposal can reduce the iteration count
without changing results. And `kmeans_plusplus` is 74.4% of all clustering time.

### The ceiling, block by block

Measured shares applied to the production stage times:

| addressable block | prod s | % of sort | sort if it went to ZERO |
|---|---:|---:|---:|
| #1 `fused_detect` fill (st0) | 156.6 | 24.8% | 474.2 s |
| #3/#4/#5 peel loop (st) | 161.7 | 25.6% | 469.1 s |
| — of which `peel_subtract` | 82.3 | 13.0% | 548.5 s |
| #2 clustering `clu0`+`clu` | 111.4 | 17.7% | 519.4 s |
| — of which `kmeans++` | 82.9 | 13.1% | 547.9 s |
| `conv1d`+`einsum` (st) | 32.7 | 5.2% | 598.1 s |
| `conv1d` B (st0) | 19.4 | 3.1% | 611.4 s |

**All three shortlist targets driven to zero simultaneously gives 201.0 s**
(3.14x further) -- and that already misses 160 s by 41 s, with three infinite
speedups spent. What survives is 61.2 s of `st` non-kernel time, 54.9 s of `st0`
non-kernel time, 32.7 s of `einsum`, 21.6 s of postprocessing, 19.4 s of
`conv1d`, 9.8 s of merge.

Plausible scenarios, for calibration:

| scenario | total | further | overall vs 1601.6 |
|---|---:|---:|---:|
| 2x fill, 1.5x peel, 1.3x clustering | 472.9 s | 1.33x | 3.4x |
| 3x fill, 2x peel, 1.5x clustering | 408.4 s | 1.54x | 3.9x |
| 5x fill, 3x peel, 2x clustering | 342.0 s | 1.84x | 4.7x |

### Conclusion

**160 s is not reachable by optimizing the kernels this shortlist names.** The
honest ceiling for the whole shortlist, fully realised, is ~200 s, and a
realistic outcome is 340-470 s (4.7x-3.4x overall). Getting under 200 s requires
changing what the sorter *computes* -- fewer templates, a different candidate
representation, or a different matching formulation -- not making the present
statements faster. Reaching 160 s additionally requires attacking `conv1d`,
`einsum` and postprocessing, which no current proposal touches.

The single largest addressable block in the sorter is the `fused_detect` fill at
**24.8% of the sort**, which is where §10's candidate-only proposal points. Note
it is already the fused Triton path, so the 54.6x candidate ratio from §10 is a
traffic bound on a statement whose traffic was already cut ~130x; `As` itself
must still be computed densely because it is the input to the threshold test.

Recorded so this target is not re-litigated: the 4x that took 1601.6 s to
637.4 s came from four independent 2-7x kernel wins landing on stages that were
then 58% and 32% of the sort. Those stages are still 77% of it, but the fat has
been taken off the statements inside them.

### 12a. Pricing Astra #1 properly: the fill is two kernels, and only one is the target

`fused_detect._run` launches exactly two kernels, and they split the fill almost
evenly (real inputs, batch 150, median of 20 reps):

| kernel | produces | ms | share of fill |
|---|---|---:|---:|
| `_tm_fused_kernel` | `As`, `imaxs` | 12.963 | **55.0%** |
| `_amax_kernel` | `Amaxs` | 10.602 | **45.0%** |

`_tm_fused_kernel` **must stay dense**: `As` is the input to the threshold test
that defines a candidate, so it cannot be deferred to candidates. Astra #1 can
only touch `_amax_kernel`, which is 45% of the fill and **11.2% of the sort** --
not the 24.8% the fill as a whole represents.

`_amax_kernel` scales essentially linearly in its neighbour count, so it is
really doing the work: NC2 = 1 / 10 / 25 / 50 / 100 gives 1.078 / 1.175 / 2.618
/ 5.446 / 10.602 ms (9.83x from 1 to 100).

**The halo, which the first estimate missed.** The peak test is not
`Amax == As`. `Amaxs` is temporally pooled over `2*nt0+1 = 41` samples *before*
the comparison, so evaluating the mask at a candidate `(k,m)` needs raw `Amax`
over `m-nt0 .. m+nt0`. A candidate-only kernel must therefore compute `Amax` on
the candidate set **dilated by ±nt0**, not on the bare candidate set. Measured
over 4 batches:

| | fraction of positions |
|---|---:|
| bare candidates `As > Th` | 1.977% |
| dilated by ±`nt0` (what the mask actually needs) | **11.840%** |
| dilation cost | **5.99x** |

So the ceiling drops from 8.15x to **4.73x on `_amax_kernel`**, and from 1.65x
to **1.55x on the fill**. Also measured: **99.01% of rows contain at least one
candidate**, which kills any row-level gating scheme — there are no empty rows
to skip.

**Sort-level value of Astra #1, at its optimistic ceiling:** the fill goes
156.6 s -> 101.0 s in production, saving **55.6 s of 630.8 s (8.8%)**, for
575.2 s total and 2.78x overall. That assumes perfect compaction, no launch
overhead on the compaction pass, and a new Triton kernel proven bit-identical --
against a precedent (§ the max/min rewrite) where a similar "obviously
equivalent" change silently altered the full-file result.

Even driving `_amax_kernel` to **zero** leaves 560.3 s. This is a real ~9% win,
not a step toward 160 s.

### 12b. The peel loop is overhead-bound, Astra #6 is dead, and max_peels=50 truncates

**The loop's cost barely depends on how many spikes it peels.** Per-iteration
measurement over 150 iterations (3 batches, 812 units, NT=10122):

    peel_subtract = 0.1788 ms fixed + 0.2717 us per spike

At the median 62 spikes the fixed part is **91%**. This is not a regression
artifact of a narrow range -- the direct observation is that iterations 0-4
(74.1 spikes) cost 0.202 ms and iterations 45-49 (22.9 spikes) cost 0.182 ms:
**3.2x fewer spikes for 10% less time.**

The consequence for Astra #4 is that "remove work from the peel loop" has almost
nothing to remove. The loop's cost is set by its *iteration count*, which is
pinned at `max_peels`, not by the work inside an iteration.

**The guard is not the cause** (hypothesis tested and rejected).
`fused_peel.py:405` does `torch.stack((phases_disjoint(...), (amp>0).all())).tolist()`
every iteration -- a D2H copy that forces a host sync, 15,000 times per
slice300 sort. It looked like the fixed cost. Measured, it is **16.7%** of
peel_subtract (0.068 ms of 0.407 ms; note this split was measured with explicit
syncs and so is inflated in absolute terms against the 0.193 ms event figure --
the ratio is the usable part). Worth ~1 s of a 5.5 s peel loop, real but not the
story. The remaining fixed cost is inside `_fused_phase`, whose grid does not
shrink with the spike count. Both guard verdicts were constant across all 50
iterations (`disjoint=1, amp_pos=1`).

**Astra #6 answered: dirty unit x time coverage is far too high to exploit.**
`torch.max(B, 0)` is recomputed over the full `(n_units, NT)` B every iteration,
while `peel_subtract` only touches a ±nt window per spike. But the max reduces
over *units*, so a column must be recomputed if any unit changed there, and the
number that matters is dirtied **time columns**:

| dirty time columns of B per iteration | |
|---|---:|
| mean | **55.4%** |
| median | 67.3% |
| min / max | 1.2% / 87.5% |

A perfect dirty-column max would skip 44.6% of `max(B,0)`, which is 8.7% of
(max + subtract) and under 2% of the loop. **Do not build this.** It confirms
with a number what the shortlist already said about reviving time-only dirty
peeling.

**max_peels=50 truncates the learned pass, and the loss is heavily skewed.**
The tuned profile sets `max_peels` 200 -> 50 (`run_kilosort4.py:60`); stock KS4
defaults to 100. Replaying the peel to 200 iterations on 30 batches:

| | |
|---|---:|
| iterations to convergence | median **57**, min 38, max 192 |
| batches still finding spikes at iteration 50 | **21 / 30** |
| per-batch share of spikes captured by 50 peels | median **98.9%**, mean 86.7%, min **29.8%** |
| aggregate share of a 200-peel total | **73.3%** |

The skew is the point: the typical batch loses ~1%, but busy batches both need
more peels *and* contribute more spikes, so in aggregate 50 peels finds 73.3% of
what 200 finds. `max_peels=100` reaches 89.8% at ~2x the peel-loop cost
(+161.7 s in production), because cost is flat per iteration.

Stated carefully: these are learned-pass *detections*, not accepted units. More
peels also means more marginal, more-overlapped events, which may split or
duplicate rather than become good units. Whether the missing 26.7% is signal or
junk is a question for the QA pipeline, not for this file -- but the tuning
decision that discarded it was never measured, and now it is.

### 12c. `kmeans_plusplus` dug into: already 3.6x'd, and the remainder is real compute

Astra #2 (sparse clustering assignment) priced `clu0`+`clu` at 111.4 s, 82.9 s
of which is `kmeans++`. Before pricing that further, what does `kmeans++`
actually spend its time on *now*? It already carries two shipped, bit-identical
optimizations (`kilosort/fast_kpp.py`, commits `6ef211a`/`468a778`, both
predating this "no code changes" pass): cutting 3 host syncs/iteration to 1 per
call (1.56x), then capturing the 200-iteration body in a CUDA graph (2.3x more).
Neither is re-tested here -- both cite their own bitwise verification. The
question is what's left inside the currently-active graph path.

**First pass was wrong, and the bug was mine, not the shipped code's.**
Bracketing `capture_begin`/body/`capture_end` with CUDA events recorded on the
default stream while the capture itself ran on a separate, *freshly-created*
stream (no `wait_stream` linking the two) gave capture = **4.1-4.4 ms**, 15.5%
of the graph path -- 24x `fast_kpp.py`'s own documented "capture itself costs
0.18 ms." Switching to synced host timing (`torch.cuda.synchronize()` +
`perf_counter`, the same technique validated for the guard-cost split in §12b)
reproduced the same 4.1-4.4 ms, so it wasn't a cross-stream artifact. Two
hypotheses tested in order:

1. *Fresh graph-pool per call, vs. production's one process-wide pool.*
   Reused one `graph_pool_handle()` across probed calls -- **no change** (still
   ~4.4 ms). Rejected.
2. *Fresh `torch.cuda.Stream()` per call, vs. production's cached
   `_CAPTURE_STREAM`.* My harness created a brand-new stream object on every
   probed call; production creates it once (`if _CAPTURE_STREAM is None`) and
   reuses it for the rest of the process. Caching the stream the same way:
   capture dropped to **0.22-0.39 ms on every call after the first**, with the
   first call in-process paying 3.81 ms (one-time stream-creation cost,
   amortized over hundreds of calls in a real sort). This matches the
   documented 0.18 ms within measurement noise. **Confirmed and fixed in the
   harness; not a defect in the shipped module.**

Corrected steady-state anatomy of the graph path (12 probed calls of ~184,
n_spikes 1,447-15,946, niter=200; first-call stream warmup excluded from the
per-call reasoning above but left in the aggregate below since it's what any
single measured call actually pays):

| phase | mean | share |
|---|---:|---:|
| `draw_noise` (200x `.exponential_()`, un-graphed -- see fast_kpp.py's RNG note) | 1.05 ms | 4.2% |
| `warmup` (3 eager `body()` calls, priming cuBLAS/dispatch before capture) | 1.50 ms | 6.0% |
| `capture` (corrected) | 0.59 ms | 2.4% |
| **`replay`** (200x `g.replay()`) | **21.64 ms** | **87.0%** |
| `final` (`count_nonzero` + the one host read) | 0.09 ms | 0.4% |

`replay` costs **108 us/iteration**, essentially flat from 1,447 to 15,946
spikes (an 11x range) -- confirming the docstring's own read that this is
overhead/launch-shaped work, not data-shaped, and that the graph already
converted ~20 launches/iteration into GPU-side execution of the same ops back
to back. Going from the pre-graph estimate (~222 us/iteration: 168 overhead +
54 data) to 108 us/iteration matches the shipped "2.3x on top" commit message.

**`warmup` is the one real, currently-unexploited cost**, at 6% of the path
(~1.2 ms/call is the 3 eager passes beyond what ~324 us of pure compute would
cost at the measured replay rate). It cannot be skipped: capture requires the
body to have already run so cuBLAS's algorithm/workspace selection isn't lazily
triggered *during* capture. And it likely cannot be cached across centers the
way the stream and pool now are, because that selection is keyed to input
shape and every center has a different `n_spikes` (the same reason the module's
own SHAPE note gives for why the graph itself can't be reused between centres).
Not verified further -- flagged, not priced, since testing it means touching
`fast_kpp.py` and this pass does not change code.

**Conclusion for Astra #2.** `kmeans_plusplus` is not an idle function waiting
on launch overhead; 87% of its current graph-path cost is the replayed
arithmetic itself, already collapsed from per-iteration launches to one
per-iteration replay. There is no further no-cost lever inside this function --
the two real optimizations it had (kill the host syncs, kill the launches) are
both already shipped. Whatever Astra #2 still has to offer lives in the 28.5 s
of `clu0`+`clu` that *isn't* `kmeans++` (`swarmsplitter.split`, `neigh_mat`, the
alternating-assignment loop), or requires changing what `kmeans++` computes
(fewer `NTRY` candidates, fewer `niter` seeds) -- which is an accuracy question,
not a speed one, and out of scope for this pass.

### 12d. `conv1d`+`einsum` (peel-loop setup): priced, and it's real compute too

Never on any Astra shortlist. `block_timing.py` had already measured the pair
combined at 16.8% of `run_matching`'s GPU time (32.7 s in production,
§"Where the detection time actually goes"); this splits the two calls and asks
whether either is overhead-shaped or FLOP-shaped, the same question asked of
`kmeans++`'s graph path and of `peel_subtract`.

    B = conv1d(X.unsqueeze(1), W.unsqueeze(1), padding=nt // 2)   # (n_chan,1,NT) x (n_pcs,1,nt)
    B = torch.einsum('ijk, kjl -> il', Us, B)                     # Us (n_units, n_chan, n_pcs)

Median-of-20, 4 probed batches (n_chan=519, NT=10122, n_pcs=3, n_units=812 --
this pass's late-sort unit count):

| call | mean | share |
|---|---:|---:|
| `conv1d` | 1.425 ms | 38.0% |
| `einsum` | 2.326 ms | **62.0%** |

`einsum` contracts over both `n_chan` (519) and `n_pcs` (3) at once, against
812 units x 10,122 samples -- ~25.6 GFLOP for this one batch. That is real
arithmetic, not launch count: at 812 units this is not a small problem.

Tested one alternative: reshape both operands and replace the einsum with a
single `(n_units, n_chan*n_pcs) @ (n_chan*n_pcs, NT)` matmul, timed only, never
substituted. **1.14x** on the einsum call alone (2.326 -> ~2.04 ms), which is
7.6% off the combined block, not the einsum's dominant term -- because the
underlying FLOPs are the same either way; cuBLAS's batched contraction was
already close to as good as a flat GEMM here. And that 1.14x was checked with
`torch.allclose(atol=1e-3, rtol=1e-3)`, not `torch.equal` -- a different
summation order over the same contraction is not guaranteed to be bit-identical,
so this number has not cleared this project's own bar and is reported only as
a ceiling estimate, not a candidate.

**Conclusion.** Like `kmeans++`'s replay phase and `peel_subtract`'s per-spike
term, this block is FLOP-bound, not overhead-bound. There is no free lever:
the realistic ceiling is ~7.6% of 32.7 s (~2.5 s in production), and even that
requires giving up the byte-identical guarantee to get. Closed.

### Running total against the 160 s target

Every block investigated since the 201.0 s ceiling (§12) was computed has come
back the same way: real, already-necessary arithmetic, not overhead left on
the table by an unoptimized implementation.

| block investigated after §12 | prod s | outcome |
|---|---:|---:|
| `kmeans_plusplus` (§12c) | 82.9 | closed -- 87% of its cost is already-graphed replay compute |
| `conv1d`+`einsum` (§12d) | 32.7 | closed -- FLOP-bound, ~7.6% ceiling, not bit-identical |

Neither adds usable headroom beyond the 201.0 s floor already on record. The
160 s target was answered in §12 and remains answered: **not reachable from
this pipeline's current computation** without changing what gets computed
(fewer peels, fewer clustering candidates/seeds, coarser detection), which is
an accuracy tradeoff outside this pass's scope. Remaining unpriced pieces
(`swarmsplitter.split`, `neigh_mat`, the alternating-assignment loop, cluster
merge's 0.88x regression, and postprocessing's untouched 1.00x) are each too
small individually to change this conclusion even in the best case -- they
sum to well under the 41 s gap this pipeline is short by.

### 12e. Correction to §12c's kmeans++ figure, and swarmsplitter.split priced

**§12c's "82.9 s of kmeans++" mixed two different denominators and is wrong.**
`fast_kpp.py`'s own docstring measured kmeans++ at 44.5% of
`clustering_qr.run`'s time; a separate `gpu_busy.py` measurement (§"Where the
detection time actually goes") measured kmeans++ at 74.4% of
`clustering_qr.cluster`'s time. `cluster()` is a strict subset of `run()` --
`run()` also calls `swarmsplitter.split`, `hierarchical.maketree`,
`get_data_cpu` outside `cluster()`. The ceiling table applied the 74.4%
(`cluster()`-relative) figure directly to 111.4 s, which is `run()`'s
production time (the `clu0`+`clu` stage totals `run_kilosort` self-reports) --
not `cluster()`'s time. Since `cluster()` is not 100% of `run()`, that
overstates kmeans++'s share of the 111.4 s.

Fixed by measuring `run()`, `cluster()`, `kmeans_plusplus`, `neigh_mat`, and
`swarmsplitter.split` all in one pass over one sort, sync'd host timing
throughout, so every share below is against the same `run()`-total denominator
and none of it is cross-derived:

| | n | share of `run()` |
|---|---:|---:|
| `cluster()` (= kmeans++ + neigh_mat + alt-loop) | 184 | 69.5% |
| ↳ `kmeans_plusplus` | 184 | **51.5%** |
| ↳ `neigh_mat` | 184 | 7.6% |
| ↳ alternating-assignment loop + `Mstats` (residual) | -- | 10.4% |
| `swarmsplitter.split` | 184 | **19.9%** |
| `get_data_cpu`/`maketree`/other bookkeeping (residual) | -- | 10.6% |

These five sum to 100.0% by construction (four measured, two residuals) --
that internal consistency is the check the earlier cross-measurement comparison
didn't have. 51.5% is also just a different number from `fast_kpp.py`'s
documented 44.5%; different real sorts have different cluster-size mixes and
this is not claimed to be more authoritative than that figure, only mutually
consistent with itself, which the 74.4%-of-111.4s arithmetic was not.

**Corrected production breakdown of the 111.4 s `clu0`+`clu` budget:**

| | prod s (corrected) | was (§12c, wrong) |
|---|---:|---:|
| `kmeans_plusplus` | **57.4** | 82.9 |
| `swarmsplitter.split` | **22.2** | (folded into "28.5 s other") |
| `neigh_mat` | **8.5** | (folded into "28.5 s other") |
| alt-loop + `Mstats` | 11.6 | (folded into "28.5 s other") |
| bookkeeping | 11.8 | (folded into "28.5 s other") |

§12c's *conclusion* (kmeans++'s graph-replay phase is 87% real compute, no
free lever) is about that function's internal anatomy and is unaffected by
which absolute number it's a percentage of -- only the stated 82.9 s figure
was wrong, now corrected to 57.4 s.

**`swarmsplitter.split`, now priced: no free lever here either, for a
different reason than compute-boundedness.** At 22.2 s / 3.5% of the whole
sort it is too small to matter much either way, but the `run()` call site's
own comment (`clustering_qr.py`, around the `cluster(...)` call) already
recorded a directly relevant experiment: truncating the alternating-assignment
loop's iteration count to cut cost was tried and **refuted** -- leaves get
cleaner but the merged *output* gets dirtier (contaminated spike mass 8.95% ->
10.73% on d007 at niter=1), and niter=0 "shatters the sort outright" (recall
0.909, 2x wall, 4x GPU, two ground-truth cells lost). That is the same
shape as `max_peels`: an already-tried speed/accuracy trade that made things
measurably worse, not an unexploited free lever. Not re-tested here -- cited
because it directly bears on whether `split`/the alternating loop are worth
attacking, and the answer already on record is no.

**Revised running total.** Every block priced this session across `st0`, `st`,
and clustering has landed the same way: real compute, an already-tried and
refuted trade, or too small to matter. The 160 s target's answer from §12
stands, now on a fully self-consistent accounting of where clustering's 111.4 s
actually goes.

### 12f. Postprocessing priced: the last stage nobody had looked at

Postprocessing sat at 1.00x through every change in this file (21.5 -> 21.6 s)
because it was never touched, not because it was checked and found optimal.
`io.save_to_phy` does three different kinds of thing -- real compute, array
assembly, and disk I/O -- so it was worth knowing which one actually dominates
before writing it off as small-and-therefore-irrelevant.

Sync'd host timing around every named sub-call, one full slice300 sort
(`save_to_phy` called once per pass):

| piece | share of `save_to_phy` |
|---|---:|
| `make_pc_features` | **42.9%** |
| `np.save` (19 calls, disk I/O) | 36.0% |
| `CCG.refract` (refractory/contamination) | 15.1% |
| `CCG.similarity` (pairwise template correlation) | 3.2% |
| `remove_duplicates` | 0.6% |

Two importing traps worth recording since they'd silently no-op a lazier hook:
`remove_duplicates` is bound into `io.py`'s namespace by
`from kilosort.postprocessing import (remove_duplicates, ...)` -- a direct name
import, so it has to be patched as `io.remove_duplicates`, not
`kilosort.postprocessing.remove_duplicates` (patching the original module
would not touch `io.py`'s already-bound copy). `CCG` is imported as
`from kilosort import CCG` -- a module reference, so `io.CCG.similarity` and
`io.CCG.refract` ARE late-bound and patchable directly, unlike `remove_duplicates`.

**`make_pc_features` is the single largest piece of postprocessing, ahead of
all disk I/O combined.** Scaled to production's 21.6 s: ~9.3 s
`make_pc_features`, ~7.8 s I/O, ~3.3 s `CCG.refract`, ~0.7 s `CCG.similarity`,
~0.1 s `remove_duplicates`. Not further decomposed -- at this absolute size
(9.3 s, 1.5% of the whole sort) even a 2x here is ~4.6 s, and postprocessing's
1.00x was never the thing standing between this pipeline and 160 s. Recorded
so "postprocessing is untouched" reads as "measured and small," not "unknown."

### Session close-out

Every stage of this sort has now been measured at least once: the two
detection passes and their dominant kernels (already exhaustively tuned before
this pass began -- config ladders, live-tile LUTs, roofline analysis, all
predating and outside this "no code changes" investigation), both clustering
passes end to end (kmeans++, `swarmsplitter.split`, `neigh_mat`, the
alternating-assignment loop), the peel-loop setup (`conv1d`+`einsum`), and now
postprocessing. Every single one came back one of three ways: real,
already-necessary compute; an accuracy trade already tried and refuted; or too
small in absolute seconds to move a 41 s gap. The 160 s target's answer,
unchanged since §12: **not reachable from this pipeline's current computation.**
