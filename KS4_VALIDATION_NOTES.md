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

## Cumulative

Whole slice300 sort across this series, all byte-identical to the stock
baseline at every step:

| | s |
|---|---:|
| stock | 118.2 |
| + adaptive niter (§1) | 113.2 |
| + fused detection (§2) | 67.6 |
| + fused peel subtract (§3) | 62.6 |

Stage shares have moved a long way: universal detection was 57.9% of the
production sort and is now the same size as clustering, which is untouched.

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
* **Clustering** (129.2 s across both passes): 235 cluster centres in a Python
  loop over small tensors.
