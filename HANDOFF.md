# Kilosort4 speed work — where it stands, and what to attack next

Handoff for someone working from a checkout that is ~a week behind. Everything
below is **measured on this machine**, not projected. The rule the whole series
follows: every change is either **bit-identical** or it does not land. No
"probably fine", no accuracy trades — there is no QA pipeline yet to catch a
regression, so byte identity is the only safe currency.

Canonical detail lives in `KS4_VALIDATION_NOTES.md`. This file is the summary
and the roadmap.

**Read this first if you are about to compare two sorts.** kilosort4 is not
byte-reproducible run to run by default -- two runs of *identical* code differ
on a few float32 of the tF-derived outputs. That is a property of the program,
not of this branch, and it makes a naive A/B diff unreadable. The cause is a
nondeterministic CUDA reduction, and `tools/run_full_sort.py --deterministic`
(torch.use_deterministic_algorithms + CUBLAS_WORKSPACE_CONFIG=:4096:8) removes
it: same code twice goes from 5 files differing to **23/23 identical**, and old
code vs this branch likewise goes to **23/23 identical** at production scale.
Costs ~7-13%. Use it for every comparison; leave it off for timing.

---

## The headline

Full production sort, `20260724A/chunk12_9-11` (42 GB, 4054 batches, 519
channels, `NT=10122`, `Nfilt=4048`, `nt=61`, `nt0min=20`):

| | s |
|---|---:|
| stock kilosort4 | 1601.6 |
| this branch | **637.4** |
| | **2.51x** |

11,050,850 spikes, with `spike_times`, `spike_clusters` and `spike_templates`
byte-identical to stock. Confirmed twice independently: the isolated benchmark
(637.40 s) and a real lab-pipeline run through `run_kilosort()` (637.11 s),
agreeing to 0.05%.

`slice300.bin` (300-batch dev benchmark) cumulative: **118.2 s → 53.3 s**.

### It now holds on a second array

Everything above was one recording, so one set of tensor shapes — which matters
because §2's bit-identity depends on reproducing cuBLAS's accumulation order
and *which* Triton block size does that is shape-dependent. `20260514A` is a
macaque recording on a **512-channel 60 um array** (array 504) against
20260724A's **519-channel 30 um** (array 1551): 2x the pitch, ~3x the area,
and **1920 universal templates instead of 4048**.

All four gates re-validated and enabled there, and `fused_detect` chose the
*same* BLOCK_M=128 that matched at the old shapes.

Validated at **production scale on the whole of `data000`** — 33,140,000
samples, 1657.0 s, 3314 batches, 33.94 GB, input verified against the run-E
file-order bug class before any GPU time. Three-way byte comparison:
**23/23 files identical in all three pairings**, including A vs A2, which is
the first run on this geometry with the reach to see the run-to-run wobble.

| arm | `Total runtime` |
|---|---:|
| stock (all five `KILOSORT_NO_*`) | 858.81 s |
| before the peel live-tile LUT | 383.60 s |
| + live-tile LUT | 319.10 s |
| **this branch** (+ both peel tails) | **309.03 s** |
| same code, second run | 311.10 s |
| | **2.78x** |

The peel alone went 397.61 -> 128.84 s, **3.09x against stock**. Every step is
23/23 byte-identical -- the final build was compared against both pre-LUT
baselines, both LUT-only baselines, and its own second run.

**An earlier version of this file said 1.45x here and blamed the geometry.
Both halves were wrong.** That number came from a 150 s slice (~9% of one
recording, and `data000` is 1 of 39), and its fused arm was carrying Triton JIT
compile cost — the warm slice comparison is 2.01x. At production scale this
array gives 2.24x against 20260724A's 2.51x, so the template-count argument
survives only as a small residual. The mechanism: detection is 60% of a slice
sort and 78% of a production sort, and detection is what these optimizations
attack — universal detect alone is **3.85x** and the peel **2.87x**, while
clustering falls to 1.16x.

One thing to know before you sort 60 um data with the lab pipeline: its `tuned`
profile applies `dmin=15, dminx=32` to both 30 and 60 um arrays, but those
numbers *are* the 30 um array's row/column pitch. On the 60 um array they give
**14,280** universal templates instead of 1,920 and OOM a 20 GB card — and
`max_channel_distance=66` culls nothing there, so the cull that holds Nfilt
down at 30 um does no work. Details and the measured grid table are in the
notes.

---

## What is optimized (and what each one actually bought)

| § | change | win | where |
|---|---|---:|---|
| 1 | Column tiling `niter = 40 → min(40, ceil(NT/1024))` | 1.04x | `spikedetect.template_match` |
| 2 | Fused detect body (Triton) | 6.9x on the loop | `fused_detect.py` |
| 3 | Fused peel subtract (Triton) | 5x on the subtract | `fused_peel.py` |
| 4 | k-means++ without per-iteration host reads | 1.56x | `fast_kpp.py` |
| 5 | k-means++ body in a CUDA graph | 2.3x on top of §4 | `fast_kpp.py` |
| 6 | **Fused peak-selection tail (new)** | **6.2x on the tail** | `fused_peaks.py` |

§6 is the one your checkout will not have. The tail after the detect body —
edge zeroing, `max_pool1d`, two comparisons, the `and` — ran on 41 M elements
and moved ~1 GB per batch to produce one bool array. One kernel does it in a
single pass and **short-circuits**: `As > Th` is brutally sparse (1,437
survivors of 41 M), so a block with no candidate skips the whole `2*nt0+1`
window max. 5.332 → 0.859 ms/batch.

That tail is the *easiest* kind of thing to make identical and worth copying as
a pattern: it contains **no floating-point arithmetic at all**. `max` is a
selection, so reduction order cannot change the result; the maxima's *indices*
are never used, so ties are harmless; `==`, `>`, `&` are exact. Contrast §2,
where identity meant reproducing cuBLAS's split-K-2 accumulation order and only
*some* block sizes agreed.

### Current stage breakdown (the 637 s sort)

| stage | s | % |
|---|---:|---:|
| spike detection (universal) | 240.6 | 37.8 |
| spike detection (learned) | 250.8 | 39.4 |
| clustering (templates + final) | 107.3 | 16.8 |
| merge + postproc | 29.6 | 4.6 |

Detection and peel are 76% of the sort. Clustering is no longer where the money
is.

---

## What to attack next, ranked

### 0. DONE — `ctc`/`U_time` zero-tile skip (commit `6cb2d94`)

Kept here because the *negative* result is the reusable part.

`fused_peel` subtracted `amp * ctc[r, iY, :]` over every unit row, but `U` is
exactly zero off each template's local channels, so 89.58% of `ctc` rows and
89.00% of `U_time` tiles are exactly `+0.0` and their subtract is a no-op.

**The recommended first step was benched and rejected.** An in-kernel early
exit — load the tile, skip the read-modify-write if dead — was exact but bought
only **1.04-1.08x**, with a 2-4% regression on dense data. That answered the
question this section used to pose: skipping 67% of per-program traffic changed
nothing, so the peel is **still launch-bound after fusion**, and cutting traffic
is the wrong lever.

**Program count is the right lever.** Holding per-program work fixed and cutting
tiles 65 -> 11 gave 3.27x, so dead tiles are dropped *before* the launch via a
per-unit live-tile LUT: grid `(n_spk, n_tiles)` -> `(n_spk, MAXT)`. Built once
per `prepare_matching` (ctc is batch-invariant), cached on the **base** tensor
because `run_matching` rebuilds `ctc_p` as a fresh permuted view every batch.

    peel   202.71 s -> 138.71 s   (2.87x against stock, was 1.96x)
    total  383.60 s -> 319.10 s   (2.69x against stock, was 2.24x)
    23/23 byte-identical, three pairings, at production scale

Two guards make it exact, and both are enforced rather than measured: a tile is
dropped only if **bitwise all `+0.0`** (a `-0.0` tile is not droppable, since
`o - (a * -0.0) = o + 0.0` flips a stored `-0.0`), and **`amp > 0` strictly**
(`>= 0` admits `-0.0`, which causes the same flip). Phases with any non-positive
amp take the full path; the check rides on the sync the peel already does.

**Trap, and it caught two of my own artefacts.** Do not benchmark or unit-test
this against uniform-random sparsity. With 89.58% of rows zeroed at random a
16-row tile survives 93% of the time, giving MAXT 62 and 1.12x. Real liveness is
spatially clustered — measured MAXT is **19 of 65**. The synthetic benchmark and
the first draft of the tests both said this change does not work.

### 1. DONE -- both peel tails fused (commits `1e8c39e`, `63db84f`)

Kernel speedups were large (condition tail 45.7 -> 8.9 us/call, 5.1x; store
tail 49.9 us -> one program per spike) but the END-TO-END return was 1.03x and
1.085x. That gap is the lesson: `tools/profile_peel_statements.py` syncs around
each statement, which inflates exactly the launch-bound blocks it is used to
find. Its shares are UPPER BOUNDS.

Also from that work: `tl.sqrt` is the approximate instruction and is NOT
bit-identical to torch's `**.5` -- use `tl.math.sqrt_rn`. The gate caught it.

### 2. Universal detection -- 30.9% of the sort, and it is DONE

Measured 2026-09-06 by wrapping whole functions over a real sort, so this is
the answer to "is anything left in the biggest stage":

| block | share of `spikedetect.run` | at production |
|---|---:|---:|
| `template_match` (fused_detect + fused_peaks) | 75.5% | ~72 s |
| per-batch tail (gather + wPCA matmul + D2H) | 13.5% | ~12.9 s |
| `extract_wPCA_wTEMP` (ONE call, fixed cost) | 10.1% | ~1.1 s |
| `yweighted`, `nearest_chans` | <1% | -- |

`template_match` is already 3.84x against stock and what remains inside it is
`conv1d` (cuDNN) and the loop einsum (cuBLAS). Beating those means matching
cuBLAS's accumulation order, which §2 established is the hard constraint of
this whole series -- high risk, and the shapes here are ones cuBLAS is good at.

The per-batch tail is the only soft spot, and it is small: 13.5% of 30.9% is
**4.2% of a sort**, of which the gather and matmul are real work. The
realistic prize is the per-batch `.cpu()` (pinned + non_blocking), worth maybe
1.5% overall for a loop restructure that has to keep buffers alive across the
copy. Not taken; ranked below clustering.

Note `extract_wPCA_wTEMP` reads 10.1% on a 300-batch slice but is ONE call, so
it is ~1.1% at production. Do not size it from a slice profile.

### 3. Clustering -- 18% of the sort, ~1.17x, the least-optimized block left

Re-measured 2026-09-06 by wrapping whole functions over a real sort (whole
functions, not statements, so the sync distortion above is much smaller):

| block | share of `clustering_qr.run` |
|---|---:|
| `kmeans_plusplus` | 44.5% |
| **`swarmsplitter.split`** | **32.1%** |
| alternating-assignment loop | ~8.3% |
| `neigh_mat` | 6.4% |
| `Mstats` | 0.7% |

Two corrections to what this file used to say. `kmeans_plusplus` is 44.5%, not
the 78-87% its own docstring claimed -- §4 and §5 already took most of it out,
and the docstring is now fixed. And `swarmsplitter.split` is 32.1% of
clustering, not the 12.3% recorded earlier.

`split` is CPU-side numpy and pure Python. A cProfile of it over one sort:
`check_split` is 24% of its own time, and ~29% WAS numba JIT compilation --
now removed by `@njit(cache=True)` on `CCG.compute_CCG` (33.63 -> 32.44 s on
slice300, byte-identical).

Be realistic about what is left here: clustering is 18% of the sort, `split` is
a third of that, and `check_split` a quarter of THAT -- so even a 3x on the hot
function is ~1% of a sort. The loop is also a sequential tree walk whose
pruning is data-dependent, so it does not parallelise without changing
semantics. Rank it below anything in detection.

### 2. `torch.max(B, 0)` — do NOT bother

Recorded so nobody repeats it. 33 MB in 85.6 µs is **384 GB/s, near peak**.
The older notes' "10.3% of the learned pass" reading invites optimising it;
there is nothing there.

### 3. The fused detect body — biggest single item, highest risk

~33.3 ms/batch × 4054 ≈ 135 s, the largest single line item in the sort (21%).
It is ~16x off peak FLOPs, so there is headroom in principle. But its bit
identity is pinned to reproducing cuBLAS's split-K-2 accumulation, and which
Triton block size agrees is **not predictable and not a constant of the code**
(BLOCK_M=128 matches at production shapes, 64 and 32 do not; at test shapes it
is the other way round). Touch this only with the gate in place and expect to
spend the time on identity, not on speed.

### 4. `swarmsplitter.split` — untouched, CPU-side

1.80 s / 12.3% of the template pass, pure Python on the host. May not need
Triton at all. Nobody has started it.

### 5. Clustering at production scale — probably not worth it

§4/§5 give 2.47x/2.38x on slice300 but only 1.11x/1.23x in production: the
remaining time is data movement, not launch overhead. Clustering is 16.8% of
the sort. A different approach would be needed and the payoff is capped.

---

## Two things that will cost you a day if you don't know them

**Kilosort4 is not bit-reproducible run to run**, independent of any of this
work. Two runs of *identical* code on identical input disagreed on 13 float32
values of 33 M, in `tF`-derived outputs (`amplitudes`, `spike_positions`,
`templates`, `pc_features`), at different spikes each time. Spike times and
cluster assignments never move. **Root cause still unknown.** Consequence: one
matching end-to-end sort is not proof of byte identity.

The technique that settles attribution is the **three-way comparison**: run
run1↔baseline, run2↔baseline, and run1↔run2, where run1/run2 are the same new
code. If the two same-code runs disagree *with each other* at different places
than either disagrees with the baseline, the diff belongs to the program, not
your change. Reusable well beyond kilosort.

**Measure before you build.** A worked example from this week: the peel loop
recomputes full-width `max`/`pool`/`nonzero` every iteration, but
`peel_subtract` only writes `B[:, iX ± nt]`, so the detection condition can
only change within ±2nt of the previous iteration's spikes — everything else is
provably redundant. Sound argument, real invariant. Then measure: the dirty set
is **99.5% of NT at iteration 0 and still 92.7% at iteration 20**, because the
peel does not converge (~61 spikes/iteration, each spreading a 245-column
window, and ~60 spikes saturate NT ≈ 10k). Total reduction **1.27x** — about 4%
of the sort for a Triton kernel plus a tie-break identity argument. Rejected
before a line of kernel was written. `tools/measure_peel_dirty_region.py` is
the harness; note it is **data-dependent** and would look different on a probe
where the peel converges.

## Harnesses

`tools/` is the durable half; much of the rest still lives in a session
scratchpad under `/tmp` and will not survive a reboot.

| tool | what it does |
|---|---|
| `tools/run_tests.py` | runs the test suite without pytest (see below) |
| `tools/run_full_sort.py` | one full sort from a flat `.bin`; both arms of every A/B |
| `tools/compare_sorts.py` | raw-byte compare of two result dirs |
| `tools/make_litke_slice.py` | cut a flat int16 slice from a Litke recording |
| `tools/measure_peel_dirty_region.py` | dirty-set census over a full sort |
| `tools/measure_ctc_sparsity.py` | exact-zero census of `ctc` + skip preconditions |
| `tools/profile_peel_statements.py` | statement/block profile of the peel loop |
| `tools/bench_fused_peaks.py` | identity + timing for §6 on real captured batches |

`tests/` pins the **gates**, which is where the safety argument lives:
`test_fused_detect.py`, `test_fused_peel.py`, `test_fast_kpp.py`,
`test_fused_peaks.py`.

**There is no pytest in any conda env on this machine.** `tools/run_tests.py`
installs a minimal shim and runs the real files anyway — `python
tools/run_tests.py`, currently **71 passed, 0 failed, 0 skipped, 0 errored**.
Before it existed the gate tests were driven by throwaway scripts under `/tmp`
that *re-implemented* the assertions, so the committed test files had never
actually been executed. Run this before trusting a change.

Every optimization sits behind a `KILOSORT_NO_*` env switch and a runtime gate
that validates against the stock path once per process. Keep that pattern.
