# Kilosort4 speed work — where it stands, and what to attack next

Handoff for someone working from a checkout that is ~a week behind. Everything
below is **measured on this machine**, not projected. The rule the whole series
follows: every change is either **bit-identical** or it does not land. No
"probably fine", no accuracy trades — there is no QA pipeline yet to catch a
regression, so byte identity is the only safe currency.

Canonical detail lives in `KS4_VALIDATION_NOTES.md`. This file is the summary
and the roadmap.

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

### 1. The learned pass's two tails — ~30% of a 250.8 s stage

Best available target. Statement profile of `run_matching` over a full
slice300 sort (14,557 peel iterations, 815 units, NT 10,122):

| block | s | share | µs/call |
|---|---:|---:|---:|
| `peel_subtract` (already fused, §3) | 4.70 | 45.7% | 323.0 |
| **condition tail** relu→square→edges→`max_pool1d`→`cnd1`/`cnd2`→`nonzero` | **1.63** | **15.8%** | 111.1 |
| **store tail** `imax[iX]`, 4 slice writes, `B[iY,iX]`, `s[iY]`, `**.5` | **1.46** | **14.2%** | 50.2 |
| `torch.max(B, 0)` | 1.25 | 12.2% | 85.6 |
| `einsum` / `conv1d` (once per batch) | 1.14 | 11.1% | — |

The two tails are **launch-overhead bound, not bandwidth bound** — roughly 20
kernels per peel iteration, each on a 10,122-element (40 KB) array. 40 KB in
31 µs is 1.3 GB/s, three orders off this card. The peel loop runs ~48.5
iterations per batch, so that overhead is multiplied 48x.

Two things to know before starting:

* The same sparse short-circuit as §6 applies. `cnd1 = cmax > Th2` is satisfied
  by ~61 of 10,122 positions, so the window max only needs computing for
  candidates.
* **Order is load-bearing.** `nonzero` returns ascending indices and `st` /
  `amps` rows are written in that order. Any compaction scheme must preserve
  it — an atomic-counter compaction will not. A block-level `cumsum` will.
* `th_amps` uses `cmax[iX]**.5`. Check whether torch lowers `**.5` to `sqrt`
  before assuming `tl.sqrt` matches it bit-for-bit; that is a real hazard and
  the gate must catch it.

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
| `tools/measure_peel_dirty_region.py` | dirty-set census over a full sort |
| `tools/profile_peel_statements.py` | statement/block profile of the peel loop |
| `tools/bench_fused_peaks.py` | identity + timing for §6 on real captured batches |

`tests/` pins the **gates**, which is where the safety argument lives:
`test_fused_detect.py`, `test_fused_peel.py`, `test_fast_kpp.py`,
`test_fused_peaks.py`. Note **there is no pytest in any conda env on this
machine** — drive them with the hand-written runners.

Every optimization sits behind a `KILOSORT_NO_*` env switch and a runtime gate
that validates against the stock path once per process. Keep that pattern.
