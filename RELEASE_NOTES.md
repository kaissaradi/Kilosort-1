# Kilosort 4.5 — release notes

A fork of Kilosort4 for large planar multi-electrode arrays, developed against
Litke 512- and 519-channel retina recordings. It sorts the same data 2.5 times
faster, adds native Litke input, and adds a deterministic mode.

**Base:** upstream `main` at `17743f2`, which reports itself as 4.1.8.dev.
**Branch:** `mea-optimizations`.

## The rule this release was built under

> Every change is bit-identical or it does not land.

There is no accuracy trade anywhere in the speed work. That is not a claim
about intent; it is the acceptance test, and it is measured in both directions
on two array geometries at production scale. It is why a fork this large is
safe to run science on.

Two things in 4.5 deliberately step outside that rule, and both are opt-in:
`--deterministic`, which *pins* a nondeterminism that upstream already has,
and the `KS4_DUMP_RESIDUAL` diagnostic, which only reads.

## Speed

Full production sort, `20260724A/chunk12_9-11` — 42 GB, 4054 batches, 519
channels, `NT=10122`, `Nfilt=4048`:

| | Total runtime |
|---|---:|
| stock Kilosort4 | 1601.6 s |
| **Kilosort 4.5** | **637.4 s** |
| | **2.51×** |

11,050,850 spikes, with `spike_times`, `spike_clusters` and `spike_templates`
**byte-identical to stock**. Confirmed twice independently — an isolated
benchmark at 637.40 s and a real pipeline run at 637.11 s, agreeing to 0.05 %.

The result holds on a second, different geometry. `20260514A` is a macaque
recording on a **512-channel 60 µm array** against the above **519-channel
30 µm** one: twice the pitch, about three times the area, and 1920 universal
templates instead of 4048. Validated on the whole of `data000` — 33,140,000
samples, 1657.0 s of data, 33.94 GB:

| arm | Total runtime |
|---|---:|
| stock | 858.81 s |
| **Kilosort 4.5** | **309.03 s** |
| | **2.78×** |

**23 of 23 output files identical in all three pairings**, including
stock-versus-stock, which is the run that has the reach to see upstream's own
run-to-run wobble.

Where the time went: the peel subtract, the universal-detection loop body, the
peak-selection tail and k-means++ were each fused into single Triton kernels,
and the `ctc` matrix was found to be 89 % exact zeros and is now skipped
tile-wise. Each fusion ships behind a gate that proves itself bit-identical on
the first real block of the sort and falls back to the stock statements
otherwise. The gates are not optional and they are not assumptions about the
data — see `KS4_VALIDATION_NOTES.md` §2, §3, §6 and §8.

## New in 4.5

**Native Litke input, no conversion step.** `kilosort.litke.LitkeRecording`
reads 12-bit packed Litke bins directly as a `file_object`, joins multi-file
folders, and drops electrode 0 — the TTL trigger channel, which is not spikes —
by default. The unpack is bit-exact against the lab's `bin2py_cythonext` on
real 519-channel data. TTL is still reachable through `get_ttl`, `save_ttl` and
`detect_ttl_onsets`.

**`--deterministic`.** Kilosort4 is not byte-reproducible run to run: two runs
of *identical* code differ in a few float32 values of the `tF`-derived outputs,
because the peel's overlapping-window scatter is last-write-wins on duplicate
indices. That is a property of the program, not of this fork, and it makes a
naive A/B diff unreadable. `tools/run_full_sort.py --deterministic` removes it —
the same code twice goes from 5 files differing to **23/23 identical**. It
costs 7–13 %. Use it for every comparison and leave it off for timing. It
*pins* rather than shifts: on a recording that was already reproducible it
gives a result identical to the non-deterministic run.

**Per-array `dmin` / `dminx`.** These are absolute micrometre values, not
multiples of the electrode pitch, and the two Litke arrays have their fine axes
swapped. Applying the 519-channel array's grid to the 512-channel array
oversamples it and runs out of GPU memory. The values are now keyed by pitch.

**Memory scaling.** Peak *allocated* memory is flat in recording length; it is
the reserved pool that grows. Several buffers that were sized from the whole
recording are now sized from what the batch actually needs.

**`KS4_DUMP_RESIDUAL`.** A read-only diagnostic that writes the pre-peel data,
the post-peel residual, the batch's spikes and the whitening matrix for a
chosen set of batches: `KS4_DUMP_RESIDUAL='dir:150-179'`. Off unless the
variable is set, and byte-identical when it is set — 24 of 24 files, measured
both ways.

## Fixed in 4.5

**The CPU path — and this one is upstream's bug, not the fork's.** Every
memory-diagnostic call site guarded on `torch.cuda.is_available()`, which
reports whether the *host* has a GPU rather than whether *this run* uses one.
On a machine with a GPU, a run with `device='cpu'` therefore entered
`torch.cuda.memory_stats('cpu')` and raised
`ValueError: Expected a cuda device, but got: cpu`. That took the entire CPU
pipeline down.

Verified as inherited, not introduced: a pristine `origin/main` worktree at
`17743f2` fails `tests/test_full_pipeline.py` with the identical error at its
own `run_kilosort.py:800`. **Anyone on a CUDA machine who asks upstream
Kilosort4 for `device='cpu'` hits this.** Worth sending upstream.

The guard now reads the device (`kilosort.utils.is_cuda_device`), which also
fixes a second, silent instance of the same mistake:
`device == torch.device('cuda')` is False for a real device, which is
`cuda:0`, so that branch never fired at all.

Byte identity re-measured after the fix, since the fork rule is measured and
not argued: 24 of 24 files identical against the pre-fix baseline on a
300-batch 519-channel sort with `--deterministic`.

**Eight gate and cache defects** inside the optimisations themselves, found by
re-auditing the fast paths rather than by a failure. See the bug register in
`KS4_VALIDATION_NOTES.md`.

**Empty-template rows** with NaN means from clustering are now zeroed before
`ctc` is built, instead of poisoning the projection for a whole batch.

**Dense cluster-label assumptions** in `merging_function`, which raised
`IndexError` whenever `Wall` had a gap.

## Tests

272 pass, 1 skipped, 2 xfail. Run them with
`~/anaconda3/envs/kilosort1/bin/python -m pytest tests/ -q` — the default
interpreter has no `faiss` and collection fails at import.

The skipped one is `tests/test_full_pipeline.py`, the upstream end-to-end
regression test against upstream's own saved reference output on a Neuropixels
probe. It needs `--runslow` and about 6 minutes on CPU — 19 minutes on stock,
which is the speedup showing up on a path that has no Triton in it.

**Nothing in this branch had ever run that test before 4.5**, and it is what
found the CPU bug above. Run it before any future release.

## Known limitations

- **The CPU integration test fails its unit-count bound**, on a reference
  upstream itself nearly misses. See "CPU reference comparison" below. This is
  the one red test in the release and it is stated rather than waived.
- **Byte identity is established on GPU only.** It has never been measured on
  CPU, in either direction.
- Drift correction is untested here. Every recording in this work uses
  `nblocks=0`, because a retina preparation does not drift the way a probe in
  cortex does. The code path is unchanged from upstream, not validated.
- The Triton fast paths need a CUDA device. Everything falls back to the stock
  statements without one, so the fork runs on CPU but at stock speed.
- `max_peels` and `EI_MODE` defaults are unchanged from upstream and from the
  lab profile respectively. Both have measured cases for changing them and both
  are decisions for the lab, not for this release.

## CPU reference comparison — read this before quoting the failure

`tests/test_full_pipeline.py` compares a CPU sort of a Neuropixels test file
against a reference output upstream saved on some other machine. Kilosort 4.5
fails its unit-count bound. **Pristine upstream, run on this machine today,
only just passes it**, so the reference is the bigger problem.

| CPU run, same input | spikes | units | unit error | test bound | verdict |
|---|---:|---:|---:|---:|---|
| upstream's saved reference | 138,666 | 267 | — | — | — |
| pristine upstream `17743f2` | 137,696 | 280 | 4.64 % | 5 % | passes |
| **Kilosort 4.5** | 138,183 | 285 | 6.32 % | 5 % | **fails** |

Upstream itself is 13 units away from its own saved reference, which is 93 % of
the way to the limit. **The CPU path is machine-dependent**, and the reference
was not regenerated on this hardware.

**Whether the remaining 5-unit gap is this fork or CPU run-to-run variability
is NOT measured.** Establishing it needs a repeat-run stability estimate on
this machine, which nothing here provides. Do not state that the fork changes
CPU results, and do not state that it does not.

What *is* measured is the GPU path, which is the fork's target: byte-identical
to stock at production scale on two array geometries, 23/23 and 24/24 files,
in both directions.

Both thresholds in that test carry upstream's own
`TODO: Make sure these are reasonable error bounds`, and three further
assertions above them are commented out for deviations upstream could not
explain either.

Evidence for the table is kept at
`scratchpad/ks_upstream/upstream_pipeline.log`, in a detached `origin/main`
worktree carrying only the device-guard fix.

## Versioning

The version comes from `setuptools_scm`, so it is the git tag. Tagging `v4.5.0`
makes the wheel `4.5.0`. Note that upstream's own tags are `v4.1.x`, so `4.5.0`
is this fork's number and does not correspond to any upstream release. A reader
who needs the exact base commit should use `git describe`, which reports the
upstream tag plus the fork's commit count.
