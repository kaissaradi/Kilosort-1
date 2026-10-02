# Kilosort4, MEA fork (branch `mea-optimizations`): quick start

The fork sorts 3–5× faster than stock Kilosort4. The GPU kernels give byte-identical output;
5 other speed changes can change float rounding or tie order (CHANGE_INVENTORY class 2).
It reads Litke raw data directly, adds MEA settings and merges, and writes Vision files.
Full list of changes: MEA-fieldlab `papers/ks4_retina_mea/qa_report/CHANGE_INVENTORY.md`.

## 1. Environment

```bash
conda activate kilosort1                       # has the fork (editable) + visionwriter
cd MEA-fieldlab/src && python -m mea doctor    # must say "All critical checks passed"
```

The `kilosort` env holds **stock** 4.0.32. Do not sort with it.

## 2. Full run: native Litke → sort → Vision `.neurons/.ei` → STA

```bash
python -m mea run 20260514A chunk1 -f "data003" -n "data003" -a 60 -l --dry-run   # check, then drop --dry-run
```

| Option | What it does |
|---|---|
| `-l`, `--native-litke` | Read the raw `dataXXX` folders directly (no bin2py). |
| `--no-flat-bin` | With `-l`: do not write the joined `{chunk}.bin` (KS4 only). |
| `-a 30` / `-a 60` | Array pitch. Picks the tuned per-array settings. |
| `-k '...'` | Pass flags to `run_kilosort4.py` (table below). |
| `--sort-only` | Sort only. No Vision export, EI or STA. |
| `--lab-path DIR` | Archive to `DIR`, not the lab share. Use it for a test run. |

Flags for `-k` (all optional):

| Flag | Effect |
|---|---|
| `--params tuned` | The default. `baseline` = the Nov 2025 stock sort settings. |
| `--deterministic` | Same output on every run. Use it for any A/B comparison. |
| `--dedup-mode auto` | The default. 60 µm: label axonal copies (`axonal`), no merge. 30 µm: merge, then label. |
| `--coincidence_frac_thresh X` | The coincidence merge. The default is 0.90 (drop-only + keep-cleaner) at 60 µm, off at 30 µm. |
| `--plots`, `--pc-features` | Write the plots / Phy PC files (off by default). |

Environment variables:

| Variable | Values |
|---|---|
| `EI_BACKEND` | `fast` (default, numba, ~20× faster than Java), `python`, `java` |
| `EI_MODE` | `exact` (default, bit-identical to Java), `accurate` (no int16 truncation bias), `truncated64` |
| `MEA_STAGE_TIMES=file` | Write the start and end time of each pipeline stage to `file`. |
| `MEA_KEEP_TRASH=1` | Do not empty `~/.local/share/Trash` at the end. |

## 3. Vision bundle from an existing sort (no re-sort)

```bash
python -m mea export-vision /path/to/kilosort4 /path/to/raw/data003 -o /path/to/vision/kilosort4 --mode exact
```

This writes `.neurons`, `.globals`, `.ei`, `vision_export.json` and `<name>.axonal.tsv`.
It writes no `.params` and no `.sta`. The STA step makes those (`mea run`, `mea analyze` or `mea sta-vision`).

## 4. Sort only, from Python

```python
from kilosort.litke import LitkeRecording
from kilosort import run_kilosort
rec = LitkeRecording('/path/to/EXP/data003')          # electrode 0 (TTL) is dropped
run_kilosort({'n_chan_bin': rec.n_chan, 'fs': int(rec.fs), 'results_dir': 'out'},
             filename=str(rec.paths[0]), file_object=rec, probe=probe)   # probe: your 512/519 Litke probe
```

See `docs/litke.rst` for the TTL and probe details.

## 5. Switches for checks (not for production)

| Variable | Effect |
|---|---|
| `KILOSORT_NO_<X>=1`, X = `FUSED_DETECT`, `FUSED_PEEL`, `FUSED_PEAKS`, `FAST_KPP`, `KPP_GRAPH`, `PEEL_COND`, `PEEL_STORE`, `PEEL_LUT`, `COMPILE` | Turn one GPU speed-up off. The output stays the same. |
| `KILOSORT_TRACE_DIR=dir` | Save the spike table after each sorter stage. |
| `KS4_DUMP_RESIDUAL=dir:batches` | Save the peel residual for the given batches. |

Test: `pytest tests/test_regression_switches.py --runslow` with `KS_REGRESSION_BIN` and
`KS_REGRESSION_OPS` set sorts 60 s twice (all speed-ups on, then all off) and checks that the outputs are identical.
