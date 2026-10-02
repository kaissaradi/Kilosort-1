# Kilosort4, MEA fork (branch `mea-optimizations`): quick start

## 1. Sort the Litke raw data directly (no bin2py, no flat .bin)

```python
import torch
from kilosort import run_kilosort, io
from kilosort.litke import LitkeRecording

raw = ['/path/EXP/data003']                              # one or more dataXXX folders, in order
recs = [LitkeRecording(p) for p in raw]                  # drops electrode 0 (TTL)
data = recs[0] if len(recs) == 1 else io.BinaryFileGroup(file_objects=recs)
probe = io.load_probe('/path/to/LITKE_512_ARRAY.mat')

settings = {                                             # production, 60 um array (512 ch)
    'n_chan_bin': 512, 'fs': 20000, 'batch_size': 10000, 'nblocks': 0,
    'dmin': 60, 'dminx': 60, 'nearest_chans': 19, 'nt': 81, 'max_channel_distance': 66,
    'n_templates': 10, 'Th_single_ch': 5.0, 'x_centers': 10, 'whitening_range': 37,
    'ccg_threshold': 0.2, 'split_ccg_threshold': 0.4,
    'coincidence_frac_thresh': 0.9,                      # the coincidence merge
    'coincidence_import_unmatched': False,               # drop-only
    'coincidence_keep_cleaner': True,
}
run_kilosort(settings, probe=probe, filename=str(recs[0].paths[0]), file_object=data,
             results_dir='out/kilosort4', data_dtype='int16', invert_sign=True, do_CAR=False,
             device=torch.device('cuda'), save_pc_features=False)
```

**30 µm array (519 ch):** use `LITKE_519_ARRAY_30UM.mat` and
`{'n_chan_bin': 519, 'fs': 20000, 'batch_size': 10000, 'nblocks': 0, 'dmin': 15, 'max_channel_distance': 33,
'n_templates': 10, 'Th_single_ch': 5.0, 'x_centers': 10, 'whitening_range': 37, 'ccg_threshold': 0.2,
'split_ccg_threshold': 0.4}`. The coincidence merge stays off there (no ground truth at 30 µm).

**Same output on every run** (for any A/B): before you import torch, set `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Then call `torch.use_deterministic_algorithms(True, warn_only=True)`.

## 2. Vision files with the EI

```bash
python tools/export_vision.py out/kilosort4 /path/EXP/data003 -o out/vision/kilosort4 --mode exact
```

This writes `kilosort4.neurons`, `.globals`, `.ei` and `vision_export.json`. It does not write `.params` or `.sta`.

| Option | Effect |
|---|---|
| `--mode exact` | The default. The EI is bit-identical to Java Vision. |
| `--mode accurate` | No int16 truncation. Java EIs are 0.83 % too small (median). |
| `--good-only` | Export only the `good` units. |
| `--left 67 --right 133` | EI window in samples (the defaults). |

## 3. Sort a flat int16 .bin

```bash
python tools/run_full_sort.py --data chunk.bin --results-dir out/kilosort4 --ops old_sort/ops.npy --deterministic
python tools/run_full_sort.py --data chunk.bin --results-dir out/kilosort4 --probe P.mat --settings S.json --invert-sign
```

`--ops` replays the settings and probe of an earlier sort. `--set KEY=VALUE` changes one setting.

## 4. Settings the fork adds (`kilosort/parameters.py`)

| Setting | Fork default | What it does |
|---|---|---|
| `coincidence_frac_thresh` | 0 (off) | Merge two units that share this fraction of spikes at one CCG lag. |
| `coincidence_import_unmatched` | True | False = drop-only: delete only the copied spikes. |
| `coincidence_keep_cleaner` | False | With drop-only: delete the copies from the unit with the higher CCG contamination. |
| `split_ccg_threshold` | 0.25 | CCG threshold for splits (stock uses `ccg_threshold`). |
| `max_merge_sweeps` | 10 | Repeat the KS duplicate merge until nothing changes. |
| `refractory_merge_veto` | True | Refuse a clustering merge whose result breaks the refractory period. |
| `isi_threshold`, `lam`, `residual_Th`, `discover_templates`, `final_merge_*` | off | Tested, not used in production. |

## 5. Switches for checks (not for production)

| Variable | Effect |
|---|---|
| `KILOSORT_NO_<X>=1`, X = `FUSED_DETECT`, `FUSED_PEEL`, `FUSED_PEAKS`, `FAST_KPP`, `KPP_GRAPH`, `PEEL_COND`, `PEEL_STORE`, `PEEL_LUT`, `COMPILE` | Turn one GPU speed-up off. The output stays the same. |
| `KILOSORT_TRACE_DIR=dir` | Save the spike table after each sorter stage. |
| `KS4_DUMP_RESIDUAL=dir:batches` | Save the peel residual for the given batches. |

Test: `pytest tests/test_regression_switches.py --runslow` with `KS_REGRESSION_BIN` and
`KS_REGRESSION_OPS` set sorts 60 s twice (all speed-ups on, then all off) and checks that the outputs are identical.
