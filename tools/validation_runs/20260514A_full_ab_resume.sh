#!/bin/bash
# Resume the production-scale three-arm run: arms B and A2 only.
#
# Arm A already completed (383.60 s, 1630 good units, 23 comparable output
# files) and is kept. The driver was killed between arms by the background-task
# monitor reporting "low memory" -- a false positive: MemAvailable never fell
# below 190 of 197 GB and kilosort's own peak was 9.30 GB. MemFree was low only
# because streaming the 34 GB input (and the 25.5 GB raw it was built from)
# filled the page cache with fully reclaimable pages. Confirmed from
# /proc/meminfo at the moment of the kill:
#     MemTotal 197.3 / MemAvailable 190.6 / MemFree 7.5 / Cached 177.1 GB
#
# CACHE FAIRNESS. Arm A ran with the input warm in page cache, because the
# build had just written it. The slice protocol runs A -> B -> A2 for exactly
# that reason: B inherits A's warm cache, which flatters B and therefore
# UNDERSTATES the A/B speedup. Reclaiming cache before B would reverse that and
# inflate the result, so the input is deliberately re-warmed to restore arm A's
# conditions before B starts. Stale cache from unrelated files is dropped for
# monitor headroom; the input itself is not.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kilosort1
cd /home/localadmin/Downloads/Kilosort-1

V=/tmp/claude-1001/-home-localadmin-Documents-Development-MEA-fieldlab-src-utilities/5436d4cb-2b4c-4887-a914-4bb3f3826c88/scratchpad/20260514A_validation
BIN=$V/full/full_514a_data000.bin
PROBE=/home/localadmin/Documents/Development/MEA-fieldlab/src/pipeline_utilities/kilosort/LITKE_512_ARRAY.mat
COMMON="--data $BIN --probe $PROBE --settings $V/settings.json --invert-sign"

echo "=== re-warm input to restore arm A's cache conditions ==="
date
cat $BIN > /dev/null
grep -E "MemFree|MemAvailable" /proc/meminfo

echo "=== arm B: all optimizations OFF (stock paths) ==="
date
KILOSORT_NO_FUSED_DETECT=1 KILOSORT_NO_FUSED_PEEL=1 KILOSORT_NO_FUSED_PEAKS=1 \
KILOSORT_NO_FAST_KPP=1 KILOSORT_NO_KPP_GRAPH=1 \
python tools/run_full_sort.py $COMMON --results-dir $V/results_full/full_B \
    > $V/logs/13_full_B_stock.log 2>&1
echo "B exit $?"
grep -E "MemFree|MemAvailable" /proc/meminfo

echo "=== arm A2: all optimizations ACTIVE, second run (wobble control) ==="
date
python tools/run_full_sort.py $COMMON --results-dir $V/results_full/full_A2 \
    > $V/logs/14_full_A2_fused.log 2>&1
echo "A2 exit $?"

echo "=== comparisons ==="
date
python tools/compare_sorts.py $V/results_full/full_A $V/results_full/full_B \
    > $V/logs/15_full_compare_A_vs_B.log 2>&1;  echo "A vs B  exit $?"
python tools/compare_sorts.py $V/results_full/full_A $V/results_full/full_A2 \
    > $V/logs/16_full_compare_A_vs_A2.log 2>&1; echo "A vs A2 exit $?"
python tools/compare_sorts.py $V/results_full/full_A2 $V/results_full/full_B \
    > $V/logs/17_full_compare_A2_vs_B.log 2>&1; echo "A2 vs B exit $?"
echo "=== done ==="
date
