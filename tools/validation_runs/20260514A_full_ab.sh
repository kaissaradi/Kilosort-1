#!/bin/bash
# Three-arm byte-identity validation at PRODUCTION SCALE on 20260514A/data000.
#
# Everything previously claimed for this array geometry was measured on a
# 150 s slice (300 batches). This runs the whole recording: 33,140,000 samples,
# 1657.0 s, 3314 batches -- 11.05x the slice.
#
# WHY THIS RUN EXISTS. KS4_VALIDATION_NOTES states that the run-to-run wobble
# (13 float32 of 33 M) was only ever observed at production scale, and that
# "a 300-batch slice does not have the statistical reach to see it". So the
# slice's clean A-vs-A2 is not evidence of bit-reproducibility -- it is a test
# that structurally cannot fail. This run is the one that can.
#
# INPUT VERIFIED BEFORE LAUNCH (logs/11_verify_full.log + cmp):
#   a. 14 part files in numeric order, no gaps
#   b. 33935360000 bytes == 33140000 samples x 512 ch x 2
#   c. first 3072000000 bytes byte-identical to slice300_514a.bin
#   d. all 13 inter-file seams + 12 random interior blocks match the reader
#   e. tail is signal (std 49.9), not zero padding
#
# PREDICTIONS, recorded before the run so the result can falsify them:
#   1. A-vs-A2 gap shrinks well below the slice's 21%. The slice blamed that
#      gap on Triton JIT/autotune plus a cold page cache -- fixed costs, now
#      amortized over 11x the work. If it stays near 21%, that explanation
#      was wrong.
#   2. A-vs-B lands BELOW the slice's 1.45x. Clustering gains fell 2.47x ->
#      1.11x and 2.38x -> 1.23x going from slice to production on 20260724A;
#      detection gains held. Same direction expected here.
#   3. Byte-identity is genuinely open. This is the first run on this geometry
#      with the reach to see the wobble.
#
# NB: no `set -u` -- conda's activate.d hooks read unset MKL_INTERFACE_LAYER.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kilosort1
cd /home/localadmin/Downloads/Kilosort-1

V=/tmp/claude-1001/-home-localadmin-Documents-Development-MEA-fieldlab-src-utilities/5436d4cb-2b4c-4887-a914-4bb3f3826c88/scratchpad/20260514A_validation
BIN=$V/full/full_514a_data000.bin
PROBE=/home/localadmin/Documents/Development/MEA-fieldlab/src/pipeline_utilities/kilosort/LITKE_512_ARRAY.mat
# Same settings.json and same --invert-sign as the slice run, deliberately:
# the only thing that differs between this and the slice A/B is --data.
COMMON="--data $BIN --probe $PROBE --settings $V/settings.json --invert-sign"
mkdir -p $V/results_full

echo "=== arm A: all optimizations ACTIVE ==="
date
python tools/run_full_sort.py $COMMON --results-dir $V/results_full/full_A \
    > $V/logs/12_full_A_fused.log 2>&1
echo "A exit $?"

echo "=== arm B: all optimizations OFF (stock paths) ==="
date
KILOSORT_NO_FUSED_DETECT=1 KILOSORT_NO_FUSED_PEEL=1 KILOSORT_NO_FUSED_PEAKS=1 \
KILOSORT_NO_FAST_KPP=1 KILOSORT_NO_KPP_GRAPH=1 \
python tools/run_full_sort.py $COMMON --results-dir $V/results_full/full_B \
    > $V/logs/13_full_B_stock.log 2>&1
echo "B exit $?"

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
