#!/bin/bash
# Three-arm byte-identity validation of the optimization series on a NEW array
# geometry: 20260514A/data000, 512 channels at 60 um pitch (array 504).
#
# Every claim in KS4_VALIDATION_NOTES so far was measured on 20260724A -- 519
# channels at 30 um (array 1551). This run asks whether the gates still find
# bit-identical Triton configs at shapes they have never seen. A FALLBACK here
# is a correct outcome, not a failure: it would mean the gate did its job.
#
# Arms A and A2 are the SAME code, run twice. That is the three-way comparison:
# kilosort4 is not bit-reproducible run to run, so A-vs-B alone cannot tell a
# real regression from the program's own wobble. If A and A2 disagree with each
# other at different places than either disagrees with B, the difference
# belongs to the program.
# NB: no `set -u` -- conda's own activate.d hooks read unset variables
# (MKL_INTERFACE_LAYER) and abort the script before the first sort starts.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kilosort1
cd /home/localadmin/Downloads/Kilosort-1

V=/tmp/claude-1001/-home-localadmin-Documents-Development-MEA-fieldlab-src-utilities/5436d4cb-2b4c-4887-a914-4bb3f3826c88/scratchpad/20260514A_validation
BIN=$V/slice300_514a.bin
PROBE=/home/localadmin/Documents/Development/MEA-fieldlab/src/pipeline_utilities/kilosort/LITKE_512_ARRAY.mat
COMMON="--data $BIN --probe $PROBE --settings $V/settings.json --invert-sign"

echo "=== arm A: all optimizations ACTIVE ==="
python tools/run_full_sort.py $COMMON --results-dir $V/results/slice_A \
    > $V/logs/01_slice_A_fused.log 2>&1
echo "A exit $?"

echo "=== arm B: all optimizations OFF (stock paths) ==="
KILOSORT_NO_FUSED_DETECT=1 KILOSORT_NO_FUSED_PEEL=1 KILOSORT_NO_FUSED_PEAKS=1 \
KILOSORT_NO_FAST_KPP=1 KILOSORT_NO_KPP_GRAPH=1 \
python tools/run_full_sort.py $COMMON --results-dir $V/results/slice_B \
    > $V/logs/02_slice_B_stock.log 2>&1
echo "B exit $?"

echo "=== arm A2: all optimizations ACTIVE, second run (wobble control) ==="
python tools/run_full_sort.py $COMMON --results-dir $V/results/slice_A2 \
    > $V/logs/03_slice_A2_fused.log 2>&1
echo "A2 exit $?"

echo "=== comparisons ==="
python tools/compare_sorts.py $V/results/slice_A $V/results/slice_B \
    > $V/logs/04_compare_A_vs_B.log 2>&1;  echo "A vs B  exit $?"
python tools/compare_sorts.py $V/results/slice_A $V/results/slice_A2 \
    > $V/logs/05_compare_A_vs_A2.log 2>&1; echo "A vs A2 exit $?"
python tools/compare_sorts.py $V/results/slice_A2 $V/results/slice_B \
    > $V/logs/06_compare_A2_vs_B.log 2>&1; echo "A2 vs B exit $?"
echo "=== done ==="
