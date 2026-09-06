#!/bin/bash
# Production-scale validation of the live-tile LUT on the whole of data000.
# Two LUT runs, so this also carries its own wobble control. The no-LUT
# baseline already exists and is byte-validated: results_full/full_A and
# full_A2, 383.60 / 383.57 s, 23/23 identical to each other and to stock.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kilosort1
cd /home/localadmin/Downloads/Kilosort-1
V=/tmp/claude-1001/-home-localadmin-Documents-Development-MEA-fieldlab-src-utilities/5436d4cb-2b4c-4887-a914-4bb3f3826c88/scratchpad/20260514A_validation
S=/tmp/claude-1001/-home-localadmin-Documents-Development-MEA-fieldlab-src-utilities/5436d4cb-2b4c-4887-a914-4bb3f3826c88/scratchpad/peel_bw
PROBE=/home/localadmin/Documents/Development/MEA-fieldlab/src/pipeline_utilities/kilosort/LITKE_512_ARRAY.mat
COMMON="--data $V/full/full_514a_data000.bin --probe $PROBE --settings $V/settings.json --invert-sign"
echo "=== re-warm input, same as the baseline arms saw ==="; date
cat $V/full/full_514a_data000.bin > /dev/null
grep -E "MemFree|MemAvailable" /proc/meminfo
for r in 1 2; do
  echo "=== full LUT run $r ==="; date
  python tools/run_full_sort.py $COMMON --results-dir $S/res/full_L$r \
      > $S/logs/full_L$r.log 2>&1
  echo "L$r exit $? : $(grep -oP 'Total runtime: \K[0-9.]+' $S/logs/full_L$r.log)"
done
echo "=== LUT gate ==="; grep -h "live-tile LUT" $S/logs/full_L1.log
echo "=== byte comparisons against the validated no-LUT baseline ==="
for pair in "full_A:$V/results_full/full_A" "full_A2:$V/results_full/full_A2"; do
  n=${pair%%:*}; d=${pair#*:}
  echo "--- L1 vs $n"; python tools/compare_sorts.py $S/res/full_L1 $d -q 2>&1 | tail -2
done
echo "--- L1 vs L2 (wobble control)"
python tools/compare_sorts.py $S/res/full_L1 $S/res/full_L2 -q 2>&1 | tail -2
echo "=== stage timings ==="
for a in full_L1 full_L2; do echo "--- $a"; grep -E "spikes extracted|clusters found|units found|Total runtime:" $S/logs/$a.log; done
echo "=== done ==="; date
