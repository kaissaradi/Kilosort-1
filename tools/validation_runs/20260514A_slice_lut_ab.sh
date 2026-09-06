#!/bin/bash
# Slice A/B for the live-tile LUT. Each arm runs TWICE and the second is
# quoted: the LUT kernel is new, so its first run pays Triton JIT that the
# no-LUT arm does not -- exactly the contamination that made the old 1.45x
# headline wrong. Order L,A,L,A so neither arm owns the cold cache.
source ~/anaconda3/etc/profile.d/conda.sh
conda activate kilosort1
cd /home/localadmin/Downloads/Kilosort-1
V=/tmp/claude-1001/-home-localadmin-Documents-Development-MEA-fieldlab-src-utilities/5436d4cb-2b4c-4887-a914-4bb3f3826c88/scratchpad/20260514A_validation
S=/tmp/claude-1001/-home-localadmin-Documents-Development-MEA-fieldlab-src-utilities/5436d4cb-2b4c-4887-a914-4bb3f3826c88/scratchpad/peel_bw
PROBE=/home/localadmin/Documents/Development/MEA-fieldlab/src/pipeline_utilities/kilosort/LITKE_512_ARRAY.mat
COMMON="--data $V/slice300_514a.bin --probe $PROBE --settings $V/settings.json --invert-sign"
mkdir -p $S/logs $S/res
for round in 1 2; do
  for arm in L A; do
    [ $arm = A ] && export KILOSORT_NO_PEEL_LUT=1 || unset KILOSORT_NO_PEEL_LUT
    python tools/run_full_sort.py $COMMON --results-dir $S/res/${arm}${round} \
      > $S/logs/${arm}${round}.log 2>&1
    echo "$arm$round exit $? : $(grep -oP 'Total runtime: \K[0-9.]+' $S/logs/${arm}${round}.log)"
  done
done
unset KILOSORT_NO_PEEL_LUT
echo "=== LUT gate lines ==="
grep -h "live-tile LUT" $S/logs/L2.log
echo "=== byte comparison: LUT vs no-LUT ==="
python tools/compare_sorts.py $S/res/L2 $S/res/A2 2>&1 | tail -4
echo "=== stage timings ==="
for a in L2 A2; do echo "--- $a"; grep -E "spikes extracted|clusters found|units found|Total runtime:" $S/logs/$a.log; done
