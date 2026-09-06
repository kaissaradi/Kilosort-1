#!/bin/bash
# Does running two sorts at once on ONE GPU beat running them back to back?
#
# The census says memory never binds (peak 9.60 GB of a 19.54 GB card), and the
# peel loop is documented launch-bound -- a launch-bound kernel leaves the SMs
# idle between launches, which is exactly the gap a second process can fill.
# nvidia-smi's "100% utilization" does not contradict that: it reports the
# fraction of time ANY kernel was resident, not SM occupancy.
#
# Interleaved baselines, per the execution discipline: solo, pair, solo. If the
# machine drifts mid-test the two solo runs disagree and the result is void.
set -euo pipefail
PY=/home/localadmin/anaconda3/envs/kilosort1/bin/python
KS=/home/localadmin/Downloads/Kilosort-1
OPS=/tmp/claude-1001/ks_fused/ops.npy
BIN=/tmp/claude-1001/-home-localadmin-Documents-Development-MEA-fieldlab-src-utilities/5436d4cb-2b4c-4887-a914-4bb3f3826c88/scratchpad/slice300.bin
OUT=/tmp/claude-1001/conc
rm -rf "$OUT"; mkdir -p "$OUT"

run_one() {  # name resultdir
    /usr/bin/time -f "%e" -o "$OUT/$1.wall" \
        "$PY" "$KS/tools/run_full_sort.py" --ops "$OPS" --data "$BIN" \
        --results-dir "$2" >"$OUT/$1.log" 2>&1
}

echo "== solo A =="
run_one soloA "$OUT/soloA"; cat "$OUT/soloA.wall"

echo "== pair (two concurrent processes) =="
start=$(date +%s.%N)
run_one pair1 "$OUT/pair1" &
p1=$!
run_one pair2 "$OUT/pair2" &
p2=$!
wait $p1 $p2
end=$(date +%s.%N)
echo "$end - $start" | bc > "$OUT/pair.wall"
cat "$OUT/pair.wall"

echo "== solo B (interleaved baseline) =="
run_one soloB "$OUT/soloB"; cat "$OUT/soloB.wall"

echo "== summary =="
"$PY" - "$OUT" <<'EOF'
import sys, pathlib
o = pathlib.Path(sys.argv[1])
g = lambda n: float((o / f'{n}.wall').read_text().strip())
a, b, pair = g('soloA'), g('soloB'), g('pair')
p1, p2 = g('pair1'), g('pair2')
solo = (a + b) / 2
print(f'solo A / solo B       : {a:.1f} s / {b:.1f} s  (drift {abs(a-b)/solo*100:.1f}%)')
print(f'two sorts back to back: {2*solo:.1f} s')
print(f'two sorts concurrent  : {pair:.1f} s   (each {p1:.1f} / {p2:.1f})')
print(f'throughput speedup    : {2*solo/pair:.2f}x')
EOF
