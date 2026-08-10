#!/bin/bash
# Status of the two things still running: the backward screen sweep and the reachability control.
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
echo "=== backward screen sweep, array 2277256 ==="
sacct -j 2277256 -X -n -o State 2>/dev/null | sort | uniq -c | tr '\n' ' '; echo
for f in slurmScreenSweep_2277256_*.out; do
  [ -f "$f" ] || continue
  screen=$(grep -oE "^screen [0-9]+" "$f" | head -1 | awk '{print $2}')
  ctrl=$(grep "^CONTROL" "$f" | grep -oE "[0-9.]+ mV")
  tgt=$(grep "^TARGET" "$f" | grep -oE "[0-9.]+ mV")
  printf "  screen %-3s control %-12s target %-12s\n" "${screen:-?}" "${ctrl:-running}" "${tgt:-running}"
done
echo "=== reachability control, array 2271430 ==="
sacct -j 2271430 -X -n -o State 2>/dev/null | sort | uniq -c | tr '\n' ' '; echo
for f in slurmRun0c_2271430_3*.out; do
  [ -f "$f" ] || continue
  rows=$(grep -E '^[0-9]+ [0-9]+ [0-9]+ ' "$f"); [ -z "$rows" ] && continue
  tk=$(basename "$f" .out | grep -oP '_\K[0-9]+$')
  first=$(echo "$rows" | head -1 | awk '{print $5}')
  last=$(echo "$rows" | tail -1 | awk '{print $5}')
  it=$(echo "$rows" | tail -1 | awk '{print $3}')
  awk -v a=$first -v b=$last -v t=$tk -v i=$it 'BEGIN{printf "%.2f %s %s %.5g\n", 100*(a-b)/a, t, i, b}'
done | sort -rn | head -4 | awk '{printf "  task %-5s iter %-5s %.5g  %.1f%% down\n", $2, $3, $4, $1}'
