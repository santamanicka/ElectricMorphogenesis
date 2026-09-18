#!/bin/bash
# Report where each face training arm has reached. The arms differ only in horizon and all learn the
# clamp alone, so the plateau at 8.35 is the number to watch: it was the quenched-tissue local
# minimum that trapped 34 of 40 runs when the transduction bias was still being learned, and with the
# bias frozen it should not appear at all.
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
# correlation runs are scored as 1 - r and their numbers do not compare with the millivolt losses
# of the globalsum arms, so they are listed separately rather than ranked against them
for spec in "2265290:horizon 1000 globalsum s4" "2265268:horizon 2500 globalsum s4" \
            "2272308:horizon 2500 CORRELATION s4" "2272314:horizon 2500 CORRELATION s10"; do
  job=${spec%%:*}; label=${spec#*:}
  echo "=== ${label}, array ${job} ==="
  states=$(sacct -j ${job} -X -n -o State 2>/dev/null | sort | uniq -c | tr '\n' ' ')
  echo "  ${states}"
  found=0
  for f in slurm*_${job}_*.out; do
    [ -f "$f" ] || continue
    rows=$(grep -E '^[0-9]+ [0-9]+ [0-9]+ ' "$f")
    [ -z "$rows" ] && continue
    found=1
    tk=$(basename "$f" .out | grep -oP '_\K[0-9]+$')
    echo "$rows" | tail -1 | awk -v t="$tk" '{print $5, t, $3}'
  done | sort -g | head -6 | awk '{printf "  task %-5s iter %-5s best %.6g\n", $2, $3, $1}'
  [ "$found" = 0 ] && echo "  (no loss lines yet)"
done
echo "=== Run 0 reachability control, array 2271430 ==="
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

# the backward chain runs locally rather than through slurm, so it is read from its log
echo "=== backward target propagation (local) ==="
CHAIN=/tmp/claude-30980/-cluster-tufts-levinlab-smanic02-Code-Git-electricmorphogenesis/b70a9035-0262-4b00-9396-8d88d8177376/scratchpad/chain.log
if [ -f "$CHAIN" ]; then
  running=$(pgrep -fc "python analyzeBackward" 2>/dev/null || echo 0)
  echo "  processes running: ${running}"
  grep -E "===|back step" "$CHAIN" | tail -14 | sed 's/^/  /'
else
  echo "  (no chain log yet)"
fi
