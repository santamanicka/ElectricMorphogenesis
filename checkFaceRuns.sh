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
echo "=== Run 0 control, array 2258342 ==="
sacct -j 2258342 -X -n -o State 2>/dev/null | sort | uniq -c | tr '\n' ' '; echo
