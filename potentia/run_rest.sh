#!/bin/bash
set -e
python3 atlas.py S1 25 potentia:2
python3 sub_scans.py lattice
python3 sub_scans.py potentia:2
for s in F1 F2; do for m in lattice potentia:2 potentia:4; do python3 fine_scans.py $s $m; done; done
echo ALL_DONE
