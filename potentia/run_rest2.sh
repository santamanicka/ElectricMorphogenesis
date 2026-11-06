#!/bin/bash
set -e
for m in lattice potentia:2; do python3 fine_scans.py F3 $m; done
echo ALL_DONE_2
