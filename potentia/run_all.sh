#!/bin/bash
# all sweeps; finite tissue, homogeneous start, no clamp
set -e
python3 atlas.py S1 11 potentia:4
python3 atlas.py S2 11 lattice
python3 atlas.py S2 11 potentia:2
python3 atlas.py S3 11 lattice
python3 atlas.py S3 11 potentia:2
python3 atlas.py S1 25 lattice
python3 atlas.py S1 25 potentia:2
echo ALL_DONE
