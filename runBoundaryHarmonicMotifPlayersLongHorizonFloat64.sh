#!/bin/bash
# sbatch --time 4:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicMotifPlayersLongHorizonFloat64.out runBoundaryHarmonicMotifPlayersLongHorizonFloat64.sh
# The long-horizon players again with the whole model in 64-bit, which keeps the tissue left-right symmetric for longer.
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=$PWD python -u analyzeBoundaryHarmonicMotifPlayers11x11.py --numIterations 50000 --stride 100 --doublePrecision
