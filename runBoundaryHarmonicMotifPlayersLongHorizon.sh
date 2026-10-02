#!/bin/bash
# sbatch --time 4:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicMotifPlayersLongHorizon.out runBoundaryHarmonicMotifPlayersLongHorizon.sh
# Replays the best code of every order for 50,000 iterations for the report's long-horizon players (frames 100 iterations apart).
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=$PWD python -u analyzeBoundaryHarmonicMotifPlayers11x11.py --numIterations 50000 --stride 100
