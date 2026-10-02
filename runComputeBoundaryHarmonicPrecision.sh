#!/bin/bash
# sbatch --time 6:00:00 -p batch -c 4 --mem 16G -o slurmComputeBoundaryHarmonicPrecision.out runComputeBoundaryHarmonicPrecision.sh
# Default model against the all-float64 model for each order's best code, and the growth of tiny differences, over 50,000 iterations.
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=$PWD python -u computeBoundaryHarmonicPrecision11x11.py --numIterations 50000 --stride 100
