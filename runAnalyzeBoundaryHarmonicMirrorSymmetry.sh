#!/bin/bash
# sbatch --time 1:00:00 -p batch -c 4 --mem 16G -o slurmAnalyzeBoundaryHarmonicMirrorSymmetry.out runAnalyzeBoundaryHarmonicMirrorSymmetry.sh
# Left-right asymmetry of each order's best-code run over 50,000 iterations, default model against the all-float64 model.
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=$PWD python -u analyzeBoundaryHarmonicMirrorSymmetry11x11.py --numIterations 50000 --stride 100
