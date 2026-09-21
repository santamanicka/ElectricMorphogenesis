#!/bin/bash
# Snapshot-by-snapshot free-or-new analysis of the runs from runBoundaryRecurrence.sh.
#sbatch --time 1:00:00 -p batch -c 8 --mem 24G -o slurmBoundaryRecurrenceAnalysis_%j.out runBoundaryRecurrenceAnalysis.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
PYTHONPATH=$PWD python ${script:-analyzeBoundaryRecurrence11x11.py}
