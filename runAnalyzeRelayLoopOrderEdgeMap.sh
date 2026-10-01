#!/bin/bash
# sbatch --time 3:00:00 -p batch -c 16 --mem 32G -o slurmAnalyzeRelayLoopOrderEdgeMap.out runAnalyzeRelayLoopOrderEdgeMap.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=$PWD python -u analyzeRelayLoopOrderEdgeMap11x11.py
