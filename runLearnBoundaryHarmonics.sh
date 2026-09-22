#!/bin/bash
# CMA-ES training of a mirror-symmetric ring code, orders 0 to maxOrder, one restart per array task
# (learnBoundaryHarmonics11x11.py). Orders run one after another so each can start from the codes of the one below:
#J=$(sbatch --parsable --array 0-19 --export=ALL,maxOrder=0 --time 3:00:00 -p batch -c 2 --mem 8G -o slurmBoundaryHarmonicTraining_order0_%a_%A.out runLearnBoundaryHarmonics.sh)
#J=$(sbatch --parsable --dependency afterany:$J --array 0-19 --export=ALL,maxOrder=1 ... runLearnBoundaryHarmonics.sh)   and so on up to maxOrder=4
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PYTHONPATH=$PWD python learnBoundaryHarmonics11x11.py --maxOrder $maxOrder --restart $SLURM_ARRAY_TASK_ID ${extraArguments}
