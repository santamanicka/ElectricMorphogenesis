#!/bin/bash
# Whole Vmem trajectories for the free-or-new check: the free run (50,000 iterations), a shifted-start control, 20 free
# runs from random symmetric starting states, and the dial held at 0, 0.6 and 1.3 (20,000 iterations each), one per task.
#sbatch --time 0:30:00 -p batch --array 0-24 --mem 4G -o slurmBoundaryRecurrence_%A_%a.out runBoundaryRecurrence.sh
#sbatch --export=ALL,runSet=tilted --time 0:30:00 -p batch --array 0-84 --mem 4G -o slurmBoundaryRecurrence_%A_%a.out runBoundaryRecurrence.sh
#sbatch --export=ALL,runSet=tiltedStarts --time 0:30:00 -p batch --array 0-107 --mem 4G -o slurmBoundaryRecurrence_%A_%a.out runBoundaryRecurrence.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
python simulateBoundaryRecurrence11x11.py --runSet ${runSet:-dial} --taskIndex ${SLURM_ARRAY_TASK_ID}
