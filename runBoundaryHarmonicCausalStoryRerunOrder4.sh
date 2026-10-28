#!/bin/bash
# Re-run of the two order-4 causal-story analyses after Amendment 2 of the registration (three validity checks re-pointed to the order-4 values;
# no criterion changed). The first-run outputs are kept as ...Order4FirstRun.json. Defaults of the registered scripts, BOUNDARY_FACE_ORDER=4.
# sbatch --array 0-1 --time 2:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicCausalStoryRerunOrder4_%a.out runBoundaryHarmonicCausalStoryRerunOrder4.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$PWD BOUNDARY_FACE_ORDER=4
case $SLURM_ARRAY_TASK_ID in
  0) python analyzeBoundaryHarmonicCausalStory11x11.py ;;
  1) python analyzeBoundaryHarmonicCausalStoryRobustness11x11.py ;;
esac
