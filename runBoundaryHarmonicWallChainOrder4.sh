#!/bin/bash
# Batch 2 of data/boundaryHarmonicOrder4ReplicationPredictions1888Hold301FaceMinus60Minus5.json: the order-4 analyses that need the Jacobian relay
# (task 0: wall wiring, then coarse-graining on the same fine sweep, then wall counterfactual, then causal story) or only replays
# (task 1: causal-story robustness). BOUNDARY_FACE_ORDER=4 tags every output Order4 and moves the landmarks to the order-4 code's. All settings
# are the scripts' registered defaults. Each script refuses to overwrite.
# sbatch --dependency=afterok:<jacobian relay job> --array 0-1 --time 4:00:00 -p batch -c 4 --mem 32G -o slurmBoundaryHarmonicWallChainOrder4_%a.out runBoundaryHarmonicWallChainOrder4.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$PWD BOUNDARY_FACE_ORDER=4
REGEN=data/boundaryHarmonicRingOnlyRelay1888Hold301FaceMinus60Minus5Order4RegeneratedRaw.npz
JAC=data/boundaryHarmonicRingOnlyRelayJacobians1888Hold301FaceMinus60Minus5Order4.npy
CACHE=data/boundaryHarmonicCoarseGrainFineSweep1888Hold301FaceMinus60Minus5Order4.npz
set -e
case $SLURM_ARRAY_TASK_ID in
  0) [ -e $CACHE ] && { echo "$CACHE exists; not overwriting"; exit 1; }
     python analyzeBoundaryHarmonicWallWiring11x11.py --jacobianPath $JAC --relayPath $REGEN --cachePath $CACHE
     python analyzeBoundaryHarmonicCoarseGrain11x11.py --jacobianPath $JAC --relayPath $REGEN --cachePath $CACHE
     python analyzeBoundaryHarmonicWallCounterfactual11x11.py
     python analyzeBoundaryHarmonicCausalStory11x11.py ;;
  1) python analyzeBoundaryHarmonicCausalStoryRobustness11x11.py ;;
esac
