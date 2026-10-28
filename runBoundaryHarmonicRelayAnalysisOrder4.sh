#!/bin/bash
# Scores the registered Relay and RingOnlyRelay criteria on the order-4 relays (BOUNDARY_FACE_ORDER=4 tags every output Order4 and moves the
# landmarks to the order-4 code's own). Registration: data/boundaryHarmonicOrder4ReplicationPredictions1888Hold301FaceMinus60Minus5.json
# sbatch --array 0-1 --time 1:00:00 -p batch -c 2 --mem 8G -o slurmBoundaryHarmonicRelayAnalysisOrder4_%a.out runBoundaryHarmonicRelayAnalysisOrder4.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=$PWD BOUNDARY_FACE_ORDER=4
case $SLURM_ARRAY_TASK_ID in
  0) python analyzeBoundaryHarmonicRelay11x11.py --relayPath data/boundaryHarmonicRelay1888Hold301FaceMinus60Minus5Order4Raw.npz ;;
  1) python analyzeBoundaryHarmonicRingOnlyRelay11x11.py --relayPath data/boundaryHarmonicRingOnlyRelay1888Hold301FaceMinus60Minus5Order4Raw.npz ;;
esac
