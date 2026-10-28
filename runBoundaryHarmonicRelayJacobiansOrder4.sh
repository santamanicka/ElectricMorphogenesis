#!/bin/bash
# The order-4 ring-only relay again, this time keeping every step's Jacobian (about 1.5 GB, not committed), for the wall-wiring, wall-
# counterfactual, coarse-graining, causal-story analyses (batch 2 of data/boundaryHarmonicOrder4ReplicationPredictions1888Hold301FaceMinus60Minus5.json).
# Same settings as the batch-1 ring-only relay: --order 4 --primaryIteration 1687 --windowStates 302,585,1001,1501,1688 --baseline extraUpdateOnly.
# sbatch --time 2:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicRelayJacobiansOrder4.out runBoundaryHarmonicRelayJacobiansOrder4.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$PWD
OUT=data/boundaryHarmonicRingOnlyRelay1888Hold301FaceMinus60Minus5Order4RegeneratedRaw.npz
JAC=data/boundaryHarmonicRingOnlyRelayJacobians1888Hold301FaceMinus60Minus5Order4.npy
for f in $OUT $JAC; do [ -e "$f" ] && { echo "$f exists; not overwriting"; exit 1; }; done
python computeBoundaryHarmonicRelay11x11.py --order 4 --primaryIteration 1687 --windowStates 302,585,1001,1501,1688 \
  --baseline extraUpdateOnly --jacobianPath $JAC --outputPath $OUT
