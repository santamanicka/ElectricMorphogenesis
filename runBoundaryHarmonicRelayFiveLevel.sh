#!/bin/bash
# The two-sided slider and 5 x 5 grid ring codes (buildBoundaryHarmonicRingCodeFiveLevelGrid11x11.py), one relay
# decomposition per array task; the task's key is the array index into the JSON's `variants`.
# sbatch --array 0-94%32 --time 1:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicRelayFiveLevel_%a.out runBoundaryHarmonicRelayFiveLevel.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
VARIANTS=data/boundaryHarmonicRingCodeFiveLevel1888Hold301FaceMinus60Minus5.json
KEY=$(python3 -c "import json,sys; print(json.load(open('$VARIANTS'))['variants'][$SLURM_ARRAY_TASK_ID]['key'])")
PYTHONPATH=$PWD python computeBoundaryHarmonicRelay11x11.py \
  --ringCodeVariantsPath $VARIANTS \
  --ringCodeKey $KEY \
  --baseline extraUpdateOnly \
  --outputPath data/boundaryHarmonicRelayFiveLevel_${KEY}1888Hold301FaceMinus60Minus5Raw.npz
