#!/bin/bash
# The sweep's ring codes (buildBoundaryHarmonicRingCodeSweep11x11.py), one relay decomposition per array task; the task's code
# is the array index into the JSON's `variants`. The raw relay file goes to a temporary directory and is deleted once
# extractRelayLoopSweepRecord11x11.py has written the small record under data/relayLoopSweep/, so 400 codes do not add 6 GB.
# A code whose record already exists is skipped, so the array can be resubmitted after a partial failure.
# sbatch --array 0-399%64 --time 1:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicRelaySweep_%a.out runBoundaryHarmonicRelaySweep.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
SUFFIX=1888Hold301FaceMinus60Minus5
VARIANTS=data/boundaryHarmonicRingCodeSweep${SUFFIX}.json
KEY=$(python3 -c "import json; print(json.load(open('$VARIANTS'))['variants'][$SLURM_ARRAY_TASK_ID]['key'])")
if [ -f data/relayLoopSweep/${KEY}.json ]; then echo "skip $KEY (record exists)"; exit 0; fi
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
PYTHONPATH=$PWD python computeBoundaryHarmonicRelay11x11.py \
  --ringCodeVariantsPath $VARIANTS --ringCodeKey $KEY --baseline extraUpdateOnly \
  --outputPath $WORK/${KEY}Raw.npz \
&& PYTHONPATH=$PWD python extractRelayLoopSweepRecord11x11.py --rawPath $WORK/${KEY}Raw.npz --key $KEY
