#!/bin/bash
# The order-4 counterpart of runBoundaryHarmonicRelaySweep.sh: one relay decomposition per sweep code (400 codes: 280 space-filling, 120 near the
# trained order-4 code; data/boundaryHarmonicRingCodeSweep1888Hold301FaceMinus60Minus5Order4.json, built by buildBoundaryHarmonicRingCodeSweep11x11.py with
# the order-3 seeds under BOUNDARY_FACE_ORDER=4). Each relay is the ring-only baseline (extraUpdateOnly), order-4 landmarks (scored 2143, primary 1687, transfer
# windows cut at states 302,585,1001,1501,1688), exactly as the order-4 relay of the trained code. The raw file is deleted once the small record is written
# to data/relayLoopSweepOrder4/<key>.json; a code whose record exists is skipped.
# The page set (Amendment 6) and the confirmation codes (Amendment 5) use the same script: VARIANTS=<variants json> RECORD_DIR=<record directory> sbatch --export=ALL --array ... (see
# runBoundaryHarmonicRelayPageSetOrder4.sh and runBoundaryHarmonicRelayConfirmationOrder4.sh).
# sbatch --array 0-399%64 --time 1:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicRelaySweepOrder4_%a.out runBoundaryHarmonicRelaySweepOrder4.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 BOUNDARY_FACE_ORDER=4
SUFFIX=1888Hold301FaceMinus60Minus5
VARIANTS=${VARIANTS:-data/boundaryHarmonicRingCodeSweep${SUFFIX}Order4.json}
RECORD_DIR=${RECORD_DIR:-data/relayLoopSweepOrder4}
KEY=$(python3 -c "import json; print(json.load(open('$VARIANTS'))['variants'][$SLURM_ARRAY_TASK_ID]['key'])")
if [ -f ${RECORD_DIR}/${KEY}.json ]; then echo "skip $KEY (record exists)"; exit 0; fi
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
PYTHONPATH=$PWD python computeBoundaryHarmonicRelay11x11.py --order 4 --primaryIteration 1687 --windowStates 302,585,1001,1501,1688 \
  --ringCodeVariantsPath $VARIANTS --ringCodeKey $KEY --baseline extraUpdateOnly --outputPath $WORK/${KEY}Raw.npz \
&& PYTHONPATH=$PWD python extractRelayLoopSweepRecord11x11.py --rawPath $WORK/${KEY}Raw.npz --key $KEY --variantsPath $VARIANTS --outputDirectory $RECORD_DIR
