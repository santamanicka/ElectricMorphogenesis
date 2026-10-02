#!/bin/bash
# The relay of each ring code of buildBoundaryHarmonicStripeEndsVariants11x11.py (the stripe's ends: both levels in the window, top only, bottom only, neither),
# one array task each, then the compact record of each net (extractRelayLoopSweepRecord11x11.py, raw files discarded after) under data/relayLoopStripeEnds/.
#   sbatch --array 0-3 --time 1:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicStripeEndsRelay_%a.out runBoundaryHarmonicStripeEndsRelays.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
SUFFIX=1888Hold301StripesInteriorMinus60Minus5
VARIANTS=data/boundaryHarmonicRingCodeStripeEnds${SUFFIX}.json
SUMMARY=data/boundaryHarmonicTrainingSummary${SUFFIX}Ceiling2Pilot.json
KEY=$(python3 -c "import json; print(json.load(open('$VARIANTS'))['variants'][$SLURM_ARRAY_TASK_ID]['key'])")
if [ -f data/relayLoopStripeEnds/${KEY}.json ]; then echo "skip $KEY (record exists)"; exit 0; fi
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
PYTHONPATH=$PWD python computeBoundaryHarmonicRelay11x11.py \
  --ringCodeVariantsPath $VARIANTS --ringCodeKey $KEY --baseline extraUpdateOnly --target stripesInterior --order 2 --summaryPath $SUMMARY --primaryIteration 504 \
  --outputPath $WORK/${KEY}Raw.npz \
&& PYTHONPATH=$PWD python extractRelayLoopSweepRecord11x11.py --rawPath $WORK/${KEY}Raw.npz --key $KEY --variantsPath $VARIANTS --outputDirectory data/relayLoopStripeEnds
