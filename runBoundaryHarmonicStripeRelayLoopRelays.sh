#!/bin/bash
# The stripe's Relay Loop ring codes (buildBoundaryHarmonicRingCodeStripeLevels11x11.py for the page's slider, grid and curated codes,
# buildBoundaryHarmonicRingCodeStripeSweep11x11.py for the steering lab), one relay decomposition per array task; the task's key is the array index
# into the JSON's `variants`. The raw relay file goes to a temporary directory and is deleted once extractRelayLoopSweepRecord11x11.py has written the
# small record, so hundreds of codes do not add gigabytes. A code whose record already exists is skipped, so the array can be resubmitted after a
# partial failure. Every code is decomposed to the stripe code's own readout (iteration 504), so all are read at the same moment.
#   the page's codes (the trained code first), with the Vmem and conductance blocks:
#     sbatch --array 0-98%32 --time 1:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicStripeRelayLoop_%a.out runBoundaryHarmonicStripeRelayLoopRelays.sh
#   the steering lab's sweep, nets only:
#     VARIANTS=data/boundaryHarmonicRingCodeStripeSweep1888Hold301StripesInteriorMinus60Minus5.json OUTDIR=data/relayLoopStripeSweep TRAJECTORY= \
#       sbatch --array 0-399%64 --time 1:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicStripeRelayLoopSweep_%a.out runBoundaryHarmonicStripeRelayLoopRelays.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
SUFFIX=1888Hold301StripesInteriorMinus60Minus5
VARIANTS=${VARIANTS:-data/boundaryHarmonicRingCodeStripeLevels${SUFFIX}.json}
OUTDIR=${OUTDIR:-data/relayLoopStripes}
TRAJECTORY=${TRAJECTORY-"--withTrajectory"}
SUMMARY=data/boundaryHarmonicTrainingSummary${SUFFIX}Ceiling2Pilot.json
KEY=$(python3 -c "import json; print(json.load(open('$VARIANTS'))['variants'][$SLURM_ARRAY_TASK_ID]['key'])")
if [ -f $OUTDIR/${KEY}.json ]; then echo "skip $KEY (record exists)"; exit 0; fi
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
PYTHONPATH=$PWD python computeBoundaryHarmonicRelay11x11.py \
  --ringCodeVariantsPath $VARIANTS --ringCodeKey $KEY --baseline extraUpdateOnly --target stripesInterior --order 2 --summaryPath $SUMMARY --primaryIteration 504 \
  --outputPath $WORK/${KEY}Raw.npz \
&& PYTHONPATH=$PWD python extractRelayLoopSweepRecord11x11.py --rawPath $WORK/${KEY}Raw.npz --key $KEY --variantsPath $VARIANTS --outputDirectory $OUTDIR $TRAJECTORY
