#!/bin/bash
# Rung 2 of the double stripes' registered ladder (Amendment 3, data/boundaryHarmonicTrainingPredictionsAmendment3_1888Hold301DoubleStripesInteriorMinus60Minus5.json):
# the even-only code sets {0,2,4} and {0,2,4,6} at ceiling 2.0, population 64, 100 generations, 32 restarts each, one set after the other
# (--dependency=afterany), in one shared folder; --libraryDirs is arm B's folder (the pilot of the same ceiling and the same target). Submitted
# only if rung 1 (contiguous orders 3-6 at both ceilings, 10 restarts each) has finished and not one of its restarts formed the pattern
# (overlap with the flank cells >= 0.9).
#   bash runBoundaryHarmonicDoubleStripesLadderRung2.sh
set -e
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
TARGET="--target doubleStripesInterior --targetName DoubleStripesInteriorMinus60Minus5"
FOLDER=data/boundaryHarmonicTraining1888Hold301DoubleStripesInteriorMinus60Minus5

python3 - <<'EOF'
import glob, sys
import numpy as np
import boundaryCodeUtilities as boundary
base = 'data/boundaryHarmonicTraining1888Hold301DoubleStripesInteriorMinus60Minus5'
best = 0.0
for folder in (base + 'Population64', base + 'Ceiling2Population64'):
    for order in (3, 4, 5, 6):
        files = sorted(glob.glob(f'{folder}/order{order}_restart*.npz'))
        if len(files) < 10:
            sys.exit(f'{folder} order {order} has {len(files)} of 10 restarts: rung 1 not finished, nothing submitted')
        overlaps = [boundary.structuralIntersectionOverUnion(np.load(f)['bestVmem'].astype(float), boundary.flankCellIndices) for f in files]
        best = max(best, max(overlaps))
if best >= 0.9:
    sys.exit(f'rung 1 formed the pattern (best overlap {best:.3f}): rung 2 is not run')
print(f'rung 1: 80 of 80 restarts, best overlap {best:.3f} < 0.9; rung 2 goes ahead')
EOF
if ls ${FOLDER}Ceiling2EvenOrdersPopulation64/*.npz > /dev/null 2>&1; then
    echo "${FOLDER}Ceiling2EvenOrdersPopulation64 already holds results: nothing submitted" >&2
    exit 1
fi

# submitStage <job name> <array> <time> <memory> <job to wait for, or empty> <maxOrder> <extraArguments> <log name>
submitStage() {
    local dependency=""
    if [ -n "$5" ]; then dependency="--dependency=afterany:$5"; fi
    extraArguments="$7" sbatch --parsable --array "$2" --time "$3" -p batch -c 2 --mem "$4" $dependency -J "$1" \
        --export=ALL,maxOrder=$6 -o "slurm${8}_%a_%A.out" runLearnBoundaryHarmonics.sh
}

EVEN="$TARGET --ceiling 2 --populationSize 64 --numGenerations 100 --outputDir ${FOLDER}Ceiling2EvenOrdersPopulation64 --libraryDirs ${FOLDER}Ceiling2"
LOG=BoundaryHarmonicTrainingDoubleStripesCeiling2EvenOrdersPopulation64
J=$(submitStage doubleStripesRung2_orders0-2-4 0-31 4:00:00 16G "" 4 "$EVEN --orders 0,2,4" ${LOG}_orders0-2-4)
echo "2 even {0,2,4}: $J"
J=$(submitStage doubleStripesRung2_orders0-2-4-6 0-31 4:00:00 16G $J 6 "$EVEN --orders 0,2,4,6" ${LOG}_orders0-2-4-6)
echo "2 even {0,2,4,6}: $J"
