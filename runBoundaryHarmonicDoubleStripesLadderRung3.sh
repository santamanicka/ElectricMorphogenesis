#!/bin/bash
# Rung 3 of the double stripes' registered ladder (Amendment 4, data/boundaryHarmonicTrainingPredictionsAmendment4_1888Hold301DoubleStripesInteriorMinus60Minus5.json):
# contiguous orders 0 to N for N = 8, 10, 12, 14, 16, 18, 20 at ceiling 2.0, 16 restarts each, the sizes one after the other in increasing N
# (--dependency=afterany, so each starts from the codes of the sizes below it, which the training script reads from the same folder).
#   bash runBoundaryHarmonicDoubleStripesLadderRung3.sh 16     rung 3a: population 16, 150 generations, 8 GB / 3 h
#   bash runBoundaryHarmonicDoubleStripesLadderRung3.sh 64     rung 3b: population 64, 100 generations, 16 GB / 4 h
# Rung 3a is submitted only if rungs 1 and 2 have finished and not one of their restarts formed the pattern (overlap with the flank cells >= 0.9);
# rung 3b only if, in addition, rung 3a has finished and not one of its restarts formed it. The checks are made here, not by hand.
set -e
POPULATION=$1
if [ "$POPULATION" != "16" ] && [ "$POPULATION" != "64" ]; then
    echo "usage: bash runBoundaryHarmonicDoubleStripesLadderRung3.sh 16|64" >&2
    exit 1
fi
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
TARGET="--target doubleStripesInterior --targetName DoubleStripesInteriorMinus60Minus5"
FOLDER=data/boundaryHarmonicTraining1888Hold301DoubleStripesInteriorMinus60Minus5

POPULATION=$POPULATION python3 - <<'EOF'
import glob, os, sys
import numpy as np
import boundaryCodeUtilities as boundary
base = 'data/boundaryHarmonicTraining1888Hold301DoubleStripesInteriorMinus60Minus5'


def bestOverlap(pattern, expected, label):
    files = sorted(glob.glob(pattern))
    if len(files) < expected:
        sys.exit(f'{label} has {len(files)} of {expected} restarts: not finished, nothing submitted')
    return max(boundary.structuralIntersectionOverUnion(np.load(f)['bestVmem'].astype(float), boundary.flankCellIndices) for f in files)


checks = {}
for folder in (base + 'Population64', base + 'Ceiling2Population64'):
    for order in (3, 4, 5, 6):
        checks[f'rung 1 {folder} order {order}'] = bestOverlap(f'{folder}/order{order}_restart*.npz', 10, f'rung 1 {folder} order {order}')
for orders in ('0-2-4', '0-2-4-6'):
    checks[f'rung 2 {orders}'] = bestOverlap(f'{base}Ceiling2EvenOrdersPopulation64/orders{orders}_restart*.npz', 32, f'rung 2 {orders}')
if os.environ['POPULATION'] == '64':
    for order in (8, 10, 12, 14, 16, 18, 20):
        checks[f'rung 3a order {order}'] = bestOverlap(f'{base}Ceiling2HigherOrdersPopulation16/order{order}_restart*.npz', 16, f'rung 3a order {order}')
best = max(checks.values())
if best >= 0.9:
    sys.exit(f'{max(checks, key=checks.get)} formed the pattern (best overlap {best:.3f}): rung 3 (population {os.environ["POPULATION"]}) is not run')
print(f'every earlier rung finished, best overlap {best:.3f} < 0.9; rung 3 (population {os.environ["POPULATION"]}) goes ahead')
EOF
OUTPUT=${FOLDER}Ceiling2HigherOrdersPopulation${POPULATION}
if ls ${OUTPUT}/*.npz > /dev/null 2>&1; then
    echo "${OUTPUT} already holds results: nothing submitted" >&2
    exit 1
fi

# submitStage <job name> <array> <time> <memory> <job to wait for, or empty> <maxOrder> <extraArguments> <log name>
submitStage() {
    local dependency=""
    if [ -n "$5" ]; then dependency="--dependency=afterany:$5"; fi
    extraArguments="$7" sbatch --parsable --array "$2" --time "$3" -p batch -c 2 --mem "$4" $dependency -J "$1" \
        --export=ALL,maxOrder=$6 -o "slurm${8}_%a_%A.out" runLearnBoundaryHarmonics.sh
}

LIBRARY=${FOLDER}Ceiling2,${FOLDER}Ceiling2Population64,${FOLDER}Ceiling2EvenOrdersPopulation64
if [ "$POPULATION" = "16" ]; then
    SETTINGS="--populationSize 16 --numGenerations 150"; TIME=3:00:00; MEMORY=8G
else
    SETTINGS="--populationSize 64 --numGenerations 100"; TIME=4:00:00; MEMORY=16G
    LIBRARY=${LIBRARY},${FOLDER}Ceiling2HigherOrdersPopulation16
fi
HIGHER="$TARGET --ceiling 2 $SETTINGS --outputDir $OUTPUT --libraryDirs $LIBRARY"
LOG=BoundaryHarmonicTrainingDoubleStripesCeiling2HigherOrdersPopulation${POPULATION}
J=""
for order in 8 10 12 14 16 18 20; do
    J=$(submitStage doubleStripesRung3p${POPULATION}_order${order} 0-15 $TIME $MEMORY "$J" $order "$HIGHER" ${LOG}_order${order})
    echo "3 population ${POPULATION} order 0-${order}: $J"
done
