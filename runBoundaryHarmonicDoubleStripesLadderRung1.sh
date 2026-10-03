#!/bin/bash
# Rung 1 of the double stripes' registered ladder (Amendment 2, data/boundaryHarmonicTrainingPredictionsAmendment2_1888Hold301DoubleStripesInteriorMinus60Minus5.json):
# the contiguous orders 3, 4, 5 and 6, population 64, 100 generations, 10 restarts each, one order after another, at ceiling 1.3 (1a) and at
# ceiling 2.0 (1b), started together; --libraryDirs is the pilot folder of the same ceiling. Submitted only if arm C's last set (the even-only
# orders 0..20) has finished and not one of its restarts formed the pattern (overlap with the flank cells >= 0.9).
#   bash runBoundaryHarmonicDoubleStripesLadderRung1.sh
set -e
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
TARGET="--target doubleStripesInterior --targetName DoubleStripesInteriorMinus60Minus5"
FOLDER=data/boundaryHarmonicTraining1888Hold301DoubleStripesInteriorMinus60Minus5

python3 - <<'EOF'
import glob, sys
import numpy as np
import boundaryCodeUtilities as boundary
folder = 'data/boundaryHarmonicTraining1888Hold301DoubleStripesInteriorMinus60Minus5EvenOrdersPopulation64'
files = sorted(glob.glob(f'{folder}/orders0-2-4-6-8-10-12-14-16-18-20_restart*.npz'))
if len(files) < 16:
    sys.exit(f'arm C {{0..20}} has {len(files)} of 16 restarts: not finished, nothing submitted')
overlaps = [boundary.structuralIntersectionOverUnion(np.load(f)['bestVmem'].astype(float), boundary.flankCellIndices) for f in files]
if max(overlaps) >= 0.9:
    sys.exit(f'arm C {{0..20}} formed the pattern (best overlap {max(overlaps):.3f}): rung 1 is not run')
print(f'arm C {{0..20}}: 16 of 16 restarts, best overlap {max(overlaps):.3f} < 0.9; rung 1 goes ahead')
EOF

# submitStage <job name> <array> <time> <memory> <job to wait for, or empty> <maxOrder> <extraArguments> <log name>
submitStage() {
    local dependency=""
    if [ -n "$5" ]; then dependency="--dependency=afterany:$5"; fi
    extraArguments="$7" sbatch --parsable --array "$2" --time "$3" -p batch -c 2 --mem "$4" $dependency -J "$1" \
        --export=ALL,maxOrder=$6 -o "slurm${8}_%a_%A.out" runLearnBoundaryHarmonics.sh
}

# ------------------------------------------------------------------------------------------ 1a: ceiling 1.3
ONEA="$TARGET --populationSize 64 --numGenerations 100 --outputDir ${FOLDER}Population64 --libraryDirs $FOLDER"
J=""
for order in 3 4 5 6; do
    J=$(submitStage doubleStripesRung1a_order$order 0-9 4:00:00 16G "$J" $order "$ONEA" BoundaryHarmonicTrainingDoubleStripesPopulation64_order$order)
    echo "1a order $order: $J"
done
# ------------------------------------------------------------------------------------------ 1b: ceiling 2.0
ONEB="$TARGET --ceiling 2 --populationSize 64 --numGenerations 100 --outputDir ${FOLDER}Ceiling2Population64 --libraryDirs ${FOLDER}Ceiling2"
J=""
for order in 3 4 5 6; do
    J=$(submitStage doubleStripesRung1b_order$order 0-9 4:00:00 16G "$J" $order "$ONEB" BoundaryHarmonicTrainingDoubleStripesCeiling2Population64_order$order)
    echo "1b order $order: $J"
done
