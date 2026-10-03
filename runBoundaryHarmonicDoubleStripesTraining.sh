#!/bin/bash
# The registered training ladder for the double stripes (data/boundaryHarmonicTrainingPredictions1888Hold301DoubleStripesInteriorMinus60Minus5.json,
# design.arms), submitted as three Slurm dependency chains (--dependency=afterany, so each order starts from the codes of the one below):
#   A  pilot, ceiling 1.3: population 16, 150 generations, 20 restarts each of orders 0-0, 0-1, 0-2, 0-3 and of the even-only set {0, 2}
#   B  the same arms at ceiling 2.0, started at the same time as A
#   C  even-only chain at ceiling 1.3, population 64, 100 generations: {0,2,4,6} (32 restarts), {0..10} (16), {0..20} (16), one shared folder,
#      --libraryDirs A's folder; starts when A has finished
# runLearnBoundaryHarmonics.sh carries no target arguments; they travel in the exported extraArguments, which is set here for each submission
# and kept by Slurm (sacct -j <jobid>_0 --env-vars -X | grep extraArguments). Nothing is run on the login node; nothing is overwritten
# (the training script refuses an existing output path). Resources are those of the single stripe's chains (8 GB / 3 h for population 16, 16 GB / 4 h for 64).
#
#   bash runBoundaryHarmonicDoubleStripesTraining.sh
set -e
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
TARGET="--target doubleStripesInterior --targetName DoubleStripesInteriorMinus60Minus5"
FOLDER=data/boundaryHarmonicTraining1888Hold301DoubleStripesInteriorMinus60Minus5

# submitStage <job name> <array> <time> <memory> <job to wait for, or empty> <maxOrder> <extraArguments> <log name>
submitStage() {
    local dependency=""
    if [ -n "$5" ]; then dependency="--dependency=afterany:$5"; fi
    extraArguments="$7" sbatch --parsable --array "$2" --time "$3" -p batch -c 2 --mem "$4" $dependency -J "$1" \
        --export=ALL,maxOrder=$6 -o "slurm${8}_%a_%A.out" runLearnBoundaryHarmonics.sh
}

# ------------------------------------------------------------------------------------------ A: pilot, ceiling 1.3
J=$(submitStage doubleStripesPilot_order0 0-19 3:00:00 8G "" 0 "$TARGET" BoundaryHarmonicTrainingDoubleStripes_order0)
echo "A order 0: $J"
for order in 1 2 3; do
    J=$(submitStage doubleStripesPilot_order$order 0-19 3:00:00 8G $J $order "$TARGET" BoundaryHarmonicTrainingDoubleStripes_order$order)
    echo "A order $order: $J"
done
J=$(submitStage doubleStripesPilot_orders0-2 0-19 3:00:00 8G $J 2 "$TARGET --orders 0,2" BoundaryHarmonicTrainingDoubleStripes_orders0-2)
echo "A even {0,2}: $J"
LAST_A=$J

# ------------------------------------------------------------------------------------------ B: pilot, ceiling 2.0
CEILING2="$TARGET --ceiling 2 --outputDir ${FOLDER}Ceiling2"
J=$(submitStage doubleStripesCeil2Pilot_order0 0-19 3:00:00 8G "" 0 "$CEILING2" BoundaryHarmonicTrainingDoubleStripesCeiling2_order0)
echo "B order 0: $J"
for order in 1 2 3; do
    J=$(submitStage doubleStripesCeil2Pilot_order$order 0-19 3:00:00 8G $J $order "$CEILING2" BoundaryHarmonicTrainingDoubleStripesCeiling2_order$order)
    echo "B order $order: $J"
done
J=$(submitStage doubleStripesCeil2Pilot_orders0-2 0-19 3:00:00 8G $J 2 "$CEILING2 --orders 0,2" BoundaryHarmonicTrainingDoubleStripesCeiling2_orders0-2)
echo "B even {0,2}: $J"

# ------------------------------------------------------------------------------------------ C: even-only chain, ceiling 1.3
EVEN="$TARGET --populationSize 64 --numGenerations 100 --outputDir ${FOLDER}EvenOrdersPopulation64 --libraryDirs $FOLDER"
LOG=BoundaryHarmonicTrainingDoubleStripesEvenOrdersPopulation64
J=$(submitStage doubleStripesEven1p3_orders0-2-4-6 0-31 4:00:00 16G $LAST_A 6 "$EVEN --orders 0,2,4,6" ${LOG}_orders0-2-4-6)
echo "C even {0,2,4,6}: $J"
J=$(submitStage doubleStripesEven1p3_orders0-to-10 0-15 4:00:00 16G $J 10 "$EVEN --orders 0,2,4,6,8,10" ${LOG}_orders0-2-4-6-8-10)
echo "C even {0..10}: $J"
J=$(submitStage doubleStripesEven1p3_orders0-to-20 0-15 4:00:00 16G $J 20 "$EVEN --orders 0,2,4,6,8,10,12,14,16,18,20" ${LOG}_orders0-to-20)
echo "C even {0..20}: $J"
