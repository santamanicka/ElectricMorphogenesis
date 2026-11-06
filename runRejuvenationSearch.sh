#!/bin/bash
# The rejuvenation search (data/canalizationTalkRejuvenationPredictions1888Hold301.json): CMA-ES for a ring code on a tissue aged to best moment + 300, one restart per array task.
# Second pass (data/canalizationTalkRejuvenationExtensionPredictions1888Hold301.json): optional env vars offset (iterations after the best moment, default 300), hold (default 301),
# population (default 16), stripeMaxOrder (default 2), suffix (appended to the output folder name, e.g. Population64, LongHold1000, Order4Family).
# First make the three states (makeAgedTissueState11x11.py), then, for each target, e.g.
#   sbatch --array 0-15 --export=ALL,label=Stripe  -p batch -c 2 --mem 8G --time 2:00:00 -o slurmRejuvenationStripe_%a_%A.out runRejuvenationSearch.sh
#   sbatch --array 0-15 --export=ALL,label=FaceOrder3 ... ;  sbatch --array 0-15 --export=ALL,label=FaceOrder4 ...
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
case $label in
  Stripe)     extra="--target stripesInterior --targetName StripesInteriorMinus60Minus5 --maxOrder ${stripeMaxOrder:-2} --ceiling 2.0" ;;
  FaceOrder3) extra="--maxOrder 3" ;;
  FaceOrder4) extra="--maxOrder 4" ;;
  *) echo "label must be Stripe, FaceOrder3 or FaceOrder4"; exit 1 ;;
esac
offset=${offset:-300}
PYTHONPATH=$PWD python learnBoundaryHarmonics11x11.py $extra --restart $SLURM_ARRAY_TASK_ID \
  --holdIterations ${hold:-301} --populationSize ${population:-16} \
  --startState data/agedTissueState1888Hold301${label}After${offset}.npz \
  --outputDir data/boundaryHarmonicRejuvenationTraining1888Hold301${label}After${offset}${suffix}
