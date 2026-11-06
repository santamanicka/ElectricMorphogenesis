#!/bin/bash
# The rejuvenation search (data/canalizationTalkRejuvenationPredictions1888Hold301.json): CMA-ES for a ring code on a tissue aged to best moment + 300, one restart per array task.
# First make the three states (makeAgedTissueState11x11.py), then, for each target, e.g.
#   sbatch --array 0-15 --export=ALL,label=Stripe  -p batch -c 2 --mem 8G --time 2:00:00 -o slurmRejuvenationStripe_%a_%A.out runRejuvenationSearch.sh
#   sbatch --array 0-15 --export=ALL,label=FaceOrder3 ... ;  sbatch --array 0-15 --export=ALL,label=FaceOrder4 ...
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
case $label in
  Stripe)     extra="--target stripesInterior --targetName StripesInteriorMinus60Minus5 --maxOrder 2 --ceiling 2.0" ;;
  FaceOrder3) extra="--maxOrder 3" ;;
  FaceOrder4) extra="--maxOrder 4" ;;
  *) echo "label must be Stripe, FaceOrder3 or FaceOrder4"; exit 1 ;;
esac
PYTHONPATH=$PWD python learnBoundaryHarmonics11x11.py $extra --restart $SLURM_ARRAY_TASK_ID \
  --startState data/agedTissueState1888Hold301${label}After300.npz \
  --outputDir data/boundaryHarmonicRejuvenationTraining1888Hold301${label}After300
