#!/bin/bash
# Stage A of the order-4 replication's third batch (registration data/boundaryHarmonicOrder4ReplicationPredictions1888Hold301FaceMinus60Minus5.json).
# Each task repeats an order-3 analysis at the order-3 settings (the scripts' own defaults, written out here); BOUNDARY_FACE_ORDER=4 takes the
# order-4 code (population-64 restart 05, best moment 2143) and tags every output Order4. The analyses refuse to overwrite an existing output.
#   0 Modes (256 codes, seed 17)             1 Layers                          2 ReadoutPanels (1024 codes, 5 folds, seed 17)
#   3 WhenMeasured (1024 codes, seed 17, window 2000,2400)   4 Knockout (all 15 subsets of orders 1-4, 20 size-matched controls, seed 11, window 1700,2400,
#   no retraining: --retrainedDir points at a folder that does not exist)   5 CodeJitter (face, sigmas 0,0.01,0.03,0.10, 100 draws, seed 20250926, hold 301,
#   read at the code's own best moment)      6 ClampContribution (trace stride 5)
#   7 record trained (best code of every order, Vmem/G_pol/field every iteration)    8 record ensemble (1024 random codes, seed 17)
#   9 record steering (trained code with one coefficient moved at a time)
# Records are intermediates and go to $INTERMEDIATES.
# sbatch --array 0-9 --time 4:00:00 -p batch -c 4 --mem 16G -o slurmBoundaryHarmonicStageAOrder4_%a.out runBoundaryHarmonicStageAOrder4.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=$PWD BOUNDARY_FACE_ORDER=4
INTERMEDIATES=/cluster/tufts/levinlab/smanic02/scratchFaceOrder4
mkdir -p $INTERMEDIATES
refuse() { [ -e "$1" ] && { echo "$1 exists; not overwriting"; exit 1; }; }
case $SLURM_ARRAY_TASK_ID in
  0) python analyzeBoundaryHarmonicModes11x11.py --numCodes 256 --seed 17 ;;
  1) python analyzeBoundaryHarmonicLayers11x11.py ;;
  2) python analyzeBoundaryHarmonicReadoutPanels11x11.py --numCodes 1024 --numFolds 5 --seed 17 ;;
  3) python analyzeBoundaryHarmonicWhenMeasured11x11.py --window 2000,2400 --numCodes 1024 --seed 17 ;;
  4) python analyzeBoundaryHarmonicKnockout11x11.py --numControls 20 --window 1700,2400 --seed 11 --retrainedDir data/noRetrainingAtOrder4 ;;
  5) python analyzeBoundaryHarmonicCodeJitter11x11.py --target face --sigmas 0.0,0.01,0.03,0.10 --draws 100 --seed 20250926 --holdIterations 301 ;;
  6) python analyzeBoundaryHarmonicClampContribution11x11.py --traceStride 5 ;;
  7) OUT=$INTERMEDIATES/recordTrainedOrder4.npz; refuse $OUT; python -u recordBoundaryHarmonicRuns11x11.py --mode trained --outputPath $OUT ;;
  8) OUT=$INTERMEDIATES/recordEnsembleOrder4.npz; refuse $OUT; python -u recordBoundaryHarmonicRuns11x11.py --mode ensemble --numCodes 1024 --seed 17 --outputPath $OUT ;;
  9) OUT=$INTERMEDIATES/recordSteeringOrder4.npz; refuse $OUT; python -u recordBoundaryHarmonicRuns11x11.py --mode steering --outputPath $OUT ;;
esac
