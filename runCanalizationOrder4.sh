#!/bin/bash
# The order-4 face code's canalization and memory runs (registration data/boundaryHarmonicOrder4ReplicationPredictions1888Hold301FaceMinus60Minus5.json,
# K1, K2, M1-M4). Every output carries the Order4 tag; none of the order-3 files is touched; the scripts refuse to overwrite.
#   task 0  K1: the 400 order-4 sweep codes replayed (float32, as at order 3) and read at 2143 -> data/canalizationTalkPatternEnsembleOrder4.json
#   task 1  K2: the dial sweeps (each order alone over its allowed range, 201 stops; and +-0.06 in 0.001 steps), ceiling 2 as at order 3
#           -> data/boundaryHarmonicRingCodePatterns1888Hold301FaceMinus60Minus5Order4.json
#   task 2  M1-M4: the face class, 20,000 iterations in float64, stride 5: the trained code, 12 noisy copies (1%, seed 20261004), 24 left-right-symmetric
#           space-filling controls from the order-4 sweep, the no-ring run; the stripe-class runs are the order-3 file's, appended unchanged
#           -> data/canalizationTalkFamilyTrajectories..Order4.npz, ...FamilyVisits...Order4.json, ...FamilyFlags...Order4.npz
#   task 3  M3: the pattern-cluster test on the combined trajectory file (after task 2)
#   task 4  M2/M4 support: the family-distance test on the same file (after task 2)
# sbatch --array 0-2 --time 3:00:00 -p batch -c 4 --mem 16G -o slurmCanalizationOrder4_%a.out runCanalizationOrder4.sh
# sbatch --array 3-4 --time 2:00:00 -p batch -c 4 --mem 16G --dependency=afterok:<job of task 2> -o slurmCanalizationOrder4_%a.out runCanalizationOrder4.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 BOUNDARY_FACE_ORDER=4 PYTHONPATH=$PWD
ORDER4=data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5Population64/order4_restart05.npz
case $SLURM_ARRAY_TASK_ID in
  0) python analyzeCanalizationTalkPatternEnsemble11x11.py ;;
  1) python simulateBoundaryHarmonicRingCodePatterns11x11.py --target face --trainedRunPath $ORDER4 --ceiling 2 --families wideSlider,zoomSlider \
       --outputPath data/boundaryHarmonicRingCodePatterns1888Hold301FaceMinus60Minus5Order4.json ;;
  2) python analyzeCanalizationTalkFamilyVisits11x11.py ;;
  3) python analyzeCanalizationTalkPatternClusters11x11.py ;;
  4) python analyzeCanalizationTalkFamilyDistance11x11.py ;;
esac
