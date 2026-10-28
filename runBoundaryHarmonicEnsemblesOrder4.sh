#!/bin/bash
# The order-4 counterparts of the order-3 ensemble analyses: the settings of each task are the settings of the order-3 output it repeats
# (recovered from runBoundaryHarmonicModeAnalyses.sh, ...FieldStrength.sh, ...Zoom.sh, ...Roughness.sh and from the stored outputs' own fields).
# BOUNDARY_FACE_ORDER=4 tags every output Order4 and takes the order-4 code (population-64 restart 05, best moment 2143); probe moments that
# were the order-3 best moment 2173 are 2143 here (the probe list in analyzeBoundaryHarmonicModeOwnership11x11.py already includes the trained moment).
# 1 CPU, 8 GB per task; the order-3 versions took 12-80 min each (sacct).
# sbatch --array 0-18 --time 4:00:00 -p batch -c 1 --mem 8G -o slurmBoundaryHarmonicEnsemblesOrder4_%a.out runBoundaryHarmonicEnsemblesOrder4.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
source ~/.bashrc
myconda
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD BOUNDARY_FACE_ORDER=4
OUT=analyzeBoundaryHarmonicOutcomes11x11.py
MO=analyzeBoundaryHarmonicModeOwnership11x11.py
MOMENTS=128,300,600,1000,1300,1853,2143,2999
case $SLURM_ARRAY_TASK_ID in
  0) python $OUT --sampling random --numCodes 1024 ;;
  1) python $OUT --sampling slice --gridSize 48 ;;
  2) python $OUT --sampling slice --gridSize 48 --sliceHalfWidth 0.12 ;;
  3) python $OUT --sampling slice --gridSize 48 --sliceHalfWidth 0.024 ;;
  4) python $MO --condition baseline --numCodes 1024 ;;
  5) python $MO --condition baseline --numCodes 1024 --interiorOnly --outputSuffix Interior --storeAmplitudesAt 300,2143 ;;
  6) python $MO --condition baseline --numCodes 4096 --seed 23 --outputSuffix Large ;;
  7) python $MO --condition baseline --numCodes 1024 --modeBasis principal --outputSuffix Principal ;;
  8) python $MO --condition baseline --numCodes 1024 --outputSuffix CosineStored --storeAmplitudesAt 300,2143 ;;
  9) python $MO --condition fieldOff --numCodes 1024 ;;
  10) python $MO --condition heldThroughout --numCodes 256 ;;
  11) python $MO --condition baseline --sampling slice --gridSize 32 --sliceHalfWidth 0.6 --outputSuffix SliceWide --storeAmplitudesAt 300,1853,2143 ;;
  12) python $MO --condition baseline --sampling slice --gridSize 32 --sliceHalfWidth 0.06 --outputSuffix SliceZoom --storeAmplitudesAt 300,1853,2143 ;;
  13) python $MO --condition baseline --sampling slice --gridSize 32 --sliceHalfWidth 0.06 --outputSuffix SliceCourse --storeAmplitudesAt $MOMENTS ;;
  14) python $MO --condition fieldOff --sampling slice --gridSize 32 --sliceHalfWidth 0.06 --outputSuffix SliceCourse --storeAmplitudesAt $MOMENTS ;;
  15|16|17|18)
     STRENGTHS=(0.25 0.5 0.75 0.0); STRENGTH=${STRENGTHS[$((SLURM_ARRAY_TASK_ID - 15))]}; TAG=Strength${STRENGTH/./p}
     python analyzeBoundaryHarmonicFieldRole11x11.py --fieldStrength $STRENGTH --outputSuffix Order4$TAG
     python $MO --condition baseline --sampling slice --gridSize 32 --sliceHalfWidth 0.06 --fieldStrength $STRENGTH \
        --outputSuffix Slice$TAG --storeAmplitudesAt 300,1000,2143 ;;
esac
