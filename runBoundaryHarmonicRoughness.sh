#!/bin/bash
# Does the field's feedback roughen the code-to-pattern map, and when? Slices with amplitudes at many moments.
# sbatch --export=ALL --time 8:00:00 -p batch --array 0-1 -e Error_%A_%a.err --mem 8G runBoundaryHarmonicRoughness.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
PYTHON=/cluster/tufts/levinlab/smanic02/condaenv/santa/bin/python
MOMENTS=128,300,600,1000,1300,1853,2173,2999
case $SLURM_ARRAY_TASK_ID in
  0) $PYTHON analyzeBoundaryHarmonicModeOwnership11x11.py --condition baseline --sampling slice --gridSize 32 \
        --sliceHalfWidth 0.06 --outputSuffix SliceCourse --storeAmplitudesAt $MOMENTS ;;
  1) $PYTHON analyzeBoundaryHarmonicModeOwnership11x11.py --condition fieldOff --sampling slice --gridSize 32 \
        --sliceHalfWidth 0.06 --outputSuffix SliceCourse --storeAmplitudesAt $MOMENTS ;;
esac
