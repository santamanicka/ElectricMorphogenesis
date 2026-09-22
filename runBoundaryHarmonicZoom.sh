#!/bin/bash
# Correlation length of the code-to-pattern map: outcome maps and mode ownership at several slice spacings.
# sbatch --export=ALL --time 8:00:00 -p batch --array 0-3 -e Error_%A_%a.err --mem 8G runBoundaryHarmonicZoom.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
PYTHON=/cluster/tufts/levinlab/smanic02/condaenv/santa/bin/python
case $SLURM_ARRAY_TASK_ID in
  0) $PYTHON analyzeBoundaryHarmonicOutcomes11x11.py --sampling slice --gridSize 48 --sliceHalfWidth 0.12 ;;
  1) $PYTHON analyzeBoundaryHarmonicOutcomes11x11.py --sampling slice --gridSize 48 --sliceHalfWidth 0.024 ;;
  2) $PYTHON analyzeBoundaryHarmonicModeOwnership11x11.py --condition baseline --sampling slice --gridSize 32 \
        --sliceHalfWidth 0.6 --outputSuffix SliceWide --storeAmplitudesAt 300,1853,2173 ;;
  3) $PYTHON analyzeBoundaryHarmonicModeOwnership11x11.py --condition baseline --sampling slice --gridSize 32 \
        --sliceHalfWidth 0.06 --outputSuffix SliceZoom --storeAmplitudesAt 300,1853,2173 ;;
esac
