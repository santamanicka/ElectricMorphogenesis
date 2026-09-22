#!/bin/bash
# Mode-ownership and outcome analyses, one per array index (PolyPatterning_Sim.md, Section 12).
# sbatch --export=ALL --time 8:00:00 -p batch --array 0-3 -e Error_%A_%a.err --mem 8G runBoundaryHarmonicModeAnalyses.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
PYTHON=/cluster/tufts/levinlab/smanic02/condaenv/santa/bin/python
case $SLURM_ARRAY_TASK_ID in
  0) $PYTHON analyzeBoundaryHarmonicOutcomes11x11.py --sampling random --numCodes 1024 ;;
  1) $PYTHON analyzeBoundaryHarmonicOutcomes11x11.py --sampling slice --gridSize 48 ;;
  2) $PYTHON analyzeBoundaryHarmonicModeOwnership11x11.py --condition baseline --numCodes 1024 --interiorOnly \
        --outputSuffix Interior --storeAmplitudesAt 300,2173 ;;
  3) $PYTHON analyzeBoundaryHarmonicModeOwnership11x11.py --condition baseline --numCodes 4096 --seed 23 \
        --outputSuffix Large ;;
esac
