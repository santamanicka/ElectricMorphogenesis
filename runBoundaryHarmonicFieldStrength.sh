#!/bin/bash
# Does turning the field's feedback down trade the face against the map's smoothness?
# sbatch --export=ALL --time 8:00:00 -p batch --array 0-3 -e Error_%A_%a.err --mem 8G runBoundaryHarmonicFieldStrength.sh
cd /cluster/tufts/levinlab/smanic02/Code/Git/electricmorphogenesis
PYTHON=/cluster/tufts/levinlab/smanic02/condaenv/santa/bin/python
STRENGTHS=(0.25 0.5 0.75 0.0)
STRENGTH=${STRENGTHS[$SLURM_ARRAY_TASK_ID]}
TAG=Strength${STRENGTH/./p}
$PYTHON analyzeBoundaryHarmonicFieldRole11x11.py --fieldStrength $STRENGTH --outputSuffix $TAG
$PYTHON analyzeBoundaryHarmonicModeOwnership11x11.py --condition baseline --sampling slice --gridSize 32 \
    --sliceHalfWidth 0.06 --fieldStrength $STRENGTH --outputSuffix Slice$TAG --storeAmplitudesAt 300,1000,2173
