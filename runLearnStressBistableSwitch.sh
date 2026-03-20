#!/bin/bash

# Learn stress bistable switch parameters
# Optimizes RD bistable stress system to classify healthy vs perturbed embryos
# based on GRN damping level:
#   damping=1.0 (healthy)  -> stress ~ 0
#   damping=0.95           -> stress ~ 0.5
#   damping=0.9 (perturbed)-> stress ~ 1.0

python learnStressBistableSwitch.py \
    --numBioSteps 1000 \
    --numStressSteps 500 \
    --numLearnIters 100 \
    --lr 0.01 \
    --dampingLevels "1.0,0.95,0.9" \
    --targetStress "0.0,0.5,1.0" \
    --fileNumber 0 \
    --verbose True
