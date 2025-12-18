#!/bin/bash

# Learn CaMKII bistability parameters
# Optimizes parameters to maximize pattern retention when Vmem decays

python learn_camkii_bistability.py \
    --gridSize 11 \
    --numBioSteps 1000 \
    --numTotalSteps 2000 \
    --numLearnIters 1000 \
    --lr 0.01 \
    --stigmergicParamsPath data/StigmergicModelParameters.dat \
    --fileNumber 0 \
    --verbose True