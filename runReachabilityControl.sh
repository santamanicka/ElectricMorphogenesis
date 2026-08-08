#!/bin/bash
# Run 0: can training find a clamp for a pattern the tissue has already produced?
#
# Every 30x30 training result is ambiguous until this is answered. The boundary-to-pattern map
# hashes at this lattice size -- 400 random clamps give 400 unrelated patterns, no two within
# 5.5 mV (Sim Appendix 14.4) -- and a hashed landscape offers gradient descent nothing to descend.
# So a failure to train some new target could mean the tissue cannot make it, or merely that the
# search cannot find it, and those demand different responses.
#
# This removes the first possibility. The target is drawn from the ensemble itself, so a clamp
# that produces it provably exists: one did. Only the clamp is learned; the tissue is held at the
# parameters that produced the target. Failure here is a statement about search alone.
#
# The protocol matches the ensemble that supplied the target exactly -- 2500 iterations, a 100
# iteration clamp (0.04), a 200 iteration averaged readout (0.08), fieldScreenSize 4 -- because a
# target produced under one protocol is not necessarily reachable under another.
source ~/.bashrc
myconda
targetIndex=${targetIndex:-0}
python learnCellularFieldNetwork.py \
  --latticeDims "(30,30)" \
  --targetPattern ensemble \
  --targetEnsembleFile data/fieldRangeSweepDense/screen04_vmem_final.npy \
  --targetEnsembleIndex ${targetIndex} \
  --fieldEnabled True --fieldScreenSize 4 --fieldStrength 1.0 \
  --fieldTransductionWeight 1000.0 --fieldTransductionGain -1.0 \
  --fieldRangeSymmetric False --fieldVector True \
  --ligandEnabled False --ligandGatingWeightRange None \
  --GJStrength 0.05 --GRNEnabled False --GRNTarget None \
  --clampMode fieldDomeTwoFoldSymmetry --clampType oscillatory \
  --clampedCellsProp 1.0 --clampDurationProp 0.04 \
  --clampAmplitudeRange "(-1.0,1.0)" --clampFrequencyRange "(100.0,1000.0)" \
  --loadExistingModel None --numSamples 1 \
  --numSimIters ${numSimIters:-2500} --numLearnIters ${numLearnIters:-2000} --numLearnTrials 1 \
  --evalDurationProp 0.08 \
  --learnedParameters "['clampFrequencies','clampPhases']" \
  --parameterGridSweep None --lossMethod globalmean --lr ${lr:-0.01} \
  --fileNumber ${SLURM_ARRAY_TASK_ID:-0} --verbose
