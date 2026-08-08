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
# The protocol matches the ensemble that supplied the target exactly, because a target produced
# under one protocol is not necessarily reachable under another. The ensemble specifies its clamp
# and readout windows as absolute iteration counts (100 and 200) while training takes proportions,
# so the two are derived here rather than written as literals: 0.04 and 0.08 happen to be correct
# at 2500 iterations and are silently wrong at any other horizon.
#
# 1000 iterations rather than the 2500 used elsewhere, because backpropagation through the
# simulation stores about 49 MB per iteration at this lattice size and 2500 needs roughly 123 GB,
# above every available node. The shorter horizon is not an easier problem: the map hashes just as
# hard there, 100 random clamps giving 100 outcomes with the nearest pair 11.19 mV apart against
# 10.82 mV at 2500, so the search question this run asks is unchanged.
source ~/.bashrc
myconda
targetIndex=${targetIndex:-0}
numSimIters=${numSimIters:-1000}
clampIters=${clampIters:-100}     # absolute, matching runFieldRangeSweep.sh
readoutIters=${readoutIters:-200} # absolute, matching runFieldRangeSweep.sh
clampDurationProp=$(awk "BEGIN{print ${clampIters}/${numSimIters}}")
evalDurationProp=$(awk "BEGIN{print ${readoutIters}/${numSimIters}}")
python learnCellularFieldNetwork.py \
  --latticeDims "(30,30)" \
  --targetPattern ensemble \
  --targetEnsembleFile data/fieldRangeSweep1000/screen04_vmem_final.npy \
  --targetEnsembleIndex ${targetIndex} \
  --fieldEnabled True --fieldScreenSize 4 --fieldStrength 1.0 \
  --fieldTransductionWeight 1000.0 --fieldTransductionGain -1.0 \
  --fieldRangeSymmetric False --fieldVector True \
  --ligandEnabled False --ligandGatingWeightRange None \
  --GJStrength 0.05 --GRNEnabled False --GRNTarget None \
  --clampMode fieldDomeTwoFoldSymmetry --clampType oscillatory \
  --clampedCellsProp 1.0 --clampDurationProp ${clampDurationProp} \
  --clampAmplitudeRange "(-1.0,1.0)" --clampFrequencyRange "(100.0,1000.0)" \
  --loadExistingModel None --numSamples 1 \
  --numSimIters ${numSimIters} --numLearnIters ${numLearnIters:-2000} --numLearnTrials 1 \
  --evalDurationProp ${evalDurationProp} \
  --learnedParameters "['clampFrequencies','clampPhases']" \
  --parameterGridSweep None --lossMethod globalmean --lr ${lr:-0.01} \
  --fileNumber ${SLURM_ARRAY_TASK_ID:-0} --verbose
