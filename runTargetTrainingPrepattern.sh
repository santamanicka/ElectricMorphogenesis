#!/bin/bash
# Control for runTargetTraining.sh: relax the boundary-only constraint and let the clamp set the
# G_pol prepattern directly on every cell, not just the dome. Same field dynamics, same clamp
# duration and readout window, same learning mechanism (per-point oscillatory frequency/phase),
# the only change is clampMode fieldDomeTwoFoldSymmetry -> tissueGpol, which switches cellIndices
# from the boundary dome to np.arange(numCells) (see learnCellularFieldNetwork.py's
# loadTissueParamsOnly branch) and switches what's forced from boundary Vmem (routed through the
# field, screen-limited) to G_pol directly at every cell. clampValues are already scaled by G_ref
# before being written to G_pol (embryo.py's 'Gpol' branch), matching how initParameters seeds
# G_pol elsewhere, so the existing amplitude range needs no rescaling.
#
# This tests whether the boundary's limited addressability (fair_CT_dims ~1 at the best screen
# size measured, see cxCt30x30Dense_metricEvaluation.npz) is what's capping the face runs, or
# whether the deeper issue is the one analyzeTrajectoryConditioning.py found earlier: the target is
# a transient, not an attractor, so even a fully-addressed prepattern may not survive to the
# readout window once the clamp releases at clampEndIter and the tissue evolves under the same
# field feedback (fieldScreenSize, fieldTransductionWeight) as every other run here.
source ~/.bashrc
myconda
targetPattern=${targetPattern:-face}
latticeRows=${latticeRows:-30}
latticeCols=${latticeCols:-30}
latticeDims="(${latticeRows},${latticeCols})"
numSimIters=${numSimIters:-2500}
clampIters=${clampIters:-100}
readoutIters=${readoutIters:-100}
clampDurationProp=$(awk "BEGIN{print ${clampIters}/${numSimIters}}")
evalDurationProp=$(awk "BEGIN{print ${readoutIters}/${numSimIters}}")
python learnCellularFieldNetwork.py \
  --latticeDims "${latticeDims}" \
  --targetPattern ${targetPattern:-face} \
  --fieldEnabled True --fieldScreenSize ${fieldScreenSize:-4} --fieldStrength ${fieldStrength:-1.0} \
  --fieldTransductionWeight ${fieldTransductionWeight:-1000.0} --fieldTransductionGain -1.0 \
  --fieldRangeSymmetric False --fieldVector True \
  --ligandEnabled False --ligandGatingWeightRange None \
  --GJStrength 0.05 --GRNEnabled False --GRNTarget None \
  --clampMode ${clampMode:-tissueGpol} --clampType oscillatory \
  --clampedCellsProp 1.0 --clampDurationProp ${clampDurationProp} \
  --clampAmplitudeRange "(-1.0,1.0)" --clampFrequencyRange "(100.0,1000.0)" \
  --loadExistingModel None --numSamples 1 \
  --numSimIters ${numSimIters} --numLearnIters ${numLearnIters:-2000} --numLearnTrials 1 \
  --evalDurationProp ${evalDurationProp} \
  --learnedParameters "['clampFrequencies','clampPhases']" \
  --parameterGridSweep None --lossMethod ${lossMethod:-globalsum} --lr ${lr:-0.01} \
  --fileNumber ${SLURM_ARRAY_TASK_ID:-0} --verbose
