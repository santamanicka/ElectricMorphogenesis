#!/bin/bash
# Train a named target: can a boundary clamp drive the tissue into a specified interior pattern?
#
# targetPattern selects it. The four form a difficulty ladder -- ap_band is half the tissue, stripes
# a central band, triangular_wave an M through the interior, face a pair of eyes with a nose and a
# mouth -- which is the point of running them together: where the ladder stops is where the boundary
# code runs out of capacity. All four had only ever been trained at 11x11.
#
# The configuration follows the 11x11 runs that produced the stored Stigmergic model -- a 1000
# iteration horizon, a 100 iteration clamp, globalsum loss, and the field transduction bias learned
# alongside the clamp frequencies and phases. latticeDims is the only intended difference, so that
# a 30x30 result can be read against the 11x11 one rather than against nothing.
#
# The face itself is now a fraction of the lattice rather than the 11x11 cell indices it used to
# be; at 30x30 that is a pair of 6x6 eyes, an eight row nose and a three row mouth. See
# faceFeatureIndices in learnCellularFieldNetwork.py, and --verifyTargets to check it.
#
# fieldScreenSize is 4 rather than the 21 of the reference run, because 4 is the screen at which
# the 30x30 tissue has actually been characterised: the clamp ensemble, the hashing measurement and
# the CX/CT numbers all come from screen 4, and a face trained at a different screen could not be
# compared against any of them.
#
# The clamp and readout windows are derived from absolute iteration counts, not written as
# proportions, because a proportion silently means a different protocol at a different horizon.
source ~/.bashrc
myconda
# Rows and columns are taken separately rather than as a "(30,30)" string because sbatch
# --export splits its value list on commas: exporting latticeDims="(30,30)" delivers "(30" to the
# job, which then dies in ast.literal_eval. Comma-free variables cannot be broken that way.
targetPattern=${targetPattern:-face}
latticeRows=${latticeRows:-30}
latticeCols=${latticeCols:-30}
latticeDims="(${latticeRows},${latticeCols})"
numSimIters=${numSimIters:-1000}
clampIters=${clampIters:-100}
readoutIters=${readoutIters:-100}
clampDurationProp=$(awk "BEGIN{print ${clampIters}/${numSimIters}}")
evalDurationProp=$(awk "BEGIN{print ${readoutIters}/${numSimIters}}")
python learnCellularFieldNetwork.py \
  --latticeDims "${latticeDims}" \
  --targetPattern ${targetPattern:-face} \
  --fieldEnabled True --fieldScreenSize ${fieldScreenSize:-4} --fieldStrength 1.0 \
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
  --learnedParameters "['fieldTransductionBias','clampFrequencies','clampPhases']" \
  --parameterGridSweep None --lossMethod globalsum --lr ${lr:-0.01} \
  --fileNumber ${SLURM_ARRAY_TASK_ID:-0} --verbose
