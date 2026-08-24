#!/bin/bash
# Train a named target: can a boundary clamp drive the tissue into a specified interior pattern?
#
# targetPattern selects it. The four form a difficulty ladder -- ap_band is half the tissue, stripes
# a central band, triangular_wave an M through the interior, face a pair of eyes with a nose and a
# mouth -- which is the point of running them together: where the ladder stops is where the boundary
# code runs out of capacity. All four had only ever been trained at 11x11.
#
# The horizon is 2500, matching the ensemble, the field range sweep, the basin map and the
# amplification and response measurements. It is not the 1000 of the 11x11 reference run: at 30x30
# the tissue is still moving at 2500 whether clamped or not (see replot_unclamped_evolution.py), so
# a shorter horizon reads the pattern mid-development, and a face trained at one horizon cannot be
# compared with an ensemble generated at another. The clamp, the loss and the learned parameters do
# follow the 11x11 run; latticeDims and the horizon are the intended differences.
#
# The face itself is now a fraction of the lattice rather than the 11x11 cell indices it used to
# be; at 30x30 that is a pair of 6x6 eyes, an eight row nose and a three row mouth. See
# faceFeatureIndices in learnCellularFieldNetwork.py, and --verifyTargets to check it.
#
# lossMethod defaults to globalsum, which is what every result so far was scored under. correlation
# drops the requirement to reach absolute voltages: the target is binary at -9.2 and -60 mV while the
# tissue's reachable mean sits within a few mV of -15, so a correct arrangement at the wrong level
# scores no better than noise under globalsum, and a uniform sheet lands within one percent of the
# best trained model. The two are not comparable numbers -- one is millivolts summed, the other is
# one minus a correlation -- so runs under each want separate file numbers.
#
# fieldScreenSize is 4 rather than the 21 of the reference run, because 4 is the screen at which
# the 30x30 tissue has actually been characterised: the clamp ensemble, the hashing measurement and
# the CX/CT numbers all come from screen 4, and a face trained at a different screen could not be
# compared against any of them.
#
# fieldTransductionWeight defaults to 1000, inherited unchanged from the 11x11 tissue like every
# other tissue parameter, but is now overridable: the pre-pattern depth profile at weight 700 most
# closely matches what the working 11x11 model natively produces (see the weight sweep against the
# 11x11 g-trend), and the clamp-sensitivity measurement found weight 1000 landscapes are two to three
# orders of magnitude more rugged than needed for search to work. Weight and screen size were also
# found to act on the same underlying lever -- how strongly and how far field feedback couples cell
# to cell -- so both are exposed as independent overrides rather than only one.
#
# The clamp and readout windows are derived from absolute iteration counts, not written as
# proportions, because a proportion silently means a different protocol at a different horizon.
#
# Only the clamp is learned. fieldTransductionBias was in this list and should not have been: it is
# a property of the tissue, so learning it asks whether some tissue can be driven into the target
# rather than whether this one can, and the reachability control learns the clamp alone.
#
# It also created the plateau. Zeroing the bias sends the transduction term to zero, G_pol relaxes
# to zero and the sheet flattens, and a flat sheet is not a good solution but it is a local minimum
# gradient descent cannot leave. Of the 40 models trained at 1000 iterations, 34 ended with the bias
# at exactly zero and all 34 are the plateau at 8.35; the six that escaped all kept it near the
# 0.0005 default, and bias against loss over the batch gives r = -0.97. So the escape rate of six in
# forty was a measure of how often training avoided switching the tissue off, and freezing the bias
# should remove the plateau rather than merely make the comparison honest.
source ~/.bashrc
myconda
# Rows and columns are taken separately rather than as a "(30,30)" string because sbatch
# --export splits its value list on commas: exporting latticeDims="(30,30)" delivers "(30" to the
# job, which then dies in ast.literal_eval. Comma-free variables cannot be broken that way.
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
  --fieldEnabled True --fieldScreenSize ${fieldScreenSize:-4} --fieldStrength 1.0 \
  --fieldTransductionWeight ${fieldTransductionWeight:-1000.0} --fieldTransductionGain -1.0 \
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
  --parameterGridSweep None --lossMethod ${lossMethod:-globalsum} --lr ${lr:-0.01} \
  --fileNumber ${SLURM_ARRAY_TASK_ID:-0} --verbose
