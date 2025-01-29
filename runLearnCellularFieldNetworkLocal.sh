#!/bin/bash
latticeDims="(11,11)"
fieldEnabled=True
fieldResolution=1
fieldAggregation="average"
fieldScreenSize=4
fieldStrength=1
fieldTransductionWeight=1000
fieldTransductionGain=-1
fieldRangeSymmetric=False
fieldVector=True
ligandEnabled=True
GJStrength=0.05
clampMode="tissueDomeLigandTwoFoldSymmetry"
#clampMode="field"
clampType="oscillatory"
clampedCellsProp=1.0
clampDurationProp=0.1
clampAmplitudeRange="(0.0,1.0)"
clampFrequencyRange="(100.0,1000.0)"
numSamples=1
numSimIters=500
numLearnIters=1000
numLearnTrials=1
parameterGridSweep="None"
evalDurationProp=0.1
lossMethod="globalsum"
lr=0.01
SLURM_ARRAY_TASK_ID=0
verbose="True"
#learnedParameters="['clampFrequencies','clampPhases']"
#learnedParameters="['fieldTransductionBias','clampFrequencies','clampPhases','clampAmplitudes']"
#learnedParameters="['fieldTransductionBias']"
#learnedParameters="['fieldTransductionWeight','fieldTransductionBias','clampFrequencies','clampPhases']"
learnedParameters="['ligandGatingWeight','ligandGatingBias','clampFrequencies','clampPhases','clampAmplitudes']"
#learnedParameters="['ligandGatingWeight','ligandGatingBias','ligandCurrentStrength','clampValuesStatic']"
#learnedParameters="['fieldTransductionWeight','fieldTransductionBias','ligandGatingWeight','ligandGatingBias','ligandCurrentStrength','vmemToLigandCurrentStrength','clampFrequencies','clampPhases']"
#python learnCellularFieldNetwork.py --latticeDims $latticeDims --fieldEnabled $fieldEnabled --fieldResolution $fieldResolution --fieldAggregation $fieldAggregation --fieldScreenSize $fieldScreenSize --GJStrength $GJStrength --clampMode $clampMode --clampType $clampType --clampedCellsProp $clampedCellsProp --clampDurationProp $clampDurationProp --clampAmplitudeRange $clampAmplitudeRange --clampFrequencyRange $clampFrequencyRange --numClampCoreSquares $numClampCoreSquares --numSamples $numSamples --numSimIters $numSimIters --numLearnIters $numLearnIters --learnedParameters $learnedParameters --fileNumber $SLURM_ARRAY_TASK_ID --verbose $verbose
python learnCellularFieldNetwork.py --latticeDims $latticeDims --fieldEnabled $fieldEnabled --fieldScreenSize $fieldScreenSize --fieldStrength $fieldStrength --fieldTransductionWeight $fieldTransductionWeight --fieldTransductionGain $fieldTransductionGain --fieldRangeSymmetric $fieldRangeSymmetric --fieldVector $fieldVector --ligandEnabled $ligandEnabled --GJStrength $GJStrength --clampMode $clampMode --clampType $clampType --clampedCellsProp $clampedCellsProp --clampDurationProp $clampDurationProp --clampAmplitudeRange $clampAmplitudeRange --clampFrequencyRange $clampFrequencyRange --numSamples $numSamples --numSimIters $numSimIters --numLearnIters $numLearnIters --numLearnTrials $numLearnTrials --evalDurationProp $evalDurationProp --learnedParameters $learnedParameters --parameterGridSweep $parameterGridSweep --lossMethod $lossMethod --lr $lr --fileNumber $SLURM_ARRAY_TASK_ID --verbose $verbose
#python learnCellularFieldNetwork.py --latticeDims $latticeDims --fieldResolution $fieldResolution --fieldAggregation $fieldAggregation --fieldScreenSize $fieldScreenSize --clampMode $clampMode --clampType $clampType --clampedCellsProp $clampedCellsProp --clampDurationProp $clampDurationProp --clampAmplitudeRange $clampAmplitudeRange --clampFrequencyRange $clampFrequencyRange --numClampCoreSquares $numClampCoreSquares --numSamples $numSamples --numSimIters $numSimIters --numLearnIters $numLearnIters --learnedParameters $learnedParameters --fileNumber $SLURM_ARRAY_TASK_ID --verbose $verbose
#sbatch --export=ALL --time 2-00:00:00 -p batch --array 1-100 -e Error_%A_%a.err --mem 12G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL --time 2-00:00:00 -p batch --array 101-200 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL,fieldResolution=4,fieldAggregation=sum,clampMode=fieldDome,clampType=oscillatory,verbose=True --time 2-00:00:00 -p batch --array 401-500 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL,fieldResolution=1,fieldAggregation=average,fieldScreenSize=4,GJStrength=0.05,clampMode=fieldDomeTwoFoldSymmetry,clampType=oscillatory,clampDurationProp=0.2,numSimIters=2000,numLearnIters=50000,verbose=False --time 2-00:00:00 -p batch --array 1301-1400 -e Error_%A_%a.err --mem 10G runLearnCellularFieldNetwork.sh
