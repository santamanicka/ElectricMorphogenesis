#!/bin/bash
latticeDims="(11,11)"
fieldResolution=1
fieldAggregation="average"
fieldScreenSize=4
GJStrength=0.05
fieldTransductionWeight=10.0
fieldTransductionBias=0.03
fieldStrengthProp=1.0
randomizeInitialStates="False"
characteristicNames=$1
numSamples=101
numSimIters=1000
numPerturbSimIters=0
analysisMode="fixBiasSweepWeightScreenGJ"
analysisRegion="representative"
StartFileNumber=1
EndFileNumber=1
fileNumberVersion=0
#SLURM_ARRAY_TASK_ID=0
verbose="True"
for (( fileNumber = $StartFileNumber; fileNumber <= $EndFileNumber; fileNumber++ ))
do
  python analyzeCellularFieldNetwork.py --latticeDims $latticeDims --fieldResolution $fieldResolution --fieldAggregation $fieldAggregation --fieldScreenSize $fieldScreenSize --fieldTransductionWeight $fieldTransductionWeight --fieldTransductionBias $fieldTransductionBias --fieldStrengthProp $fieldStrengthProp --GJStrength $GJStrength --randomizeInitialStates $randomizeInitialStates --characteristicNames $characteristicNames --numSamples $numSamples --numSimIters $numSimIters --numPerturbSimIters $numPerturbSimIters --analysisMode $analysisMode --analysisRegion $analysisRegion --fileNumber $fileNumber --fileNumberVersion $fileNumberVersion --verbose $verbose
done
#python learnCellularFieldNetwork.py --latticeDims $latticeDims --fieldResolution $fieldResolution --fieldAggregation $fieldAggregation --fieldScreenSize $fieldScreenSize --clampMode $clampMode --clampType $clampType --clampedCellsProp $clampedCellsProp --clampDurationProp $clampDurationProp --clampAmplitudeRange $clampAmplitudeRange --clampFrequencyRange $clampFrequencyRange --numClampCoreSquares $numClampCoreSquares --numSamples $numSamples --numSimIters $numSimIters --numLearnIters $numLearnIters --learnedParameters $learnedParameters --fileNumber $SLURM_ARRAY_TASK_ID --verbose $verbose
#sbatch --export=ALL --time 2-00:00:00 -p batch --array 1-100 -e Error_%A_%a.err --mem 12G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL --time 2-00:00:00 -p batch --array 101-200 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL,fieldResolution=4,fieldAggregation=sum,clampMode=fieldDome,clampType=oscillatory,verbose=True --time 2-00:00:00 -p batch --array 401-500 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL,fieldResolution=1,fieldAggregation=average,fieldScreenSize=4,GJStrength=0.05,clampMode=fieldDomeTwoFoldSymmetry,clampType=oscillatory,clampDurationProp=0.2,numSimIters=2000,numLearnIters=50000,verbose=False --time 2-00:00:00 -p batch --array 1301-1400 -e Error_%A_%a.err --mem 10G runLearnCellularFieldNetwork.sh
