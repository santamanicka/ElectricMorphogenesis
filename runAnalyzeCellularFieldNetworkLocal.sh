#!/bin/bash
latticeDims="(3,3)"
fieldEnabled="True"
fieldResolution=1
fieldAggregation="average"
fieldScreenSize=4
GJStrength=0.05
fieldTransductionWeight=10.0
fieldTransductionBias=0.03
fieldStrengthProp=1.0
ligandEnabled="False"
randomizeInitialStates="True"
characteristicNames=$1
numSamples=2
numSimIters=50
numPerturbSimIters=0
numGradientTimePoints=100
analysisMode="sensitivity"
analysisRegion='[29,30,40,41]'
#StartFileNumber=0.1
#EndFileNumber=0.1
fileNumber=1939
fileNumberVersion=0
#SLURM_ARRAY_TASK_ID=0
verbose="True"
#for (( fileNumber = $StartFileNumber; fileNumber <= $EndFileNumber; fileNumber++ ))
#do
python analyzeCellularFieldNetwork.py --latticeDims $latticeDims --fieldEnabled $fieldEnabled --fieldResolution $fieldResolution --fieldAggregation $fieldAggregation --fieldScreenSize $fieldScreenSize --fieldTransductionWeight $fieldTransductionWeight --fieldTransductionBias $fieldTransductionBias --fieldStrengthProp $fieldStrengthProp --ligandEnabled $ligandEnabled --GJStrength $GJStrength --randomizeInitialStates $randomizeInitialStates --characteristicNames $characteristicNames --numGradientTimePoints $numGradientTimePoints --numSamples $numSamples --numSimIters $numSimIters --numPerturbSimIters $numPerturbSimIters --analysisMode $analysisMode --analysisRegion $analysisRegion --fileNumber $fileNumber --fileNumberVersion $fileNumberVersion --verbose $verbose
#done
#python learnCellularFieldNetwork.py --latticeDims $latticeDims --fieldResolution $fieldResolution --fieldAggregation $fieldAggregation --fieldScreenSize $fieldScreenSize --clampMode $clampMode --clampType $clampType --clampedCellsProp $clampedCellsProp --clampDurationProp $clampDurationProp --clampAmplitudeRange $clampAmplitudeRange --clampFrequencyRange $clampFrequencyRange --numClampCoreSquares $numClampCoreSquares --numSamples $numSamples --numSimIters $numSimIters --numLearnIters $numLearnIters --learnedParameters $learnedParameters --fileNumber $SLURM_ARRAY_TASK_ID --verbose $verbose
#sbatch --export=ALL --time 2-00:00:00 -p batch --array 1-100 -e Error_%A_%a.err --mem 12G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL --time 2-00:00:00 -p batch --array 101-200 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL,fieldResolution=4,fieldAggregation=sum,clampMode=fieldDome,clampType=oscillatory,verbose=True --time 2-00:00:00 -p batch --array 401-500 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL,fieldResolution=1,fieldAggregation=average,fieldScreenSize=4,GJStrength=0.05,clampMode=fieldDomeTwoFoldSymmetry,clampType=oscillatory,clampDurationProp=0.2,numSimIters=2000,numLearnIters=50000,verbose=False --time 2-00:00:00 -p batch --array 1301-1400 -e Error_%A_%a.err --mem 10G runLearnCellularFieldNetwork.sh
