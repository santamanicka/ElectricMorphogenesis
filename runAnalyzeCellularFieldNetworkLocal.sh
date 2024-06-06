#!/bin/bash
latticeDims="(3,3)"
fieldResolution=1
fieldAggregation="average"
fieldScreenSize=4
GJStrength=0.05
numSamples=1
numSimIters=100
NumFileNumbers=1
#SLURM_ARRAY_TASK_ID=0
verbose="True"
for (( fileNumber = 0; fileNumber <= $NumFileNumbers; fileNumber++ ))
do
  python analyzeCellularFieldNetwork.py --latticeDims $latticeDims --fieldResolution $fieldResolution --fieldAggregation $fieldAggregation --fieldScreenSize $fieldScreenSize --GJStrength $GJStrength --numSamples $numSamples --numSimIters $numSimIters --fileNumber $fileNumber --verbose $verbose
done
#python learnCellularFieldNetwork.py --latticeDims $latticeDims --fieldResolution $fieldResolution --fieldAggregation $fieldAggregation --fieldScreenSize $fieldScreenSize --clampMode $clampMode --clampType $clampType --clampedCellsProp $clampedCellsProp --clampDurationProp $clampDurationProp --clampAmplitudeRange $clampAmplitudeRange --clampFrequencyRange $clampFrequencyRange --numClampCoreSquares $numClampCoreSquares --numSamples $numSamples --numSimIters $numSimIters --numLearnIters $numLearnIters --learnedParameters $learnedParameters --fileNumber $SLURM_ARRAY_TASK_ID --verbose $verbose
#sbatch --export=ALL --time 2-00:00:00 -p batch --array 1-100 -e Error_%A_%a.err --mem 12G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL --time 2-00:00:00 -p batch --array 101-200 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL,fieldResolution=4,fieldAggregation=sum,clampMode=fieldDome,clampType=oscillatory,verbose=True --time 2-00:00:00 -p batch --array 401-500 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL,fieldResolution=1,fieldAggregation=average,fieldScreenSize=4,GJStrength=0.05,clampMode=fieldDomeTwoFoldSymmetry,clampType=oscillatory,clampDurationProp=0.2,numSimIters=2000,numLearnIters=50000,verbose=False --time 2-00:00:00 -p batch --array 1301-1400 -e Error_%A_%a.err --mem 10G runLearnCellularFieldNetwork.sh
