#!/bin/bash
latticeDims="(11,11)"
#fieldResolution=4
#fieldAggregation="average"
#fieldScreenSize=4
#GJStrength=0.05
#clampMode="fieldDomeTwoFoldSymmetry"
#clampType="oscillatory"
clampedCellsProp=1.0
#clampDurationProp=0.1
clampAmplitudeRange="(-10.0,10.0)"
clampFrequencyRange="(100.0,1000.0)"
numSamples=1
#numSimIters=100
#numLearnIters=100
#SLURM_ARRAY_TASK_ID=0
#verbose="True"
learnedParameters="['fieldTransductionWeight','fieldTransductionBias','clampFrequencies','clampPhases']"
#python learnCellularFieldNetwork.py --latticeDims $latticeDims --fieldResolution $fieldResolution --fieldAggregation $fieldAggregation --fieldScreenSize $fieldScreenSize --GJStrength $GJStrength --clampMode $clampMode --clampType $clampType --clampedCellsProp $clampedCellsProp --clampDurationProp $clampDurationProp --clampAmplitudeRange $clampAmplitudeRange --clampFrequencyRange $clampFrequencyRange --numClampCoreSquares $numClampCoreSquares --numSamples $numSamples --numSimIters $numSimIters --numLearnIters $numLearnIters --learnedParameters $learnedParameters --fileNumber $SLURM_ARRAY_TASK_ID --verbose $verbose
python learnCellularFieldNetwork.py --latticeDims $latticeDims --fieldEnabled $fieldEnabled --fieldScreenSize $fieldScreenSize --ligandEnabled $ligandEnabled --GJStrength $GJStrength --clampMode $clampMode --clampType $clampType --clampedCellsProp $clampedCellsProp --clampDurationProp $clampDurationProp --clampAmplitudeRange $clampAmplitudeRange --clampFrequencyRange $clampFrequencyRange --numSamples $numSamples --numSimIters $numSimIters --numLearnIters $numLearnIters --learnedParameters $learnedParameters --lossMethod $lossMethod --fileNumber $SLURM_ARRAY_TASK_ID --verbose $verbose
#python learnCellularFieldNetwork.py --latticeDims $latticeDims --fieldResolution $fieldResolution --fieldAggregation $fieldAggregation --fieldScreenSize $fieldScreenSize --clampMode $clampMode --clampType $clampType --clampedCellsProp $clampedCellsProp --clampDurationProp $clampDurationProp --clampAmplitudeRange $clampAmplitudeRange --clampFrequencyRange $clampFrequencyRange --numClampCoreSquares $numClampCoreSquares --numSamples $numSamples --numSimIters $numSimIters --numLearnIters $numLearnIters --learnedParameters $learnedParameters --fileNumber $SLURM_ARRAY_TASK_ID --verbose $verbose
#sbatch --export=ALL --time 2-00:00:00 -p batch --array 1-100 -e Error_%A_%a.err --mem 12G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL --time 2-00:00:00 -p batch --array 101-200 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL,fieldResolution=4,fieldAggregation=sum,clampMode=fieldDome,clampType=oscillatory,verbose=True --time 2-00:00:00 -p batch --array 401-500 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL,fieldResolution=1,fieldAggregation=average,fieldScreenSize=1,GJStrength=1.0,clampMode=fieldDomeTwoFoldSymmetry,clampType=oscillatory,clampDurationProp=0.1,numSimIters=1000,numLearnIters=50000,verbose=False --time 2-00:00:00 -p batch --array 1701-1800 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL,fieldEnabled=True,fieldScreenSize=4,ligandEnabled=False,GJStrength=0.05,clampMode=fieldDomeTwoFoldSymmetry,clampType=oscillatory,clampDurationProp=0.1,numSimIters=1000,numLearnIters=50000,lossMethod=globalsum,verbose=False --time 2-00:00:00 -p batch --array 401-500 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
