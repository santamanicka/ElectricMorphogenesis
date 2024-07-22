#!/bin/bash
latticeDims="(11,11)"
#fieldResolution=4
#fieldAggregation="average"
#fieldScreenSize=4
#GJStrength=0.05
#clampMode="tissueLigand"
#clampType="staticRandom"
#clampedCellsProp=1.0
#clampDurationProp=0.1
clampAmplitudeRange=$1
vmemToLigandCurrentStrengthRange=$2
learnedParameters=$3
clampFrequencyRange="(100.0,1000.0)"
numSamples=1
#numSimIters=100
#numLearnIters=100
#SLURM_ARRAY_TASK_ID=0
#verbose="True"
#learnedParameters1="['fieldTransductionWeight','fieldTransductionBias','clampFrequencies','clampPhases']"
#learnedParameters2="['ligandGatingWeight','ligandGatingBias','ligandCurrentStrength','clampFrequencies','clampPhases']"
#learnedParameters3="['ligandGatingWeight','ligandGatingBias','ligandCurrentStrength','clampValuesStatic']"
#python learnCellularFieldNetwork.py --latticeDims $latticeDims --fieldResolution $fieldResolution --fieldAggregation $fieldAggregation --fieldScreenSize $fieldScreenSize --GJStrength $GJStrength --clampMode $clampMode --clampType $clampType --clampedCellsProp $clampedCellsProp --clampDurationProp $clampDurationProp --clampAmplitudeRange $clampAmplitudeRange --clampFrequencyRange $clampFrequencyRange --numClampCoreSquares $numClampCoreSquares --numSamples $numSamples --numSimIters $numSimIters --numLearnIters $numLearnIters --learnedParameters $learnedParameters --fileNumber $SLURM_ARRAY_TASK_ID --verbose $verbose
python learnCellularFieldNetwork.py --latticeDims $latticeDims --fieldEnabled $fieldEnabled --fieldScreenSize $fieldScreenSize --fieldStrength $fieldStrength --ligandEnabled $ligandEnabled --vmemToLigandCurrentStrengthRange $vmemToLigandCurrentStrengthRange --GJStrength $GJStrength --clampMode $clampMode --clampType $clampType --clampedCellsProp $clampedCellsProp --clampDurationProp $clampDurationProp --clampAmplitudeRange $clampAmplitudeRange --clampFrequencyRange $clampFrequencyRange --numSamples $numSamples --numSimIters $numSimIters --numLearnIters $numLearnIters --learnedParameters $learnedParameters --lossMethod $lossMethod --fileNumber $SLURM_ARRAY_TASK_ID --verbose $verbose
#python learnCellularFieldNetwork.py --latticeDims $latticeDims --fieldResolution $fieldResolution --fieldAggregation $fieldAggregation --fieldScreenSize $fieldScreenSize --clampMode $clampMode --clampType $clampType --clampedCellsProp $clampedCellsProp --clampDurationProp $clampDurationProp --clampAmplitudeRange $clampAmplitudeRange --clampFrequencyRange $clampFrequencyRange --numClampCoreSquares $numClampCoreSquares --numSamples $numSamples --numSimIters $numSimIters --numLearnIters $numLearnIters --learnedParameters $learnedParameters --fileNumber $SLURM_ARRAY_TASK_ID --verbose $verbose
#sbatch --export=ALL --time 2-00:00:00 -p batch --array 1-100 -e Error_%A_%a.err --mem 12G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL --time 2-00:00:00 -p batch --array 101-200 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL,fieldResolution=4,fieldAggregation=sum,clampMode=fieldDome,clampType=oscillatory,verbose=True --time 2-00:00:00 -p batch --array 401-500 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL,fieldResolution=1,fieldAggregation=average,fieldScreenSize=1,GJStrength=1.0,clampMode=fieldDomeTwoFoldSymmetry,clampType=oscillatory,clampDurationProp=0.1,numSimIters=1000,numLearnIters=50000,verbose=False --time 2-00:00:00 -p batch --array 1701-1800 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
#sbatch --export=ALL,fieldEnabled=False,fieldScreenSize=0,ligandEnabled=True,GJStrength=0.05,clampMode=tissueDomeLigandTwoFoldSymmetry,clampType=staticRandom,clampDurationProp=0.1,numSimIters=1000,numLearnIters=50000,lossMethod=globalsum,verbose=False --time 2-00:00:00 -p batch --array 301-400 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh
#learnedParameters1="['fieldTransductionWeight','fieldTransductionBias','clampFrequencies','clampPhases']"
#sbatch --export=ALL,fieldEnabled=True,fieldScreenSize=21,ligandEnabled=False,GJStrength=0.05,clampMode=fieldDomeTwoFoldSymmetry,clampType=oscillatory,clampedCellsProp=1.0,clampDurationProp=0.1,numSimIters=1000,numLearnIters=50000,lossMethod=globalsum,verbose=False --time 2-00:00:00 -p batch --array 901-1000 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh "(-10.0,10.0)" "['fieldTransductionWeight','fieldTransductionBias','clampFrequencies','clampPhases']"
#sbatch --export=ALL,fieldEnabled=True,fieldScreenSize=21,fieldStrength=1.0,ligandEnabled=False,GJStrength=0.05,clampMode=fieldDomeTwoFoldSymmetry,clampType=oscillatory,clampedCellsProp=1.0,clampDurationProp=0.1,numSimIters=1000,numLearnIters=50000,lossMethod=globalsum,verbose=False --time 2-00:00:00 -p batch --array 1001-1100 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh "(-100.0,100.0)" "['fieldTransductionWeight','fieldTransductionBias','clampFrequencies','clampPhases']"
#sbatch --export=ALL,fieldEnabled=False,fieldScreenSize=0,ligandEnabled=True,GJStrength=0.05,clampMode=tissueLigand,clampType=oscillatory,clampedCellsProp=0.5,clampDurationProp=0.1,numSimIters=1000,numLearnIters=50000,lossMethod=globalsum,verbose=False --time 2-00:00:00 -p batch --array 601-700 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh "(0.0,1.0)" "['ligandGatingWeight','ligandGatingBias','ligandCurrentStrength','clampFrequencies','clampPhases']"
#sbatch --export=ALL,fieldEnabled=False,fieldScreenSize=0,ligandEnabled=True,GJStrength=0.05,clampMode=tissueLigand,clampType=staticRandom,clampedCellsProp=1.0,clampDurationProp=0.1,numSimIters=1000,numLearnIters=50000,lossMethod=globalsum,verbose=False --time 2-00:00:00 -p batch --array 1701-1800 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh "(0.0,1.0)" "(0.1,10.0)" "['ligandGatingWeight','ligandGatingBias','ligandCurrentStrength','vmemToLigandCurrentStrength','clampValuesStatic']"
#sbatch --export=ALL,fieldEnabled=True,fieldScreenSize=1,fieldStrength=10.0,ligandEnabled=True,GJStrength=0.05,clampMode=fieldDomeTwoFoldSymmetry,clampType=oscillatory,clampedCellsProp=1.0,clampDurationProp=0.1,numSimIters=1000,numLearnIters=50000,lossMethod=globalsum,verbose=False --time 2-00:00:00 -p batch --array 1601-1700 -e Error_%A_%a.err --mem 4G runLearnCellularFieldNetwork.sh "(-100.0,100.0)" "['fieldTransductionWeight','fieldTransductionBias','ligandGatingWeight','ligandGatingBias','ligandCurrentStrength','vmemToLigandCurrentStrength','clampFrequencies','clampPhases']"
