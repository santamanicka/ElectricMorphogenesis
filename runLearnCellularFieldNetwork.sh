#!/bin/bash
latticeDims="(11,11)"
fieldResolution=4
fieldAggregation="sum"
fieldScreenSize=4
clampMode="fieldCore"
clampType="static"
clampedCellsProp=1.0
clampDurationProp=0.1
clampAmplitudeRange="(-1.0,1.0)"
clampFrequencyRange="(100.0,1000.0)"
numClampCoreSquares=1
numSamples=1
numSimIters=100
numLearnIters=1000
fileNumber=0
verbose="True"
learnedParameters="['clampFrequencies','clampPhases','fieldTransductionWeight','fieldTransductionBias']"
python learnCellularFieldNetwork.py --latticeDims $latticeDims --fieldResolution $fieldResolution --fieldAggregation $fieldAggregation --fieldScreenSize $fieldScreenSize --clampMode $clampMode --clampType $clampType --clampedCellsProp $clampedCellsProp --clampDurationProp $clampDurationProp --clampAmplitudeRange $clampAmplitudeRange --clampFrequencyRange $clampFrequencyRange --numClampCoreSquares $numClampCoreSquares --numSamples $numSamples --numSimIters $numSimIters --numLearnIters $numLearnIters --learnedParameters $learnedParameters --verbose $verbose
# sbatch --export=ALL,RandomModel=True --time 7-00:00:00 -p largemem --array 1-100 -e Error_%A_%a.err --mem 12G runLearnCellularFieldNetwork.sh
# sbatch --export=ALL,RandomModel=True,AsymmetricInterGRN=False,MaxNumGenes=10,SummaryFunction=mean,Noise=0.1,NumNoisySamples=1,IncludeVmemTarget=True,Backpropped=False --time 7-00:00:00 -p largemem --array 1101-1200 -e Error_%A_%a.err --mem 12G runBackpropagate.sbatch
# sbatch --export=ALL,RandomModel=True,AsymmetricInterGRN=True,MaxNumGenes=10,SummaryFunction=mean,Noise=0.1,NumNoisySamples=1,IncludeVmemTarget=True,Backpropped=False --time 7-00:00:00 -p batch --array 1201-1300 -e Error_%A_%a.err --mem 12G runBackpropagate.sbatch
# sbatch --export=ALL,RandomModel=True,AsymmetricInterGRN=True,MaxNumGenes=20,SummaryFunction=mean,Noise=0.1,NumNoisySamples=1,IncludeVmemTarget=True,Backpropped=False --time 7-00:00:00 -p largemem --array 1201-1300 -e Error_%A_%a.err --mem 25G runBackpropagate.sbatch
# sbatch --export=ALL,RandomModel=True,AsymmetricInterGRN=True,PCPAxes=Horizontal,MaxNumGenes=20,SummaryFunction=mean,Noise=0.1,NumNoisySamples=1,IncludeVmemTarget=True,Backpropped=False --time 7-00:00:00 -p batch --array 1301-1400 -e Error_%A_%a.err --mem 40G runBackpropagate.sbatch
# sbatch --export=ALL,RandomModel=True,AsymmetricInterGRN=True,PCPAxes=HorizontalBiDir,MaxNumGenes=20,SummaryFunction=mean,Noise=0.1,NumNoisySamples=1,IncludeVmemTarget=True,Backpropped=False --time 7-00:00:00 -p batch --array 1301-1400 -e Error_%A_%a.err --mem 40G runBackpropagate.sbatch
# sbatch --export=ALL,RandomModel=True,AsymmetricInterGRN=True,PCPAxes=HorizontalUniDir,MaxNumGenes=20,SummaryFunction=mean,Noise=0.1,NumNoisySamples=1,IncludeVmemTarget=True,Backpropped=False --time 7-00:00:00 -p largemem --array 1401-1500 -e Error_%A_%a.err --mem 40G runBackpropagate.sbatch
# sbatch --export=ALL,RandomModel=True,AsymmetricInterGRN=False,PCPAxes=None,MaxNumGenes=20,SummaryFunction=mean,Noise=0.1,NumNoisySamples=1,IncludeVmemTarget=True,Backpropped=False --time 7-00:00:00 -p batch --array 1501-1600 -e Error_%A_%a.err --mem 40G runBackpropagate.sbatch
# sbatch --export=ALL,RandomModel=False,AsymmetricInterGRN=False,PCPAxes=None,MaxNumGenes=20,SummaryFunction=mean,Noise=0.1,NumNoisySamples=1,IncludeVmemTarget=False,Backpropped=False --time 7-00:00:00 -p batch --array 1-30 -e Error_%A_%a.err --mem 40G runBackpropagate.sbatch
