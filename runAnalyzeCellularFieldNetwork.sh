#!/bin/bash
latticeDims="(11,11)"
fieldTransductionWeight=10.0
fieldTransductionBias=0.03
numSimIters=1000
for (( fileNumber = $StartFileNumber; fileNumber <= $EndFileNumber; fileNumber++ ))
do
  python analyzeCellularFieldNetwork.py --latticeDims $latticeDims --fieldResolution $fieldResolution --fieldAggregation $fieldAggregation --fieldScreenSize $fieldScreenSize --fieldTransductionWeight $fieldTransductionWeight --fieldTransductionBias $fieldTransductionBias --fieldStrengthProp $fieldStrengthProp --GJStrength $GJStrength --randomizeInitialStates $randomizeInitialStates --numSamples $numSamples --numSimIters $numSimIters --analysisMode $analysisMode --analysisRegion $analysisRegion --fileNumber $fileNumber --fileNumberVersion $fileNumberVersion --verbose $verbose
done
#sbatch --export=ALL,StartFileNumber=483,EndFileNumber=483,fileNumberVersion=1,fieldResolution=0,fieldAggregation=average,fieldScreenSize=0,fieldStrengthProp=1.0,GJStrength=0.05,randomizeInitialStates=True,numSamples=101,analysisMode=sensitivity,analysisRegion=leftHalf,verbose=False --time 2-00:00:00 -p batch -e Error_%A.err --mem 10G runAnalyzeCellularFieldNetwork.sh
