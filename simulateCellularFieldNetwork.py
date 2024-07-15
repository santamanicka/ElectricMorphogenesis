import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import utilities

circuitRows,circuitCols = latticeDims = (11,11)
fieldEnabled = True
ligandEnabled = False
fieldResolution = 1
fieldStrength = 10  # original value 10.0
fieldScreenSize = 2*max(latticeDims) - 1 # max: 2*max(latticeDims) - 1
fieldTransductionBias = torch.DoubleTensor([0.0214])  # 0.0214
fieldTransductionWeight = torch.DoubleTensor([9.4505])  # 9.4505
fieldTransductionTimeConstant = torch.DoubleTensor([10.0])
# RelativePermittivity = 10**7
GapJunctionStrength = 0.05  # meaningful values in [0.05,1.0]
ligandGatingWeight = torch.DoubleTensor([10.0])
ligandGatingBias = torch.DoubleTensor([-0.5])
ligandCurrentStrength = torch.DoubleTensor([1.0])
clampMode = 'fieldDomeTwoFoldSymmetry'  # possible values: field, fieldDome, tissueVmem, tissueDomeVmem, tissueLigand, tissueDomeLigand, tissueGpol, tissueDomeGpol, None
clampType = 'oscillatory'  # possible values: oscillatory, staticConstant, staticRandom
clampValue = 1.0  # static clamp value when clampType = 'staticConstant'
minClampAmplitude, maxClampAmplitude = -100.0, 100.0  # field: (-100.0,100.0); ligand: (0.0,1.0)
minClampFrequency, maxClampFrequency = 100.0, 1000.0
clampedCellsProp = 1.0
if clampedCellsProp == 0.0:
    clampMode = None
clampDurationProp = 0.5
perturbMode = None  # possible values: tissueDome, tissueDomePartial, None
perturbStartIter, perturbEndIter = 12000, 13000
perturbedCellsProp = 0.0
numSamples = 10
numSimIters = 1000
RandomizeInitialIonChannelState = True
RandomizeInitialField = False
stochasticIonChannels = False
BlockGapJunctions = False
AmplifyGapJunctions = False

VmemBins = np.arange(-0.0, -0.1, -0.04)

fieldParameters = dict()
fieldParameters['fieldEnabled'] = fieldEnabled
fieldParameters['fieldResolution'] = fieldResolution
fieldParameters['fieldStrength'] = fieldStrength
fieldParameters['fieldAggregation'] = 'average'
fieldParameters['fieldScreenSize'] = fieldScreenSize
fieldParameters['fieldTransductionWeight'] = fieldTransductionWeight
fieldParameters['fieldTransductionBias'] = fieldTransductionBias
fieldParameters['fieldTransductionTimeConstant'] = fieldTransductionTimeConstant
ligandParameters = dict()
ligandParameters['ligandEnabled'] = ligandEnabled
ligandParameters['ligandGatingWeight'] = ligandGatingWeight
ligandParameters['ligandGatingBias'] = ligandGatingBias
ligandParameters['ligandCurrentStrength'] = ligandCurrentStrength

modelparameters = dict()
modelparameters['fieldParameters'] = fieldParameters
modelparameters['GJParameters'] = None
modelparameters['GRNParameters'] = None
modelparameters['ligandParameters'] = ligandParameters

circuit = cellularFieldNetwork(latticeDims=latticeDims,parameters=modelparameters,numSamples=numSamples)

numCells = circuit.numCells
numExtracellularGridPoints = circuit.numExtracellularGridPoints

initialValues = dict()
initVmem = list(chain([-9.2e-3] * numSamples))
# initVmem = list(chain([-0.03] * numSamples))
initialValues['Vmem'] = torch.repeat_interleave(torch.DoubleTensor(initVmem),numCells,0).view(numSamples,numCells,1)
if RandomizeInitialField:
    initialValues['eV'] = torch.rand((numSamples,numExtracellularGridPoints,1),dtype=torch.float64)
else:
    initialValues['eV'] = torch.zeros((numSamples,numExtracellularGridPoints,1),dtype=torch.float64)
initialValues['ligandConc'] = torch.zeros((numSamples,numCells,1),dtype=torch.float64)
initialValues['G_pol'] = dict()
AllCells = list(range(numCells))
initialValues['G_pol']['cells'] = [[AllCells]] * numSamples
if RandomizeInitialIonChannelState:
    initialValues['G_pol']['values'] = [[torch.rand(numCells,dtype=torch.float64)*2] for _ in  range(numSamples)]  # covers a range of unistable and bistable values
else:
    initialValues['G_pol']['values'] = [torch.DoubleTensor([1.0])] * numSamples  # bistable
initialValues['G_dep'] = dict()
initialValues['G_dep']['cells'] = []
initialValues['G_dep']['values'] = torch.DoubleTensor([])

circuit.initVariables(initialValues)
circuit.initParameters(initialValues)
circuit.G_0 = GapJunctionStrength * circuit.G_ref
# circuit.relativePermittivity = RelativePermittivity

utils = utilities.utilities()
fieldDomeIndices = utils.computeDomeIndices(circuit,mode='field')

# block gap junctions by zeroing GJ current
if BlockGapJunctions:
    circuit.G_0 = 0.0
    circuit.G_res = 0.0
elif AmplifyGapJunctions:
    circuit.G_0 = 0.05 * circuit.G_ref
    circuit.G_res = 0.0

print("Initial Vmem:")
print(circuit.Vmem.view(numSamples,*latticeDims))
timeseriesVmem = torch.DoubleTensor([-999]*numSimIters*numSamples*numCells).view(numSimIters,numSamples,numCells,1)
timeserieseV = torch.DoubleTensor([-999]*numSimIters*numSamples*numExtracellularGridPoints).view(numSimIters,numSamples,numExtracellularGridPoints,1)

if clampMode == 'field':
    numTotalCells = circuit.numExtracellularGridPoints
    cellIndices = np.arange(numTotalCells)
elif clampMode == 'fieldDome':
    numTotalCells = len(fieldDomeIndices)
    cellIndices = fieldDomeIndices
elif clampMode == 'fieldDomeTwoFoldSymmetry':
    fieldDomeLeftHalfIndices = utils.computeDomeIndices(circuit,mode='field',region='leftHalf')
    numTotalCells = len(fieldDomeLeftHalfIndices)
    cellIndices = fieldDomeLeftHalfIndices
elif (clampMode == 'tissueDomeVmem') or (clampMode == 'tissueDomeLigand') or (clampMode == 'tissueDomeGpol'):
    tissueDomeIndices = utils.computeDomeIndices(circuit,mode='tissue')
    numTotalCells = len(tissueDomeIndices)
    cellIndices = tissueDomeIndices
elif (clampMode == 'tissueVmem') or (clampMode == 'tissueLigand') or (clampMode == 'tissueGpol'):
    numTotalCells = circuit.numCells
    cellIndices = np.arange(numTotalCells)
elif clampMode == 'tissueDomeLigandTwoFoldSymmetry':
    tissueDomeLeftHalfIndices = utils.computeDomeIndices(circuit,mode='tissue',region='leftHalf')
    numTotalCells = len(tissueDomeLeftHalfIndices)
    cellIndices = tissueDomeLeftHalfIndices

if clampMode != None:
    numClampPoints = int(clampedCellsProp*numTotalCells)
    clampPointIndices = np.array([np.random.choice(cellIndices,numClampPoints,replace=False)
                                             for _ in range(numSamples)]).reshape(numSamples,-1).tolist()
    sampleIndices = np.repeat(range(numSamples),numClampPoints).reshape(numSamples,numClampPoints).tolist()
    # clampIndices = (sampleIndices,clampPointIndices)
    clampStartIter, clampEndIter = 0, int(clampDurationProp * numSimIters)
    numClampIters = clampEndIter - clampStartIter + 1
    timeIndices = torch.linspace(0,0.5,numClampIters).view(-1,1)
    if clampType == 'oscillatory':
        clampValues = torch.zeros(numClampIters).view(numClampIters,1)
        for sample in range(numSamples):
            clampFrequencies = torch.rand(numClampPoints,dtype=torch.double)*(maxClampFrequency-minClampFrequency) + minClampFrequency
            clampPhases = torch.rand(numClampPoints,dtype=torch.double)*2*torch.pi
            if 'Symmetry' in clampMode:
                if 'FourFoldSymmetry' in clampMode:
                    if 'field' in clampMode:
                        verticalReflectedIndices, horizontalReflectedIndices, diagonalReflectedIndices = \
                            utils.computeSymmetricalIndices(circuit,clampPointIndices[sample],mode='field',symmetry='fourfold')
                    clampFrequenciesActual = torch.tile(clampFrequencies,(4,))
                    clampPhasesActual = torch.tile(clampPhases,(4,))
                    clampPointIndices[sample] = np.concatenate((clampPointIndices[sample],verticalReflectedIndices,horizontalReflectedIndices,
                                                                diagonalReflectedIndices))
                elif 'TwoFoldSymmetry' in clampMode:
                    if 'field' in clampMode:
                        verticalReflectedIndices = utils.computeSymmetricalIndices(circuit,clampPointIndices[sample],mode='field',symmetry='twofold')
                    elif 'tissue' in clampMode:
                        verticalReflectedIndices = utils.computeSymmetricalIndices(circuit,clampPointIndices[sample],mode='tissue',symmetry='twofold')
                    clampFrequenciesActual = torch.tile(clampFrequencies,(2,))
                    clampPhasesActual = torch.tile(clampPhases,(2,))
                    clampPointIndices[sample] = np.concatenate((clampPointIndices[sample],verticalReflectedIndices))
                _, uniqueClampPointIndices = np.unique(clampPointIndices[sample],return_index=True)  # first-occurrence indices
                clampPointIndices[sample] = clampPointIndices[sample][uniqueClampPointIndices]  # this will always be a sorted array
                numClampPoints = len(clampPointIndices[sample])
                sampleIndices[sample] = np.repeat(sample,numClampPoints)
                # clampIndices = (sampleIndices,clampPointIndices)
                clampValuesSample = torch.cos(timeIndices*clampFrequenciesActual + clampPhasesActual)
                clampValuesSample = ((clampValuesSample+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
                clampValuesSample = clampValuesSample[:,uniqueClampPointIndices]
            else:
                clampValuesSample = torch.cos(timeIndices*clampFrequencies + clampPhases)
                clampValuesSample = ((clampValuesSample+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
            clampValues = torch.hstack((clampValues,clampValuesSample))
        clampValues = clampValues[:,1:]  # first column is a junk buffer
        sampleIndices = np.concatenate(sampleIndices)
        clampPointIndices = np.concatenate(clampPointIndices)
    elif clampType == 'staticConstant':
        clampValues = (torch.ones(numSamples*numClampPoints*numClampIters,dtype=torch.double)*clampValue).view(numClampIters,numClampPoints)
    elif clampType == 'staticRandom':
        clampValues = (torch.rand(numSamples*numClampPoints*numClampIters,dtype=torch.double)*clampValue).view(numClampIters,numClampPoints)
    clampIndices = (sampleIndices,clampPointIndices)
else:
    clampParameters = None

if clampMode != None:
    clampParameters = dict()
    clampParameters['clampMode'] = clampMode
    clampParameters['clampIndices'] = clampIndices
    clampParameters['clampValues'] = clampValues
    clampParameters['clampStartIter'] = clampStartIter
    clampParameters['clampEndIter'] = clampEndIter
else:
    clampParameters = None

if perturbMode != None:
    numPerturbedCells = int(perturbedCellsProp*numTotalCells)
    perturbPointIndices = np.array([np.random.choice(cellIndices,numPerturbedCells,replace=False)
                                             for _ in range(numSamples)]).reshape(-1,)
    perturbSampleIndices = np.repeat(range(numSamples),numPerturbedCells)
    perturbIndices = (perturbSampleIndices,perturbPointIndices)
    perturbationParameters = (perturbStartIter,perturbEndIter,perturbIndices)
else:
    perturbationParameters = None

externalInputs = {'gene':None}
circuit.simulate(externalInputs=externalInputs,clampParameters=clampParameters,perturbationParameters=None,
				 numSimIters=numSimIters,stochasticIonChannels=False,setGradient=False,retainGradients=False,saveData=True)
print("\nFinal Vmem:")
np.set_printoptions(precision=2, suppress=True)  # suppresses scientific notation such as the suffix in 100e+02
print(circuit.Vmem.view(numSamples,*latticeDims))
# counts = [np.unique(np.digitize(circuit.Vmem.round(decimals=2)[i],VmemBins),return_counts=True)[1] for i in range(circuit.Vmem.shape[0])]
# print(*counts,sep='\n')
# counts = torch.unique(circuit.Vmem.round(decimals=2),return_counts=True)
# print("\nCounts of unique Vmems: ",counts)