from model import model
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import utilities

circuitRows,circuitCols = latticeDims = (11,11)
hardCodeInitSingleCell = False
hardCodeInitTissue = False
fieldEnabled = True
ligandEnabled = False
GRNEnabled = True
AsymmetricInterGRN = False
PCPAxes = None
fieldResolution = 1
fieldStrength = 1.0  # default value 1.0
fieldScreenSize = 4 # max: 2*max(latticeDims) - 1 relative to the cell at the corner of the lattice
fieldTransductionBias = torch.DoubleTensor([0.0005])  # 0.0214
fieldTransductionWeight = torch.DoubleTensor([1000.0])  # 9.4505
fieldTransductionGain = -1.0
fieldTransductionTimeConstant = torch.DoubleTensor([10.0])
fieldRangeSymmetric = False
fieldVector = True
# RelativePermittivity = 10**5
GapJunctionStrength = 0.05  # meaningful values in [0.05,1.0]
ligandGatingWeight = torch.DoubleTensor([10.0])  # default: 1.0
ligandGatingBias = torch.DoubleTensor([-0.5])  # default: 0.0
ligandDiffusionStrength = torch.DoubleTensor([1.0])  # default: 1.0
vmemToLigandTransductionWeight = torch.DoubleTensor([1.0])  # default: 1.0
clampMode = 'learned'  # possible values: learned, field, fieldDome, fieldDomeTwoFoldSymmetry, tissueVmem, tissueDomeVmem, tissueLigand, tissueDomeLigand, tissueGpol, tissueDomeGpol, None
if clampMode is not None:
    if clampMode != 'learned':
        clampType = 'oscillatory'  # possible values: oscillatory, staticConstant, staticRandom
        clampValue = 1.0  # static clamp value when clampType = 'staticConstant'
        minClampAmplitude, maxClampAmplitude = -1000.0, 1000.0  # field: (-100.0,100.0); ligand: (0.0,1.0)
        minClampFrequency, maxClampFrequency = 100.0, 1000.0
        clampedCellsProp = 1.0
        if clampedCellsProp == 0.0:
            clampMode = None
        clampDurationProp = 0.1
perturbationMode = None  # possible values: setLigand, tissueDome, tissueDomePartial, None
numSamples = 1
numSimIters = 1000
RandomizeInitialIonChannelState = False
RandomizeInitialField = False
stochasticIonChannels = False
BlockGapJunctions = False
AmplifyGapJunctions = False
numCells = circuitRows * circuitCols

VmemBins = np.arange(-0.0, -0.1, -0.04)

utils = utilities.utilities()

fieldParameters = dict()
fieldParameters['fieldEnabled'] = fieldEnabled
fieldParameters['fieldResolution'] = fieldResolution
fieldParameters['fieldStrength'] = fieldStrength
fieldParameters['fieldAggregation'] = 'average'
fieldParameters['fieldScreenSize'] = fieldScreenSize
fieldParameters['fieldTransductionWeight'] = fieldTransductionWeight
fieldParameters['fieldTransductionBias'] = fieldTransductionBias
fieldParameters['fieldTransductionGain'] = fieldTransductionGain
fieldParameters['fieldTransductionTimeConstant'] = fieldTransductionTimeConstant
fieldParameters['fieldRangeSymmetric'] = fieldRangeSymmetric
fieldParameters['fieldVector'] = fieldVector
GJParameters = dict()
GJParameters['GJStrength'] = GapJunctionStrength
ligandParameters = dict()
ligandParameters['ligandEnabled'] = ligandEnabled
ligandParameters['ligandGatingWeight'] = ligandGatingWeight
ligandParameters['ligandGatingBias'] = ligandGatingBias
ligandParameters['ligandDiffusionStrength'] = ligandDiffusionStrength
ligandParameters['vmemToLigandTransductionWeight'] = vmemToLigandTransductionWeight

def GenerateRandomGRNModel():
    numGenes = torch.randint(minNumGenes,maxNumGenes+1,(1,)).item()
    numVariables = numGenes * numCells
    GRNtoVmemWeights = torch.rand(1,numGenes)*(maxWeight-minWeight) + torch.DoubleTensor([minWeight])
    GRNGains = torch.rand(1,numGenes)*(maxGRNGain-minGRNGain) + torch.DoubleTensor([minGRNGain])
    # GRNGains = torch.ones(1,numGenes)
    GRNBiases = torch.rand(1,numGenes)*(maxGRNBias-minGRNBias) + torch.DoubleTensor([minGRNBias])
    GRNtoVmemWeightsTimeconstant = torch.rand(1)*(maxWeightTimeconstant-minWeightTimeconstant) + torch.DoubleTensor([minWeightTimeconstant])
    GRNWeights = torch.rand(numGenes,numGenes)*(maxWeight-minWeight) + torch.DoubleTensor([minWeight])
    GRNWeights[range(numGenes),range(numGenes)] = 0.0  # remove self-loops
    minGRNEdges, maxGRNEdges = metaParameterMinMaxMap['NumGRNEdges'](numGenes)
    numGRNEdges = torch.randint(minGRNEdges,maxGRNEdges+1,(1,)).item()
    edgeIndicesX, edgeIndicesY = torch.where(GRNWeights != 0.0)
    numEdgesToRemove = numGenes*(numGenes-1) - numGRNEdges
    removeIndices = np.random.choice(range(len(edgeIndicesX)),numEdgesToRemove)
    for i in removeIndices:
        GRNWeights[edgeIndicesX[i],edgeIndicesY[i]] = 0.0
    InterGRNWeights = torch.rand(numGenes,numGenes)*(maxWeight-minWeight) + torch.DoubleTensor([minWeight])
    minInterGRNEdges, maxInterGRNEdges = metaParameterMinMaxMap['NumInterGRNEdges'](numGenes)
    numInterGRNEdges = torch.randint(minInterGRNEdges,maxInterGRNEdges+1,(1,)).item()
    edgeIndicesX, edgeIndicesY = torch.where(InterGRNWeights != 0.0)
    numEdgesToRemove = numGenes*(numGenes) - numInterGRNEdges
    removeIndices = np.random.choice(range(len(edgeIndicesX)),numEdgesToRemove)
    for i in removeIndices:
        InterGRNWeights[edgeIndicesX[i],edgeIndicesY[i]] = 0.0
    # InterGRNWeights = torch.zeros(numGenes,numGenes)
    # VmemToGRNWeights = torch.zeros(1,numGenes)  # no Vmem->GRN connectivity
    VmemToGRNWeights = torch.rand(1,numGenes)*(maxWeight-minWeight) + torch.DoubleTensor([minWeight])
    minVmemToGRNEdges, maxVmemToGRNEdges = metaParameterMinMaxMap['NumVmemToGRNEdges'](numGenes)
    numVmemToGRNEdges = torch.randint(minVmemToGRNEdges,maxVmemToGRNEdges+1,(1,)).item()
    edgeIndicesX, edgeIndicesY = torch.where(VmemToGRNWeights != 0.0)
    numEdgesToRemove = numGenes - numVmemToGRNEdges
    removeIndices = np.random.choice(range(len(edgeIndicesX)),numEdgesToRemove)
    for i in removeIndices:
        VmemToGRNWeights[edgeIndicesX[i],edgeIndicesY[i]] = 0.0
    GRNToVmemWeights = torch.rand(1,numGenes)*(maxWeight-minWeight) + torch.DoubleTensor([minWeight])
    minGRNToVmemEdges, maxGRNToVmemEdges = metaParameterMinMaxMap['NumGRNToVmemEdges'](numGenes)
    numGRNToVmemEdges = torch.randint(minGRNToVmemEdges,maxGRNToVmemEdges+1,(1,)).item()
    edgeIndicesX, edgeIndicesY = torch.where(GRNToVmemWeights != 0.0)
    numEdgesToRemove = numGenes*(numGenes) - numGRNToVmemEdges
    removeIndices = np.random.choice(range(len(edgeIndicesX)),numEdgesToRemove)
    for i in removeIndices:
        GRNToVmemWeights[edgeIndicesX[i],edgeIndicesY[i]] = 0.0
    VmemGain = torch.rand(1)*(maxVmemGain-minVmemGain) + torch.DoubleTensor([minVmemGain])
    VmemBias = torch.rand(1)*(maxVmemBias-minVmemBias) + torch.DoubleTensor([minVmemBias])
    GRNTimeconstants = torch.rand(1,numGenes)*(maxTimeconstant-minTimeconstant) + torch.DoubleTensor([minTimeconstant])
    # GRNTimeconstants = torch.ones(1,numGenes)
    InterGRNWeightsTimeconstant = torch.rand(1)*(maxWeightTimeconstant-minWeightTimeconstant) + torch.DoubleTensor([minWeightTimeconstant])
    VmemToGRNWeightsTimeconstant = torch.rand(1)*(maxWeightTimeconstant-minWeightTimeconstant) + torch.DoubleTensor([minWeightTimeconstant])
    parameters = numGenes, AsymmetricInterGRN, PCPAxes, \
                    GRNtoVmemWeights, GRNtoVmemWeightsTimeconstant, \
                    GRNWeights, InterGRNWeights, VmemToGRNWeights, VmemGain, GRNGains, GRNBiases, VmemBias, \
                    GRNTimeconstants, InterGRNWeightsTimeconstant, VmemToGRNWeightsTimeconstant
    return parameters

if GRNEnabled:
    minNumGenes, maxNumGenes = 1, 6
    if AsymmetricInterGRN:  # include the PCP genes; 2 of them for a 2D lattice: horizontal and vertical (2 for each with opp signs)
        numPCPGenes = 2
        minNumGenes, maxNumGenes = minNumGenes + numPCPGenes, maxNumGenes + numPCPGenes
    minTimeconstant, maxTimeconstant = 0.1, 30
    minWeightTimeconstant, maxWeightTimeconstant = 0.5, 2
    minWeight, maxWeight = -16, 16
    minGRNGain, maxGRNGain = -1, 1
    minGRNBias, maxGRNBias = -16, 16
    minVmemGain, maxVmemGain = 0, 7
    minVmemBias, maxVmemBias = -0.1, 0
    metaParameterMinMaxFunctions = [lambda numGenes: (minNumGenes,maxNumGenes),
                                     lambda numGenes: (numGenes-1, numGenes*(numGenes-1)),
                                     lambda numGenes: (1, numGenes**2),
                                     lambda numGenes: (1, numGenes),
                                     lambda numGenes: (1, numGenes)]
    metaParameterNames = ['NumGenes','NumGRNEdges','NumInterGRNEdges','NumGRNToVmemEdges','NumVmemToGRNEdges']
    metaParameterMinMaxMap = dict(zip(metaParameterNames,metaParameterMinMaxFunctions))
    parameterList = GenerateRandomGRNModel()
    numGenes, AsymmetricInterGRN, PCPAxes, \
    GRNtoVmemWeights, GRNtoVmemWeightsTimeconstant, \
    GRNWeights, InterGRNWeights, VmemToGRNWeights, VmemGain, GRNGains, GRNBiases, VmemBias, \
    GRNTimeconstants, InterGRNWeightsTimeconstant, VmemToGRNWeightsTimeconstant = parameterList
    GRNParameters = dict()
    GRNParameters['GRNEnabled'] = True
    GRNParameters['GRNNumGenes'] = numGenes
    GRNParameters['GRNWeights'] = GRNWeights
    GRNParameters['GRNBiases'] = GRNBiases
    GRNParameters['InterGRNWeights'] = InterGRNWeights
    GRNParameters['GRNtoVmemWeights'] = GRNtoVmemWeights
    GRNParameters['VmemToGRNWeights'] = VmemToGRNWeights
    GRNParameters['VmemGain'] = VmemGain
    GRNParameters['GRNGains'] = GRNGains
    GRNParameters['VmemBias'] = VmemBias
    GRNParameters['GRNTimeconstants'] = GRNTimeconstants
    GRNParameters['InterGRNWeightsTimeconstant'] = InterGRNWeightsTimeconstant
    GRNParameters['GRNtoVmemWeightsTimeconstant'] = GRNtoVmemWeightsTimeconstant
    GRNParameters['VmemToGRNWeightsTimeconstant'] = VmemToGRNWeightsTimeconstant
    GRNParameters['AsymmetricInterGRN'] = AsymmetricInterGRN
    GRNParameters['PCPAxes'] = PCPAxes
else:
    GRNParameters = None

def setupExperimentalConditions(model,RandomizeInitialField=False,RandomizeInitialIonChannelState=False):
    numFieldGridPoints = model.electricNetwork.numFieldGridPoints
    initialValues = dict()
    initVmem = list(chain([-9.2e-3] * numSamples))
    # initVmem = list(chain([-0.055] * numSamples))
    initialValues['Vmem'] = torch.repeat_interleave(torch.DoubleTensor(initVmem),numCells,0).view(numSamples,numCells,1)
    if RandomizeInitialField:
        initialValues['eV'] = torch.rand((numSamples,numFieldGridPoints,1),dtype=torch.float64)
    else:
        initialValues['eV'] = torch.zeros((numSamples,numFieldGridPoints,1),dtype=torch.float64)
    initialValues['ligandConc'] = torch.zeros((numSamples,numCells,1),dtype=torch.float64)
    initialValues['G_pol'] = dict()
    AllCells = list(range(numCells))
    initialValues['G_pol']['cells'] = [[AllCells]] * numSamples
    if RandomizeInitialIonChannelState:
        initialValues['G_pol']['values'] = [[torch.rand(numCells,dtype=torch.float64)*2] for _ in  range(numSamples)]  # covers a range of unistable and bistable values
    else:
        initialValues['G_pol']['values'] = [torch.DoubleTensor([1.0])] * numSamples  # 0.0=dep(-5mV);1.0=bistable;2.0=hyp(-55mV)
    initialValues['G_dep'] = dict()
    initialValues['G_dep']['cells'] = []
    initialValues['G_dep']['values'] = torch.DoubleTensor([])
    experimentalConditions = (initialValues, numSamples)
    return experimentalConditions

parameters = dict()
parameters['latticeDims'] = latticeDims
parameters['fieldParameters'] = fieldParameters
parameters['GJParameters'] = GJParameters
parameters['GRNParameters'] = GRNParameters
parameters['ligandParameters'] = ligandParameters

model = model(parameters,numSamples)

if clampMode == 'field':
    numTotalCells = model.electricNetwork.numFieldGridPoints
    cellIndices = np.arange(numTotalCells)
elif clampMode == 'fieldDome':
    fieldDomeIndices = utils.computeDomeIndices(model,mode='field')
    numTotalCells = len(fieldDomeIndices)
    cellIndices = fieldDomeIndices
elif clampMode == 'fieldDomeTwoFoldSymmetry':
    fieldDomeLeftHalfIndices = utils.computeDomeIndices(model,mode='field',region='leftHalf')
    numTotalCells = len(fieldDomeLeftHalfIndices)
    cellIndices = fieldDomeLeftHalfIndices
elif (clampMode == 'tissueDomeVmem') or (clampMode == 'tissueDomeLigand') or (clampMode == 'tissueDomeGpol'):
    tissueDomeIndices = utils.computeDomeIndices(model,mode='tissue')
    numTotalCells = len(tissueDomeIndices)
    cellIndices = tissueDomeIndices
elif (clampMode == 'tissueVmem') or (clampMode == 'tissueLigand') or (clampMode == 'tissueGpol'):
    numTotalCells = model.electricNetwork.numCells
    cellIndices = np.arange(numTotalCells)
elif clampMode == 'tissueDomeLigandTwoFoldSymmetry':
    tissueDomeLeftHalfIndices = utils.computeDomeIndices(model,mode='tissue',region='leftHalf')
    numTotalCells = len(tissueDomeLeftHalfIndices)
    cellIndices = tissueDomeLeftHalfIndices

if clampMode == 'learned':  # retrieve clamp parameters from a file
    parameterfilename = './data/StigmergicModelParameters.dat'
    clampParametersData = torch.load(parameterfilename)
elif clampMode != None:
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
                            utils.computeSymmetricalIndices(model,clampPointIndices[sample],mode='field',symmetry='fourfold')
                    clampFrequenciesActual = torch.tile(clampFrequencies,(4,))
                    clampPhasesActual = torch.tile(clampPhases,(4,))
                    clampPointIndices[sample] = np.concatenate((clampPointIndices[sample],verticalReflectedIndices,horizontalReflectedIndices,
                                                                diagonalReflectedIndices))
                elif 'TwoFoldSymmetry' in clampMode:
                    if 'field' in clampMode:
                        verticalReflectedIndices = utils.computeSymmetricalIndices(model,clampPointIndices[sample],mode='field',symmetry='twofold')
                    elif 'tissue' in clampMode:
                        verticalReflectedIndices = utils.computeSymmetricalIndices(model,clampPointIndices[sample],mode='tissue',symmetry='twofold')
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

if clampMode == 'learned':
    clampParameters = clampParametersData['clampParameters']
elif clampMode != None:
    clampParameters = dict()
    clampParameters['clampMode'] = clampMode
    clampParameters['clampIndices'] = clampIndices
    clampParameters['clampValues'] = clampValues
    clampParameters['clampStartIter'] = clampStartIter
    clampParameters['clampEndIter'] = clampEndIter
else:
    clampParameters = None

if perturbationMode == 'setLigand':
    perturbation = dict()
    # indices = [0]
    indices = np.arange(model.electricNetwork.numCells)
    perturbPointIndicesA = np.tile(indices,numSamples)
    perturbPointIndicesB = None
    # perturbValues = 1.0
    perturbValues = torch.tensor(np.random.rand(len(indices))).view(-1,1)
    perturbStartIter, perturbEndIter = 1000, 1005
    numPerturbPoints = len(perturbPointIndicesA)
    sampleIndices = np.repeat(range(numSamples),numPerturbPoints)  # assuming that there's only one sample in which the eye block is shifted
    perturbation['mode'] = perturbationMode
    perturbation['data'] = (sampleIndices,(perturbPointIndicesA,perturbPointIndicesB),perturbValues)
    perturbation['time'] = (perturbStartIter,perturbEndIter)
else:
    perturbation = None

experimentalConditions = setupExperimentalConditions(model,RandomizeInitialField,RandomizeInitialIonChannelState)
_, numSamples = experimentalConditions
model.setExperimentalConditions(experimentalConditions)
model.simulate(clampParameters=clampParameters,perturbation=perturbation,numSimIters=numSimIters)
