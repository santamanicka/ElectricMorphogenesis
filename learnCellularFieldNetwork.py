from model import model
import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import utilities
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--latticeDims', type=str, default='(5,5)')
parser.add_argument('--fieldEnabled', type=str, default='True')
parser.add_argument('--fieldResolution', type=int, default=1)
parser.add_argument('--fieldStrength', type=float, default=1.0)
parser.add_argument('--fieldAggregation', type=str, default='average')
parser.add_argument('--fieldScreenSize', type=int, default=1)
parser.add_argument('--fieldTransductionWeight', type=float, default=1000.0)
parser.add_argument('--fieldTransductionBias', type=float, default=0.0005)
parser.add_argument('--fieldTransductionGain', type=float, default=1.0)
parser.add_argument('--fieldRangeSymmetric', type=str, default='False')
parser.add_argument('--fieldVector', type=str, default='False')
parser.add_argument('--ligandEnabled', type=str, default='False')
parser.add_argument('--ligandGatingWeight', type=float, default=10.0)
parser.add_argument('--ligandGatingWeightRange', type=str, default='None')
parser.add_argument('--ligandGatingBias', type=float, default=-0.5)
parser.add_argument('--ligandDiffusionStrength', type=float, default=1.0)
parser.add_argument('--ligandDiffusionStrengthRange', type=str, default='(1.0,10.0)')
parser.add_argument('--vmemToLigandTransductionWeight', type=float, default=1.0)
parser.add_argument('--vmemToLigandTransductionWeightRange', type=str, default='(1.0,10.0)')
parser.add_argument('--GJStrength', type=float, default=0.05)
parser.add_argument('--GRNEnabled', type=str, default='False')
parser.add_argument('--parameterGridSweep', type=str, default='None')
parser.add_argument('--clampMode', type=str, default='field')
parser.add_argument('--clampType', type=str, default='static')
parser.add_argument('--clampValue', type=float, default=1.0)
parser.add_argument('--clampedCellsProp', type=float, default=1.0)
parser.add_argument('--clampDurationProp', type=float, default=0.1)
parser.add_argument('--clampAmplitudeRange', type=str, default='(-1.0,1.0)')
parser.add_argument('--clampFrequencyRange', type=str, default='(100.0,1000.0)')
parser.add_argument('--loadExistingModel', type=str, default='None')
parser.add_argument('--numClampCoreSquares', type=int, default=1)
parser.add_argument('--numSamples', type=int, default=1)
parser.add_argument('--numSimIters', type=int, default=100)
parser.add_argument('--numLearnIters', type=int, default=100)
parser.add_argument('--numLearnTrials', type=int, default=1)
parser.add_argument('--evalDurationProp', type=float, default=0.1)
parser.add_argument('--learnedParameters', type=str, default='None')
parser.add_argument('--lossMethod', type=str, default='global')
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--fileNumber', type=int, default=0)
parser.add_argument('--verbose', type=str, default='True')

args = parser.parse_args()

circuitRows,circuitCols = latticeDims = ast.literal_eval(args.latticeDims)
fieldEnabled = ast.literal_eval(args.fieldEnabled)
fieldResolution = args.fieldResolution
fieldStrength = args.fieldStrength
fieldAggregation = args.fieldAggregation
fieldScreenSize = args.fieldScreenSize
fieldTransductionWeight = args.fieldTransductionWeight
fieldTransductionBias = args.fieldTransductionBias
fieldTransductionGain = args.fieldTransductionGain
fieldRangeSymmetric = ast.literal_eval(args.fieldRangeSymmetric)
fieldVector = ast.literal_eval(args.fieldVector)
ligandEnabled = ast.literal_eval(args.ligandEnabled)
ligandGatingWeight = args.ligandGatingWeight
ligandGatingWeightRange = args.ligandGatingWeightRange
if ligandGatingWeightRange != 'None':
    minligandGatingWeight, maxligandGatingWeight = ast.literal_eval(args.ligandGatingWeightRange)
else:
    minligandGatingWeight, maxligandGatingWeight = 0.0, 0.0
ligandGatingBias = args.ligandGatingBias
ligandDiffusionStrength = args.ligandDiffusionStrength
ligandDiffusionStrengthRange = ast.literal_eval(args.ligandDiffusionStrengthRange)
vmemToLigandTransductionWeight = args.vmemToLigandTransductionWeight
vmemToLigandTransductionWeightRange = ast.literal_eval(args.vmemToLigandTransductionWeightRange)
GJStrength = args.GJStrength
GRNEnabled = ast.literal_eval(args.GRNEnabled)
parameterGridSweep = args.parameterGridSweep
clampMode = args.clampMode
clampType = args.clampType
clampValue = args.clampValue
clampedCellsProp = args.clampedCellsProp
clampDurationProp = args.clampDurationProp
minClampAmplitude, maxClampAmplitude = ast.literal_eval(args.clampAmplitudeRange)
minClampFrequency, maxClampFrequency = ast.literal_eval(args.clampFrequencyRange)
loadExistingModel = args.loadExistingModel
numClampCoreSquares = args.numClampCoreSquares
numSamples = args.numSamples
numSimIters = args.numSimIters
numLearnIters = args.numLearnIters
numLearnTrials = args.numLearnTrials
evalDurationProp = args.evalDurationProp
learnedParameterNames = ast.literal_eval(args.learnedParameters)
lossMethod = args.lossMethod
lr = args.lr
fileNumber = args.fileNumber
verbose = ast.literal_eval(args.verbose)

def defineInitialValues(circuit):
    initialValues = dict()
    initVmem = torch.FloatTensor(list(chain([-9.2e-3] * numSamples)))
    initialValues['Vmem'] = torch.repeat_interleave(initVmem,circuit.numCells,0).double().view(numSamples,circuit.numCells,1)
    initialValues['eV'] = torch.zeros((numSamples,circuit.numFieldGridPoints,1),dtype=torch.float64)
    initialValues['ligandConc'] = torch.zeros((numSamples,circuit.numCells,1),dtype=torch.float64)
    # initialValues['ligandConc'] = torch.rand((numSamples,circuit.numCells,1), dtype=torch.float64)
    # initialValues['ligandConc'] = torch.ones((numSamples,circuit.numCells,1),dtype=torch.float64) * 0.5
    initialValues['G_pol'] = dict()
    initialValues['G_pol']['cells'] = [[[0]]] * numSamples
    initialValues['G_pol']['values'] = [torch.DoubleTensor([1.0])] * numSamples  # bistable
    initialValues['G_dep'] = dict()
    initialValues['G_dep']['cells'] = []
    initialValues['G_dep']['values'] = torch.DoubleTensor([])
    return initialValues

def defineTargetVmem():
    targetVmem = torch.FloatTensor(list(chain([-9.2e-3] * numSamples)))
    targetVmem = torch.repeat_interleave(targetVmem,circuit.numCells,0).view(numSamples,circuit.numCells,1)
    targetVmem[:,skinIndices] = -0.06  # Skin
    targetVmem[:,eyeIndices] = -0.06  # Eyes 1 and 2
    targetVmem[:,noseIndices] = -0.06  # Nose
    targetVmem[:,mouthIndices] = -0.06  # Mouth
    # ## Dot pattern in a 3x3 tissue
    # targetVmem[:,[4]] = -0.0  # Dot
    return targetVmem

def defineTargetdGpol():
    targetdGpol = torch.zeros(numSamples * circuit.numCells).view(numSamples,circuit.numCells,1)
    return targetdGpol

def computeLoss(method='globalsum'):
    if method == 'globalsum':
        loss = ((targetVmem - system.timeseriesVmem[-evalDuration:]) ** 2).sum().sqrt()
    elif method == 'globalmean':
        loss = ((targetVmem - system.timeseriesVmem[-evalDuration:]) ** 2).mean().sqrt()
    elif method == 'partitioned':
        observedVmem = circuit.timeseriesVmem[-evalDuration:,:,:,0]  # shape = (numEvalIters,numSamples,numCells)
        lossSkin = ((targetVmem[:,skinIndices,0] - observedVmem[:,:,skinIndices])**2).sum().sqrt() / len(skinIndices)
        lossEyes = ((targetVmem[:,eyeIndices,0] - observedVmem[:,:,eyeIndices])**2).sum().sqrt() / len(eyeIndices)
        lossNose = ((targetVmem[:,noseIndices,0] - observedVmem[:,:,noseIndices])**2).sum().sqrt() / len(noseIndices)
        lossMouth = ((targetVmem[:,mouthIndices,0] - observedVmem[:,:,mouthIndices])**2).sum().sqrt() / len(mouthIndices)
        loss = (lossSkin + lossEyes + lossNose + lossMouth)
    elif method == 'globalsumWithdGpol':
        dGpolValues = system.timeseriesdGpol[-evalDuration:]
        observedMax = dGpolValues.abs().max()
        normalization = min(0.05, observedMax)
        dGpolValues = dGpolValues * (normalization / observedMax)  # scale it to be comparable to Vmem with expected mean -0.03
        # observed = torch.cat((system.timeseriesVmem[-evalDuration:],dGpolValues),axis=2)
        # loss = ((target - observed)**2).sum().sqrt()
        loss1 = ((targetVmem - system.timeseriesVmem[-evalDuration:]) ** 2).sum().sqrt()
        loss2 = ((0 - dGpolValues) ** 2).sum().sqrt()  # target dG_pol = 0
        loss = (loss1 + loss2) / 2
    return loss

# Simulation parameters (typically fixed, except clampParameters)
perturbationParameters = None
stochasticIonChannels = False
externalInputs = {'gene': None}
setGradient = False
retainGradients = False
saveData = True

if parameterGridSweep == 'fixBiasSweepWeightScreenGJ':
    fieldTransductionWeights = np.linspace(0,50,10)
    fieldScreenSizes = np.array([1,4,10,15,20])
    GJStrengths = np.array([0,0.05,0.1,0.25,0.5,0.6,0.7,0.8,0.9,1.0])
    parameterGrid = [(screensize,gj,weight) for screensize in fieldScreenSizes for gj in GJStrengths for weight in fieldTransductionWeights]
    fieldTransductionTimeConstant = torch.DoubleTensor([10.0])
    parameterCombination = parameterGrid[fileNumber - 1]  # so file numbers can start from 1
    fieldScreenSize = parameterCombination[0]
    GJStrength = parameterCombination[1]
    fieldTransductionWeight = torch.DoubleTensor([parameterCombination[2]])

GJParameterNames = ['GJStrength']
fieldParameterNames = ['fieldEnabled','fieldResolution','fieldStrength','fieldAggregation','fieldScreenSize','fieldTransductionGain',
                       'fieldTransductionWeight','fieldTransductionBias','fieldTransductionTimeConstant','fieldRangeSymmetric','fieldVector']
ligandParameterNames = ['ligandEnabled','ligandGatingWeight','ligandGatingBias','ligandDiffusionStrength','vmemToLigandTransductionWeight']
# GRNParameterNames = ['GRNtoVmemWeights','GRNBiases','GRNtoVmemWeightsTimeconstant','GRNNumGenes']
GRNParameterNames = ['GRNEnabled','GRNNumGenes',
                     'GRNtoVmemWeights','GRNtoVmemWeightsTimeconstant',   # bioelectric parameters
                     'GRNWeights','InterGRNWeights','VmemToGRNWeights','VmemGain','GRNGains','GRNBiases','VmemBias',  # genetic parameters
                     'GRNTimeconstants','InterGRNWeightsTimeconstant','VmemToGRNWeightsTimeconstant',
                     'AsymmetricInterGRN','PCPAxes']
clampParameterNames = ['clampMode','clampIndices','clampValues','clampStartIter','clampEndIter']  # clampValues is not included as it'll be generated from clampFrequencies and clampPhases
simParameterNames = ['initialValues','externalInputs','numSamples','numSimIters']
trainParameterNames = ['targetVmem','actualVmem','numLearnIters','lr','evalDurationProp','bestLoss','bestLossHistory','lossMethod']

utils = utilities.utilities()

if 'fieldTransductionWeight' in learnedParameterNames:
    maxfieldTransductionWeight = 1000
    minfieldTransductionWeight = 10
if 'fieldTransductionBias' in learnedParameterNames:
    if fieldVector:  # range should be negative if fieldTransductionGain is negative, otherwise it should be positive
        maxfieldTransductionBias = 0.1
        minfieldTransductionBias = 0.0
    else:
        maxfieldTransductionBias = 1.0
        minfieldTransductionBias = -maxfieldTransductionBias
if 'ligandGatingWeight' in learnedParameterNames:
    minligandGatingWeight, maxligandGatingWeight = torch.DoubleTensor([minligandGatingWeight]), torch.DoubleTensor([maxligandGatingWeight])
if 'ligandGatingBias' in learnedParameterNames:
    maxligandGatingBias = 1.0
    minligandGatingBias = 0.0
if 'ligandDiffusionStrength' in learnedParameterNames:
    minligandDiffusionStrength, maxligandDiffusionStrength = ligandDiffusionStrengthRange
if 'vmemToLigandTransductionWeight' in learnedParameterNames:
    minVmemToLigandTransductionWeightRange, maxVmemToLigandTransductionWeightRange = vmemToLigandTransductionWeightRange

def GenerateRandomGRNModel():
    minNumGenes, maxNumGenes = 1, 6
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
    numGenes = torch.randint(minNumGenes,maxNumGenes+1,(1,)).item()
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
    AsymmetricInterGRN = False
    PCPAxes = None
    GRNEnabled = True
    parameters = GRNEnabled, numGenes, AsymmetricInterGRN, PCPAxes, \
                    GRNtoVmemWeights, GRNtoVmemWeightsTimeconstant, \
                    GRNWeights, InterGRNWeights, VmemToGRNWeights, VmemGain, GRNGains, GRNBiases, VmemBias, \
                    GRNTimeconstants, InterGRNWeightsTimeconstant, VmemToGRNWeightsTimeconstant
    return parameters

if parameterGridSweep == 'fixBiasSweepWeightScreenGJ':
    trialData = dict()
for trial in range(1,numLearnTrials+1):
    if loadExistingModel != 'None':
        parameterfilename = './data/' + loadExistingModel
        parameters = torch.load(parameterfilename)
        vars()['fieldParameters'] = parameters['fieldParameters']
        vars()['GJParameters'] = parameters['GJParameters']
        vars()['ligandParameters'] = parameters['ligandParameters']
        vars()['GRNParameters'] = parameters['GRNParameters']
        vars()['clampParameters'] = parameters['clampParameters']
        for param in fieldParameterNames:
            vars()[param] = parameters['fieldParameters'][param]
        for param in GJParameterNames:
            vars()[param] = parameters['GJParameters'][param]
        for param in ligandParameterNames:
            vars()[param] = parameters['ligandParameters'][param]
        for param in GRNParameterNames:
            vars()[param] = parameters['GRNParameters'][param]
        for param in parameters['clampParameters'].keys():
            vars()[param] = parameters['clampParameters'][param]
        GRNWeights = GRNWeights * 0.9
        parameters['GRNParameters']['GRNWeights'] = GRNWeights
        parameters['latticePeriodicBoundary'] = True
        boundaryEdgeDiffusionStrength = 0.3
        boundaryEdgeDiffusionStrength = torch.FloatTensor([boundaryEdgeDiffusionStrength])
        parameters['boundaryEdgeDiffusionStrength'] = boundaryEdgeDiffusionStrength
        system = model(parameters,numSamples)
        circuit = system.electricNetwork
        fieldDomeIndices = utils.computeDomeIndices(circuit,mode='field')
        tissueDomeIndices = utils.computeDomeIndices(circuit,mode='tissue')
        ## Smiley pattern in a 11x11 tissue
        skinIndices = tissueDomeIndices
        eyeIndices = [24,25,35,36,29,30,40,41]  # left and right eyes
        noseIndices = [49,60,71]
        mouthIndices = [92,93,94]
    else:
        if 'fieldTransductionWeight' in learnedParameterNames:
            fieldTransductionWeight = torch.rand(1,dtype=torch.double)*(maxfieldTransductionWeight-minfieldTransductionWeight) + minfieldTransductionWeight # a good value is 9.4505
        else:
            fieldTransductionWeight = torch.DoubleTensor([fieldTransductionWeight])
        if 'fieldTransductionBias' in learnedParameterNames:
            if fieldVector:  # range should be negative if fieldTransductionGain is negative, otherwise it should be positive
                fieldTransductionBias = torch.rand(1,dtype=torch.double)*(maxfieldTransductionBias-minfieldTransductionBias) + minfieldTransductionBias  # so that threshold lies in [0,0.1]
            else:
                fieldTransductionBias = torch.rand(1,dtype=torch.double)*2*maxfieldTransductionBias - maxfieldTransductionBias  # a good value is 0.0214
        else:
            fieldTransductionBias = torch.DoubleTensor([fieldTransductionBias])
        fieldTransductionTimeConstant = torch.DoubleTensor([10.0])
        if 'ligandGatingWeight' in learnedParameterNames:
            ligandGatingWeight = torch.rand(1,dtype=torch.double)*(maxligandGatingWeight-minligandGatingWeight) + minligandGatingWeight
        else:
            ligandGatingWeight = torch.DoubleTensor([ligandGatingWeight])
        if 'ligandGatingBias' in learnedParameterNames:
            ligandGatingBias = torch.rand(1,dtype=torch.double)*(maxligandGatingBias-minligandGatingBias) + minligandGatingBias  # [-1.0,1.0]
        else:
            ligandGatingBias = torch.DoubleTensor([ligandGatingBias])
        if 'ligandDiffusionStrength' in learnedParameterNames:
            ligandDiffusionStrength = (torch.rand(1,dtype=torch.double)*(maxligandDiffusionStrength-minligandDiffusionStrength) +
                                     minligandDiffusionStrength) # [min,max]
        else:
            ligandDiffusionStrength = torch.DoubleTensor([ligandDiffusionStrength])
        if 'vmemToLigandTransductionWeight' in learnedParameterNames:
            vmemToLigandTransductionWeight = (torch.rand(1,dtype=torch.double)*
                                           (maxVmemToLigandTransductionWeightRange-minVmemToLigandTransductionWeightRange) +
                                           minVmemToLigandTransductionWeightRange) # [min,max]
        else:
            vmemToLigandTransductionWeight = torch.DoubleTensor([vmemToLigandTransductionWeight])
        minClampAmplitude, maxClampAmplitude = torch.DoubleTensor([minClampAmplitude]), torch.DoubleTensor([maxClampAmplitude])
        if GRNEnabled:
            parameterList = GenerateRandomGRNModel()
            GRNEnabled, GRNNumGenes, AsymmetricInterGRN, PCPAxes, \
            GRNtoVmemWeights, GRNtoVmemWeightsTimeconstant, \
            GRNWeights, InterGRNWeights, VmemToGRNWeights, VmemGain, GRNGains, GRNBiases, VmemBias, \
            GRNTimeconstants, InterGRNWeightsTimeconstant, VmemToGRNWeightsTimeconstant = parameterList
            GRNParameters = dict()
            for param in GRNParameterNames:
                GRNParameters['param'] = eval(param)
        else:
            for param in GRNParameterNames:
                vars()[param] = None  # dynamically create variables and assign values
            GRNEnabled = False

        GJParameters = dict()
        for param in GJParameterNames:
            GJParameters[param] = eval(param)
        fieldParameters = dict()
        for param in fieldParameterNames:
            fieldParameters[param] = eval(param)
        ligandParameters = dict()
        for param in ligandParameterNames:
            ligandParameters[param] = eval(param)
        GRNParameters = dict()
        for param in GRNParameterNames:
            GRNParameters[param] = eval(param)
        parameters = dict()
        parameters['latticeDims'] = latticeDims
        parameters['GJParameters'] = GJParameters
        parameters['fieldParameters'] = fieldParameters
        parameters['ligandParameters'] = ligandParameters
        parameters['GRNParameters'] = GRNParameters

        system = model(parameters,numSamples)
        circuit = system.electricNetwork
        # circuit = cellularFieldNetwork(latticeDims,parameters=parameters,numSamples=numSamples)

        fieldDomeIndices = utils.computeDomeIndices(circuit,mode='field')
        tissueDomeIndices = utils.computeDomeIndices(circuit,mode='tissue')
        ## Smiley pattern in a 11x11 tissue
        skinIndices = tissueDomeIndices
        eyeIndices = [24,25,35,36,29,30,40,41]  # left and right eyes
        noseIndices = [49,60,71]
        mouthIndices = [92,93,94]

        if clampMode == 'fieldDome':
            numTotalCells = len(fieldDomeIndices)
            cellIndices = fieldDomeIndices
        elif clampMode == 'field':
            numTotalCells = circuit.numFieldGridPoints
            cellIndices = np.arange(numTotalCells)
        elif clampMode == 'fieldDomeFourFoldSymmetry':
            fieldDomeTopLeftQuadrantIndices = utils.computeDomeIndices(circuit,mode='field',region='topLeftQuadrant')
            numTotalCells = len(fieldDomeTopLeftQuadrantIndices)
            cellIndices = fieldDomeTopLeftQuadrantIndices
        elif clampMode == 'fieldDomeTwoFoldSymmetry':
            fieldDomeLeftHalfIndices = utils.computeDomeIndices(circuit,mode='field',region='leftHalf')
            numTotalCells = len(fieldDomeLeftHalfIndices)
            cellIndices = fieldDomeLeftHalfIndices
        elif clampMode == 'fieldDomeLeftHalf':
            fieldDomeLeftHalfIndices = utils.computeDomeIndices(circuit,mode='field',region='leftHalf')
            numTotalCells = len(fieldDomeLeftHalfIndices)
            cellIndices = fieldDomeLeftHalfIndices
        elif clampMode == 'fieldCore':
            fieldCoreIndices = utils.computeCoreIndices(circuit,mode='field',numCoreSquares=numClampCoreSquares)  # 4x4 square
            numTotalCells = len(fieldCoreIndices)
            cellIndices = fieldCoreIndices
            numFieldCells = numTotalCells
        elif (clampMode == 'tissueDomeVmem') or (clampMode == 'tissueDomeLigand') or (clampMode == 'tissueDomeGpol'):
            tissueDomeIndices = utils.computeDomeIndices(circuit,mode='tissue')
            numTotalCells = len(tissueDomeIndices)
            cellIndices = tissueDomeIndices
        elif (clampMode == 'tissueVmem') or (clampMode == 'tissueLigand') or (clampMode == 'tissueGpol'):
            numTotalCells = circuit.numCells
            cellIndices = np.arange(numTotalCells)
        elif clampMode == 'tissueDomeLigandTwoFoldSymmetry':
            fieldDomeLeftHalfIndices = utils.computeDomeIndices(circuit,mode='tissue',region='leftHalf')
            numTotalCells = len(fieldDomeLeftHalfIndices)
            cellIndices = fieldDomeLeftHalfIndices

        if clampMode != "None":
            numClampPoints = int(clampedCellsProp*numTotalCells*numSamples)
            clampPointIndices = np.array([np.random.choice(cellIndices,numClampPoints,replace=False)
                                                     for _ in range(numSamples)]).reshape(-1,)
            sampleIndices = np.repeat(range(numSamples),numClampPoints)
            clampIndices = (sampleIndices,clampPointIndices)
            clampStartIter, clampEndIter = 0, int(clampDurationProp * numSimIters)
            numClampIters = clampEndIter - clampStartIter + 1
            timeIndices = torch.linspace(0,0.5,numClampIters).view(-1,1)
            if clampType == 'oscillatory':
                clampFrequencies = torch.rand(numSamples*numClampPoints,dtype=torch.double)*(maxClampFrequency-minClampFrequency) + minClampFrequency
                clampPhases = torch.rand(numSamples*numClampPoints,dtype=torch.double)*2*torch.pi
                clampAmplitudes = torch.rand(numSamples*numClampPoints,dtype=torch.double)*(maxClampAmplitude-minClampAmplitude) + minClampAmplitude
                if 'Symmetry' in clampMode:
                    if 'FourFoldSymmetry' in clampMode:
                        if 'field' in clampMode:
                            verticalReflectedIndices, horizontalReflectedIndices, diagonalReflectedIndices = \
                                utils.computeSymmetricalIndices(circuit,clampPointIndices,mode='field',symmetry='fourfold')
                        clampFrequenciesActual = torch.tile(clampFrequencies,(4,))
                        clampPhasesActual = torch.tile(clampPhases,(4,))
                        clampAmplitudesActual = torch.tile(clampAmplitudes,(4,))
                        clampPointIndices = np.concatenate((clampPointIndices,verticalReflectedIndices,horizontalReflectedIndices,
                                                diagonalReflectedIndices))
                    elif 'TwoFoldSymmetry' in clampMode:
                        if 'field' in clampMode:
                            verticalReflectedIndices = utils.computeSymmetricalIndices(circuit,clampPointIndices,mode='field',symmetry='twofold')
                        elif 'tissue' in clampMode:
                            verticalReflectedIndices = utils.computeSymmetricalIndices(circuit,clampPointIndices,mode='tissue',symmetry='twofold')
                        clampFrequenciesActual = torch.tile(clampFrequencies,(2,))
                        clampPhasesActual = torch.tile(clampPhases,(2,))
                        clampAmplitudesActual = torch.tile(clampAmplitudes,(2,))
                        clampPointIndices = np.concatenate((clampPointIndices,verticalReflectedIndices))
                    _, uniqueClampPointIndices = np.unique(clampPointIndices,return_index=True)  # first-occurrence indices
                    clampPointIndices = clampPointIndices[uniqueClampPointIndices]  # this will always be a sorted array
                    numClampPoints = len(clampPointIndices)
                    sampleIndices = np.repeat(range(numSamples),numClampPoints)
                    clampIndices = (sampleIndices,clampPointIndices)
                    clampValues = torch.cos(timeIndices * clampFrequenciesActual + clampPhasesActual)
                    if 'Ligand' in clampMode:
                        clampValues = (clampValues + 1) / 2
                    # clampValues = ((clampValues+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
                    # clampValues = ((clampValues+1)/2)*clampFrequenciesActual
                    clampValues = clampValues * clampAmplitudesActual
                    clampValues = clampValues[:,uniqueClampPointIndices]
                else:
                    clampValues = torch.cos(timeIndices * clampFrequencies + clampPhases)
                    if 'Ligand' in clampMode:
                        clampValues = (clampValues + 1) / 2
                    # clampValues = ((clampValues+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
                    # clampValues = ((clampValues+1)/2)*clampFrequencies
                    clampValues = clampValues * clampAmplitudes
            elif clampType == 'staticConstant':
                clampValuesStatic = (torch.ones(numClampPoints,dtype=torch.double)*clampValue)
            elif clampType == 'staticRandom':
                clampValuesStatic = (torch.rand(numClampPoints,dtype=torch.double)*clampValue)
        else:
            clampParameters = None

    LearnedParameters = []
    for parameterName in learnedParameterNames:
        parameter = eval(parameterName)
        parameter.requires_grad = True
        LearnedParameters.append(parameter)

    targetVmem = defineTargetVmem()
    targetdGpol = defineTargetdGpol()
    target = torch.cat((targetVmem,targetdGpol),axis=1)
    optimizer = torch.optim.Rprop(LearnedParameters,lr=lr)
    bestLoss = 99999
    evalDuration = int(evalDurationProp*numSimIters)
    bestModelParameters = dict()
    bestModelParameters['latticeDims'] = latticeDims
    bestModelParameters['GJParameters'] = dict()
    bestModelParameters['fieldParameters'] = dict()
    bestModelParameters['ligandParameters'] = dict()
    bestModelParameters['GRNParameters'] = dict()
    bestModelParameters['clampParameters'] = dict()
    bestModelParameters['simParameters'] = dict()
    bestModelParameters['trainParameters'] = dict()
    bestLossHistory = []
    if parameterGridSweep == 'fixBiasSweepWeightScreenGJ':
        trialData[trial] = bestModelParameters
        Sfx = 'ModelCharacteristics_FixedBias_Patternability_'
    else:
        if fieldVector:
            if ligandEnabled:
                Sfx = 'bestModelParameters_fieldVector_Ligand_'
            else:
                Sfx = 'bestModelParameters_fieldVector_'
        else:
            Sfx = 'bestModelParameters_'
        if GRNEnabled:
            Sfx += 'GRN_'
    savefilename = './data/' + Sfx + str(fileNumber) + '.dat'
    for iter in range(numLearnIters):
        parameters = dict()
        GJParameters = dict()
        for param in GJParameterNames:  # learned field parameters will be automatically updated in the model
            GJParameters[param] = eval(param)
        fieldParameters = dict()
        for param in fieldParameterNames:  # learned field parameters will be automatically updated in the model
            fieldParameters[param] = eval(param)
        ligandParameters = dict()
        for param in ligandParameterNames:  # learned field parameters will be automatically updated in the model
            ligandParameters[param] = eval(param)
        parameters['latticeDims'] = latticeDims
        parameters['GJParameters'] = GJParameters
        parameters['fieldParameters'] = fieldParameters
        parameters['ligandParameters'] = ligandParameters
        parameters['GRNParameters'] = GRNParameters  # just a tuple of Nones at the moment
        parameters['latticePeriodicBoundary'] = True
        parameters['boundaryEdgeDiffusionStrength'] = boundaryEdgeDiffusionStrength
        system = model(parameters,numSamples)
        circuit = system.electricNetwork
        # circuit = cellularFieldNetwork(latticeDims,parameters=parameters,numSamples=numSamples)
        initialValues = defineInitialValues(circuit)
        system.setExperimentalConditions((initialValues,numSamples))
        # circuit.initVariables(initialValues)
        # circuit.initParameters(initialValues)
        if 'fieldTransductionBias' in learnedParameterNames:
            fieldTransductionBias.data = torch.clip(fieldTransductionBias.data,minfieldTransductionBias,maxfieldTransductionBias)
        if 'ligandGatingWeight' in learnedParameterNames:
            ligandGatingWeight.data = torch.clip(ligandGatingWeight.data,minligandGatingWeight,maxligandGatingWeight)
        if 'ligandGatingBias' in learnedParameterNames:
            ligandGatingBias.data = torch.clip(ligandGatingBias.data,minligandGatingBias,maxligandGatingBias)
        if 'ligandDiffusionStrength' in learnedParameterNames:
            ligandDiffusionStrength.data = torch.clip(ligandDiffusionStrength.data,minligandDiffusionStrength,maxligandDiffusionStrength)
        if 'vmemToLigandTransductionWeight' in learnedParameterNames:
            vmemToLigandTransductionWeight.data = torch.clip(vmemToLigandTransductionWeight.data,minVmemToLigandTransductionWeightRange,maxVmemToLigandTransductionWeightRange)
        if loadExistingModel == 'None':  # else clampParameters would have been preloaded
            if clampMode != 'None':
                if clampType == 'oscillatory':
                    clampFrequencies.data = torch.clip(clampFrequencies.data,minClampFrequency,maxClampFrequency)
                    clampPhases.data = torch.clip(clampPhases.data,0.0,2*torch.pi)
                    clampAmplitudes.data = torch.clip(clampAmplitudes.data,minClampAmplitude,maxClampAmplitude)
                    if 'FourFoldSymmetry' in clampMode:
                        clampFrequenciesActual = torch.tile(clampFrequencies,(4,))
                        clampPhasesActual = torch.tile(clampPhases,(4,))
                        clampAmplitudesActual = torch.tile(clampAmplitudes,(4,))
                        clampValues = torch.cos(timeIndices*clampFrequenciesActual + clampPhasesActual)
                        if 'Ligand' in clampMode:
                            clampValues = (clampValues + 1) / 2
                        # clampValues = ((clampValues+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
                        # clampValues = ((clampValues+1)/2)*clampAmplitudesActual
                        clampValues = clampValues * clampAmplitudesActual
                        clampValues = clampValues[:,uniqueClampPointIndices]
                    elif 'TwoFoldSymmetry' in clampMode:
                        clampFrequenciesActual = torch.tile(clampFrequencies,(2,))
                        clampPhasesActual = torch.tile(clampPhases,(2,))
                        clampAmplitudesActual = torch.tile(clampAmplitudes,(2,))
                        clampValues = torch.cos(timeIndices*clampFrequenciesActual + clampPhasesActual)
                        if 'Ligand' in clampMode:
                            clampValues = (clampValues + 1) / 2
                        # clampValues = ((clampValues+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
                        # clampValues = ((clampValues+1)/2)*clampAmplitudesActual
                        clampValues = clampValues * clampAmplitudesActual
                        clampValues = clampValues[:,uniqueClampPointIndices]
                    else:
                        clampValues = torch.cos(timeIndices*clampFrequencies + clampPhases)
                        if 'Ligand' in clampMode:
                            clampValues = (clampValues + 1) / 2
                        # clampValues = ((clampValues+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
                        # clampValues = ((clampValues+1)/2)*clampAmplitudes
                        clampValues = clampValues * clampAmplitudes
                elif 'static' in clampType:
                    clampValuesStatic.data = torch.clip(clampValuesStatic.data,minClampAmplitude,maxClampAmplitude)
                    clampValues = clampValuesStatic.repeat((numClampIters,1))
                clampParameters = dict()
                for param in clampParameterNames:  # learned field parameters will be automatically updated in the model
                    clampParameters[param] = eval(param)
            else:
                clampParameters = None
        system.simulate(clampParameters=clampParameters,perturbation=perturbationParameters,numSimIters=numSimIters)
        # circuit.simulate(externalInputs=externalInputs,clampParameters=clampParameters,perturbationParameters=perturbationParameters,
        #                  numSimIters=numSimIters,stochasticIonChannels=stochasticIonChannels,setGradient=setGradient,
        #                  retainGradients=retainGradients,saveData=saveData)
        loss = computeLoss(method=lossMethod)
        currentLoss = loss.data #.round(decimals=2)
        if currentLoss < bestLoss:
            actualVmem = circuit.Vmem
            bestLoss = currentLoss
            bestLossHistory.append((iter,bestLoss.item()))
            for param in GJParameterNames:
                variable = eval(param)
                if torch.is_tensor(variable):
                    bestModelParameters['GJParameters'][param] = variable.detach().clone()
                else:
                    bestModelParameters['GJParameters'][param] = variable
            for param in fieldParameterNames:
                variable = eval(param)
                if torch.is_tensor(variable):
                    bestModelParameters['fieldParameters'][param] = variable.detach().clone()
                else:
                    bestModelParameters['fieldParameters'][param] = variable
            for param in ligandParameterNames:
                variable = eval(param)
                if torch.is_tensor(variable):
                    bestModelParameters['ligandParameters'][param] = variable.detach().clone()
                else:
                    bestModelParameters['ligandParameters'][param] = variable
            for param in GRNParameterNames:
                variable = eval(param)
                if torch.is_tensor(variable):
                    bestModelParameters['GRNParameters'][param] = variable.detach().clone()
                else:
                    bestModelParameters['GRNParameters'][param] = variable
            if clampMode != 'None':
                for param in clampParameterNames:
                    variable = eval(param)
                    if torch.is_tensor(variable):
                        bestModelParameters['clampParameters'][param] = variable.detach().clone()
                    else:
                        bestModelParameters['clampParameters'][param] = variable
            for param in simParameterNames:
                variable = eval(param)
                if torch.is_tensor(variable) and (variable.dim()<=1):
                    bestModelParameters['simParameters'][param] = variable.detach().clone().item()
                else:
                    bestModelParameters['simParameters'][param] = variable
            for param in trainParameterNames:
                variable = eval(param)
                if torch.is_tensor(variable) and (variable.dim()<=1):
                    bestModelParameters['trainParameters'][param] = variable.detach().clone().item()
                else:
                    bestModelParameters['trainParameters'][param] = variable
            bestModelParameters['latticePeriodicBoundary'] = True
            bestModelParameters['boundaryEdgeDiffusionStrength'] = boundaryEdgeDiffusionStrength.detach().clone().item()
        loss.backward(retain_graph=False)
        optimizer.step()
        optimizer.zero_grad()
        if ((iter+1) % 20) == 0:
            if parameterGridSweep == 'fixBiasSweepWeightScreenGJ':
                torch.save(trialData, savefilename)
            else:
                torch.save(bestModelParameters, savefilename)
        if verbose:
            print(fileNumber,trial,iter,currentLoss.item(),bestLoss.item())

if parameterGridSweep == 'fixBiasSweepWeightScreenGJ':
    torch.save(trialData, savefilename)
else:
    torch.save(bestModelParameters, savefilename)

if verbose:
    np.set_printoptions(precision=2,suppress=True)
    print("\nFinal Vmem:")
    print(circuit.Vmem.data.view(numSamples,*latticeDims).detach().numpy())

print("File number ",fileNumber," completed!")


