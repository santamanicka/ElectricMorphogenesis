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
parser.add_argument('--fieldStrength', type=float, default=10.0)
parser.add_argument('--fieldAggregation', type=str, default='average')
parser.add_argument('--fieldScreenSize', type=int, default=1)
parser.add_argument('--fieldTransductionWeight', type=float, default=10.0)
parser.add_argument('--fieldTransductionBias', type=float, default=0.03)
parser.add_argument('--ligandEnabled', type=str, default='False')
parser.add_argument('--ligandGatingWeight', type=float, default=10.0)
parser.add_argument('--ligandGatingBias', type=float, default=-0.5)
parser.add_argument('--ligandCurrentStrength', type=float, default=10.0)
parser.add_argument('--ligandCurrentStrengthRange', type=str, default='(1.0,10.0)')
parser.add_argument('--vmemToLigandCurrentStrength', type=float, default=1.0)
parser.add_argument('--vmemToLigandCurrentStrengthRange', type=str, default='(0.1,10.0)')
parser.add_argument('--GJStrength', type=float, default=0.05)
parser.add_argument('--parameterGridSweep', type=str, default='None')
parser.add_argument('--clampMode', type=str, default='field')
parser.add_argument('--clampType', type=str, default='static')
parser.add_argument('--clampValue', type=float, default=1.0)
parser.add_argument('--clampedCellsProp', type=float, default=1.0)
parser.add_argument('--clampDurationProp', type=float, default=0.1)
parser.add_argument('--clampAmplitudeRange', type=str, default='(-1.0,1.0)')
parser.add_argument('--clampFrequencyRange', type=str, default='(100.0,1000.0)')
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

circuitRows,circuitCols = circuitDims = ast.literal_eval(args.latticeDims)
fieldEnabled = ast.literal_eval(args.fieldEnabled)
fieldResolution = args.fieldResolution
fieldStrength = args.fieldStrength
fieldAggregation = args.fieldAggregation
fieldScreenSize = args.fieldScreenSize
fieldTransductionWeight = args.fieldTransductionWeight
fieldTransductionBias = args.fieldTransductionBias
ligandEnabled = ast.literal_eval(args.ligandEnabled)
ligandGatingWeight = args.ligandGatingWeight
ligandGatingBias = args.ligandGatingBias
ligandCurrentStrength = args.ligandCurrentStrength
ligandCurrentStrengthRange = ast.literal_eval(args.ligandCurrentStrengthRange)
vmemToLigandCurrentStrength = args.vmemToLigandCurrentStrength
vmemToLigandCurrentStrengthRange = ast.literal_eval(args.vmemToLigandCurrentStrengthRange)
GJStrength = args.GJStrength
parameterGridSweep = args.parameterGridSweep
clampMode = args.clampMode
clampType = args.clampType
clampValue = args.clampValue
clampedCellsProp = args.clampedCellsProp
clampDurationProp = args.clampDurationProp
minClampAmplitude, maxClampAmplitude = ast.literal_eval(args.clampAmplitudeRange)
minClampFrequency, maxClampFrequency = ast.literal_eval(args.clampFrequencyRange)
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
    initialValues['eV'] = torch.zeros((numSamples,circuit.numExtracellularGridPoints,1),dtype=torch.float64)
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

def computeLoss(method='globalsum'):
    if method == 'globalsum':
        loss = ((targetVmem - circuit.timeseriesVmem[-evalDuration:]) ** 2).sum().sqrt()
    elif method == 'globalmean':
        loss = ((targetVmem - circuit.timeseriesVmem[-evalDuration:]) ** 2).mean().sqrt()
    elif method == 'partitioned':
        observedVmem = circuit.timeseriesVmem[-evalDuration:,:,:,0]  # shape = (numEvalIters,numSamples,numCells)
        lossSkin = ((targetVmem[:,skinIndices,0] - observedVmem[:,:,skinIndices])**2).sum().sqrt() / len(skinIndices)
        lossEyes = ((targetVmem[:,eyeIndices,0] - observedVmem[:,:,eyeIndices])**2).sum().sqrt() / len(eyeIndices)
        lossNose = ((targetVmem[:,noseIndices,0] - observedVmem[:,:,noseIndices])**2).sum().sqrt() / len(noseIndices)
        lossMouth = ((targetVmem[:,mouthIndices,0] - observedVmem[:,:,mouthIndices])**2).sum().sqrt() / len(mouthIndices)
        loss = (lossSkin + lossEyes + lossNose + lossMouth)
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
fieldParameterNames = ['fieldEnabled','fieldResolution','fieldStrength','fieldAggregation','fieldScreenSize',
                       'fieldTransductionWeight','fieldTransductionBias','fieldTransductionTimeConstant']
ligandParameterNames = ['ligandEnabled','ligandGatingWeight','ligandGatingBias','ligandCurrentStrength','vmemToLigandCurrentStrength']
GRNParameterNames = ['GRNtoVmemWeights','GRNBiases','GRNtoVmemWeightsTimeconstant','GRNNumGenes']
clampParameterNames = ['clampMode','clampIndices','clampValues','clampStartIter','clampEndIter']  # clampValues is not included as it'll be generated from clampFrequencies and clampPhases
simParameterNames = ['initialValues','externalInputs','numSamples','numSimIters']
trainParameterNames = ['targetVmem','actualVmem','numLearnIters','lr','evalDurationProp','bestLoss','bestLossHistory','lossMethod']

utils = utilities.utilities()

if parameterGridSweep == 'fixBiasSweepWeightScreenGJ':
    trialData = dict()
for trial in range(1,numLearnTrials+1):
    if 'fieldTransductionWeight' in learnedParameterNames:
        fieldTransductionWeight = torch.rand(1,dtype=torch.double)*2*10 - 10  # a good value is 9.4505
    else:
        fieldTransductionWeight = torch.DoubleTensor([fieldTransductionWeight])
    if 'fieldTransductionBias' in learnedParameterNames:
        maxfieldTransductionBias = 1.0
        minfieldTransductionBias = -maxfieldTransductionBias
        fieldTransductionBias = torch.rand(1,dtype=torch.double)*2*maxfieldTransductionBias - maxfieldTransductionBias  # a good value is 0.0214
    else:
        fieldTransductionBias = torch.DoubleTensor([fieldTransductionBias])
    fieldTransductionTimeConstant = torch.DoubleTensor([10.0])
    if 'ligandGatingWeight' in learnedParameterNames:
        ligandGatingWeight = torch.rand(1,dtype=torch.double)*2*10 - 10  # [-10.0,10.0]
    else:
        ligandGatingWeight = torch.DoubleTensor([ligandGatingWeight])
    if 'ligandGatingBias' in learnedParameterNames:
        maxligandGatingBias = 1.0
        minligandGatingBias = -maxligandGatingBias
        ligandGatingBias = torch.rand(1,dtype=torch.double)*2*maxligandGatingBias - maxligandGatingBias  # [-1.0,1.0]
    else:
        ligandGatingBias = torch.DoubleTensor([ligandGatingBias])
    if 'ligandCurrentStrength' in learnedParameterNames:
        minligandCurrentStrength, maxligandCurrentStrength = ligandCurrentStrengthRange
        ligandCurrentStrength = (torch.rand(1,dtype=torch.double)*(maxligandCurrentStrength-minligandCurrentStrength) +
                                 minligandCurrentStrength) # [min,max]
    else:
        ligandCurrentStrength = torch.DoubleTensor([ligandCurrentStrength])
    if 'vmemToLigandCurrentStrength' in learnedParameterNames:
        minVmemToLigandCurrentStrengthRange, maxVmemToLigandCurrentStrengthRange = vmemToLigandCurrentStrengthRange
        vmemToLigandCurrentStrength = (torch.rand(1,dtype=torch.double)*
                                       (maxVmemToLigandCurrentStrengthRange-minVmemToLigandCurrentStrengthRange) +
                                       minVmemToLigandCurrentStrengthRange) # [min,max]
    else:
        vmemToLigandCurrentStrength = torch.DoubleTensor([vmemToLigandCurrentStrength])
    minClampAmplitude, maxClampAmplitude = torch.DoubleTensor([minClampAmplitude]), torch.DoubleTensor([maxClampAmplitude])
    if clampType == 'static':
        minClampFrequency, maxClampFrequency = 0.0, 0.0

    GRNtoVmemWeights,GRNBiases,GRNtoVmemWeightsTimeconstant,GRNNumGenes = None,None,None,None

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
    parameters['GJParameters'] = GJParameters
    parameters['fieldParameters'] = fieldParameters
    parameters['ligandParameters'] = ligandParameters
    parameters['GRNParameters'] = GRNParameters

    circuit = cellularFieldNetwork(circuitDims,parameters=parameters,numSamples=numSamples)

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
        numTotalCells = circuit.numExtracellularGridPoints
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

    if clampMode != None:
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
            if 'Symmetry' in clampMode:
                if 'FourFoldSymmetry' in clampMode:
                    if 'field' in clampMode:
                        verticalReflectedIndices, horizontalReflectedIndices, diagonalReflectedIndices = \
                            utils.computeSymmetricalIndices(circuit,clampPointIndices,mode='field',symmetry='fourfold')
                    clampFrequenciesActual = torch.tile(clampFrequencies,(4,))
                    clampPhasesActual = torch.tile(clampPhases,(4,))
                    clampPointIndices = np.concatenate((clampPointIndices,verticalReflectedIndices,horizontalReflectedIndices,
                                            diagonalReflectedIndices))
                elif 'TwoFoldSymmetry' in clampMode:
                    if 'field' in clampMode:
                        verticalReflectedIndices = utils.computeSymmetricalIndices(circuit,clampPointIndices,mode='field',symmetry='twofold')
                    elif 'tissue' in clampMode:
                        verticalReflectedIndices = utils.computeSymmetricalIndices(circuit,clampPointIndices,mode='tissue',symmetry='twofold')
                    clampFrequenciesActual = torch.tile(clampFrequencies,(2,))
                    clampPhasesActual = torch.tile(clampPhases,(2,))
                    clampPointIndices = np.concatenate((clampPointIndices,verticalReflectedIndices))
                _, uniqueClampPointIndices = np.unique(clampPointIndices,return_index=True)  # first-occurrence indices
                clampPointIndices = clampPointIndices[uniqueClampPointIndices]  # this will always be a sorted array
                numClampPoints = len(clampPointIndices)
                sampleIndices = np.repeat(range(numSamples),numClampPoints)
                clampIndices = (sampleIndices,clampPointIndices)
                clampValues = torch.cos(timeIndices * clampFrequenciesActual + clampPhasesActual)
                clampValues = ((clampValues+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
                clampValues = clampValues[:,uniqueClampPointIndices]
            else:
                clampValues = torch.cos(timeIndices * clampFrequencies + clampPhases)
                clampValues = ((clampValues+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
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
    optimizer = torch.optim.Rprop(LearnedParameters,lr=lr)
    bestLoss = 99999
    evalDuration = int(evalDurationProp*numSimIters)
    bestModelParameters = dict()
    bestModelParameters['latticeDims'] = circuitDims
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
        Sfx = 'bestModelParameters_'
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
        parameters['GJParameters'] = GJParameters
        parameters['fieldParameters'] = fieldParameters
        parameters['ligandParameters'] = ligandParameters
        parameters['GRNParameters'] = GRNParameters  # just a tuple of Nones at the moment
        circuit = cellularFieldNetwork(circuitDims,parameters=parameters,numSamples=numSamples)
        initialValues = defineInitialValues(circuit)
        circuit.initVariables(initialValues)
        circuit.initParameters(initialValues)
        if 'fieldTransductionBias' in learnedParameterNames:
            fieldTransductionBias.data = torch.clip(fieldTransductionBias.data,minfieldTransductionBias,maxfieldTransductionBias)
        if 'ligandGatingBias' in learnedParameterNames:
            ligandGatingBias.data = torch.clip(ligandGatingBias.data,minligandGatingBias,maxligandGatingBias)
        if 'ligandCurrentStrength' in learnedParameterNames:
            ligandCurrentStrength.data = torch.clip(ligandCurrentStrength.data,minligandCurrentStrength,maxligandCurrentStrength)
        if 'vmemToLigandCurrentStrength' in learnedParameterNames:
            vmemToLigandCurrentStrength.data = torch.clip(vmemToLigandCurrentStrength.data,minVmemToLigandCurrentStrengthRange,maxVmemToLigandCurrentStrengthRange)
        if clampType == 'oscillatory':
            clampFrequencies.data = torch.clip(clampFrequencies.data,minClampFrequency,maxClampFrequency)
            clampPhases.data = torch.clip(clampPhases.data,0.0,2*torch.pi)
            if 'FourFoldSymmetry' in clampMode:
                clampFrequenciesActual = torch.tile(clampFrequencies,(4,))
                clampPhasesActual = torch.tile(clampPhases,(4,))
                clampValues = torch.cos(timeIndices*clampFrequenciesActual + clampPhasesActual)
                clampValues = ((clampValues+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
                clampValues = clampValues[:,uniqueClampPointIndices]
            elif 'TwoFoldSymmetry' in clampMode:
                clampFrequenciesActual = torch.tile(clampFrequencies,(2,))
                clampPhasesActual = torch.tile(clampPhases,(2,))
                clampValues = torch.cos(timeIndices*clampFrequenciesActual + clampPhasesActual)
                clampValues = ((clampValues+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
                clampValues = clampValues[:,uniqueClampPointIndices]
            else:
                clampValues = torch.cos(timeIndices*clampFrequencies + clampPhases)
                clampValues = ((clampValues+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
        elif 'static' in clampType:
            clampValuesStatic.data = torch.clip(clampValuesStatic.data,minClampAmplitude,maxClampAmplitude)
            clampValues = clampValuesStatic.repeat((numClampIters,1))
        clampParameters = dict()
        for param in clampParameterNames:  # learned field parameters will be automatically updated in the model
            clampParameters[param] = eval(param)
        circuit.simulate(externalInputs=externalInputs,clampParameters=clampParameters,perturbationParameters=perturbationParameters,
                         numSimIters=numSimIters,stochasticIonChannels=stochasticIonChannels,setGradient=setGradient,
                         retainGradients=retainGradients,saveData=saveData)
        loss = computeLoss(method=lossMethod)
        currentLoss = loss.data #.round(decimals=2)
        if currentLoss < bestLoss:
            actualVmem = circuit.Vmem
            bestLoss = currentLoss
            bestLossHistory.append((iter,bestLoss.item()))
            for param in GJParameterNames:
                variable = eval(param)
                if torch.is_tensor(variable):
                    bestModelParameters['GJParameters'][param] = variable.detach()
                else:
                    bestModelParameters['GJParameters'][param] = variable
            for param in fieldParameterNames:
                variable = eval(param)
                if torch.is_tensor(variable):
                    bestModelParameters['fieldParameters'][param] = variable.detach()
                else:
                    bestModelParameters['fieldParameters'][param] = variable
            for param in ligandParameterNames:
                variable = eval(param)
                if torch.is_tensor(variable):
                    bestModelParameters['ligandParameters'][param] = variable.detach()
                else:
                    bestModelParameters['ligandParameters'][param] = variable
            for param in GRNParameterNames:
                variable = eval(param)
                if torch.is_tensor(variable):
                    bestModelParameters['GRNParameters'][param] = variable.detach()
                else:
                    bestModelParameters['GRNParameters'][param] = variable
            for param in clampParameterNames:
                variable = eval(param)
                if torch.is_tensor(variable):
                    bestModelParameters['clampParameters'][param] = variable.detach()
                else:
                    bestModelParameters['clampParameters'][param] = variable
            for param in simParameterNames:
                variable = eval(param)
                if torch.is_tensor(variable) and (variable.dim()<=1):
                    bestModelParameters['simParameters'][param] = variable.detach().item()
                else:
                    bestModelParameters['simParameters'][param] = variable
            for param in trainParameterNames:
                variable = eval(param)
                if torch.is_tensor(variable) and (variable.dim()<=1):
                    bestModelParameters['trainParameters'][param] = variable.detach().item()
                else:
                    bestModelParameters['trainParameters'][param] = variable
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

if verbose:
    np.set_printoptions(precision=2,suppress=True)
    print("\nFinal Vmem:")
    print(circuit.Vmem.data.view(numSamples,*circuitDims).detach().numpy())

print("File number ",fileNumber," completed!")


