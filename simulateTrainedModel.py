from embryo import model
from cellularFieldNetwork import cellularFieldNetwork
import numpy as np
import torch
import utilities
from itertools import chain
import matplotlib.pyplot as plt

Model = 'Stigmergic'  # optoions: 'Stigmergic', 'Mosaic', None

fieldVector = True
fieldRangeSymmetric = False
ligandEnabled = False
GRNEnabled = False
if Model == 'Stigmergic':
    parameterfilename = './data/StigmergicModelParameters.dat'
elif Model == 'Mosaic':
    parameterfilename = './data/MosaicModelParameters.dat'
else:
    filenum = '253'  # weakly sensitive: 1294; strongly sensitive: 1576
    if fieldVector:
        if ligandEnabled:
            Sfx = '_fieldVector_Ligand'
            if GRNEnabled:
                Sfx += '_GRN'
        elif GRNEnabled:
            Sfx = '_fieldVector_GRN'
        else:
            Sfx = '_fieldVector'
    else:
        Sfx = ''
    parameterfilename = './data/bestModelParameters' + Sfx + '_' + str(filenum) + '.dat'  # 472 (fr=4); OLD: 483 (fieldRange=4); 759 (fieldRange=1); 825 (fieldRange=21)

parameters = torch.load(parameterfilename)

numSampleRepeats = 1
Autonomous = False  # impose homogenous initial conditions under unclamped conditions
randomizeInitialState = False  # applies only if Autonomous=True
tempParamsOverride = True
Perturb = False
perturbationMode = 'setGpol'  # options: swapVmem, permuteVmem, permuteVmemBoundary, swapGpol, setFieldTransductionWeight
Freeze = False
activeBlockCellIndexCoords = ((0,0),(7,7))
MultiCircuit = False
newSimulationLength = (False,5000)
newVmemLigandStrength = (False,3.0)
TurnoffField = False
TurnoffLigand = False
TurnoffGRN = False
TurnonATP = True
numSimRuns = 1

latticeDims = parameters['latticeDims']
# GJParameters = parameters['GJParameters']
# fieldParameters = parameters['fieldParameters']
# if 'fieldRangeSymmetric' not in fieldParameters.keys():
#     fieldParameters['fieldRangeSymmetric'] = fieldRangeSymmetric
# if 'fieldVector' not in fieldParameters.keys():
#     fieldParameters['fieldVector'] = False
# if 'fieldTransductionGain' not in fieldParameters.keys():
#     fieldParameters['fieldTransductionGain'] = 1.0
# if 'ligandParameters' in parameters.keys():
#     ligandParameters = parameters['ligandParameters']
#     if 'vmemToLigandTransductionWeight' not in ligandParameters.keys():
#         ligandParameters['vmemToLigandTransductionWeight'] = 1.0
# else:
#     ligandParameters = None
# GRNParameters = parameters['GRNParameters']
numRows, numCols = latticeDims[0], latticeDims[1]
numCells = numRows * numCols
numSamples = parameters['simParameters']['numSamples'] * numSampleRepeats
initialValues = parameters['simParameters']['initialValues']
if 'ligandConc' not in initialValues.keys():
    initialValues['ligandConc'] = torch.zeros((numSamples,numCells,1),dtype=torch.float64)
clampParameters = parameters['clampParameters']
externalInputs = parameters['simParameters']['externalInputs']
numSimIters = parameters['simParameters']['numSimIters']
evalDurationProp = parameters['trainParameters']['evalDurationProp']
targetVmem = parameters['trainParameters']['targetVmem'].repeat((numSamples,1,1))
if 'lossMethod' in parameters['trainParameters'].keys():
    lossMethod = parameters['trainParameters']['lossMethod']
else:
    lossMethod = 'globalsum'
parameters['latticePeriodicBoundaryGJ'] = False
if numSampleRepeats > 1:
    initialValues['G_pol']['cells'] = initialValues['G_pol']['cells'] * numSampleRepeats
    initialValues['G_pol']['values'] = initialValues['G_pol']['values'] * numSampleRepeats
    initialValues['G_dep']['cells'] = initialValues['G_dep']['cells'] * numSampleRepeats
    initialValues['G_dep']['values'] = initialValues['G_dep']['values'] * numSampleRepeats
    clampParameters['clampIndices'] = torch.FloatTensor(clampParameters['clampIndices']).int()
    clampParameters['clampIndices'] = clampParameters['clampIndices'].repeat((1,numSampleRepeats))
    # clampParameters['clampValues'] = torch.FloatTensor(clampParameters['clampValues'])
    clampParameters['clampValues'] = clampParameters['clampValues'].repeat((1,numSampleRepeats))

utils = utilities.utilities()

if TurnoffField:
    parameters['fieldParameters']['fieldEnabled'] = False

if TurnoffLigand:
    parameters['ligandParameters']['ligandEnabled'] = False
elif newVmemLigandStrength[0]:
    parameters['ligandParameters']['vmemToLigandCurrentStrength'] = newVmemLigandStrength[1]

if TurnoffGRN:
    parameters['GRNParameters']['GRNEnabled'] = False

if TurnonATP:
    parameters['ATPParameters'] = dict()
    parameters['ATPParameters']['ATPEnabled'] = True
    parameters['ATPParameters']['ATPReactionStrength'] = 1.0  # 0.0
    parameters['ATPParameters']['ATPDiffusionStrength'] = 10.0  # 10.0
    parameters['ATPParameters']['tissueConnectivity'] = utils.computeLatticeAdjacencyMatrix(latticeDims=parameters['latticeDims'],periodicBoundary=False)
else:
    parameters['ATPParameters'] = None

if tempParamsOverride:
    # numSamples = 10
    # parameters['GRNParameters']['GRNtoLigandWeights'] *= 0.1  # 1.03
    # parameters['GRNParameters']['GRNGains'] *= 0.5  # 1.03
    # parameters['fieldParameters']['fieldTransductionWeight'] /= 10
    parameters['fieldParameters']['fieldTransductionGain'] /= 11.5
    # parameters['GRNParameters']['GRNWeights'] /= 11.5   # 1.03  # set for ATP model 262 to work
    # parameters['GRNParameters']['InterGRNWeights'] /= 11.5  # 0.7, 0.8  # set for ATP model 262 to work
    # parameters['latticePeriodicBoundaryGRN'] = False  # set for ATP model 262 to work
    # parameters['boundaryEdgeDiffusionStrengthGRN'] = 1.0  # 3.0  # set for ATP model 262 to work
    # parameters['latticePeriodicBoundaryLigand'] = False  # set for ATP model 262 to work
    # parameters['boundaryEdgeDiffusionStrengthLigand'] = 1.0  # 0.8  # set for ATP model 262 to work
    # parameters['GRNParameters']['GRNTimeconstants'] *= 1.1
    # clampParameters['clampValues'] *= 2
    # fieldParameters['fieldTransductionWeight'] = 500
    # fieldParameters['fieldTransductionBias'] = torch.DoubleTensor([0.0])
    # fieldParameters['fieldTransductionTimeConstant'] = 10.0
    # fieldParameters['fieldStrength'] *= 4.0
    # fieldParameters['fieldRangeSymmetric'] = True
    # fieldParameters['fieldScreenSize'] = 21
    # GJParameters['GJStrength'] = 1.0

if newSimulationLength[0]:
    numSimIters = newSimulationLength[1]

# indices of the features of the 11x11 smiley
eyeIndices = np.array([24,25,35,36,29,30,40,41])  # left and right eyes
noseIndices = np.array([49,60,71])
mouthIndices = np.array([92,93,94])
allTissueIndices = np.arange(numCells)

def defineTargetdGpol():
    targetdGpol = torch.zeros(numSamples*numCells).view(numSamples,numCells,1)
    return targetdGpol

targetdGpol = defineTargetdGpol()
target = torch.cat((targetVmem,targetdGpol),axis=1)

def computeLoss(method='globalsum'):
    if method == 'globalsum':
        loss = ((targetVmem - modelinstance.timeseriesVmem[-evalDuration:]) ** 2).sum().sqrt()
    elif method == 'globalmean':
        loss = ((targetVmem - modelinstance.timeseriesVmem[-evalDuration:]) ** 2).mean().sqrt()
    elif method == 'partitioned':
        skinIndices = utils.computeDomeIndices(circuit,mode='tissue')
        observedVmem = circuit.timeseriesVmem[-evalDuration:,:,:,0]  # shape = (numEvalIters,numSamples,numCells)
        lossSkin = ((targetVmem[:,skinIndices,0] - observedVmem[:,:,skinIndices])**2).sum().sqrt() / len(skinIndices)
        lossEyes = ((targetVmem[:,eyeIndices,0] - observedVmem[:,:,eyeIndices])**2).sum().sqrt() / len(eyeIndices)
        lossNose = ((targetVmem[:,noseIndices,0] - observedVmem[:,:,noseIndices])**2).sum().sqrt() / len(noseIndices)
        lossMouth = ((targetVmem[:,mouthIndices,0] - observedVmem[:,:,mouthIndices])**2).sum().sqrt() / len(mouthIndices)
        loss = (lossSkin + lossEyes + lossNose + lossMouth)
    elif method == 'globalsumWithdGpol':
        dGpolValues = modelinstance.timeseriesdGpol[-evalDuration:]
        observedMax = dGpolValues.abs().max()
        normalization = min(0.05, observedMax)
        dGpolValues = dGpolValues * (normalization / observedMax)  # scale it to be comparable to Vmem with expected mean -0.03
        # dGpolValues = dGpolValues * (0.05 / dGpolValues.abs().max())  # scale it to be comparable to Vmem with expected mean -0.03
        # observed = torch.cat((modelinstance.timeseriesVmem[-evalDuration:],dGpolValues),axis=2)
        # loss = ((target - observed)**2).sum().sqrt()
        loss1 = ((targetVmem - modelinstance.timeseriesVmem[-evalDuration:]) ** 2).sum().sqrt()
        loss2 = ((0 - dGpolValues) ** 2).sum().sqrt()  # target dG_pol = 0
        loss = (loss1 + loss2) / 2
    elif method == 'globalsumWithdVmem':
        dVmemValues = modelinstance.timeseriesdVmem[-evalDuration:]
        observedMax = dVmemValues.abs().max()
        normalization = min(0.05, observedMax)
        dVmemValues = dVmemValues * (normalization / observedMax)  # scale it to be comparable to Vmem with expected mean -0.03
        loss1 = ((targetVmem - modelinstance.timeseriesVmem[-evalDuration:]) ** 2).sum().sqrt()
        loss2 = ((0 - dVmemValues) ** 2).sum().sqrt()  # target dG_pol = 0
        loss = (loss1 + loss2) / 2
    return loss

# modelparameters = dict()
# modelparameters['GJParameters'] = GJParameters
# modelparameters['fieldParameters'] = fieldParameters
# modelparameters['ligandParameters'] = ligandParameters
# modelparameters['GRNParameters'] = GRNParameters

if Model == None:
    print("Model num = ",filenum)
else:
    print("Model name = ",Model)

losses = []
for run in range(numSimRuns):
    modelinstance = model(parameters,numSamples)
    modelinstance.setExperimentalConditions((initialValues,numSamples))
    circuit = modelinstance.electricNetwork
    if TurnonATP:
        if 'ATPConc' not in initialValues.keys():
            initialValues['ATPConc'] = torch.ones((numSamples,numCells,1),dtype=torch.float64) * (2.5-0.0000)  # 0.8, 0.75
            boundaryCells = utils.computeDomeIndices(circuit,mode='tissue')
            initialValues['ATPConc'][:,boundaryCells,0] = (2.5-0.0000)  # 0.0, 1.8, 1.95
            modelinstance.setExperimentalConditions((initialValues,numSamples))
        boundaryCells = utils.computeDomeIndices(circuit, mode='tissue')
        externalInputs['ATP'] = torch.zeros((numSamples,numSimIters,numCells,1),dtype=torch.float64)
        # Rescue values (for ATPReactionStrength = 0): 2.3 (init = 0.0), 2.2 (init = 0.1), 1.2 (init = 1.0), 1.2 (init = 0.5), 2.5 (init = -0.197)
        # Rescue values (for ATPReactionStrength = 1, equilibrium ATP = 1): 1.5 (init = 0.1)
        # Rescue values (for ATPReactionStrength = 1, equilibrium ATP = 10): 10 (init = 2.5)
        # inputs = torch.DoubleTensor(torch.load('./data/Current_dims1,4,10_210.dat')[2]).unsqueeze(1).repeat(1,len(boundaryCells))
        # externalInputs['ATP'][:,:,boundaryCells,0] = inputs * 0.26
        inputs = torch.DoubleTensor(torch.load('./data/Current_dims4,6,10,15_262.dat')[3]).unsqueeze(1).repeat(1,numCells)
        # inputs = torch.DoubleTensor(torch.load('./data/Current_dims4,6,10,15_262.dat')[0]).unsqueeze(1).repeat(1,len(boundaryCells))
        externalInputs['ATP'][:,:,:,0] = inputs[0:1000] # * 0.26
        # externalInputs['ATP'][:,:,boundaryCells,0] = inputs # * 0.26
        # externalInputs['ATP'][:,500:,boundaryCells,0] = 0
    # circuit = cellularFieldNetwork(latticeDims=latticeDims,parameters=modelparameters,numSamples=numSamples)
    # circuit.initVariables(initialValues)
    # circuit.initParameters(initialValues)

    if Autonomous:
        initVmem = list(chain([-9.2e-3] * numSamples))
        initialValues['Vmem'] = torch.repeat_interleave(torch.DoubleTensor(initVmem),circuit.numCells,0).view(numSamples,circuit.numCells,1)
        initialValues['eV'] = torch.zeros((numSamples,circuit.numFieldGridPoints,1),dtype=torch.float64)
        initialValues['ligandConc'] = torch.ones((numSamples,circuit.numCells,1),dtype=torch.float64) * 0.5
        if randomizeInitialState:
            AllCells = list(range(circuit.numCells))
            initialValues['G_pol']['cells'] = [[AllCells]] * numSamples
            initialValues['G_pol']['values'] = [[torch.rand(circuit.numCells,dtype=torch.float64)*2] for _ in  range(numSamples)]  # covers a range of unistable and bistable values
        else:
            AllCells = list(range(circuit.numCells))
            initialValues['G_pol']['cells'] = [[AllCells]] * numSamples
            initialValues['G_pol']['values'] = [torch.DoubleTensor([1.0]*numCells)] * numSamples  # bistable
        circuit.initVariables(initialValues)
        circuit.initParameters(initialValues)
        clampParameters = None

    if Perturb:
        perturbation = dict()
        if perturbationMode == 'swapVmem':  # swap a block of Vmems with another
            perturbPointIndicesA = eyeIndices[0:4]
            perturbPointIndicesB = perturbPointIndicesA + 22  # shift the entire eye down by one block
            perturbValues = None
            perturbStartIter, perturbEndIter = 1000, 1000
        elif perturbationMode == 'permuteVmem':  # randomly shuffle the tissue
            perturbPointIndicesA = np.tile(allTissueIndices,numSamples)
            perturbPointIndicesB = np.concatenate([torch.randperm(numCells) for _ in range(numSamples)])
            perturbValues = None
            perturbStartIter, perturbEndIter = 102, 102
        elif perturbationMode == 'permuteVmemBoundary':  # randomly shuffle the boundary tissue
            boundaryIndices = circuit.utils.computeDomeIndices(circuit, mode='tissue')
            numBoundareCells = len(boundaryIndices)
            perturbPointIndicesA = np.tile(boundaryIndices,numSamples)
            perturbPointIndicesB = np.concatenate([torch.randperm(numBoundareCells) for _ in range(numSamples)])
            perturbValues = None
            perturbStartIter, perturbEndIter = 102, 102
        elif perturbationMode == 'permuteGpol':
            tissueboundIndices = circuit.utils.computeDomeIndices(circuit, mode='tissue')
            # tissuebulkIndices = np.setdiff1d(np.arange(numCells),tissueboundIndices)
            # perturbPointIndicesA = np.tile(tissuebulkIndices,numSamples)
            perturbPointIndicesA = np.tile(tissueboundIndices,numSamples)
            perturbPointIndicesB = np.concatenate([np.random.permutation(tissueboundIndices) for _ in range(numSamples)])
            perturbValues = None
            perturbStartIter, perturbEndIter = 102, 102
        elif perturbationMode == 'swapGpol':
            # perturbPointIndicesA = [13]
            # perturbPointIndicesB = [112]
            perturbPointIndicesA = eyeIndices[0:4]
            perturbPointIndicesB = perturbPointIndicesA + 22  # shift the entire eye down by one block
            perturbValues = None
            perturbStartIter, perturbEndIter = 1000, 1000
        elif perturbationMode == 'setGpol':
            # indices = circuit.utils.computeDomeIndices(circuit,mode='tissue')
            indices = [77,87,88,98]
            perturbPointIndicesA = np.tile(indices,numSamples)
            perturbPointIndicesB = None
            perturbValues = 0.1 * circuit.G_ref
            perturbStartIter, perturbEndIter = 103, 103
        elif perturbationMode == 'setFieldTransductionWeight':
            perturbPointIndicesA, perturbPointIndicesB = [], []
            perturbValues = 0.0
            perturbStartIter, perturbEndIter = 1000, 1001
        numPerturbPoints = len(perturbPointIndicesA)
        sampleIndices = np.repeat(range(numSamples),numPerturbPoints)  # assuming that there's only one sample in which the eye block is shifted
        perturbation['mode'] = perturbationMode
        perturbation['data'] = (sampleIndices,(perturbPointIndicesA,perturbPointIndicesB),perturbValues)
        perturbation['time'] = (perturbStartIter,perturbEndIter)
    else:
        perturbation = None

    if Freeze:
        freeze = dict()
        freezeCellIndices, freezeFieldIndices = circuit.utils.computeFreezeIndices(circuit,activeBlockCellIndexCoords=activeBlockCellIndexCoords)
        freezeValues = 0
        freezeCellPointIndices = np.tile(freezeCellIndices,numSamples)
        freezeFieldPointIndices = np.tile(freezeFieldIndices,numSamples)
        numFreezeCellPoints = len(freezeCellPointIndices)
        numFreezeFieldPoints = len(freezeFieldPointIndices)
        sampleIndicesCell = np.repeat(range(numSamples),numFreezeCellPoints)
        sampleIndicesField = np.repeat(range(numSamples),numFreezeFieldPoints)
        freezePointIndices = (freezeCellPointIndices,freezeFieldPointIndices)
        sampleIndices = (sampleIndicesCell,sampleIndicesField)
        freezeStartIter, freezeEndIter = 999, 4999
        freeze['data'] = (sampleIndices,freezePointIndices)
        freeze['time'] = (freezeStartIter,freezeEndIter)
    else:
        freeze = None

    # boundaryFieldPoints = utils.computeDomeIndices(circuit, mode='field')
    modelinstance.electricNetwork.eVModulator = torch.ones(1,numCells,1)
    # modelinstance.electricNetwork.eVModulator[0,boundaryFieldPoints,0] = 1.2
    modelinstance.simulate(externalInputs=externalInputs,clampParameters=clampParameters,perturbation=perturbation,numSimIters=numSimIters)
    evalDuration = int(evalDurationProp*numSimIters)
    # loss = ((targetVmem - circuit.timeseriesVmem[-evalDuration:]) ** 2).sum().sqrt()
    loss = computeLoss(method=lossMethod)
    np.set_printoptions(precision=2,suppress=True)
    print("Run = ",run)
    print("Recorded loss: ",parameters['trainParameters']['bestLoss'])
    print("Actual loss: ",loss.item())
    losses.append(loss.item())

if MultiCircuit:
    circuitLarge = cellularFieldNetwork(latticeDims=(11,23),parameters=parameters,numSamples=numSamples)
    circuit1Indices = np.concatenate([np.arange(11)+(i*23) for i in range(11)])
    circuit2Indices = np.concatenate([np.arange(12,23)+(i*23) for i in range(11)])
    dividerIndices = np.concatenate([np.arange(11,12)+(i*23) for i in range(11)])
    circuitLarge.Adjacency[:] = 0
    circuitLarge.Adjacency[np.repeat(circuit1Indices,len(circuit1Indices)),np.tile(circuit1Indices,len(circuit1Indices))] = circuit.Adjacency.flatten().clone()
    circuitLarge.Adjacency[np.repeat(circuit2Indices,len(circuit2Indices)),np.tile(circuit2Indices,len(circuit2Indices))] = circuit.Adjacency.flatten().clone()
    circuitLarge.Vmem[:,circuit1Indices,:] = circuit.Vmem.clone()
    # circuitLarge.Vmem[:,circuit2Indices,:] = circuit.Vmem.clone()
    field1Indices = np.concatenate([np.arange(12)+(i*24) for i in range(12)])
    field2Indices = np.concatenate([np.arange(12,24)+(i*24) for i in range(12)])
    circuitLarge.eV[:,field1Indices,:] = circuit.eV.clone()
    # circuitLarge.eV[:,field2Indices,:] = circuit.eV.clone()
    circuitLarge.G_pol[:,circuit1Indices,:] = circuit.G_pol.clone()
    # circuitLarge.G_pol[:,circuit2Indices,:] = circuit.G_pol.clone()
    freeze = dict()
    freezeCellIndices, freezeFieldIndices = dividerIndices, []
    freezeValues = 0
    freezeCellPointIndices = np.tile(freezeCellIndices,numSamples)
    freezeFieldPointIndices = np.tile(freezeFieldIndices,numSamples)
    numFreezeCellPoints = len(freezeCellPointIndices)
    numFreezeFieldPoints = len(freezeFieldPointIndices)
    sampleIndicesCell = np.repeat(range(numSamples),numFreezeCellPoints)
    sampleIndicesField = np.repeat(range(numSamples),numFreezeFieldPoints)
    freezePointIndices = (freezeCellPointIndices,freezeFieldPointIndices)
    sampleIndices = (sampleIndicesCell,sampleIndicesField)
    freezeStartIter, freezeEndIter = 0, 4999
    freeze['data'] = (sampleIndices,freezePointIndices)
    freeze['time'] = (freezeStartIter,freezeEndIter)
    circuitLarge.simulate(externalInputs=None,clampParameters=None,perturbationParameters=None,freezeParameters=freeze,
                         numSimIters=5000,stochasticIonChannels=False,setGradient=False,retainGradients=False,saveData=True)
    # print(circuit.Vmem.shape)

if numSimRuns > 1:
    from scipy.stats import bootstrap
    recLoss = parameters['trainParameters']['bestLoss'].item()
    losses = np.array(losses)
    losses = (losses - recLoss)/recLoss
    resLoss = bootstrap(losses.reshape(1,-1), np.mean, confidence_level=0.9)
    print(losses.mean(),resLoss.confidence_interval)

# ## TEST CODE
# VmemBins = np.arange(-0.0, -0.1, -0.04)
# vbin = 2 - np.digitize(circuit.timeseriesVmem[:,0,:,0].detach(),VmemBins)
# flips = vbin[1:] - vbin[0:-1]
# numFlips0to1 = (flips==1).sum(0)
# numFlips1to0 = (flips==-1).sum(0)
# # cellfreqs = numFlips0to1+numFlips1to0
# numones = vbin.sum(0)
# numzeros = np.amax((numSimIters-numones).reshape(1,-1),axis=0,initial=1)
# cellfreqs = ((numFlips0to1/numones)+(numFlips1to0/numzeros))/2
# print(len(np.unique(cellfreqs))/numCells)
