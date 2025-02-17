from model import model
from cellularFieldNetwork import cellularFieldNetwork
import numpy as np
import torch
import utilities
from itertools import chain
import matplotlib.pyplot as plt

Model = None  # optoions: 'Stigmergic', 'Mosaic', None

fieldVector = True
fieldRangeSymmetric = False
ligandEnabled = False
if Model == 'Stigmergic':
    parameterfilename = './data/StigmergicModelParameters.dat'
elif Model == 'Mosaic':
    parameterfilename = './data/MosaicModelParameters.dat'
else:
    filenum = 0  # weakly sensitive: 1294; strongly sensitive: 1576
    if fieldVector:
        if ligandEnabled:
            Sfx = '_fieldVector_Ligand'
        else:
            Sfx = '_fieldVector'
    else:
        Sfx = ''
    parameterfilename = './data/bestModelParameters' + Sfx + '_' + str(filenum) + '.dat'  # 472 (fr=4); OLD: 483 (fieldRange=4); 759 (fieldRange=1); 825 (fieldRange=21)

parameters = torch.load(parameterfilename)

Autonomous = False  # impose homogenous initial conditions under unclamped conditions
randomizeInitialState = False  # applies only if Autonomous=True
tempFieldParamsOverride = False
Perturb = False
perturbationMode = 'setFieldTransductionWeight'  # options: swapVmem, permuteVmem, permuteVmemBoundary, swapGpol
Freeze = False
activeBlockCellIndexCoords = ((0,0),(7,7))
MultiCircuit = False
newSimulationLength = (False,100)
newVmemLigandStrength = (False,3.0)
TurnoffField = False
TurnoffLigand = False
numSimRuns = 1

latticeDims = parameters['latticeDims']
GJParameters = parameters['GJParameters']
fieldParameters = parameters['fieldParameters']
if 'fieldRangeSymmetric' not in fieldParameters.keys():
    fieldParameters['fieldRangeSymmetric'] = fieldRangeSymmetric
if 'fieldVector' not in fieldParameters.keys():
    fieldParameters['fieldVector'] = False
if 'fieldTransductionGain' not in fieldParameters.keys():
    fieldParameters['fieldTransductionGain'] = 1.0
if 'ligandParameters' in parameters.keys():
    ligandParameters = parameters['ligandParameters']
    if 'vmemToLigandTransductionWeight' not in ligandParameters.keys():
        ligandParameters['vmemToLigandTransductionWeight'] = 1.0
else:
    ligandParameters = None
GRNParameters = parameters['GRNParameters']
numRows, numCols = latticeDims[0], latticeDims[1]
numCells = numRows * numCols
numSamples = parameters['simParameters']['numSamples']
initialValues = parameters['simParameters']['initialValues']
if 'ligandConc' not in initialValues.keys():
    initialValues['ligandConc'] = torch.zeros((numSamples,numCells,1),dtype=torch.float64)
clampParameters = parameters['clampParameters']
externalInputs = parameters['simParameters']['externalInputs']
numSimIters = parameters['simParameters']['numSimIters']
evalDurationProp = parameters['trainParameters']['evalDurationProp']
targetVmem = parameters['trainParameters']['targetVmem']
if 'lossMethod' in parameters['trainParameters'].keys():
    lossMethod = parameters['trainParameters']['lossMethod']
else:
    lossMethod = 'globalsum'

if tempFieldParamsOverride:
    numSamples = 10
    # clampParameters['clampValues'] *= 2
    # fieldParameters['fieldTransductionWeight'] = 500
    # fieldParameters['fieldTransductionBias'] = torch.DoubleTensor([0.0])
    # fieldParameters['fieldTransductionTimeConstant'] = 10.0
    # fieldParameters['fieldStrength'] *= 4.0
    # fieldParameters['fieldRangeSymmetric'] = True
    # fieldParameters['fieldScreenSize'] = 21
    # GJParameters['GJStrength'] = 1.0

# indices of the features of the 11x11 smiley
eyeIndices = np.array([24,25,35,36,29,30,40,41])  # left and right eyes
noseIndices = np.array([49,60,71])
mouthIndices = np.array([92,93,94])
allTissueIndices = np.arange(numCells)

def computeLoss(method='globalsum'):
    if method == 'globalsum':
        loss = ((targetVmem - modelinstance.timeseriesVmem[-evalDuration:]) ** 2).sum().sqrt()
    elif method == 'globalmean':
        loss = ((targetVmem - modelinstance.timeseriesVmem[-evalDuration:]) ** 2).mean().sqrt()
    elif method == 'partitioned':
        utils = utilities.utilities()
        skinIndices = utils.computeDomeIndices(circuit,mode='tissue')
        observedVmem = circuit.timeseriesVmem[-evalDuration:,:,:,0]  # shape = (numEvalIters,numSamples,numCells)
        lossSkin = ((targetVmem[:,skinIndices,0] - observedVmem[:,:,skinIndices])**2).sum().sqrt() / len(skinIndices)
        lossEyes = ((targetVmem[:,eyeIndices,0] - observedVmem[:,:,eyeIndices])**2).sum().sqrt() / len(eyeIndices)
        lossNose = ((targetVmem[:,noseIndices,0] - observedVmem[:,:,noseIndices])**2).sum().sqrt() / len(noseIndices)
        lossMouth = ((targetVmem[:,mouthIndices,0] - observedVmem[:,:,mouthIndices])**2).sum().sqrt() / len(mouthIndices)
        loss = (lossSkin + lossEyes + lossNose + lossMouth)
    return loss

# fieldParameters['fieldTransductionWeight'] = 0.0  # override field parameters
modelparameters = dict()
modelparameters['GJParameters'] = GJParameters
modelparameters['fieldParameters'] = fieldParameters
modelparameters['ligandParameters'] = ligandParameters
modelparameters['GRNParameters'] = GRNParameters

if TurnoffField:
    modelparameters['fieldParameters']['fieldEnabled'] = False

if TurnoffLigand:
    modelparameters['ligandParameters']['ligandEnabled'] = False
elif newVmemLigandStrength[0]:
    modelparameters['ligandParameters']['vmemToLigandCurrentStrength'] = newVmemLigandStrength[1]

if Model == None:
    print("Model num = ",filenum)
else:
    print("Model name = ",Model)

losses = []
for run in range(numSimRuns):
    modelinstance = model(parameters,numSamples)
    modelinstance.setExperimentalConditions((initialValues,numSamples))
    circuit = modelinstance.electricNetwork
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
            perturbPointIndicesA = [13]
            perturbPointIndicesB = [112]
            perturbValues = None
            perturbStartIter, perturbEndIter = 102, 102
        elif perturbationMode == 'setGpol':
            tissueboundIndices = circuit.utils.computeDomeIndices(circuit,mode='tissue')
            perturbPointIndicesA = np.tile(tissueboundIndices,numSamples)
            perturbPointIndicesB = None
            perturbValues = 2.0 * circuit.G_ref
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

    if newSimulationLength[0]:
        numSimIters = newSimulationLength[1]

    def simulate(circuit,clampParameters=None,perturbation=None,numSimIters=1):
        numCells = circuit.numCells
        if circuit.GRNEnabled:
            numGenes = circuit.geneNetwork.numGenes
            numVariables = numGenes * numCells
            circuit.timeseriesGRN = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numGenes*numCells).view(numSimIters,circuit.numSamples,numGenes*numCells,1)
            circuit.timeseriesGRNExternalInputs = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numVariables).view(numSimIters,circuit.numSamples,numVariables,1)
        circuit.timeseriesVmem = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numCells).view(numSimIters,circuit.numSamples,numCells,1)
        # the below are recorded for debugging purpose only
        circuit.timeseriesGdep = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numCells).view(numSimIters,circuit.numSamples,numCells,1)
        circuit.timeseriesIncurrent = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numCells).view(numSimIters,circuit.numSamples,numCells,1)
        circuit.timeseriesOutcurrent = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numCells).view(numSimIters,circuit.numSamples,numCells,1)
        circuit.timeseriesGij = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numCells*numCells).view(numSimIters,circuit.numSamples,numCells,numCells)
        circuit.timeseriesGJcurrent = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numCells).view(numSimIters,circuit.numSamples,numCells,1)
        if clampParameters is not None:
            clampMode = clampParameters['clampMode']
            clampIndices = clampParameters['clampIndices']
            clampValues = clampParameters['clampValues']
            clampStartIter =  clampParameters['clampStartIter']
            clampEndIter = clampParameters['clampEndIter']
            sampleIndices, clampPointIndices = clampIndices
            # Compute the field distance matrix consisting of the pairwise distances between the clamp points and extracellular coordinates
            # shape = (numSamples,numClampPoints,numFieldGridPoints)
            if 'field' in clampMode:
                circuit.fieldClampSampleIndices = sampleIndices
                circuit.fieldClampPointIndices1D = clampPointIndices
                circuit.numFieldClampPoints = int(len(circuit.fieldClampPointIndices1D)/circuit.numSamples)
                circuit.clampFieldPointCoordinates = (circuit.extracellularCoordinates[0][:,circuit.fieldClampPointIndices1D].view(circuit.numSamples,circuit.numFieldClampPoints),
                                                                    circuit.extracellularCoordinates[1][:,circuit.fieldClampPointIndices1D].view(circuit.numSamples,circuit.numFieldClampPoints))
                # NOTE: The setdiff would have to be done separately for each set of clamp points
                circuit.fieldClampPointIndices2D = circuit.fieldClampPointIndices1D.reshape(circuit.numSamples,circuit.numFieldClampPoints)
                circuit.freeFieldPointIndices1D = np.concatenate([np.setdiff1d(range(circuit.numFieldGridPoints),indices)
                                                                 for indices in circuit.fieldClampPointIndices2D])
                circuit.freeFieldPointCoordinates = (circuit.extracellularCoordinates[0][:,circuit.freeFieldPointIndices1D].view(circuit.numSamples,-1),
                                                  circuit.extracellularCoordinates[1][:,circuit.freeFieldPointIndices1D].view(circuit.numSamples,-1))  # shape = (numSamples,numFreeFieldPoints)
                circuit.fieldClampDistanceMatrix = (circuit.utils.computePairwiseDistances(circuit.clampFieldPointCoordinates,circuit.freeFieldPointCoordinates).double()
                                                 .view(circuit.numSamples,-1,circuit.numFieldClampPoints))
                circuit.numFreeFieldPoints = circuit.numFieldGridPoints - circuit.numFieldClampPoints
                circuit.fieldFreeSampleIndices = np.repeat(range(circuit.numSamples),circuit.numFreeFieldPoints)
            elif 'tissue' in clampMode:
                sampleIndices, clampPointIndices = clampIndices
        else:
            clampMode, sampleIndices, clampPointIndices, clampValues, clampStartIter, clampEndIter = None, None, None, None, 0, -1
        if perturbation is not None:
            perturbStartIter, perturbEndIter = perturbation['time']
        else:
            perturbStartIter, perturbEndIter = 0, -1
        for iter in range(numSimIters):
            if circuit.GRNEnabled:
                circuit.timeseriesGRN[iter] = circuit.geneNetwork.state
                circuit.timeseriesGRNExternalInputs[iter] = circuit.geneNetwork.tissueExternalInputs
            circuit.timeseriesVmem[iter] = circuit.Vmem
            # the below are recorded for debugging purpose only
            circuit.timeseriesGdep[iter] = circuit.G_dep
            # circuit.timeseriesIncurrent[iter] = circuit.InCurrent
            # circuit.timeseriesOutcurrent[iter] = circuit.OutCurrent
            circuit.timeseriesGij[iter] = circuit.G_ij
            circuit.timeseriesGJcurrent[iter] = circuit.GapJunctionCurrent

            if circuit.GRNEnabled:
                externalInputs = {'gene':circuit.geneNetwork.state}
            else:
                externalInputs = {'gene':None}
            circuit.simulate(externalInputs=externalInputs,numSimIters=1,stochasticIonChannels=False,
                                setGradient=False,retainGradients=False,saveData=False)  # shape = (numSamples,numGenes*numCells,1)
            if (iter >= perturbStartIter) and (iter <= perturbEndIter):
                circuit.perturb(perturbation=perturbation,currentIter=iter)
            if (iter >= clampStartIter) and (iter <= clampEndIter):
                if ('field' in clampMode) and circuit.fieldEnabled:
                    circuit.eV[sampleIndices,clampPointIndices,0] = clampValues[iter,:]  # clamped points act like field sources themselves
                    circuit.updateExtracellularVoltage(source='eVClamp')
                    circuit.updateIonChannelConductance(inputSource='field',stochasticIonChannels=False,fieldAggregation=circuit.fieldAggregation,perturbation=None)
                    if circuit.ligandEnabled:
                        circuit.updateLigandConcentration(source='Vmem')
                        circuit.updateLigandConcentration(source='ligand')
                        # circuit.updateIonChannelConductance(inputSource='ligand',stochasticIonChannels=stochasticIonChannels,perturbation=None)
                        circuit.updateFieldSensitivity(inputSource='ligand')
                    circuit.updateCurrent()
                    circuit.updateVmem()
                elif 'Vmem' in clampMode:
                    circuit.Vmem[sampleIndices,clampPointIndices,0] = clampValues[iter,:]
                elif ('Ligand' in clampMode) and circuit.ligandEnabled:
                    circuit.ligandConc[sampleIndices,clampPointIndices,0] = clampValues[iter,:]
                    circuit.updateLigandConcentration(source='ligand')
                    # circuit.updateIonChannelConductance(inputSource='ligand',stochasticIonChannels=stochasticIonChannels,perturbation=None)
                    circuit.updateFieldSensitivity(inputSource='ligand')
                    circuit.updateCurrent()
                    circuit.updateVmem()
                elif 'Gpol' in clampMode:
                    circuit.G_pol[sampleIndices,clampPointIndices,0] = clampValues[iter,:] * circuit.G_ref
                    circuit.updateCurrent()
                    circuit.updateVmem()

    # simulate(circuit,clampParameters=clampParameters,perturbation=perturbation,numSimIters=numSimIters)
    modelinstance.simulate(clampParameters=clampParameters,perturbation=perturbation,numSimIters=numSimIters)
    evalDuration = int(evalDurationProp*numSimIters)
    # loss = ((targetVmem - circuit.timeseriesVmem[-evalDuration:]) ** 2).sum().sqrt()
    loss = computeLoss(method=lossMethod)
    np.set_printoptions(precision=2,suppress=True)
    print("Run = ",run)
    print("Recorded loss: ",parameters['trainParameters']['bestLoss'])
    print("Actual loss: ",loss.item())
    losses.append(loss.item())

if MultiCircuit:
    circuitLarge = cellularFieldNetwork(latticeDims=(11,23),parameters=modelparameters,numSamples=numSamples)
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
