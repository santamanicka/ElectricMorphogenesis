import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import utilities

circuitRows,circuitCols = 11,11
circuitDims = (circuitRows,circuitCols)  # (rows,columns) of lattice
fieldEnabled = True
fieldResolution = 1
fieldStrength = 10.0  # 10.0
minNumBoundingSquares, maxNumBoundingSquares = 1, 2*max(circuitDims) - 1
numBoundingSquares = torch.rand(1)*(maxNumBoundingSquares - minNumBoundingSquares) + minNumBoundingSquares
numBoundingSquares.requires_grad = True
print("Initial numBoundingSquares: ", numBoundingSquares)
# eVBias = torch.DoubleTensor([0.0214])  # 0.0214
# eVWeight = torch.DoubleTensor([9.4505])  # 9.4505
evTimeConstant = torch.DoubleTensor([10.0])
maxeVBias = 1.0
mineVBias = -maxeVBias
eVBias = torch.rand(1,dtype=torch.double)*2*maxeVBias - maxeVBias
eVBias.requires_grad = True
eVWeight = torch.rand(1,dtype=torch.double)*2*10 - 10
eVWeight.requires_grad = True
clampMode = 'fieldDomeTwoFoldSymmetry'  # possible values: field, fieldDome, fieldDomeFourFoldSymmetry, tissueVmem, tissueDomeVmem, tissueGpol, tissueDomeGpol, None
clampType = 'static'  # possible values: oscillatory, static
clampedCellsProp = 1.0
if clampedCellsProp == 0.0:
    clampMode = None
clampDurationProp = 0.1
minClampAmplitude, maxClampAmplitude = torch.DoubleTensor([-100.0]), torch.DoubleTensor([100.0])  # [-100,100] for eV and Vmem and [0,2] for Gpol
minClampAmplitude.requires_grad = True
maxClampAmplitude.requires_grad = True
if clampType == 'oscillatory':
    minClampOscillationFrequency, maxClampOscillationFrequency = 100.0, 1000.0
elif clampType == 'static':
    minClampOscillationFrequency, maxClampOscillationFrequency = 0.0, 0.0
evalDurationProp = 0.1
numSamples = 1
numSimIters = 500
numLearnIters = 100

utils = utilities.utilities()

fieldParameters = (fieldResolution, fieldStrength, None)
circuit = cellularFieldNetwork(circuitDims,GRNParameters=(None,None,None,None),
                               fieldParameters=fieldParameters,numSamples=numSamples)

fieldDomeIndices = utils.computeDomeIndices(circuit,mode='field')
tissueDomeIndices = utils.computeDomeIndices(circuit,mode='tissue')
if clampMode == 'field':
    numTotalCells = circuit.numExtracellularGridPoints
    cellIndices = np.arange(numTotalCells)
    numFieldCells = numTotalCells
elif clampMode == 'fieldDome':
    numTotalCells = len(fieldDomeIndices)
    cellIndices = fieldDomeIndices
    cellIndices = utils.computeDomeIndices(circuit,mode='field',region='full')
    numFieldCells = numTotalCells
elif clampMode == 'fieldDomeFourFoldSymmetry':
    fieldDomeTopLeftQuadrantIndices = utils.computeDomeIndices(circuit,mode='field',region='topLeftQuadrant')
    numTotalCells = len(fieldDomeTopLeftQuadrantIndices)
    cellIndices = fieldDomeTopLeftQuadrantIndices
    numFieldCells = numTotalCells
elif clampMode == 'fieldDomeTwoFoldSymmetry':
    fieldDomeLeftHalfIndices = utils.computeDomeIndices(circuit,mode='field',region='leftHalf')
    numTotalCells = len(fieldDomeLeftHalfIndices)
    cellIndices = fieldDomeLeftHalfIndices
    numFieldCells = numTotalCells
elif (clampMode == 'tissueVmem') or (clampMode == 'tissueGpol'):
    numTotalCells = circuit.numCells
    cellIndices = np.arange(numTotalCells)
    numFieldCells = circuit.numExtracellularGridPoints
elif (clampMode == 'tissueDomeVmem') or (clampMode == 'tissueDomeGpol'):
    numTotalCells = len(tissueDomeIndices)
    cellIndices = tissueDomeIndices
    numFieldCells = circuit.numExtracellularGridPoints
if clampMode != None:
    numClampPoints = int(clampedCellsProp*numTotalCells)
    clampPointIndices = np.array([np.random.choice(cellIndices,numClampPoints,replace=False)
                                             for _ in range(numSamples)]).reshape(-1,)
    sampleIndices = np.repeat(range(numSamples),numClampPoints)
    clampIndices = (sampleIndices,clampPointIndices)
    # clampIndices = [4,5,6,7,8,84,85,86,87,88]  # surrounding the middle two columns of 4x6
    clampStartIter, clampEndIter = 0, int(clampDurationProp * numSimIters)
    numClampIters = clampEndIter - clampStartIter + 1
    # clampValue = torch.FloatTensor([-10.1]*numClampPoints*numSamples)
    timeIndices = torch.linspace(0,0.5,numClampIters).view(-1,1)
    if clampType == 'oscillatory':
        clampFrequencies = torch.rand(numSamples*numClampPoints,dtype=torch.double)*(maxClampOscillationFrequency-minClampOscillationFrequency) + minClampOscillationFrequency
        clampFrequencies.requires_grad = True
    elif clampType == 'static':
        clampFrequencies = torch.zeros(numSamples*numClampPoints,dtype=torch.double)
    clampPhases = torch.rand(numSamples*numClampPoints,dtype=torch.double)*2*torch.pi
    clampPhases.requires_grad = True
    if 'Symmetry' in clampMode:
        if 'FourFoldSymmetry' in clampMode:
            verticalReflectedIndices, horizontalReflectedIndices, diagonalReflectedIndices = \
                utils.computeSymmetricalIndices(circuit,clampPointIndices,mode='field',symmetry='fourfold')
            clampFrequenciesActual = torch.tile(clampFrequencies,(4,))
            clampPhasesActual = torch.tile(clampPhases,(4,))
            clampPointIndices = np.concatenate((clampPointIndices,verticalReflectedIndices,horizontalReflectedIndices,
                                    diagonalReflectedIndices))
        elif 'TwoFoldSymmetry' in clampMode:
            verticalReflectedIndices = utils.computeSymmetricalIndices(circuit,clampPointIndices,mode='field',symmetry='twofold')
            clampFrequenciesActual = torch.tile(clampFrequencies,(2,))
            clampPhasesActual = torch.tile(clampPhases,(2,))
            clampPointIndices = np.concatenate((clampPointIndices,verticalReflectedIndices))
        _, uniqueClampPointIndices = np.unique(clampPointIndices,return_index=True)  # first-occurrence indices
        clampPointIndices = clampPointIndices[uniqueClampPointIndices]  # this will always be a sorted array
        clampValues = torch.cos(timeIndices*clampFrequenciesActual + clampPhasesActual)
        clampValues = ((clampValues+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
        clampValues = clampValues[:,uniqueClampPointIndices]
        numClampPoints = len(clampPointIndices)
        sampleIndices = np.repeat(range(numSamples),numClampPoints)
        clampIndices = (sampleIndices,clampPointIndices)
    else:
        clampValues = torch.cos(timeIndices*clampFrequencies + clampPhases)
        clampValues = ((clampValues+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
    clampParameters = (clampMode,clampIndices,clampValues,(clampStartIter,clampEndIter))
else:
    clampParameters = None

def defineInitialValues(circuit):
    initialValues = dict()
    initVmem = torch.FloatTensor(list(chain([-9.2e-3] * numSamples)))
    initialValues['Vmem'] = torch.repeat_interleave(initVmem,circuit.numCells,0).double().view(numSamples,circuit.numCells,1)
    initialValues['eV'] = torch.zeros((numSamples,circuit.numExtracellularGridPoints,1),dtype=torch.float64)
    initialValues['G_pol'] = dict()
    initialValues['G_pol']['cells'] = [[[0]]] * numSamples
    initialValues['G_pol']['values'] = [torch.DoubleTensor([1.0])] * numSamples  # bistable
    initialValues['G_dep'] = dict()
    initialValues['G_dep']['cells'] = []
    initialValues['G_dep']['values'] = torch.DoubleTensor([])
    return initialValues

targetVmem = torch.FloatTensor(list(chain([-9.2e-3] * numSamples)))
targetVmem = torch.repeat_interleave(targetVmem,circuit.numCells,0).view(numSamples,circuit.numCells,1)
targetVmem[:,tissueDomeIndices] = -0.06
# targetVmem[:,[12]] = -0.06  # indices of 1x1 core in a 5x5 lattice
# targetVmem[:,[14,15,20,21]] = -0.06  # indices of 2x2 core in a 6x6 lattice
# targetVmem[:,[33,34,35,36,43,44,45,46,53,54,55,56,63,64,65,66]] = -0.06  # indices of 4x4 core in a 10x10 lattice
## Smiley core in a 11x11 tissue
targetVmem[:,[24,25,35,36]] = -0.06  # Eye 1
targetVmem[:,[29,30,40,41]] = -0.06  # Eye 2
targetVmem[:,[49,60,71]] = -0.06  # Nose
targetVmem[:,[92,93,94]] = -0.06  # Mouth

LearnableParameters = [clampFrequencies,clampPhases,minClampAmplitude,maxClampAmplitude,eVWeight,eVBias,numBoundingSquares]
# LearnableParameters = [numBoundingSquares]
optimizer = torch.optim.Rprop(LearnableParameters,lr=0.02)
bestLoss = 99999
evalDuration = int(evalDurationProp*numSimIters)
for iter in range(numLearnIters):
    fieldParameters = (fieldResolution,fieldStrength,(eVBias,eVWeight,evTimeConstant))
    # fieldParameters = (fieldResolution,fieldStrength,None)
    circuit = cellularFieldNetwork(circuitDims,GRNParameters=(None,None,None,None),
                                                   fieldParameters=fieldParameters,numSamples=numSamples)
    initialValues = defineInitialValues(circuit)
    circuit.initVariables(initialValues)
    circuit.initParameters(initialValues)
    # clampDurationProp.data = torch.clip(clampDurationProp.data,0.0,1.0)
    numBoundingSquares.data = torch.clip(numBoundingSquares.data,minNumBoundingSquares,maxNumBoundingSquares)
    eVBias.data = torch.clip(eVBias.data,mineVBias,maxeVBias)
    if 'Gpol' in clampMode:
        clampValues.data = torch.clip(clampValues.data,0.01,2.0)
    else:
        clampFrequencies.data = torch.clip(clampFrequencies.data,minClampOscillationFrequency,maxClampOscillationFrequency)
        clampPhases.data = torch.clip(clampPhases.data,0.0,2*torch.pi)
        if clampMode == 'fieldDomeFourFoldSymmetry':
            clampFrequenciesActual = torch.tile(clampFrequencies,(4,))
            clampPhasesActual = torch.tile(clampPhases,(4,))
            clampValues = torch.cos(timeIndices*clampFrequenciesActual + clampPhasesActual)
            clampValues = ((clampValues+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
            clampValues = clampValues[:,uniqueClampPointIndices]
        elif clampMode == 'fieldDomeTwoFoldSymmetry':
            clampFrequenciesActual = torch.tile(clampFrequencies,(2,))
            clampPhasesActual = torch.tile(clampPhases,(2,))
            clampValues = torch.cos(timeIndices*clampFrequenciesActual + clampPhasesActual)
            clampValues = ((clampValues+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
            clampValues = clampValues[:,uniqueClampPointIndices]
        else:
            clampValues = torch.cos(timeIndices*clampFrequencies + clampPhases)
            clampValues = ((clampValues+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
    clampParameters = (clampMode,clampIndices,clampValues,(clampStartIter,clampEndIter))
    externalInputs = {'gene': None}
    fieldScreenParameters = {'numBoundingSquares': numBoundingSquares}
    circuit.simulate(externalInputs=externalInputs,fieldEnabled=fieldEnabled,clampParameters=clampParameters,fieldScreenParameters=fieldScreenParameters,
                             perturbationParameters=None,numSimIters=numSimIters,stochasticIonChannels=False,
                             setGradient=False,retainGradients=False,saveData=True)
    # loss = ((targetVmem - circuit.Vmem)**2).sum().sqrt()
    loss = ((targetVmem - circuit.timeseriesVmem[-evalDuration:]) ** 2).sum().sqrt()
    if loss.data < bestLoss:
        bestLoss = loss.data
        bestParameters = [param.clone().detach() for param in LearnableParameters]
    loss.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()
    print(iter,bestLoss)

np.set_printoptions(precision=2,suppress=True)
print("\nFinal Vmem:")
print(circuit.Vmem.data.view(numSamples,*circuitDims).detach().numpy())
if 'field' in clampMode:
    clampValueFull = (np.ones_like(circuit.eV.detach().numpy())*-99)
else:
    clampValueFull = (np.ones_like(circuit.Vmem.detach().numpy()) * -99)
clampValueFull[0,clampPointIndices,0] = clampPhasesActual[uniqueClampPointIndices].detach().numpy().round(decimals=2)
print("\nClamp phases:")
if 'field' in clampMode:
    dims = (circuitRows+1,circuitCols+1)
else:
    dims = (circuitRows,circuitCols)
print(clampValueFull.reshape(numSamples,*dims))

torch.save(bestParameters,'./data/bestParameters.dat')


