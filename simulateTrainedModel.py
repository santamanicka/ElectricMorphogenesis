from cellularFieldNetwork import cellularFieldNetwork
import numpy as np
import torch
import utilities
from itertools import chain

parameterfilename = './data/bestModelParameters_483.dat'  # 483 (fieldRange=4); 759 (fieldRange=1); 825 (fieldRange=21)
parameters = torch.load(parameterfilename)

Autonomous = True  # impose homogenous initial conditions under unclamped conditions
randomizeInitialState = False  # applies only of Autonomous=True
tempFieldParamsOverride = True
Perturb = False
perturbationMode = None  # options: swapVmem, permuteVmem
newSimulationLength = (True,5000)
newVmemLigandStrength = (False,3.0)
TurnoffField = False
TurnoffLigand = False

latticeDims = parameters['latticeDims']
GJParameters = parameters['GJParameters']
fieldParameters = parameters['fieldParameters']
fieldParameters['fieldStrength'] = 10.0
if 'ligandParameters' in parameters.keys():
    ligandParameters = parameters['ligandParameters']
    if 'vmemToLigandCurrentStrength' not in ligandParameters.keys():
        ligandParameters['vmemToLigandCurrentStrength'] = 0.1
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
    fieldParameters['fieldTransductionWeight'] = torch.DoubleTensor([11.11])
    fieldParameters['fieldTransductionBias'] = 0.03
    # fieldParameters['fieldScreenSize'] = 21

# indices of the features of the 11x11 smiley
eyeIndices = np.array([24,25,35,36,29,30,40,41])  # left and right eyes
noseIndices = np.array([49,60,71])
mouthIndices = np.array([92,93,94])
allTissueIndices = np.arange(numCells)

def computeLoss(method='globalsum'):
    if method == 'globalsum':
        loss = ((targetVmem - circuit.timeseriesVmem[-evalDuration:]) ** 2).sum().sqrt()
    elif method == 'globalmean':
        loss = ((targetVmem - circuit.timeseriesVmem[-evalDuration:]) ** 2).mean().sqrt()
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

circuit = cellularFieldNetwork(latticeDims=latticeDims,parameters=modelparameters,numSamples=numSamples)
circuit.initVariables(initialValues)
circuit.initParameters(initialValues)

if Autonomous:
    initVmem = list(chain([-9.2e-3] * numSamples))
    initialValues['Vmem'] = torch.repeat_interleave(torch.DoubleTensor(initVmem),circuit.numCells,0).view(numSamples,circuit.numCells,1)
    initialValues['eV'] = torch.zeros((numSamples,circuit.numExtracellularGridPoints,1),dtype=torch.float64)
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
    elif perturbationMode == 'permuteVmem':  # randomly shuffle the entire tissue
        perturbPointIndicesA = np.tile(allTissueIndices,numSamples)
        perturbPointIndicesB = np.concatenate([torch.randperm(numCells) for _ in range(numSamples)])
    numPerturbPoints = len(perturbPointIndicesA)
    sampleIndices = np.repeat(range(numSamples),numPerturbPoints)  # assuming that there's only one sample in which the eye block is shifted
    perturbStartIter, perturbEndIter = 1000, 1000
    perturbation['mode'] = perturbationMode
    perturbation['data'] = (sampleIndices,(perturbPointIndicesA,perturbPointIndicesB))
    perturbation['time'] = (perturbStartIter,perturbEndIter)
else:
    perturbation = None

if newSimulationLength[0]:
    numSimIters = newSimulationLength[1]

circuit.simulate(externalInputs=externalInputs,clampParameters=clampParameters,perturbationParameters=perturbation,
                 numSimIters=numSimIters,stochasticIonChannels=False,setGradient=False,retainGradients=False,saveData=True)
# circuit.G_0 = 10.0 * circuit.G_ref  # override GJ parameters
# circuit.fieldScreenSize = 0  # override field parameters
# circuit.simulate(externalInputs=externalInputs,clampParameters=None,perturbationParameters=None,
# 				 numSimIters=numSimIters*2,stochasticIonChannels=False,setGradient=False,retainGradients=False,resume=True,saveData=True)
evalDuration = int(evalDurationProp*numSimIters)
# loss = ((targetVmem - circuit.timeseriesVmem[-evalDuration:]) ** 2).sum().sqrt()
loss = computeLoss(method=lossMethod)
np.set_printoptions(precision=2,suppress=True)
print("Recorded loss: ",parameters['trainParameters']['bestLoss'])
print("Actual loss: ",loss)
