import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import utilities
import matplotlib.pyplot as plt

circuitRows,circuitCols = 1,1
circuitDims = (circuitRows,circuitCols)  # (rows,columns) of lattice
fieldEnabled = False
fieldResolution = 1
fieldStrength = 10.0
fieldAggregation = 'average'
fieldScreenSize = 1
fieldTransductionWeight = 10.0
fieldTransductionBias = 0.03
ligandEnabled = False
ligandGatingWeight = 10.0
ligandGatingBias = -0.5
ligandCurrentStrength = 10.0
ligandCurrentStrength = 1.0
vmemToLigandCurrentStrength = 0.1
GJStrength = 0.05
clampMode = None  # possible values: field, tissueVmem, tissueGpol, fieldDome, tissueDomeVmem, tissueDomeVGpol, None
clampValue = 0.6  # eV:[-0.0332,-0.005]; vmem:[-0.05,-0.01]; G_pol:[0.6,1.6]
clampedCellsProp = 0.0
if clampedCellsProp == 0.0:
    clampMode = None
# clampDurationProp = 0.0
clampStartIter, clampEndIter = 450, 700  # 450-700
numBasicSamples = 1
numNoisySamples = 1
noise = 0.0  # std of normal distribution
numSamples = numBasicSamples * numNoisySamples
numSimIters = 10
BlockGapJunctions = False
AmplifyGapJunctions = False
eVBias = torch.DoubleTensor([0.0214])  # 0.0214
eVWeight = torch.DoubleTensor([9.4505])  # 9.4505
evTimeConstant = torch.DoubleTensor([10.0])

numParams = 1
plot = False

numBoundingSquares = 2*max(circuitDims) - 1
perturbMode = None  # possible values: tissueDome, tissueDomePartial, None
fieldEnabled = True

GRNtoVmemWeights,GRNBiases,GRNtoVmemWeightsTimeconstant,GRNNumGenes = None,None,None,None

fieldParameterNames = ['fieldEnabled','fieldResolution','fieldStrength','fieldAggregation','fieldScreenSize',
                       'fieldTransductionWeight','fieldTransductionBias','fieldTransductionTimeConstant']
ligandParameterNames = ['ligandEnabled','ligandGatingWeight','ligandGatingBias','ligandCurrentStrength','vmemToLigandCurrentStrength']
GRNParameterNames = ['GRNtoVmemWeights','GRNBiases','GRNtoVmemWeightsTimeconstant','GRNNumGenes']

GRNParameters = dict()
for param in GRNParameterNames:
    GRNParameters[param] = eval(param)

parameters = dict()
parameters['GJParameters'] = GJParameters
parameters['fieldParameters'] = fieldParameters
parameters['ligandParameters'] = ligandParameters
parameters['GRNParameters'] = GRNParameters

fieldParameters = (fieldResolution,fieldStrength,(eVBias,eVWeight,evTimeConstant))
circuit = cellularFieldNetwork(circuitDims,GRNParameters=(None,None,None,None),
                               fieldParameters=fieldParameters,numSamples=numSamples)

numCells = circuit.numCells

initialValues = dict()
initVmem = torch.linspace(-0.2,0,numSamples,dtype=torch.float64)
# initVmem = torch.linspace(0.0,0,numSamples,dtype=torch.float64)
initVmem = list(chain([-9.2e-3] * numSamples))
initialValues['Vmem'] = torch.repeat_interleave(torch.DoubleTensor(initVmem),numCells,0).view(numSamples,numCells,1)
initialValues['eV'] = torch.zeros((numSamples,circuit.numFieldGridPoints,1),dtype=torch.float64)
initialValues['G_pol'] = dict()
initialValues['G_pol']['cells'] = [[[0]]] * numSamples
initialValues['G_pol']['values'] = [torch.DoubleTensor([0.7])] * numSamples  # Equilibrium Vmems: 0.1=dep{-0.0053}; 1.0=bistable{-0.05,-0.0092}; 2.0=hyp{-0.05}
initialValues['G_dep'] = dict()
initialValues['G_dep']['cells'] = []
# initialValues['G_dep']['values'] = torch.FloatTensor([])
initialValues['G_dep']['values'] = [torch.DoubleTensor([1.5])] * numSamples

circuit.initVariables(initialValues)
circuit.initParameters(initialValues)

utils = utilities.utilities()
fieldDomeIndices = utils.computeDomeIndices(circuit,mode='field')
# print("Initial Vmem:")
# print(circuit.Vmem.view(numSamples,*circuitDims))
if clampMode == None:
    pass
elif clampMode == 'field':
    numTotalCells = circuit.numFieldGridPoints
    cellIndices = np.arange(numTotalCells)
    numFieldCells = numTotalCells
elif clampMode == 'fieldDome':
    numTotalCells = len(fieldDomeIndices)
    cellIndices = fieldDomeIndices
    numFieldCells = numTotalCells
elif 'tissueDome' in clampMode:  # covers both 'tissueDomeVmem' and 'tissueDomeGpol'
    tissueDomeIndices = utils.computeDomeIndices(circuit,mode='tissue')
    numTotalCells = len(tissueDomeIndices)
    cellIndices = tissueDomeIndices
    numFieldCells = circuit.numFieldGridPoints
elif 'tissue' in clampMode:  # 'tissueDome' condition must precede 'tissue' since the latter is contained in the former
    numTotalCells = circuit.numCells
    cellIndices = np.arange(numTotalCells)
    numFieldCells = circuit.numFieldGridPoints
if clampMode != None:
    numClampedCells = int(clampedCellsProp*numTotalCells)
    clampPointIndices = np.array([np.random.choice(cellIndices,numClampedCells,replace=False)
                                             for _ in range(numSamples)]).reshape(-1,)
    sampleIndices = np.repeat(range(numSamples),numClampedCells)
    clampIndices = (sampleIndices,clampPointIndices)
    # clampIndices = [4,5,6,7,8,84,85,86,87,88]  # surrounding the middle two columns of 4x6
    clampParameters = (clampMode,clampIndices,clampValue,(clampStartIter,clampEndIter))
else:
    clampParameters = None
externalInputs = {'gene':None}
screenParameters = {'numBoundingSquares':numBoundingSquares}
circuit.simulate(externalInputs=externalInputs,fieldEnabled=fieldEnabled,clampParameters=clampParameters,fieldScreenParameters=screenParameters,
                 perturbationParameters=None,numSimIters=numSimIters,stochasticIonChannels=False,
                 setGradient=True,retainGradients=True,saveData=True)
# print("\nFinal Vmem:")
# np.set_printoptions(precision=2, suppress=True)  # suppresses scientific notation such as the suffix in 100e+02
# print(circuit.Vmem.view(numSamples,*circuitDims))

if plot:
    # for s in range(numSamples):
    #     plt.plot(circuit.timeseriesVmem[:,s,:,0])
    vt = circuit.timeseriesVmem[:,0,:,0].clone().detach().numpy()
    vt /= vt.__abs__().max()
    evt = circuit.timeserieseV[:,0,:,0].clone().detach().numpy()
    evt /= evt.__abs__().max()
    gt = circuit.timeseriesGpol[:,0,:,0].clone().detach().numpy()
    gt /= gt.__abs__().max()
    _,_ = plt.subplots()
    plt.plot(vt,color='red')
    for g in range(circuit.numFieldGridPoints):
        plt.plot(evt[:,g],color='blue')
    plt.plot(gt,color='red')
    # plt.show()
    plt.xlabel('Time')
    plt.ylabel('Vmem')
    plt.title('Clamp mode = ' + str(clampMode))
    plt.show()

# params = torch.linspace(10**4,10**7,numParams)
# timeseriesVmem = torch.FloatTensor([-999]*numParams*numSimIters*numSamples*numCells).view(numParams,numSimIters,numSamples,numCells,1)
# for i in range(len(params)):
#     param = params[i]
#     circuit.relativePermittivity = param
#     circuit.initVariables(initialValues)
#     circuit.initParameters(initialValues)
#     for t in range(numSimIters):
#         circuit.simulate(numSimIters=1)
#         timeseriesVmem[i][t] = circuit.Vmem
#
# maxt = -1
# for i in range(len(params)):
#     plt.plot(timeseriesVmem[i][0:maxt,:,0,0])
# plt.xlabel('Time')
# plt.ylabel('Vmem')
# plt.show()

# params = torch.linspace(1.0,2.0,numParams)
# timeseriesVmem = torch.FloatTensor([-999]*numParams*numSimIters*numSamples*numCells).view(numParams,numSimIters,numSamples,numCells,1)
# circuit.relativePermittivity = 10**7
# for i in range(len(params)):
#     param = params[i]
#     initialValues['G_pol']['values'] = [torch.FloatTensor([param])] * numSamples
#     circuit.initVariables(initialValues)
#     circuit.initParameters(initialValues)
#     for t in range(numSimIters):
#         circuit.simulate(numSimIters=1)
#         timeseriesVmem[i][t] = circuit.Vmem

# if plot:
#     maxt = -1
#     for i in range(len(params)):
#         plt.plot(timeseriesVmem[i][0:maxt,:,0,0])
#     plt.xlabel('Time')
#     plt.ylabel('Vmem')
#     plt.show()

# timeseriesVmem1 = torch.FloatTensor([-999]*numSimIters*numSamples*numCells).view(numSimIters,numSamples,numCells,1)
# for t in range(numSimIters):
#     circuit.simulate(numSimIters=1)
#     timeseriesVmem1[t] = circuit.Vmem
#
# initialValues['G_pol']['values'] = [torch.FloatTensor([1.8])] * numSamples
# circuit.initVariables(initialValues)
# circuit.initParameters(initialValues)
# circuit.relativePermittivity = 10**(4.5)
#
# timeseriesVmem2 = torch.FloatTensor([-999]*numParams*numSimIters*numSamples*numCells).view(numSimIters,numSamples,numCells,1)
#
# for t in range(numSimIters):
#     circuit.simulate(numSimIters=1)
#     timeseriesVmem2[t] = circuit.Vmem
#
# initialValues['G_pol']['values'] = [torch.FloatTensor([1.8])] * numSamples
# circuit.initVariables(initialValues)
# circuit.initParameters(initialValues)
# circuit.relativePermittivity = 10**(4.0)
#
# timeseriesVmem3 = torch.FloatTensor([-999]*numSimIters*numSamples*numCells).view(numSimIters,numSamples,numCells,1)
# for t in range(numSimIters):
#     circuit.simulate(numSimIters=1)
#     timeseriesVmem3[t] = circuit.Vmem
#
# initialValues['G_pol']['values'] = [torch.FloatTensor([1.8])] * numSamples
# circuit.initVariables(initialValues)
# circuit.initParameters(initialValues)
# circuit.relativePermittivity = 10**(3.0)
#
# timeseriesVmem4 = torch.FloatTensor([-999]*numSimIters*numSamples*numCells).view(numSimIters,numSamples,numCells,1)
# for t in range(numSimIters):
#     circuit.simulate(numSimIters=1)
#     timeseriesVmem4[t] = circuit.Vmem
#
# maxt = 40
# plt.plot(timeseriesVmem1[0:maxt,:,0,0])
# plt.plot(timeseriesVmem2[0:maxt,:,0,0])
# plt.plot(timeseriesVmem3[0:maxt,:,0,0])
# plt.plot(timeseriesVmem4[0:maxt,:,0,0])
# plt.show()
