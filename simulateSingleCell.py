import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import utilities
import matplotlib.pyplot as plt

circuitRows,circuitCols = 1,2
circuitDims = (circuitRows,circuitCols)  # (rows,columns) of lattice
fieldResolution = 1
fieldStrength = 10
clampVoltage = -0.1
clampDurationPercent = 0.032
numBasicSamples = 1
numNoisySamples = 1
noise = 0.0  # std of normal distribution
numSamples = numBasicSamples * numNoisySamples
numSimIters = 1000
BlockGapJunctions = False
AmplifyGapJunctions = False

numParams = 1
plot = False

circuit = cellularFieldNetwork(circuitDims,GRNParameters=(None,None,None,None),
                               fieldResolution=fieldResolution,fieldStrength=fieldStrength,
                               numSamples=numSamples)

numCells = circuit.numCells

initialValues = dict()
# initVmem = torch.linspace(-0.1,0,numSamples)
initVmem = list(chain([-9.2e-3] * numSamples))
initialValues['Vmem'] = torch.repeat_interleave(torch.FloatTensor(initVmem),numCells,0).view(numSamples,numCells,1)
initialValues['G_pol'] = dict()
initialValues['G_pol']['cells'] = [[[0]]] * numSamples
initialValues['G_pol']['values'] = [torch.FloatTensor([1.0])] * numSamples  # bistable
initialValues['G_dep'] = dict()
initialValues['G_dep']['cells'] = []
initialValues['G_dep']['values'] = torch.FloatTensor([])

circuit.initVariables(initialValues)
circuit.initParameters(initialValues)

utils = utilities.utilities()
electrodomeIndices = utils.computeElectrodomeIndices(circuit.LatticeDims,circuit.fieldResolution)
print("Initial Vmem:")
print(circuit.Vmem.view(numSamples,*circuitDims))
clampIndices = electrodomeIndices
clampFieldParameters = (clampIndices,clampVoltage,clampDurationPercent)
circuit.simulate(clampFieldParameters=clampFieldParameters,numSimIters=numSimIters,saveData=True)
print("\nFinal Vmem:")
np.set_printoptions(precision=2, suppress=True)  # suppresses scientific notation such as the suffix in 100e+02
print(circuit.Vmem.view(numSamples,*circuitDims))

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

if plot:
    maxt = -1
    for i in range(len(params)):
        plt.plot(timeseriesVmem[i][0:maxt,:,0,0])
    plt.xlabel('Time')
    plt.ylabel('Vmem')
    plt.show()

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
