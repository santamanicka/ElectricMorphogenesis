import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import utilities

circuitRows,circuitCols = 4,6
circuitDims = (circuitRows,circuitCols)  # (rows,columns) of lattice
fieldResolution = 1
fieldStrength = 10
clampMode = 'fieldDome'
# clampVoltage = -0.2
# clampedCellsProp = 0.25
clampDurationProp = 0.1
numSamples = 5
numSimIters = 1000
numLearnIters = 100

utils = utilities.utilities()

circuit = cellularFieldNetwork(circuitDims,GRNParameters=(None,None,None,None),
                               fieldResolution=fieldResolution,fieldStrength=fieldStrength,
                               numSamples=numSamples)

fieldDomeIndices = utils.computeDomeIndices(circuit.LatticeDims,circuit.fieldResolution,mode='field')
tissueDomeIndices = utils.computeDomeIndices(circuit.LatticeDims, mode='tissue')
clampCellIndices = fieldDomeIndices
clampCellIndices = np.tile(np.array(clampCellIndices),numSamples)
numCells = circuit.numCells
numClampPoints = len(fieldDomeIndices)

sampleIndices = np.repeat(range(numSamples),numClampPoints)
clampIndices = (sampleIndices,clampCellIndices)

initialValues = dict()
initVmem = torch.FloatTensor(list(chain([-9.2e-3] * numSamples)))
initialValues['Vmem'] = torch.repeat_interleave(initVmem,numCells,0).view(numSamples,numCells,1)
initialValues['G_pol'] = dict()
initialValues['G_pol']['cells'] = [[[0]]] * numSamples
initialValues['G_pol']['values'] = [torch.FloatTensor([1.0])] * numSamples  # bistable
initialValues['G_dep'] = dict()
initialValues['G_dep']['cells'] = []
initialValues['G_dep']['values'] = torch.FloatTensor([])

targetVmem = torch.FloatTensor(list(chain([-9.2e-3] * numSamples)))
targetVmem = torch.repeat_interleave(targetVmem,numCells,0).view(numSamples,numCells,1)
targetVmem[:,tissueDomeIndices] = -0.06

# clampVoltage = torch.FloatTensor([0]*numClampPoints*numSamples)
clampVoltage = (torch.rand(numSamples*numClampPoints)*0.06 - 0.06)
clampVoltage.requires_grad = True

LearnableParameters = [clampVoltage]
optimizer = torch.optim.Rprop(LearnableParameters,lr=0.01)
bestLoss = 99999
for iter in range(numLearnIters):
    circuit = cellularFieldNetwork(circuitDims,GRNParameters=(None,None,None,None),
                                                   fieldResolution=fieldResolution,fieldStrength=fieldStrength,
                                                   numSamples=numSamples)
    circuit.initVariables(initialValues)
    circuit.initParameters(initialValues)
    clampParameters = (clampMode,clampIndices,clampVoltage,clampDurationProp)
    circuit.simulate(clampParameters=clampParameters,numSimIters=numSimIters,saveData=True)
    loss = ((targetVmem - circuit.Vmem)**2).sum().sqrt()
    if loss.data < bestLoss:
        bestLoss = loss.data
        bestClampVoltage = clampVoltage.clone().detach()
    loss.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()
    print(iter,bestLoss)

print("\nFinal Vmem:")
print(circuit.Vmem.data.view(numSamples,*circuitDims))


