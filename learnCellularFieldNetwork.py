import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import utilities

circuitRows,circuitCols = 4,6
circuitDims = (circuitRows,circuitCols)  # (rows,columns) of lattice
fieldResolution = 1
fieldStrength = 10.0
clampMode = 'fieldDome'
# clampVoltage = -0.2
# clampedCellsProp = 0.25
clampDurationProp = 0.1
numSamples = 1
numSimIters = 1000
numLearnIters = 100

utils = utilities.utilities()

fieldParameters = (fieldResolution, fieldStrength, None)
circuit = cellularFieldNetwork(circuitDims,GRNParameters=(None,None,None,None),
                               fieldParameters=fieldParameters,numSamples=numSamples)

fieldDomeIndices = utils.computeDomeIndices(circuit.LatticeDims,circuit.fieldResolution,mode='field')
tissueDomeIndices = utils.computeDomeIndices(circuit.LatticeDims,mode='tissue')
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

# clampVoltage = torch.FloatTensor([-10.1]*numClampPoints*numSamples)
clampVoltage = (torch.rand(numSamples*numClampPoints)*0.06 - 0.06)
# clampVoltage = torch.FloatTensor([-0.0145, -0.4293, -0.4830, -0.3990, -0.4832, -0.4146, -0.0760, -0.4377,
#         -0.4102, -0.4868, -0.5175, -0.4429, -0.4051,  0.0173, -0.4569, -0.4553,
#         -0.4256, -0.4570, -0.4516, -0.0283])
clampVoltage.requires_grad = True
eVBias = torch.FloatTensor([-0.03])
eVBias.requires_grad = True
eVWeight = torch.rand(1)*2-1
# eVWeight = torch.FloatTensor([-2])
eVWeight.requires_grad = True
# evTimeConstant = torch.rand(1)*4
evTimeConstant = torch.FloatTensor([1.0])
evTimeConstant.requires_grad = True

LearnableParameters = [clampVoltage,eVBias,eVWeight]
# LearnableParameters = [clampVoltage]
optimizer = torch.optim.Rprop(LearnableParameters,lr=0.1)
bestLoss = 99999
for iter in range(numLearnIters):
    fieldParameters = (fieldResolution,fieldStrength,(eVBias,eVWeight,evTimeConstant))
    # fieldParameters = (fieldResolution,fieldStrength,None)
    circuit = cellularFieldNetwork(circuitDims,GRNParameters=(None,None,None,None),
                                                   fieldParameters=fieldParameters,numSamples=numSamples)
    circuit.initVariables(initialValues)
    circuit.initParameters(initialValues)
    clampParameters = (clampMode,clampIndices,clampVoltage,clampDurationProp)
    inputs = {'gene': None}
    circuit.simulate(inputs=inputs,clampParameters=clampParameters,numSimIters=numSimIters,saveData=True)
    loss = ((targetVmem - circuit.Vmem)**2).sum().sqrt()
    if loss.data < bestLoss:
        bestLoss = loss.data
        bestParameters = [param.clone().detach() for param in LearnableParameters]
    # print(iter, clampVoltage.min().data, eVBias.data, eVWeight.data, evTimeConstant.data, bestLoss.data)
    loss.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()
    print(iter,bestLoss)

print("\nFinal Vmem:")
print(circuit.Vmem.data.view(numSamples,*circuitDims))

clampVoltageFull = (torch.ones(*circuit.eV.shape)*-99)
clampVoltageFull[0,fieldDomeIndices,0] = clampVoltage
print("\nClamp voltage:")
print(clampVoltageFull.view(numSamples,circuitRows+1,circuitCols+1))

torch.save(bestParameters,'./data/bestParameters.dat')


