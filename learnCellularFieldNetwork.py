import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import utilities

circuitRows,circuitCols = 6,6
circuitDims = (circuitRows,circuitCols)  # (rows,columns) of lattice
fieldEnabled = True
fieldResolution = 1
fieldStrength = 10.0
numBoundingSquares = 2
clampMode = 'fieldDome'
# clampVoltage = -0.2
# clampedCellsProp = 0.25
# clampDurationProp = 0.1
evalDurationProp = 0.1
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

def defineInitialValues(circuit):
    initialValues = dict()
    initVmem = torch.FloatTensor(list(chain([-9.2e-3] * numSamples)))
    initialValues['Vmem'] = torch.repeat_interleave(initVmem,circuit.numCells,0).double().view(numSamples,numCells,1)
    initialValues['eV'] = torch.zeros((numSamples,circuit.numExtracellularGridPoints,1),dtype=torch.float64)
    initialValues['G_pol'] = dict()
    initialValues['G_pol']['cells'] = [[[0]]] * numSamples
    initialValues['G_pol']['values'] = [torch.DoubleTensor([1.0])] * numSamples  # bistable
    initialValues['G_dep'] = dict()
    initialValues['G_dep']['cells'] = []
    initialValues['G_dep']['values'] = torch.DoubleTensor([])
    return initialValues

targetVmem = torch.FloatTensor(list(chain([-9.2e-3] * numSamples)))
targetVmem = torch.repeat_interleave(targetVmem,numCells,0).view(numSamples,numCells,1)
targetVmem[:,tissueDomeIndices] = -0.06
targetVmem[:,[14,15,20,21]] = -0.06

# clampDurationProp = torch.FloatTensor([0.1])
# clampDurationProp.requires_grad = True
clampStartIter, clampEndIter = 0, int(0.1*numSimIters)
# clampVoltage = torch.FloatTensor([-10.1]*numClampPoints*numSamples)
clampVoltage = (torch.rand(numSamples*numClampPoints,dtype=torch.double)*0.06 - 0.06)
# clampVoltage = torch.FloatTensor([-0.0145, -0.4293, -0.4830, -0.3990, -0.4832, -0.4146, -0.0760, -0.4377,
#         -0.4102, -0.4868, -0.5175, -0.4429, -0.4051,  0.0173, -0.4569, -0.4553,
#         -0.4256, -0.4570, -0.4516, -0.0283])
clampVoltage.requires_grad = True
eVBias = torch.DoubleTensor([0.0214])  # 0.0214
eVWeight = torch.DoubleTensor([9.4505])  # 9.4505
evTimeConstant = torch.DoubleTensor([10.0])
# eVBias = torch.FloatTensor([-0.03])
# eVBias.requires_grad = True
# eVWeight = torch.rand(1)*2-1
# eVWeight = torch.FloatTensor([-2])
# eVWeight.requires_grad = True
# evTimeConstant = torch.rand(1)*4
# evTimeConstant = torch.FloatTensor([1.0])
# evTimeConstant.requires_grad = True

LearnableParameters = [clampVoltage]
# LearnableParameters = [clampVoltage]
optimizer = torch.optim.Rprop(LearnableParameters,lr=0.08)
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
    clampParameters = (clampMode,clampIndices,clampVoltage,(clampStartIter,clampEndIter))
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

print("\nFinal Vmem:")
print(circuit.Vmem.data.view(numSamples,*circuitDims))

clampVoltageFull = (np.ones_like(circuit.eV.detach().numpy())*-99)
clampVoltageFull[0,fieldDomeIndices,0] = clampVoltage.detach().numpy().round(decimals=2)
print("\nClamp voltage:")
print(clampVoltageFull.reshape(numSamples,circuitRows+1,circuitCols+1))

torch.save(bestParameters,'./data/bestParameters.dat')


