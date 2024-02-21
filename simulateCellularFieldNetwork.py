import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import utilities

circuitRows,circuitCols = 10,10
circuitDims = (circuitRows,circuitCols)  # (rows,columns) of lattice
fieldResolution = 1
fieldStrength = 10.0
clampMode = 'fieldDome'
clampVoltage = -0.1
clampedCellsProp = 0.0
if clampedCellsProp == 0.0:
    clampMode = None
clampDurationProp = 0.0
# numBoundingSquares = 2*max(circuitDims) - 1  # Max value of numBoundingSquares so the field will permeate the entire tissue = 2(l-1)+1, where l is the max of circuitDims
numBoundingSquares = 4
eVBias = torch.DoubleTensor([0.0214])  # 0.0214
eVWeight = torch.DoubleTensor([9.4505])  # 9.4505
evTimeConstant = torch.DoubleTensor([10.0])
numSamples = 1
numSimIters = 10000
RandomizeInitialState = True
BlockGapJunctions = False
AmplifyGapJunctions = False

VmemBins = np.arange(-0.0, -0.1, -0.04)

fieldParameters = (fieldResolution,fieldStrength,(eVBias,eVWeight,evTimeConstant))
circuit = cellularFieldNetwork(circuitDims,GRNParameters=(None,None,None,None),
                               fieldParameters=fieldParameters,numSamples=numSamples)

numCells = circuit.numCells
numExtracellularGridPoints = circuit.numExtracellularGridPoints

initialValues = dict()
initVmem = list(chain([-9.2e-3] * numSamples))
initialValues['Vmem'] = torch.repeat_interleave(torch.DoubleTensor(initVmem),numCells,0).view(numSamples,numCells,1)
initialValues['G_pol'] = dict()
AllCells = list(range(numCells))
initialValues['G_pol']['cells'] = [[AllCells]] * numSamples
if RandomizeInitialState:
    initialValues['G_pol']['values'] = [[torch.rand(numCells,dtype=torch.float64)*2]] * numSamples
else:
    initialValues['G_pol']['values'] = [torch.DoubleTensor([1.0])] * numSamples  # bistable
initialValues['G_dep'] = dict()
initialValues['G_dep']['cells'] = []
initialValues['G_dep']['values'] = torch.DoubleTensor([])

circuit.initVariables(initialValues)
circuit.initParameters(initialValues)

utils = utilities.utilities()
fieldDomeIndices = utils.computeDomeIndices(circuit.LatticeDims,circuit.fieldResolution,mode='field')

# block gap junctions by zeroing GJ current
if BlockGapJunctions:
    circuit.G_0 = 0.0
    circuit.G_res = 0.0
elif AmplifyGapJunctions:
    circuit.G_0 = 0.05 * circuit.G_ref
    circuit.G_res = 0.0

print("Initial Vmem:")
print(circuit.Vmem.view(numSamples,*circuitDims))
timeseriesVmem = torch.DoubleTensor([-999]*numSimIters*numSamples*numCells).view(numSimIters,numSamples,numCells,1)
timeserieseV = torch.DoubleTensor([-999]*numSimIters*numSamples*numExtracellularGridPoints).view(numSimIters,numSamples,numExtracellularGridPoints,1)
if clampMode == 'field':
    numTotalCells = circuit.numExtracellularGridPoints
    cellIndices = np.arange(numTotalCells)
    numFieldCells = numTotalCells
elif clampMode == 'tissue':
    numTotalCells = circuit.numCells
    cellIndices = np.arange(numTotalCells)
    numFieldCells = circuit.numExtracellularGridPoints
elif clampMode == 'fieldDome':
    numTotalCells = len(fieldDomeIndices)
    cellIndices = fieldDomeIndices
    numFieldCells = numTotalCells
elif clampMode == 'tissueDome':
    tissueDomeIndices = utils.computeDomeIndices(circuit.LatticeDims,mode='tissue')
    numTotalCells = len(tissueDomeIndices)
    cellIndices = tissueDomeIndices
    numFieldCells = circuit.numExtracellularGridPoints
fieldParameters = (fieldResolution,fieldStrength,(eVBias,eVWeight,evTimeConstant))
if clampMode != None:
    numClampedCells = int(clampedCellsProp*numTotalCells)
    clampPointIndices = np.array([np.random.choice(cellIndices,numClampedCells,replace=False)
                                             for _ in range(numSamples)]).reshape(-1,)
    sampleIndices = np.repeat(range(numSamples),numClampedCells)
    clampIndices = (sampleIndices,clampPointIndices)
    # clampIndices = [4,5,6,7,8,84,85,86,87,88]  # surrounding the middle two columns of 4x6
    clampParameters = (clampMode,clampIndices,clampVoltage,clampDurationProp)
else:
    clampParameters = None
inputs = {'gene':None}
screenParameters = {'numBoundingSquares':numBoundingSquares}
circuit.simulate(inputs=inputs,clampParameters=clampParameters,screenParameters = screenParameters,numSimIters=numSimIters,saveData=True)
print("\nFinal Vmem:")
np.set_printoptions(precision=2, suppress=True)  # suppresses scientific notation such as the suffix in 100e+02
print(circuit.Vmem.view(numSamples,*circuitDims))
counts = [np.unique(np.digitize(circuit.Vmem.round(decimals=2)[i],VmemBins),return_counts=True)[1] for i in range(circuit.Vmem.shape[0])]
print(*counts,sep='\n')
# counts = torch.unique(circuit.Vmem.round(decimals=2),return_counts=True)
# print("\nCounts of unique Vmems: ",counts)