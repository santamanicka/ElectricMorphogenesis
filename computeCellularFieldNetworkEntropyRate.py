import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import utilities
from scipy.stats import entropy
import math

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
maxNumBoundingSquares = 2*max(circuitDims) - 1  # Max value of numBoundingSquares so the field will permeate the entire tissue = 2(l-1)+1, where l is the max of circuitDims
# numBoundingSquares = 4
eVBias = torch.DoubleTensor([0.0214])  # 0.0214
eVWeight = torch.DoubleTensor([9.4505])  # 9.4505
evTimeConstant = torch.DoubleTensor([10.0])
numSamples = 1
numSimIters = 100000
numCells = circuitRows * circuitCols
BlockGapJunctions = False
AmplifyGapJunctions = False

VmemBins = np.arange(-0.0, -0.1, -0.04)

fieldParameters = (fieldResolution,fieldStrength,(eVBias,eVWeight,evTimeConstant))

initialValues = dict()
initVmem = list(chain([-9.2e-3] * numSamples))
initialValues['Vmem'] = torch.repeat_interleave(torch.DoubleTensor(initVmem),numCells,0).view(numSamples,numCells,1)
initialValues['G_pol'] = dict()
initialValues['G_pol']['cells'] = [[[0]]] * numSamples
initialValues['G_pol']['values'] = [torch.DoubleTensor([1.0])] * numSamples  # bistable
initialValues['G_dep'] = dict()
initialValues['G_dep']['cells'] = []
initialValues['G_dep']['values'] = torch.DoubleTensor([])

circuit = cellularFieldNetwork(circuitDims,GRNParameters=(None,None,None,None),
                               fieldParameters=fieldParameters,numSamples=numSamples)
circuit.initVariables(initialValues)
circuit.initParameters(initialValues)

numExtracellularGridPoints = circuit.numExtracellularGridPoints

utils = utilities.utilities()
fieldDomeIndices = utils.computeDomeIndices(circuit.LatticeDims,circuit.fieldResolution,mode='field')

# block gap junctions by zeroing GJ current
if BlockGapJunctions:
    circuit.G_0 = 0.0
    circuit.G_res = 0.0
elif AmplifyGapJunctions:
    circuit.G_0 = 0.05 * circuit.G_ref
    circuit.G_res = 0.0

def computeEntropy():
    nr = math.ceil(circuitDims[0]/2); nc = math.ceil(circuitDims[1]/2); r=circuit.cell_radius
    topQuadrantCoords = ((circuit.cellularCoordinates[0] <= (r*(2*nr-1))) & (circuit.cellularCoordinates[1] <= (r*(2*nc-1))))[0]
    topQuadrantIdx = np.arange(circuit.numCells)[topQuadrantCoords]
    boundaryCoords = ((circuit.cellularCoordinates[0] == r) | (circuit.cellularCoordinates[1] == r))[0]
    topQuadrantBoundaryIdx = np.arange(circuit.numCells)[boundaryCoords & topQuadrantCoords]
    topQuadrantBulkIdx = np.setdiff1d(topQuadrantIdx,topQuadrantBoundaryIdx)
    boundaryStates, bulkStates, allStates = [], [], []
    vmemBinary = np.digitize(circuit.timeseriesVmem[:,0,:,0],VmemBins)
    for t in range(vmemBinary.shape[0]):
        boundaryStates.append(int(''.join(str(i) for i in vmemBinary[t,topQuadrantBoundaryIdx]),2))
        bulkStates.append(int(''.join(str(i) for i in vmemBinary[t,topQuadrantBulkIdx]),2))
        allStates.append(int(''.join(str(i) for i in vmemBinary[t,topQuadrantIdx]),2))
    HBoundary, HBulk, HAll = [], [], []
    for t in range(vmemBinary.shape[0]):
        countsBoundary = np.unique(boundaryStates[:t],return_counts=True)[1]
        probsBoundary = countsBoundary/sum(countsBoundary)
        HBoundary.append(entropy(probsBoundary))
        countsBulk = np.unique(bulkStates[:t], return_counts=True)[1]
        probsBulk = countsBulk / sum(countsBulk)
        HBulk.append(entropy(probsBulk))
        countsAll = np.unique(allStates[:t], return_counts=True)[1]
        probsAll = countsAll / sum(countsAll)
        HAll.append(entropy(probsAll))
    Entropies = (HBoundary, HBulk, HAll)
    return Entropies

data = dict()
for numBoundingSquares in range(1,maxNumBoundingSquares+1):
    print("numBoundingSquares = ",numBoundingSquares)
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
    Entropies = computeEntropy()
    HBoundary, HBulk, HAll = Entropies
    data[numBoundingSquares] = dict()
    data[numBoundingSquares]['HBoundary'] = HBoundary
    data[numBoundingSquares]['HBulk'] = HBulk
    data[numBoundingSquares]['HAll'] = HAll
    torch.save(data,'./data/EntropyRates.dat')
