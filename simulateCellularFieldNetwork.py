import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import utilities

circuitRows,circuitCols = 4,6
circuitDims = (circuitRows,circuitCols)  # (rows,columns) of lattice
fieldResolution = 1
fieldStrength = 10
clampVoltage = -0.1
clampDurationPercent = 0.1
numBasicSamples = 1
numNoisySamples = 1
noise = 0.0  # std of normal distribution
numSamples = numBasicSamples * numNoisySamples
numSimIters = 10000
BlockGapJunctions = False
AmplifyGapJunctions = False

circuit = cellularFieldNetwork(circuitDims,GRNParameters=(None,None,None,None),
                               fieldResolution=fieldResolution,fieldStrength=fieldStrength,
                               numSamples=numSamples)

numCells = circuit.numCells
numExtracellularGridPoints = circuit.numExtracellularGridPoints

def setupExperimentalConditions(latticeDims):
    circuitRows,circuitCols = latticeDims
    numCells = circuitRows * circuitCols
    initialValues = dict()
    initVmem = list(chain([-9.2e-3] * numBasicSamples * numNoisySamples))
    initialValues['Vmem'] = torch.repeat_interleave(torch.FloatTensor(initVmem),numCells,0).view(numSamples,numCells,1)
    bandWidth = int(circuitCols/3)  # 3 bands: depolarized, hyperpolarized, depolarized
    HyperpolarizedColumns = list(range(bandWidth,2*bandWidth)) # middle band
    HyperpolarizedCells = np.concatenate([np.arange(col,numCells,circuitCols) for col in HyperpolarizedColumns])
    DepolarizedCells = np.setdiff1d(range(numCells),HyperpolarizedCells).tolist()
    AllCells = list(range(numCells))
    initialValues['G_pol'] = dict()
    # ordering: incorrect pattern: all depolarized; correct pattern: V-Shape; incorrect pattern: all polarized
    initVems = []
    if numSamples == 1:  # DHD only
        initialValues['G_pol']['cells'] = list(chain([[DepolarizedCells,HyperpolarizedCells]] * numNoisySamples))
        # DepVmem, HypVmem = torch.FloatTensor([1.0,1.8]) + np.random.normal(0,noise,2).astype(np.float32)
        DepVmem, HypVmem = torch.FloatTensor([1.0,1.0]) + np.random.normal(0,noise,2).astype(np.float32)
        initVems.extend([[DepVmem,HypVmem]])
    elif numSamples > 1:
        initialValues['G_pol']['cells'] = [[AllCells,[]],[DepolarizedCells,HyperpolarizedCells],[[],AllCells]] * numNoisySamples
        for s in range(numNoisySamples):
            DepVmem, HypVmem = torch.FloatTensor([1.0,1.8]) + np.random.normal(0,noise,2).astype(np.float32)
            initVems.extend([[DepVmem,HypVmem],[DepVmem,HypVmem],[DepVmem,HypVmem]])
    initialValues['G_pol']['values'] = initVems
    initialValues['G_dep'] = dict()
    initialValues['G_dep']['cells'] = []  # if this is an empty list then 'values' below won't be applied during initialization
    initialValues['G_dep']['values'] = torch.FloatTensor([np.nan]*numSamples).view(numSamples,1,1)
    return initialValues

initialValues = setupExperimentalConditions(circuitDims)
circuit.initVariables(initialValues)
circuit.initParameters(initialValues)

utils = utilities.utilities()
electrodomeIndices = utils.computeElectrodomeIndices(circuit.LatticeDims,circuit.fieldResolution)

# block gap junctions by zeroing GJ current
if BlockGapJunctions:
    circuit.G_0 = 0.0
    circuit.G_res = 0.0
elif AmplifyGapJunctions:
    circuit.G_0 = 0.05 * circuit.G_ref
    circuit.G_res = 0.0

print("Initial Vmem:")
print(circuit.Vmem.view(numSamples,*circuitDims))
timeseriesVmem = torch.FloatTensor([-999]*numSimIters*numSamples*numCells).view(numSimIters,numSamples,numCells,1)
timeserieseV = torch.FloatTensor([-999]*numSimIters*numSamples*numExtracellularGridPoints).view(numSimIters,numSamples,numExtracellularGridPoints,1)
clampMode = 'field'
# clampIndices = electrodomeIndices
# clampIndices = np.random.choice(electrodomeIndices,int(0.45*len(electrodomeIndices)),replace=False)
# clampIndices = [4,5,6,7,8,84,85,86,87,88]  # surrounding the middle two columns of 4x6
clampIndices = np.random.choice(circuit.numExtracellularGridPoints,int(0.2*circuit.numExtracellularGridPoints))
clampParameters = (clampMode,clampIndices,clampVoltage,clampDurationPercent)
circuit.simulate(clampParameters=clampParameters,numSimIters=numSimIters,saveData=True)
print("\nFinal Vmem:")
np.set_printoptions(precision=2, suppress=True)  # suppresses scientific notation such as the suffix in 100e+02
print(circuit.Vmem.view(numSamples,*circuitDims))
counts = torch.unique(circuit.Vmem.round(decimals=2),return_counts=True)
print("\nCounts of unique Vmems: ",counts)