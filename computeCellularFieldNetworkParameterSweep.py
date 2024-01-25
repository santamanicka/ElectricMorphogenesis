import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import utilities
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--clampMode', type=str, default='field')

args = parser.parse_args()
clampMode = args.clampMode

circuitRows,circuitCols = 4,6
numParameterValues = 10
numFieldResolutions = 5
numSamples = 100
numSimIters = 10000
# numTimeIndices = 10
fieldStrength = 10
circuitDims = (circuitRows,circuitCols)  # (rows,columns) of lattice
# timeIndices = np.linspace(0.1*numSimIters,numSimIters-1,numTimeIndices,dtype=np.int32)
numCells = circuitRows * circuitCols

fieldResolutions = torch.linspace(1,9,numFieldResolutions,dtype=torch.int8)
clampVoltages = torch.linspace(-0.01,-0.2,numParameterValues) # -0.01 to -0.2
clampDurationProps = torch.linspace(0.1,0.9,numParameterValues)  # 0.1-0.9
clampedCellsProps = torch.linspace(0.05,0.95,numParameterValues)  # 0.05-0.95

clampModeFileSuffix = {'field':'Field','tissue':'Tissue','fieldDome':'FieldDome','tissueDome':'TissueDome'}

initialValues = dict()
initVmem = list(chain([-9.2e-3] * numSamples))
initialValues['Vmem'] = torch.repeat_interleave(torch.FloatTensor(initVmem),numCells,0).view(numSamples,numCells,1)
initialValues['G_pol'] = dict()
initialValues['G_pol']['cells'] = [[[0]]] * numSamples
initialValues['G_pol']['values'] = [torch.FloatTensor([1.0])] * numSamples  # bistable
initialValues['G_dep'] = dict()
initialValues['G_dep']['cells'] = []
initialValues['G_dep']['values'] = torch.FloatTensor([])

utils = utilities.utilities()

data = dict()
paramCombination = 0
for fieldResolution in fieldResolutions:
    for clampVoltage in clampVoltages:
        for clampDurationProp in clampDurationProps:
            for clampedCellsProp in clampedCellsProps:
                print(paramCombination,clampMode,fieldResolution,clampVoltage,clampDurationProp,clampedCellsProp)
                circuit = cellularFieldNetwork(circuitDims,GRNParameters=(None,None,None,None),
                                               fieldResolution=fieldResolution,fieldStrength=fieldStrength,
                                               numSamples=numSamples)

                numExtracellularGridPoints = circuit.numExtracellularGridPoints

                circuit.initVariables(initialValues)
                circuit.initParameters(initialValues)

                if clampMode == 'field':
                    numClampableCells = circuit.numExtracellularGridPoints
                    cellIndices = np.arange(numClampableCells)
                    numFieldCells = numClampableCells
                elif clampMode == 'fieldDome':
                    fieldDomeIndices = utils.computeDomeIndices(circuit.LatticeDims,circuit.fieldResolution,mode='field')
                    numClampableCells = len(fieldDomeIndices)
                    cellIndices = fieldDomeIndices
                    numFieldCells = numClampableCells
                elif clampMode == 'tissue':
                    numClampableCells = circuit.numCells
                    cellIndices = np.arange(numClampableCells)
                    numFieldCells = circuit.numExtracellularGridPoints
                elif clampMode == 'tissueDome':
                    tissueDomeIndices = utils.computeDomeIndices(circuit.LatticeDims,mode='tissue')
                    numClampableCells = len(tissueDomeIndices)
                    cellIndices = tissueDomeIndices
                    numFieldCells = circuit.numExtracellularGridPoints
                numClampedCells = int(clampedCellsProp*numClampableCells)
                clampCellIndices = np.array([np.random.choice(cellIndices,numClampedCells,replace=False)
                                         for _ in range(numSamples)]).reshape(-1,)
                sampleIndices = np.repeat(range(numSamples),numClampedCells)
                clampIndices = (sampleIndices,clampCellIndices)
                clampParameters = (clampMode,clampIndices,clampVoltage,clampDurationProp)
                circuit.simulate(clampParameters=clampParameters,numSimIters=numSimIters,saveData=True)

                data[paramCombination] = dict()
                data[paramCombination]['tissueDimensions'] = (circuitRows,circuitCols)
                data[paramCombination]['clampMode'] = clampMode
                data[paramCombination]['fieldResolution'] = fieldResolution
                data[paramCombination]['clampVoltage'] = clampVoltage
                data[paramCombination]['clampDurationProp'] = clampDurationProp
                data[paramCombination]['clampedCellsProp'] = clampedCellsProp
                data[paramCombination]['clampedCellsPropNorm'] = numClampedCells/(numFieldCells+circuit.numCells)
                data[paramCombination]['clampCellIndices'] = clampCellIndices
                data[paramCombination]['Vmem'] = circuit.Vmem
                data[paramCombination]['eV'] = circuit.eV
                # data[paramCombination]['timeseriesVmem'] = circuit.timeseriesVmem[timeIndices]
                data[paramCombination]['numSamples'] = numSamples

                paramCombination += 1

    Sfx = clampModeFileSuffix[clampMode]
    torch.save(data, './data/parameterSweep'+Sfx+'.dat')

