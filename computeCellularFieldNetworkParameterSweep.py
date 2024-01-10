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
numTimeIndices = 10
fieldResolutions = torch.linspace(1,9,numFieldResolutions,dtype=torch.int8)
fieldStrength = 10
numSimIters = 10000
circuitDims = (circuitRows,circuitCols)  # (rows,columns) of lattice
clampVoltages = torch.linspace(-0.01,-0.2,numParameterValues) # -0.01 to -0.2
clampDurationProps = torch.linspace(0.1,0.9,numParameterValues)  # 0.1-0.9
clampedCellsProps = torch.linspace(0.05,0.95,numParameterValues)  # 0.05-0.95
numBasicSamples = 1
numNoisySamples = 1
noise = 0.0  # std of normal distribution
numSamples = numBasicSamples * numNoisySamples
timeIndices = np.linspace(0.1*numSimIters,numSimIters-1,numTimeIndices,dtype=np.int32)
numCells = circuitRows * circuitCols
clampModeFileSuffix = {'field':'Field','tissue':'Tissue','fieldDome':'FieldDome'}

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

                electrodomeIndices = utils.computeElectrodomeIndices(circuit.LatticeDims,circuit.fieldResolution)

                numExtracellularGridPoints = circuit.numExtracellularGridPoints

                circuit.initVariables(initialValues)
                circuit.initParameters(initialValues)

                if clampMode == 'field':
                    numTotalCells = circuit.numExtracellularGridPoints
                    cellIndices = np.arange(numTotalCells)
                    numFieldCells = numTotalCells
                elif clampMode == 'tissue':
                    numTotalCells = circuit.numCells
                    cellIndices = np.arange(numTotalCells)
                    numFieldCells = circuit.numExtracellularGridPoints
                elif clampMode == 'fieldDome':
                    numTotalCells = len(electrodomeIndices)
                    cellIndices = electrodomeIndices
                    numFieldCells = numTotalCells
                numClampedCells = int(clampedCellsProp*numTotalCells)
                clampIndices = np.random.choice(cellIndices,numClampedCells,replace=False)
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
                data[paramCombination]['clampIndices'] = clampIndices
                data[paramCombination]['Vmem'] = circuit.Vmem
                # data[paramCombination]['eV'] = circuit.eV
                data[paramCombination]['timeseriesVmem'] = circuit.timeseriesVmem[timeIndices]

                paramCombination += 1

    Sfx = clampModeFileSuffix[clampMode]
    torch.save(data, './data/parameterSweep'+Sfx+'.dat')

