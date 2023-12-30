import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork

circuitRows,circuitCols = 4,6
circuitDims = (circuitRows,circuitCols)  # (rows,columns) of lattice
numParameterValues = 20
fieldResolutions = torch.arange(1,11)
fieldStrength = 10
clampModes = ['field','tissue']
clampVoltages = torch.linspace(-0.01,-0.2,numParameterValues)
clampDurationProps = torch.linspace(0.1,0.9,numParameterValues)
clampedCellsProps = torch.linspace(0.05,0.95,numParameterValues)
numBasicSamples = 1
numNoisySamples = 1
noise = 0.0  # std of normal distribution
numSamples = numBasicSamples * numNoisySamples
numSimIters = 300
numCells = circuitRows * circuitCols

initialValues = dict()
initVmem = list(chain([-9.2e-3] * numSamples))
initialValues['Vmem'] = torch.repeat_interleave(torch.FloatTensor(initVmem),numCells,0).view(numSamples,numCells,1)
initialValues['G_pol'] = dict()
initialValues['G_pol']['cells'] = [[[0]]] * numSamples
initialValues['G_pol']['values'] = [torch.FloatTensor([1.0])] * numSamples  # bistable
initialValues['G_dep'] = dict()
initialValues['G_dep']['cells'] = []
initialValues['G_dep']['values'] = torch.FloatTensor([])

data = dict()
paramCombination = 0
for clampMode in clampModes:
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
                        numTotalCells = circuit.numExtracellularGridPoints
                        numClampedCells = int(clampedCellsProp*circuit.numExtracellularGridPoints)
                    elif clampMode == 'tissue':
                        numTotalCells = circuit.numCells
                        numClampedCells = int(clampedCellsProp*circuit.numCells)
                    clampIndices = np.random.choice(numTotalCells,numClampedCells)
                    clampParameters = (clampMode,clampIndices,clampVoltage,clampDurationProp)
                    circuit.simulate(clampParameters=clampParameters,numSimIters=numSimIters,saveData=True)

                    data[paramCombination] = dict()
                    data[paramCombination]['clampMode'] = clampMode
                    if clampMode == 'field':
                        data[paramCombination]['fieldResolution'] = fieldResolution
                    data[paramCombination]['clampVoltage'] = clampVoltage
                    data[paramCombination]['clampDurationProp'] = clampDurationProp
                    data[paramCombination]['clampedCellsProp'] = clampedCellsProp
                    data[paramCombination]['clampIndices'] = clampIndices
                    data[paramCombination]['timeseriesVmem'] = circuit.timeseriesVmem
                    data[paramCombination]['timeserieseV'] = circuit.timeserieseV

                    paramCombination += 1

        torch.save(data, './data/parameterSweep.dat')

