import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import utilities
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--fieldEnabled', type=str, default='True')
parser.add_argument('--latticeDims', type=str, default='(5,5)')
parser.add_argument('--fieldResolution', type=int, default=1)
parser.add_argument('--fieldStrength', type=float, default=10.0)
parser.add_argument('--fieldAggregation', type=str, default='average')
parser.add_argument('--fieldScreenSize', type=int, default=1)
parser.add_argument('--GJStrength', type=float, default=0.05)
parser.add_argument('--clampMode', type=str, default='field')
parser.add_argument('--clampType', type=str, default='static')
parser.add_argument('--clampedCellsProp', type=float, default=1.0)
parser.add_argument('--clampDurationProp', type=float, default=0.1)
parser.add_argument('--clampAmplitudeRange', type=str, default='(-1.0,1.0)')
parser.add_argument('--clampFrequencyRange', type=str, default='(100.0,1000.0)')
parser.add_argument('--numClampCoreSquares', type=int, default=1)
parser.add_argument('--numSamples', type=int, default=1)
parser.add_argument('--numSimIters', type=int, default=100)
parser.add_argument('--numLearnIters', type=int, default=100)
parser.add_argument('--learnedParameters', type=str, default='None')
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--fileNumber', type=int, default=0)
parser.add_argument('--verbose', type=str, default='True')

args = parser.parse_args()

circuitRows,circuitCols = circuitDims = ast.literal_eval(args.latticeDims)
fieldEnabled = ast.literal_eval(args.fieldEnabled)
fieldResolution = args.fieldResolution
fieldStrength = args.fieldStrength
fieldAggregation = args.fieldAggregation
fieldScreenSize = args.fieldScreenSize
GJStrength = args.GJStrength
numSamples = args.numSamples
numSimIters = args.numSimIters
fileNumber = args.fileNumber
verbose = ast.literal_eval(args.verbose)

# Simulation parameters (typically fixed, except clampParameters)
perturbationParameters = None
stochasticIonChannels = False
externalInputs = {'gene': None}
setGradient = False
retainGradients = False
saveData = True

fieldTransductionWeights = np.linspace(0,50,50)
fieldTransductionBiases = np.linspace(0,0.1,10)
fieldTransductionTimeConstant = torch.DoubleTensor([10.0])
parameterGrid = list(zip(np.repeat(fieldTransductionWeights,10),np.tile(fieldTransductionBiases,50)))

parameterCombination = parameterGrid[fileNumber]
fieldTransductionWeight = torch.DoubleTensor([parameterCombination[0]])
fieldTransductionBias = torch.DoubleTensor([parameterCombination[1]])

GRNtoVmemWeights,GRNBiases,GRNtoVmemWeightsTimeconstant,GRNNumGenes = None,None,None,None

GJParameterNames = ['GJStrength']
fieldParameterNames = ['fieldEnabled','fieldResolution','fieldStrength','fieldAggregation','fieldScreenSize',
                       'fieldTransductionWeight','fieldTransductionBias','fieldTransductionTimeConstant']
GRNParameterNames = ['GRNtoVmemWeights','GRNBiases','GRNtoVmemWeightsTimeconstant','GRNNumGenes']
simParameterNames = ['initialValues','externalInputs','numSamples','numSimIters']
characteristicNames = ['VarMaxValues']

utils = utilities.utilities()

GJParameters = dict()
for param in GJParameterNames:
    GJParameters[param] = eval(param)
fieldParameters = dict()
for param in fieldParameterNames:
    fieldParameters[param] = eval(param)
GRNParameters = dict()
for param in GRNParameterNames:
    GRNParameters[param] = eval(param)
parameters = dict()
parameters['GJParameters'] = GJParameters
parameters['fieldParameters'] = fieldParameters
parameters['GRNParameters'] = GRNParameters
circuit = cellularFieldNetwork(circuitDims,parameters=parameters,numSamples=numSamples)

def defineInitialValues(circuit):
    initialValues = dict()
    initVmem = torch.FloatTensor(list(chain([-9.2e-3] * numSamples)))
    initialValues['Vmem'] = torch.repeat_interleave(initVmem,circuit.numCells,0).double().view(numSamples,circuit.numCells,1)
    initialValues['eV'] = torch.zeros((numSamples,circuit.numExtracellularGridPoints,1),dtype=torch.float64)
    initialValues['G_pol'] = dict()
    initialValues['G_pol']['cells'] = [[[0]]] * numSamples
    initialValues['G_pol']['values'] = [torch.DoubleTensor([1.0])] * numSamples  # bistable
    initialValues['G_dep'] = dict()
    initialValues['G_dep']['cells'] = []
    initialValues['G_dep']['values'] = torch.DoubleTensor([])
    return initialValues

def computeVmemRangeDynamics(circuit):
    timeseriesLength = circuit.timeseriesVmem.shape[0]
    VarMaxValues = [(torch.var(circuit.timeseriesVmem[t]).item(),circuit.timeseriesVmem[t].abs().max().item()) for t in range(timeseriesLength)]
    return VarMaxValues

modelCharacteristics = dict()
modelCharacteristics['latticeDims'] = circuitDims
modelCharacteristics['GJParameters'] = dict()
modelCharacteristics['fieldParameters'] = dict()
modelCharacteristics['GRNParameters'] = dict()
modelCharacteristics['simParameters'] = dict()
modelCharacteristics['characteristics'] = dict()

savefilename = './data/modelCharacteristics_' + str(fileNumber) + '.dat'
parameters = dict()
GJParameters = dict()
for param in GJParameterNames:  # learned field parameters will be automatically updated in the model
    GJParameters[param] = eval(param)
fieldParameters = dict()
for param in fieldParameterNames:  # learned field parameters will be automatically updated in the model
    fieldParameters[param] = eval(param)
parameters['GJParameters'] = GJParameters
parameters['fieldParameters'] = fieldParameters
parameters['GRNParameters'] = GRNParameters  # just a tuple of Nones at the moment
circuit = cellularFieldNetwork(circuitDims,parameters=parameters,numSamples=numSamples)
initialValues = defineInitialValues(circuit)
circuit.initVariables(initialValues)
circuit.initParameters(initialValues)
circuit.simulate(externalInputs=externalInputs,clampParameters=None,perturbationParameters=perturbationParameters,
                 numSimIters=numSimIters,stochasticIonChannels=stochasticIonChannels,setGradient=setGradient,
                 retainGradients=retainGradients,saveData=saveData)
VarMaxValues = computeVmemRangeDynamics(circuit)
for param in GJParameterNames:
    variable = eval(param)
    if torch.is_tensor(variable):
        modelCharacteristics['GJParameters'][param] = variable.detach()
    else:
        modelCharacteristics['GJParameters'][param] = variable
for param in fieldParameterNames:
    variable = eval(param)
    if torch.is_tensor(variable):
        modelCharacteristics['fieldParameters'][param] = variable.detach()
    else:
        modelCharacteristics['fieldParameters'][param] = variable
for param in GRNParameterNames:
    variable = eval(param)
    if torch.is_tensor(variable):
        modelCharacteristics['GRNParameters'][param] = variable.detach()
    else:
        modelCharacteristics['GRNParameters'][param] = variable
for param in simParameterNames:
    variable = eval(param)
    if torch.is_tensor(variable) and (variable.dim()<=1):
        modelCharacteristics['simParameters'][param] = variable.detach().item()
    else:
        modelCharacteristics['simParameters'][param] = variable
for param in characteristicNames:
    variable = eval(param)
    if torch.is_tensor(variable) and (variable.dim()<=1):
        modelCharacteristics['characteristics'][param] = variable.detach().item()
    else:
        modelCharacteristics['characteristics'][param] = variable
torch.save(modelCharacteristics, savefilename)

if verbose:
    print("File number ",fileNumber," completed!")


