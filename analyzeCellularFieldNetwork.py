import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import utilities
import argparse
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import dit
import math

parser = argparse.ArgumentParser()
parser.add_argument('--fieldEnabled', type=str, default='True')
parser.add_argument('--latticeDims', type=str, default='(5,5)')
parser.add_argument('--fieldResolution', type=int, default=1)
parser.add_argument('--fieldStrength', type=float, default=10.0)
parser.add_argument('--fieldAggregation', type=str, default='average')
parser.add_argument('--fieldScreenSize', type=int, default=1)
parser.add_argument('--fieldTransductionWeight', type=float, default=10.0)
parser.add_argument('--fieldTransductionBias', type=float, default=0.03)
parser.add_argument('--GJStrength', type=float, default=0.05)
parser.add_argument('--numSamples', type=int, default=1)
parser.add_argument('--numSimIters', type=int, default=100)
parser.add_argument('--analysisMode', type=str, default='fixScreenGJSweepWeightBias')
parser.add_argument('--fileNumber', type=int, default=0)
parser.add_argument('--verbose', type=str, default='True')

args = parser.parse_args()

circuitRows,circuitCols = circuitDims = ast.literal_eval(args.latticeDims)
fieldEnabled = ast.literal_eval(args.fieldEnabled)
fieldResolution = args.fieldResolution
fieldStrength = args.fieldStrength
fieldAggregation = args.fieldAggregation
fieldScreenSize = args.fieldScreenSize
fieldTransductionWeight = args.fieldTransductionWeight
fieldTransductionBias = args.fieldTransductionBias
GJStrength = args.GJStrength
numSamples = args.numSamples
numSimIters = args.numSimIters
analysisMode = args.analysisMode
fileNumber = args.fileNumber
verbose = ast.literal_eval(args.verbose)

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

def computeDimensionality(circuit,ndims=2,startTime=0):
    evData = circuit.timeserieseV[startTime:,0,:,0]
    evData = StandardScaler().fit_transform(evData)
    pca = PCA(n_components=ndims)
    eVPCA = pca.fit_transform(evData)
    evPCAProps = pca.explained_variance_ratio_
    evCellWiseMeanData = (circuit.timeserieseV * circuit.fieldCellNeighborhoodBitmap).sum(2) / circuit.numFieldNeighbors
    evCellWiseMeanData = StandardScaler().fit_transform(evCellWiseMeanData[:,0,:])
    pca = PCA(n_components=ndims)
    eVCellWiseMeanPCA = pca.fit_transform(evCellWiseMeanData)
    eVCellWiseMeanPCAProps = pca.explained_variance_ratio_
    vmemData = circuit.timeseriesVmem[startTime:,0,:,0]
    vmemData = StandardScaler().fit_transform(vmemData)
    pca = PCA(n_components=ndims)
    vmemPCA = pca.fit_transform(vmemData)
    vmemPCAProps = pca.explained_variance_ratio_
    return ([evPCAProps,eVCellWiseMeanPCAProps,vmemPCAProps])

def computeInformationMeasures(circuit):
    VmemBins = np.arange(-0.0, -0.1, -0.04)
    vbin = 2 - np.digitize(circuit.timeseriesVmem[:,0,:,0].detach(),VmemBins)
    topLeftQuadrantIdx = utils.computeRegionIndices(circuit,mode='tissue',region='topLeftQuadrant')
    tlqstates = vbin[:,topLeftQuadrantIdx]
    uniquetlqstates, countstlqstates = np.unique(tlqstates,axis=0,return_counts=True)
    probstlqstates = countstlqstates / sum(countstlqstates)
    tlqstatestr = [''.join(str(bit) for bit in state) for state in uniquetlqstates]
    tlqdistrdict = dict(zip(tlqstatestr,probstlqstates))
    tlqdistr = dit.Distribution(tlqdistrdict)
    tlqTotalCorr = dit.multivariate.binding_information(tlqdistr)
    tlqEntropy = dit.multivariate.entropy(tlqdistr)
    return ([tlqTotalCorr,tlqEntropy])

def computeSensitivity(circuit):
    topQuadrantVmemVariables = utils.computeRegionIndices(circuit,mode='tissue',region='topLeftQuadrant')
    numTargetVmemVariables = len(topQuadrantVmemVariables)
    eVToVmemSensitivity = torch.zeros(numSimIters,circuit.numExtracellularGridPoints,numTargetVmemVariables)
    VmemToVemSensitivity = torch.zeros(numSimIters,circuit.numCells,numTargetVmemVariables)
    for t in range(1,numSimIters+1):
        for variableIdx in range(numTargetVmemVariables):
            variable = topQuadrantVmemVariables[variableIdx]
            # circuit.Vmem[0,variable,0].backward(retain_graph=True)
            circuit.timeseriesVmem[t-1,0,variable,0].backward(retain_graph=True)
            VmemToVemSensitivity[t-1,:,variableIdx] = circuit.VmemInit.grad.data[0,:,0]
            if circuit.fieldEnabled:
                eVToVmemSensitivity[t-1,:,variableIdx] = circuit.eVInit.grad.data[0,:,0]
                circuit.eVInit.grad.data.zero_()
            circuit.VmemInit.grad.data.zero_()
            circuit.G_polInit.grad.data.zero_()
    if circuit.fieldEnabled:
        return([eVToVmemSensitivity,VmemToVemSensitivity])
    else:
        return ([VmemToVemSensitivity])

# Simulation parameters (typically fixed, except clampParameters)
perturbationParameters = None
stochasticIonChannels = False
externalInputs = {'gene': None}
saveData = True

if analysisMode == 'fixScreenGJSweepWeightBias':
    fieldTransductionWeights = np.linspace(0,50,50)
    fieldTransductionBiases = np.linspace(0,0.1,10)
    parameterGrid = list(zip(np.repeat(fieldTransductionWeights,len(fieldTransductionBiases)),
                             np.tile(fieldTransductionBiases,len(fieldTransductionWeights))))
    fieldTransductionTimeConstant = torch.DoubleTensor([10.0])
    parameterCombination = parameterGrid[fileNumber]
elif analysisMode == 'fixWeightBiasSweepScreenGJ':
    maxFieldScreenSize = 2*max(circuitDims)-1  # the field will permeate the entire tissue = 2(l-1)+1, where l is the max of circuitDims
    fieldScreenSizes = np.arange(1,maxFieldScreenSize,1)
    GJStrengths = np.linspace(0,1.0,20)
    parameterGrid = list(zip(np.repeat(fieldScreenSizes,len(GJStrengths)),
                             np.tile(GJStrengths,len(fieldScreenSizes))))
    fieldTransductionTimeConstant = torch.DoubleTensor([10.0])
    parameterCombination = parameterGrid[fileNumber]
elif analysisMode == 'sensitivity':
    parameterfilename = './data/bestModelParameters_' + str(fileNumber) + '.dat'
    parameters = torch.load(parameterfilename)
    circuitRows,circuitCols = circuitDims = parameters['latticeDims']
    GJParameters = parameters['GJParameters']
    fieldParameters = parameters['fieldParameters']
    GRNParameters = parameters['GRNParameters']
    numSamples = parameters['simParameters']['numSamples']
    initialValues = parameters['simParameters']['initialValues']
    clampParameters = parameters['clampParameters']
    externalInputs = parameters['simParameters']['externalInputs']
    initialValues = parameters['simParameters']['initialValues']

if analysisMode == 'fixScreenGJSweepWeightBias':
    fieldTransductionWeight = torch.DoubleTensor([parameterCombination[0]])
    fieldTransductionBias = torch.DoubleTensor([parameterCombination[1]])
elif analysisMode == 'fixWeightBiasSweepScreenGJ':
    fieldScreenSize = parameterCombination[0]
    GJStrength = parameterCombination[1]
# Note that if analysisMode is 'sensitivity' then the parameters would be loaded from a file

GRNtoVmemWeights,GRNBiases,GRNtoVmemWeightsTimeconstant,GRNNumGenes = None,None,None,None

GJParameterNames = ['GJStrength']
fieldParameterNames = ['fieldEnabled','fieldResolution','fieldStrength','fieldAggregation','fieldScreenSize',
                       'fieldTransductionWeight','fieldTransductionBias','fieldTransductionTimeConstant']
GRNParameterNames = ['GRNtoVmemWeights','GRNBiases','GRNtoVmemWeightsTimeconstant','GRNNumGenes']
simParameterNames = ['initialValues','externalInputs','numSamples','numSimIters']
if analysisMode == 'fixScreenGJSweepWeightBias':
    characteristicNames = ['VarMaxValues','Dimensionality']
elif analysisMode == 'fixWeightBiasSweepScreenGJ':
    characteristicNames = ['Dimensionality','Information']
elif analysisMode == 'sensitivity':
    characteristicNames = ['Sensitivity']

if analysisMode == 'sensitivity':  # parameters loaded from file
    parameters = dict()
    parameters['GJParameters'] = GJParameters
    parameters['fieldParameters'] = fieldParameters
    parameters['GRNParameters'] = GRNParameters
    setGradient = True
    retainGradients = False
    circuit = cellularFieldNetwork(circuitDims,parameters=parameters,numSamples=numSamples)
else:
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
    setGradient = False
    retainGradients = False
    circuit = cellularFieldNetwork(circuitDims,parameters=parameters,numSamples=numSamples)
    initialValues = defineInitialValues(circuit)

utils = utilities.utilities()

modelCharacteristics = dict()
modelCharacteristics['latticeDims'] = circuitDims
modelCharacteristics['GJParameters'] = dict()
modelCharacteristics['fieldParameters'] = dict()
modelCharacteristics['GRNParameters'] = dict()
modelCharacteristics['simParameters'] = dict()
modelCharacteristics['characteristics'] = dict()

if analysisMode == 'fixScreenGJSweepWeightBias':
    Sfx = 'FixedScreenSizeGJ_'
elif analysisMode == 'fixWeightBiasSweepScreenGJ':
    Sfx = 'FixedWeightBias_'
elif analysisMode == 'sensitivity':
    Sfx = 'Sensitivity_'
savefilename = './data/modelCharacteristics_' + Sfx + str(fileNumber) + '.dat'

circuit.initVariables(initialValues)
circuit.initParameters(initialValues)
circuit.simulate(externalInputs=externalInputs,clampParameters=None,perturbationParameters=perturbationParameters,
                 numSimIters=numSimIters,stochasticIonChannels=stochasticIonChannels,
                 setGradient=setGradient,retainGradients=retainGradients,saveData=saveData)
if analysisMode == 'fixScreenGJSweepWeightBias':
    VarMaxValues = computeVmemRangeDynamics(circuit)
    Dimensionality = computeDimensionality(circuit, ndims=3)
elif analysisMode == 'fixWeightBiasSweepScreenGJ':
    Dimensionality = computeDimensionality(circuit,ndims=3)
    Information = computeInformationMeasures(circuit)
elif analysisMode == 'sensitivity':
    Sensitivity = computeSensitivity(circuit)
for param in GJParameterNames:
    modelCharacteristics['GJParameters'][param] = GJParameters[param]
for param in fieldParameterNames:
    modelCharacteristics['fieldParameters'][param] = fieldParameters[param]
for param in GRNParameterNames:
    modelCharacteristics['GRNParameters'][param] = GRNParameters[param]
for param in simParameterNames:
    variable = eval(param)
    modelCharacteristics['simParameters'][param] = variable
for param in characteristicNames:
    variable = eval(param)
    modelCharacteristics['characteristics'][param] = variable
torch.save(modelCharacteristics, savefilename)

if verbose:
    print("File number ",fileNumber," completed!")


