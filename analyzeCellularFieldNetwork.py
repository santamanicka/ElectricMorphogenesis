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
parser.add_argument('--fieldStrengthProp', type=float, default=1.0)
parser.add_argument('--ligandEnabled', type=str, default='False')
parser.add_argument('--ligandGatingWeight', type=float, default=10.0)
parser.add_argument('--ligandGatingBias', type=float, default=-0.5)
parser.add_argument('--ligandCurrentStrength', type=float, default=10.0)
parser.add_argument('--GJStrength', type=float, default=0.05)
parser.add_argument('--randomizeInitialStates', type=str, default='False')
parser.add_argument('--numSamples', type=int, default=1)
parser.add_argument('--numSimIters', type=int, default=100)
parser.add_argument('--analysisMode', type=str, default='fixScreenGJSweepWeightBias')
parser.add_argument('--analysisRegion', type=str, default='topLeftQuadrant')
parser.add_argument('--fileNumber', type=int, default=0)
parser.add_argument('--fileNumberVersion', type=int, default=0)
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
fieldStrengthProp = args.fieldStrengthProp
ligandEnabled = ast.literal_eval(args.ligandEnabled)
ligandGatingWeight = args.ligandGatingWeight
ligandGatingBias = args.ligandGatingBias
ligandCurrentStrength = args.ligandCurrentStrength
GJStrength = args.GJStrength
randomizeInitialStates = ast.literal_eval(args.randomizeInitialStates)
numSamples = args.numSamples
numSimIters = args.numSimIters
analysisMode = args.analysisMode
analysisRegion = args.analysisRegion
fileNumber = args.fileNumber
fileNumberVersion = args.fileNumberVersion
verbose = ast.literal_eval(args.verbose)

def defineInitialValues(circuit,randomize=False):
    numCells = circuit.numCells
    initialValues = dict()
    initVmem = torch.FloatTensor(list(chain([-9.2e-3] * numSamples)))
    initialValues['Vmem'] = torch.repeat_interleave(initVmem,numCells,0).double().view(numSamples,numCells,1)
    initialValues['eV'] = torch.zeros((numSamples,circuit.numExtracellularGridPoints,1),dtype=torch.float64)
    initialValues['ligandConc'] = torch.zeros((numSamples,numCells,1),dtype=torch.float64)
    initialValues['G_pol'] = dict()
    if randomize:  # 0th sample = homogenous; rest = heterogeneous
        AllCells = list(range(circuit.numCells))
        initialValues['G_pol']['cells'] = [[AllCells]] * numSamples
        initialValues['G_pol']['values'] = [[torch.rand(numCells,dtype=torch.float64)*2] for _ in  range(numSamples)]  # covers a range of unistable and bistable values
        initialValues['G_pol']['values'][0] = torch.DoubleTensor([1.0]*numCells)  # homogenous state; bistable Vmem
    else:
        initialValues['G_pol']['cells'] = [[[0]]] * numSamples
        initialValues['G_pol']['values'] = [torch.DoubleTensor([1.0])] * numSamples  # bistable
    initialValues['G_dep'] = dict()
    initialValues['G_dep']['cells'] = []
    initialValues['G_dep']['values'] = torch.DoubleTensor([])
    return initialValues

def computeVmemRangeDynamics(circuit):
    timeseriesLength = circuit.timeseriesVmem.shape[0]
    VarMaxValues = [[(torch.var(circuit.timeseriesVmem[t,s]).item(),circuit.timeseriesVmem[t].abs().max().item())
                    for t in range(timeseriesLength)] for s in range(numSamples)]
    return VarMaxValues

def computeDimensionality(circuit,ndims=2,startTime=0):
    evPCAProps, eVCellWiseMeanPCAProps, vmemPCAProps = [], [], []
    for sample in range(numSamples):
        evData = circuit.timeserieseV[startTime:,sample,:,0]
        evData = StandardScaler().fit_transform(evData)
        pca = PCA(n_components=ndims)
        eVPCA = pca.fit_transform(evData)
        evPCAProps.append(pca.explained_variance_ratio_)
        evCellWiseMeanData = (circuit.timeserieseV * circuit.fieldScreenMatrix).sum(2) / circuit.numFieldNeighbors
        evCellWiseMeanData = StandardScaler().fit_transform(evCellWiseMeanData[:,sample,:])
        pca = PCA(n_components=ndims)
        eVCellWiseMeanPCA = pca.fit_transform(evCellWiseMeanData)
        eVCellWiseMeanPCAProps.append(pca.explained_variance_ratio_)
        vmemData = circuit.timeseriesVmem[startTime:,sample,:,0]
        vmemData = StandardScaler().fit_transform(vmemData)
        pca = PCA(n_components=ndims)
        vmemPCA = pca.fit_transform(vmemData)
        vmemPCAProps.append(pca.explained_variance_ratio_)
    return ([evPCAProps,eVCellWiseMeanPCAProps,vmemPCAProps])

def computeInformationMeasures(circuit):
    VmemBins = np.arange(-0.0, -0.1, -0.04)
    tlqTotalCorr, tlqEntropy = [], []
    for sample in range(numSamples):
        vbin = 2 - np.digitize(circuit.timeseriesVmem[:,sample,:,0].detach(),VmemBins)
        topLeftQuadrantIdx = utils.computeBulkIndices(circuit,mode='tissue',region='topLeftQuadrant')
        tlqstates = vbin[:,topLeftQuadrantIdx]
        uniquetlqstates, countstlqstates = np.unique(tlqstates,axis=0,return_counts=True)
        probstlqstates = countstlqstates / sum(countstlqstates)
        tlqstatestr = [''.join(str(bit) for bit in state) for state in uniquetlqstates]
        tlqdistrdict = dict(zip(tlqstatestr,probstlqstates))
        tlqdistr = dit.Distribution(tlqdistrdict)
        tlqTotalCorr.append(dit.multivariate.binding_information(tlqdistr))
        tlqEntropy.append(dit.multivariate.entropy(tlqdistr))
    return ([tlqTotalCorr,tlqEntropy])

def computeSensitivity(circuit,region=analysisRegion):
    targetVariables = utils.computeBulkIndices(circuit,mode='tissue',region=region)
    numTargetVmemVariables = len(targetVariables)
    if circuit.fieldEnabled:
        eVToVmemSensitivity = torch.zeros(numSimIters,circuit.numExtracellularGridPoints,numTargetVmemVariables)
    if circuit.ligandEnabled:
        ligandToVmemSensitivity = torch.zeros(numSimIters,circuit.numCells,numTargetVmemVariables)
    VmemToVemSensitivity = torch.zeros(numSimIters,circuit.numCells,numTargetVmemVariables)
    for t in range(setGradientIter+1,numSimIters+1,2):
        for variableIdx in range(numTargetVmemVariables):
            print(t,variableIdx)
            variable = targetVariables[variableIdx]
            # circuit.Vmem[0,variable,0].backward(retain_graph=True)
            circuit.timeseriesVmem[t-1,0,variable,0].backward(retain_graph=True)
            VmemToVemSensitivity[t-1,:,variableIdx] = circuit.VmemInit.grad.data[0,:,0]
            if circuit.fieldEnabled:
                eVToVmemSensitivity[t-1,:,variableIdx] = circuit.eVInit.grad.data[0,:,0]
                circuit.eVInit.grad.data.zero_()
            if circuit.ligandEnabled:
                ligandToVmemSensitivity[t-1,:,variableIdx] = circuit.ligandConcInit.grad.data[0,:,0]
                circuit.ligandConcInit.grad.data.zero_()
            circuit.VmemInit.grad.data.zero_()
            circuit.G_polInit.grad.data.zero_()
    if circuit.fieldEnabled:
        return([eVToVmemSensitivity,VmemToVemSensitivity])
    elif circuit.ligandEnabled:
        return ([ligandToVmemSensitivity,VmemToVemSensitivity])
    else:
        return ([VmemToVemSensitivity])

# Simulation parameters (typically fixed, except clampParameters)
perturbationParameters = None
stochasticIonChannels = False
externalInputs = {'gene': None}
saveData = True

# The particular parameter combination will be chosen from a grid whose location will be determined by fileNumber
if analysisMode == 'fixScreenGJSweepWeightBias':  # total parameter combinations = 30x10 = 300
    fieldTransductionWeights = np.linspace(0,50,30)
    fieldTransductionBiases = np.linspace(0,0.1,10)
    parameterGrid = list(zip(np.repeat(fieldTransductionWeights,len(fieldTransductionBiases)),
                             np.tile(fieldTransductionBiases,len(fieldTransductionWeights))))
    fieldTransductionTimeConstant = torch.DoubleTensor([10.0])
    parameterCombination = parameterGrid[fileNumber - 1]  # so file numbers can start from 1
    clampParameters = None
elif analysisMode == 'fixWeightBiasSweepScreenGJ':  # total parameter combinations = 15x20 = 300
    maxFieldScreenSize = 2*max(circuitDims)-1  # the field will permeate the entire tissue = 2(l-1)+1, where l is the max of circuitDims
    fieldScreenSizes = np.linspace(1,maxFieldScreenSize,15,dtype=np.int8)
    GJStrengths = np.linspace(0,1.0,20)
    parameterGrid = list(zip(np.repeat(fieldScreenSizes,len(GJStrengths)),
                             np.tile(GJStrengths,len(fieldScreenSizes))))
    fieldTransductionTimeConstant = torch.DoubleTensor([10.0])
    parameterCombination = parameterGrid[fileNumber - 1]  # so file numbers can start from 1
    clampParameters = None
elif analysisMode == 'sensitivity':
    parameterfilename = './data/bestModelParameters_' + str(fileNumber) + '.dat'
    parameters = torch.load(parameterfilename)
    circuitRows,circuitCols = circuitDims = parameters['latticeDims']
    GJParameters = parameters['GJParameters']
    fieldParameters = parameters['fieldParameters']
    fieldParameters['fieldStrength'] *= fieldStrengthProp
    ligandParameters = parameters['ligandParameters']
    GRNParameters = parameters['GRNParameters']
    numSamples = parameters['simParameters']['numSamples']
    initialValues = parameters['simParameters']['initialValues']
    numSimIters = parameters['simParameters']['numSimIters']
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
ligandParameterNames = ['ligandEnabled','ligandGatingWeight','ligandGatingBias','ligandCurrentStrength']
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
    parameters['ligandParameters'] = ligandParameters
    parameters['GRNParameters'] = GRNParameters
    setGradient = True
    setGradientIter = clampParameters['clampEndIter'] + 1
    retainGradients = False
    circuit = cellularFieldNetwork(circuitDims,parameters=parameters,numSamples=numSamples)
else:
    GJParameters = dict()
    for param in GJParameterNames:
        GJParameters[param] = eval(param)
    fieldParameters = dict()
    for param in fieldParameterNames:
        fieldParameters[param] = eval(param)
    ligandParameters = dict()
    for param in ligandParameterNames:
        ligandParameters[param] = eval(param)
    GRNParameters = dict()
    for param in GRNParameterNames:
        GRNParameters[param] = eval(param)
    parameters = dict()
    parameters['GJParameters'] = GJParameters
    parameters['fieldParameters'] = fieldParameters
    parameters['ligandParameters'] = ligandParameters
    parameters['GRNParameters'] = GRNParameters
    setGradient = False
    setGradientIter = 0
    retainGradients = False
    circuit = cellularFieldNetwork(circuitDims,parameters=parameters,numSamples=numSamples)
    initialValues = defineInitialValues(circuit,randomize=randomizeInitialStates)

utils = utilities.utilities()

modelCharacteristics = dict()
modelCharacteristics['latticeDims'] = circuitDims
modelCharacteristics['GJParameters'] = dict()
modelCharacteristics['fieldParameters'] = dict()
modelCharacteristics['ligandParameters'] = dict()
modelCharacteristics['GRNParameters'] = dict()
modelCharacteristics['simParameters'] = dict()
modelCharacteristics['analysisMode'] = analysisMode
modelCharacteristics['characteristics'] = dict()

if analysisMode == 'fixScreenGJSweepWeightBias':
    Sfx = 'FixedScreenSizeGJ_'
elif analysisMode == 'fixWeightBiasSweepScreenGJ':
    Sfx = 'FixedWeightBias_'
elif analysisMode == 'sensitivity':
    Sfx = 'Sensitivity_'
if fileNumberVersion > 0:
    fileVersionSfx = '_V' + str(fileNumberVersion)
else:
    fileVersionSfx = ''
savefilename = './data/modelCharacteristics_' + Sfx + str(fileNumber) + fileVersionSfx + '.dat'

circuit.initVariables(initialValues)
circuit.initParameters(initialValues)
circuit.simulate(externalInputs=externalInputs,clampParameters=clampParameters,perturbationParameters=perturbationParameters,
                 numSimIters=numSimIters,stochasticIonChannels=stochasticIonChannels,
                 setGradient=setGradient,setGradientIter=setGradientIter,retainGradients=retainGradients,saveData=saveData)
if analysisMode == 'fixScreenGJSweepWeightBias':
    VarMaxValues = computeVmemRangeDynamics(circuit)
    Dimensionality = computeDimensionality(circuit, ndims=3)
elif analysisMode == 'fixWeightBiasSweepScreenGJ':
    Dimensionality = computeDimensionality(circuit,ndims=3)
    Information = computeInformationMeasures(circuit)
elif analysisMode == 'sensitivity':
    Sensitivity = computeSensitivity(circuit,region=analysisRegion)
for param in GJParameterNames:
    modelCharacteristics['GJParameters'][param] = GJParameters[param]
for param in fieldParameterNames:
    modelCharacteristics['fieldParameters'][param] = fieldParameters[param]
for param in ligandParameterNames:
    modelCharacteristics['ligandParameters'][param] = ligandParameters[param]
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


