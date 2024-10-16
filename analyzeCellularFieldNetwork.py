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
import os

parser = argparse.ArgumentParser()
parser.add_argument('--fieldEnabled', type=str, default='True')
parser.add_argument('--latticeDims', type=str, default='(5,5)')
parser.add_argument('--fieldResolution', type=int, default=1)
parser.add_argument('--fieldStrength', type=float, default=1.0)
parser.add_argument('--fieldAggregation', type=str, default='average')
parser.add_argument('--fieldScreenSize', type=int, default=1)
parser.add_argument('--fieldTransductionWeight', type=float, default=10.0)
parser.add_argument('--fieldTransductionBias', type=float, default=0.03)
parser.add_argument('--fieldTransductionGain', type=float, default=1.0)
parser.add_argument('--fieldTransductionTimeConstant', type=float, default=10.0)
parser.add_argument('--fieldStrengthProp', type=float, default=1.0)
parser.add_argument('--fieldRangeSymmetric', type=str, default='False')
parser.add_argument('--fieldVector', type=str, default='True')
parser.add_argument('--ligandEnabled', type=str, default='False')
parser.add_argument('--ligandGatingWeight', type=float, default=10.0)
parser.add_argument('--ligandGatingBias', type=float, default=-0.5)
parser.add_argument('--ligandCurrentStrength', type=float, default=10.0)
parser.add_argument('--vmemToLigandCurrentStrength', type=float, default=5.0)
parser.add_argument('--GJStrength', type=float, default=0.05)
parser.add_argument('--randomizeInitialStates', type=str, default='False')
parser.add_argument('--characteristicNames', type=str, default='Default')
parser.add_argument('--numSamples', type=int, default=1)
parser.add_argument('--numSimIters', type=int, default=100)
parser.add_argument('--numPerturbSimIters', type=int, default=100)
parser.add_argument('--perturbationMode', type=str, default='None')
parser.add_argument('--analysisMode', type=str, default='fixScreenGJSweepWeightBias')
parser.add_argument('--analysisRegion', type=str, default='"topLeftQuadrant"')
parser.add_argument('--numGradientTimePoints', type=int, default=10)
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
fieldTransductionGain = args.fieldTransductionGain
fieldTransductionTimeConstant = args.fieldTransductionTimeConstant
fieldStrengthProp = args.fieldStrengthProp
fieldRangeSymmetric = ast.literal_eval(args.fieldRangeSymmetric)
fieldVector = ast.literal_eval(args.fieldVector)
ligandEnabled = ast.literal_eval(args.ligandEnabled)
ligandGatingWeight = args.ligandGatingWeight
ligandGatingBias = args.ligandGatingBias
ligandCurrentStrength = args.ligandCurrentStrength
vmemToLigandCurrentStrength = args.vmemToLigandCurrentStrength
GJStrength = args.GJStrength
randomizeInitialStates = ast.literal_eval(args.randomizeInitialStates)
perturbationMode = args.perturbationMode
characteristicNames = ast.literal_eval(args.characteristicNames)
numSamples = args.numSamples
numSimIters = args.numSimIters
numPerturbSimIters = args.numPerturbSimIters
analysisMode = args.analysisMode
analysisRegion = ast.literal_eval(args.analysisRegion)
numGradientTimePoints = args.numGradientTimePoints
fileNumber = args.fileNumber
fileNumberVersion = args.fileNumberVersion
verbose = ast.literal_eval(args.verbose)

GJParameterNames = ['GJStrength']
fieldParameterNames = ['fieldEnabled','fieldResolution','fieldStrength','fieldAggregation','fieldScreenSize','fieldTransductionGain',
                       'fieldTransductionWeight','fieldTransductionBias','fieldTransductionTimeConstant','fieldRangeSymmetric','fieldVector']
ligandParameterNames = ['ligandEnabled','ligandGatingWeight','ligandGatingBias','ligandCurrentStrength','vmemToLigandCurrentStrength']
GRNParameterNames = ['GRNtoVmemWeights','GRNBiases','GRNtoVmemWeightsTimeconstant','GRNNumGenes']
simParameterNames = ['initialValues','externalInputs','numSamples','numSimIters']
if characteristicNames == 'Default':
    if analysisMode == 'fixScreenGJSweepWeightBias':
        characteristicNames = ['VarMaxValues','Dimensionality']
    elif analysisMode == 'fixWeightBiasSweepScreenGJ':
        characteristicNames = ['Dimensionality','Information','Robustness']
    elif analysisMode == 'fixBiasSweepWeightScreenGJ':
        characteristicNames = ['Dimensionality','Information','TSEComplexity','CelluarFrequency','Robustness','RobustnessGpol',
                               'RobustnessSwapVmem','Persistence','CorrelationDistance','Correlation','Covariance','Sensitivity','Hessian']
    elif analysisMode == 'sweepBiasWeightScreenGJFieldVector':
        characteristicNames = ['Dimensionality','Information','TSEComplexity','CelluarFrequency','Robustness','RobustnessGpol',
                               'RobustnessSwapVmem','Persistence','CorrelationDistance','Correlation','Covariance','Sensitivity','Hessian']
    elif (analysisMode == 'sensitivity') or (analysisMode == 'sensitivityFieldVector'):
        characteristicNames = ['Sensitivity']
    elif analysisMode == 'robustness':
        characteristicNames = ['Perturbation','Robustness']

def defineInitialValues(circuit,randomize=False):
    numCells = circuit.numCells
    initialValues = dict()
    initVmem = torch.FloatTensor(list(chain([-9.2e-3] * numSamples)))
    initialValues['Vmem'] = torch.repeat_interleave(initVmem,numCells,0).double().view(numSamples,numCells,1)
    initialValues['eV'] = torch.zeros((numSamples,circuit.numExtracellularGridPoints,1),dtype=torch.float64)
    initialValues['ligandConc'] = torch.zeros((numSamples,numCells,1),dtype=torch.float64)
    initialValues['G_pol'] = dict()
    AllCells = list(range(circuit.numCells))
    if randomize:  # 0th sample = homogenous; rest = heterogeneous
        initialValues['G_pol']['cells'] = [[AllCells]] * numSamples
        initialValues['G_pol']['values'] = [[torch.rand(numCells,dtype=torch.float64)*2] for _ in  range(numSamples)]  # covers a range of unistable and bistable values
        initialValues['G_pol']['values'][0] = torch.DoubleTensor([1.0]*numCells)  # homogenous state for 1st sample; bistable Vmem
    else:
        initialValues['G_pol']['cells'] = [[AllCells]] * numSamples
        initialValues['G_pol']['values'] = [torch.DoubleTensor([1.0]*numCells)] * numSamples  # bistable
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
    evPCAProps, eVCellWiseMeanPCAProps, vmemPCAProps, eVAggVmemJointPCAProps = [], [], [], []
    for sample in range(numSamples):
        evData = circuit.timeserieseV[startTime:,sample,:,0]
        evDataScaled = StandardScaler().fit_transform(evData)
        pca = PCA(n_components=ndims)
        eVPCA = pca.fit_transform(evDataScaled)
        evPCAProps.append(pca.explained_variance_ratio_)
        evCellWiseMeanData = (circuit.timeserieseV[:,sample,:] * circuit.fieldScreenMatrix).sum(1) / circuit.numFieldNeighbors
        evCellWiseMeanDataScaled = StandardScaler().fit_transform(evCellWiseMeanData)
        pca = PCA(n_components=ndims)
        eVCellWiseMeanPCA = pca.fit_transform(evCellWiseMeanDataScaled)
        eVCellWiseMeanPCAProps.append(pca.explained_variance_ratio_)
        vmemData = circuit.timeseriesVmem[startTime:,sample,:,0]
        vmemDataScaled = StandardScaler().fit_transform(vmemData)
        pca = PCA(n_components=ndims)
        vmemPCA = pca.fit_transform(vmemDataScaled)
        vmemPCAProps.append(pca.explained_variance_ratio_)
        eVAggVmemJointData = torch.cat((evCellWiseMeanData,vmemData),axis=1)
        eVAggVmemJointDataScaled = StandardScaler().fit_transform(eVAggVmemJointData)
        pca = PCA(n_components=ndims)
        eVAggVmemJointPCA = pca.fit_transform(eVAggVmemJointDataScaled)
        eVAggVmemJointPCAProps.append(pca.explained_variance_ratio_)
    return ([evPCAProps,eVCellWiseMeanPCAProps,vmemPCAProps,eVAggVmemJointPCAProps])

def computeInformationMeasures(circuit,region='topLeftQuadrant'):
    if region == 'full':
        targetIndices = np.array(range(circuit.numCells))
    else:
        targetIndices = np.array(utils.computeBulkIndices(circuit,mode='tissue',region=region))
    VmemBins = np.arange(-0.0, -0.1, -0.04)
    TotalCorr, Entropy = [], []
    for sample in range(numSamples):
        vbin = 2 - np.digitize(circuit.timeseriesVmem[:,sample,:,0].detach(),VmemBins)
        states = vbin[:,targetIndices]
        uniquestates, countsstates = np.unique(states,axis=0,return_counts=True)
        probsstates = countsstates / sum(countsstates)
        statestr = [''.join(str(bit) for bit in state) for state in uniquestates]
        distrdict = dict(zip(statestr,probsstates))
        distr = dit.Distribution(distrdict)
        TotalCorr.append(dit.multivariate.binding_information(distr))
        Entropy.append(dit.multivariate.entropy(distr))
    return ([TotalCorr,Entropy])

def computeTSEComplexity(circuit,region='topLeftQuadrant'):
    if region == 'full':
        cellIndicesAll = np.array(range(circuit.numCells))
    else:
        cellIndicesAll = np.array(utils.computeBulkIndices(circuit,mode='tissue',region=region))
    VmemBins = np.arange(-0.0, -0.1, -0.04)
    TSEComplexity = []
    numCells = len(cellIndicesAll)
    for sample in range(numSamples):
        vbin = 2 - np.digitize(circuit.timeseriesVmem[:,sample,:,0].detach(),VmemBins)
        scales = np.linspace(2,numCells-1,50,dtype=np.int16)
        totalSubScalesComplexity = 0
        for scale in scales:
            totalComplexityScale = 0
            for subsetsample in range(100):
                cellIndicesSubset = np.random.choice(cellIndicesAll,scale,replace=False)
                states = vbin[:,cellIndicesSubset]
                uniquestates, countsstates = np.unique(states,axis=0,return_counts=True)
                probsstates = countsstates / sum(countsstates)
                statestr = [''.join(str(bit) for bit in state) for state in uniquestates]
                distrdict = dict(zip(statestr,probsstates))
                distr = dit.Distribution(distrdict)
                entropy = dit.multivariate.entropy(distr)
                totalComplexityScale += entropy
            totalComplexityScale /= 100
            totalSubScalesComplexity += totalComplexityScale
        states = vbin[:,cellIndicesAll]
        uniquestates, countsstates = np.unique(states,axis=0,return_counts=True)
        probsstates = countsstates / sum(countsstates)
        statestr = [''.join(str(bit) for bit in state) for state in uniquestates]
        distrdict = dict(zip(statestr,probsstates))
        distr = dit.Distribution(distrdict)
        fullScaleComplexity = dit.multivariate.entropy(distr)
        complexity = totalSubScalesComplexity - (np.sum(scales) * fullScaleComplexity / circuit.numCells)
        TSEComplexity.append(complexity)
    return (TSEComplexity)

def computeCellularFrequency(circuit,region='topLeftQuadrant'):
    if region == 'full':
        cellIndicesAll = np.array(range(circuit.numCells))
    else:
        cellIndicesAll = np.array(utils.computeBulkIndices(circuit,mode='tissue',region=region))
    VmemBins = np.arange(-0.0, -0.1, -0.04)
    numSamples = circuit.timeseriesVmem.shape[1]
    numCells = len(cellIndicesAll)
    numTimeSteps = circuit.timeseriesVmem.shape[0]
    NumOnesPerCell = np.zeros((numSamples,numCells))
    NumFlips1To0PerCell = np.zeros((numSamples,numCells))
    NumFlips0To1PerCell = np.zeros((numSamples, numCells))
    for sample in range(numSamples):
        vbin = 2 - np.digitize(circuit.timeseriesVmem[:,sample,cellIndicesAll,0].detach(),VmemBins)
        numones = vbin.sum(0)
        NumOnesPerCell[sample] = numones
        flips = vbin[1:] - vbin[0:-1]
        numFlips0to1 = (flips==1).sum(0)
        NumFlips0To1PerCell[sample] = numFlips0to1
        numFlips1to0 = (flips==-1).sum(0)
        NumFlips1To0PerCell[sample] = numFlips1to0
    return (NumOnesPerCell,NumFlips1To0PerCell,NumFlips0To1PerCell)

def computeSensitivity(circuit,timePoints=[-1],region='topLeftQuadrant',order=1,returnPreviousOrders=False):
    if isinstance(region,str):
        targetVariables = utils.computeBulkIndices(circuit,mode='tissue',region=region)
    elif isinstance(region,list):
        targetVariables = region
    numTargetVmemVariables = len(targetVariables)
    numTimePoints = len(timePoints)
    VmemToVemSensitivity = torch.zeros(numTimePoints,circuit.numCells,numTargetVmemVariables)
    if circuit.fieldEnabled:
        if order == 1:
            eVToVmemSensitivity = torch.zeros(numTimePoints,circuit.numExtracellularGridPoints,numTargetVmemVariables)
        elif order == 2:
            if returnPreviousOrders:
                eVToVmemSensitivity = torch.zeros(numTimePoints,circuit.numExtracellularGridPoints,numTargetVmemVariables)
            eVToVmemToVmemHessian = torch.zeros(numTimePoints,circuit.numExtracellularGridPoints,circuit.numCells,numTargetVmemVariables)
    if circuit.ligandEnabled:
        if order == 1:
            ligandToVmemSensitivity = torch.zeros(numTimePoints,circuit.numCells,numTargetVmemVariables)
        if order == 2:
            ligandToVmemToVmemHessian = torch.zeros(numTimePoints,circuit.numCells,circuit.numCells,numTargetVmemVariables)
    if order > 1:
        createGraph = True
    else:
        createGraph = False
    for tIdx in range(numTimePoints):
        t = timePoints[tIdx]
        for targetVariable in range(numTargetVmemVariables):
            print(fileNumber,t,targetVariables[targetVariable])
            variable = targetVariables[targetVariable]
            JacobianVmem = torch.autograd.grad(circuit.timeseriesVmem[t-1,0,variable,0],circuit.VmemInit,
                               retain_graph=True,create_graph=createGraph)[0]
            VmemToVemSensitivity[tIdx,:,targetVariable] = JacobianVmem[0,:,0]
            if circuit.fieldEnabled:
                if order == 1:
                    JacobianeV = torch.autograd.grad(circuit.timeseriesVmem[t-1,0,variable,0],circuit.eVInit,
                                               retain_graph=True,create_graph=createGraph)[0]
                    eVToVmemSensitivity[tIdx,:,targetVariable] = JacobianeV[0,:,0]
                elif order == 2:
                    if returnPreviousOrders:
                        JacobianeV = torch.autograd.grad(circuit.timeseriesVmem[t-1,0,variable,0],circuit.eVInit,
                                               retain_graph=True,create_graph=False)[0]
                        eVToVmemSensitivity[tIdx,:,targetVariable] = JacobianeV[0,:,0]
                    for cell in range(circuit.numCells):
                        HessianeVVmem = torch.autograd.grad(JacobianVmem[0,cell,0],circuit.eVInit,
                                                            retain_graph=True,create_graph=False)[0]
                        eVToVmemToVmemHessian[tIdx,:,cell,targetVariable] = HessianeVVmem[0,:,0]
            if circuit.ligandEnabled:
                if order == 1:
                    JacobianLigand = torch.autograd.grad(circuit.timeseriesVmem[t-1,0,variable,0],circuit.ligandConcInit,
                                               retain_graph=True,create_graph=createGraph)[0]
                    ligandToVmemSensitivity[tIdx,:,targetVariable] = JacobianLigand[0,:,0]
                elif order == 2:
                    for cell in range(circuit.numCells):
                        HessianLigandVmem = torch.autograd.grad(JacobianVmem[0,cell,0],circuit.ligandConcInit,
                                                            retain_graph=True,create_graph=False)[0]
                        ligandToVmemToVmemHessian[tIdx,:,cell,targetVariable] = HessianLigandVmem[0,:,0]
    if circuit.fieldEnabled:
        if order == 1:
            Sensitivity = {'Derivatives':[eVToVmemSensitivity,VmemToVemSensitivity],'timePoints':timePoints}
            return (Sensitivity)  # only Hessian is returned
            # return([eVToVmemSensitivity,VmemToVemSensitivity])
        elif order == 2:
            Hessian = {'Derivatives':eVToVmemToVmemHessian,'timePoints':timePoints}
            if returnPreviousOrders:
                Sensitivity = {'Derivatives':[eVToVmemSensitivity,VmemToVemSensitivity],'timePoints':timePoints}
                return([Sensitivity,Hessian])
            return (Hessian)  # only Hessian is returned
    elif circuit.ligandEnabled:
        if order == 1:
            return ([ligandToVmemSensitivity,VmemToVemSensitivity])
        elif order == 2:
            return ({'Derivatives':[VmemToVemSensitivity,ligandToVmemToVmemHessian],'timePoints':timePoints})  # both Jacobian and Hessian are returned
    else:
        return ([VmemToVemSensitivity])

def computeCorrelationDistance(circuit,region='topLeftQuadrant',thresholdRank=1):
    if region == 'full':
        targetIndices = np.array(range(circuit.numCells))
    else:
        targetIndices = np.array(utils.computeBulkIndices(circuit,mode='tissue',region=region))
    correlationDistances = []
    for sample in range(numSamples):
        obs = circuit.timeseriesVmem[:,sample,targetIndices,0].numpy()
        correlationMatrix = np.corrcoef(obs.transpose()).__abs__()
        correlationMatrixFiltered = correlationMatrix.copy()
        correlationMatrixFiltered[range(correlationMatrixFiltered.shape[0]),range(correlationMatrixFiltered.shape[0])] = 0.0  # ignore self-correlations by setting them to 0
        thresholdCorrelations = np.sort(np.unique(correlationMatrixFiltered,axis=1),axis=1)[:,-thresholdRank].reshape(-1,1)  # row-wise thresholds
        correlationMatrixFiltered[correlationMatrixFiltered < thresholdCorrelations] = 0.0  # in each row set corr to 0 for values below corresponding threshold
        dists = utils.computePairwiseDistances(circuit.cellularCoordinates,circuit.cellularCoordinates)
        correlationDistanceMatrix = correlationMatrixFiltered.copy()
        for row in range(correlationDistanceMatrix.shape[0]):
            nonZeroIndices = np.nonzero(correlationDistanceMatrix[row])
            correlationDistanceMatrix[row,nonZeroIndices] = dists[0,targetIndices[row],targetIndices[nonZeroIndices]]
        correlationDistances.append(correlationDistanceMatrix.mean())
    return {'ThresholdRank':thresholdRank,'CorrelationDistances':np.array(correlationDistances)}

def computePearsonCorrelation(circuit,region='topLeftQuadrant'):
    if region == 'full':
        targetIndices = np.array(range(circuit.numCells))
    else:
        targetIndices = np.array(utils.computeBulkIndices(circuit,mode='tissue',region=region))
    correlations = []
    for sample in range(numSamples):
        obs = circuit.timeseriesVmem[:,sample,targetIndices,0].numpy()
        correlationMatrix = np.corrcoef(obs.transpose()).__abs__()
        correlations.append(correlationMatrix.mean())
    return np.array(correlations)

def computeCovariance(circuit,region='topLeftQuadrant'):
    if region == 'full':
        targetIndices = np.array(range(circuit.numCells))
    else:
        targetIndices = np.array(utils.computeBulkIndices(circuit,mode='tissue',region=region))
    numTargetIndices = len(targetIndices)
    covarianceMatrices = np.zeros((numSamples,numTargetIndices,numTargetIndices))
    for sample in range(numSamples):
        obs = circuit.timeseriesVmem[:,sample,targetIndices,0].numpy()
        covarianceMatrix = np.cov(obs.transpose()).__abs__()
        covarianceMatrices[sample] = covarianceMatrix
    return covarianceMatrices

def computeRobustness(circuit,referenceSample=0):
    referenceTrajectory = circuit.timeseriesVmem[-100:,referenceSample,:,0].view(100,1,-1)
    perturbedTrajectories = circuit.timeseriesVmem[-100:,1:,:,0].view(100,numSamples-1,-1)
    robustness = ((perturbedTrajectories - referenceTrajectory)**2).sum(2).sqrt().mean(0)
    return robustness

def computePersistence(circuit,referenceTimePoint=0):
    referencePattern = circuit.timeseriesVmem[referenceTimePoint,0,:,0].view(1,-1)
    observedPatterns = circuit.timeseriesVmem[-100:,0,:,0].view(100,-1)
    persistence = ((observedPatterns - referencePattern)**2).sum(1).sqrt().mean()
    return persistence

# Simulation parameters (typically fixed, unless otherwise specified below)
perturbationParameters = None
stochasticIonChannels = False
externalInputs = {'gene': None}
saveData = True

# indices of the features of a smiley in a 11x11 tissue
eyeIndices = np.array([24,25,35,36,29,30,40,41])  # left and right eyes
noseIndices = np.array([49,60,71])
mouthIndices = np.array([92,93,94])

# Default model parameters that will otherwise get updated in the routines below
GRNtoVmemWeights,GRNBiases,GRNtoVmemWeightsTimeconstant,GRNNumGenes = None,None,None,None

# The particular parameter combination will be chosen from a grid whose location will be determined by fileNumber
if analysisMode == 'fixScreenGJSweepWeightBias':  # total parameter combinations = 30x10 = 300
    fieldTransductionWeights = np.linspace(0,50,30)
    fieldTransductionBiases = np.linspace(0,0.1,10)
    parameterGrid = list(zip(np.repeat(fieldTransductionWeights,len(fieldTransductionBiases)),
                             np.tile(fieldTransductionBiases,len(fieldTransductionWeights))))
    parameterCombination = parameterGrid[int(fileNumber) - 1]  # so file numbers can start from 1
    fieldStrength *= fieldStrengthProp
    clampParameters = None
elif analysisMode == 'fixWeightBiasSweepScreenGJ':  # total parameter combinations = 15x20 = 300
    maxFieldScreenSize = 2*max(circuitDims)-1  # the field will permeate the entire tissue = 2(l-1)+1, where l is the max of circuitDims
    fieldScreenSizes = np.linspace(1,maxFieldScreenSize,15,dtype=np.int8)
    GJStrengths = np.linspace(0,1.0,20)
    parameterGrid = list(zip(np.repeat(fieldScreenSizes,len(GJStrengths)),
                             np.tile(GJStrengths,len(fieldScreenSizes))))
    parameterCombination = parameterGrid[int(fileNumber) - 1]  # so file numbers can start from 1
    fieldStrength *= fieldStrengthProp
    clampParameters = None
    # Robustness parameters
    if perturbationMode != 'None':
        Perturbation = dict()
        numCells = circuitRows * circuitCols
        perturbPointIndicesA = np.tile(np.arange(numCells),numSamples-1) # first sample is unperturbed and serves as the reference
        perturbPointIndicesB = np.concatenate([torch.randperm(numCells) for _ in range(numSamples-1)])
        numPerturbPoints = len(perturbPointIndicesA) / (numSamples-1)
        sampleIndices = np.repeat(range(1,numSamples),numPerturbPoints)  # assuming that there's only one sample in which the eye block is shifted
        perturbStartIter, perturbEndIter = numSimIters, numSimIters  # original numSimIters at the end of which a perturbation will be applied
        Perturbation['mode'] = perturbationMode
        Perturbation['data'] = (sampleIndices,(perturbPointIndicesA,perturbPointIndicesB))
        Perturbation['time'] = (perturbStartIter,perturbEndIter)
        numSimIters = numPerturbSimIters
        perturbationParameters = Perturbation
elif analysisMode == 'fixBiasSweepWeightScreenGJ':  # total parameter combinations = 10*5x10 = 500
    fieldTransductionWeights = np.linspace(0,50,10)
    fieldScreenSizes = np.array([1,4,10,15,20])
    GJStrengths = np.array([0,0.05,0.1,0.25,0.5,0.6,0.7,0.8,0.9,1.0])
    if fileNumber >= 1:  # choose from parameter grid
        parameterGrid = [(screensize,gj,weight) for screensize in fieldScreenSizes for gj in GJStrengths for weight in fieldTransductionWeights]
        parameterCombination = parameterGrid[int(fileNumber) - 1]  # so file numbers can start from 1
    else:  # choose from passed arguments
        parameterCombination = fieldScreenSize, GJStrength, fieldTransductionWeight
    fieldStrength *= fieldStrengthProp
    clampParameters = None
    # Robustness parameters
    if perturbationMode != 'None':
        Perturbation = dict()
        numCells = circuitRows * circuitCols
        Perturbation['mode'] = perturbationMode
        if perturbationMode == 'swapVmem':  # swap an eye block with the block below
            assert numSamples == 2
            perturbPointIndicesA = eyeIndices[0:4]
            perturbPointIndicesB = perturbPointIndicesA + 22  # shift the entire eye down by one block
            perturbStartIter, perturbEndIter = numSimIters, numSimIters
            numPerturbPoints = len(perturbPointIndicesA) / (numSamples-1)
            sampleIndices = np.repeat(range(1,numSamples),numPerturbPoints)  # assuming that there's only one sample in which the eye block is shifted
        elif perturbationMode == 'swapClampVmem':  # shift an eye block and transiently fix it
            assert numSamples == 1
            perturbPointIndicesA = eyeIndices[0:4]
            perturbPointIndicesB = perturbPointIndicesA + 22  # shift the entire eye down by one block
            perturbStartIter, perturbEndIter = numSimIters, numSimIters + 100
            numPerturbPoints = len(perturbPointIndicesA)
            sampleIndices = np.repeat(0,numPerturbPoints)  # assuming that there's only one sample in which the eye block is shifted
        else:  # 'permuteVmem' or 'permuteGpol'
            perturbPointIndicesA = np.tile(np.arange(numCells),numSamples-1) # first sample is unperturbed and serves as the reference
            perturbPointIndicesB = np.concatenate([torch.randperm(numCells) for _ in range(numSamples-1)])
            perturbStartIter, perturbEndIter = numSimIters, numSimIters  # original numSimIters at the end of which a perturbation will be applied
            numPerturbPoints = len(perturbPointIndicesA) / (numSamples-1)
            sampleIndices = np.repeat(range(1,numSamples),numPerturbPoints)  # assuming that there's only one sample in which the eye block is shifted
        Perturbation['data'] = (sampleIndices,(perturbPointIndicesA,perturbPointIndicesB))
        Perturbation['time'] = (perturbStartIter,perturbEndIter)
        numSimIters = numPerturbSimIters
        perturbationParameters = Perturbation
    else:
        perturbationParameters = None
elif analysisMode == 'sweepBiasWeightScreenGJFieldVector':
    fieldTransductionWeights = np.array([10,100,200,500,1000])
    fieldTransductionBiases = np.array([0.0005,0.002,0.005,0.01,0.02])
    fieldScreenSizes = np.array([1,4,10,15,21])
    GJStrengths = np.array([0,0.05,0.2,0.5,1.0])
    parameterGrid = [(screensize,gj,weight,bias) for screensize in fieldScreenSizes for gj in GJStrengths for weight in fieldTransductionWeights for bias in fieldTransductionBiases]
    fieldTransductionTimeConstant = torch.DoubleTensor([10.0])
    parameterCombination = parameterGrid[int(fileNumber) - 1]  # so file numbers can start from 1
    fieldStrength *= fieldStrengthProp
    clampParameters = None
    perturbationParameters = None
elif analysisMode == 'fixBiasSweepWeightLigandGJ':  # total parameter combinations = 10*5x10 = 500
    ligandGatingWeights = np.linspace(0,50,10)
    vmemToLigandCurrentStrengths = np.linspace(0,10,5)
    # GJStrengths = np.array([0.05,1.0])
    GJStrengths = np.array([0,0.05,0.1,0.25,0.5,0.6,0.7,0.8,0.9,1.0])
    if fileNumber >= 1:  # choose from parameter grid
        parameterGrid = [(ligand,gj,weight) for ligand in vmemToLigandCurrentStrengths for gj in GJStrengths for weight in ligandGatingWeights]
        parameterCombination = parameterGrid[int(fileNumber) - 1]  # so file numbers can start from 1
    else:  # choose from passed arguments
        parameterCombination = vmemToLigandCurrentStrength, GJStrength, ligandGatingWeight
    fieldStrength *= fieldStrengthProp
    clampParameters = None
    perturbationParameters = None
elif (analysisMode == 'sensitivity') or (analysisMode == 'robustness') or (analysisMode == 'sensitivityFieldVector'):
    if analysisMode == 'sensitivityFieldVector':
        parameterfilename = './data/bestModelParameters_fieldVector_' + str(int(fileNumber)) + '.dat'
    else:
        parameterfilename = './data/bestModelParameters_' + str(int(fileNumber)) + '.dat'
    parameters = torch.load(parameterfilename)
    circuitRows,circuitCols = circuitDims = parameters['latticeDims']
    GJParameters = parameters['GJParameters']
    fieldParameters = parameters['fieldParameters']
    fieldParameters['fieldStrength'] *= fieldStrengthProp
    ligandParameters = parameters['ligandParameters']
    GRNParameters = parameters['GRNParameters']
    externalInputs = parameters['simParameters']['externalInputs']
    if (analysisMode == 'sensitivity') or (analysisMode == 'sensitivityFieldVector'):
        numSamples = parameters['simParameters']['numSamples']
        initialValues = parameters['simParameters']['initialValues']
        clampParameters = parameters['clampParameters']
        numSimIters = parameters['simParameters']['numSimIters']
    elif analysisMode == 'robustness':
        clampParameters = None
        Perturbation = dict()
        numCells = circuitRows * circuitCols
        perturbPointIndicesA = np.tile(np.arange(numCells),numSamples-1) # first sample is unperturbed and serves as the reference
        perturbPointIndicesB = np.concatenate([torch.randperm(numCells) for _ in range(numSamples-1)])
        numPerturbPoints = len(perturbPointIndicesA) / (numSamples-1)
        sampleIndices = np.repeat(range(1,numSamples),numPerturbPoints)  # assuming that there's only one sample in which the eye block is shifted
        perturbStartIter, perturbEndIter = numSimIters, numSimIters  # original numSimIters at the end of which a perturbation will be applied
        Perturbation['mode'] = perturbationMode
        Perturbation['data'] = (sampleIndices,(perturbPointIndicesA,perturbPointIndicesB))
        Perturbation['time'] = (perturbStartIter,perturbEndIter)
        numSimIters = numPerturbSimIters
        perturbationParameters = Perturbation

if analysisMode == 'fixScreenGJSweepWeightBias':
    fieldTransductionWeight = torch.DoubleTensor([parameterCombination[0]])
    fieldTransductionBias = torch.DoubleTensor([parameterCombination[1]])
elif analysisMode == 'fixWeightBiasSweepScreenGJ':
    fieldScreenSize = parameterCombination[0]
    GJStrength = parameterCombination[1]
elif analysisMode == 'fixBiasSweepWeightScreenGJ':
    fieldScreenSize = parameterCombination[0]
    GJStrength = parameterCombination[1]
    fieldTransductionWeight = torch.DoubleTensor([parameterCombination[2]])
elif analysisMode == 'sweepBiasWeightScreenGJFieldVector':
    fieldScreenSize = parameterCombination[0]
    GJStrength = parameterCombination[1]
    fieldTransductionWeight = torch.DoubleTensor([parameterCombination[2]])
    fieldTransductionBias = torch.DoubleTensor([parameterCombination[3]])
elif analysisMode == 'fixBiasSweepWeightLigandGJ':
    vmemToLigandCurrentStrength = parameterCombination[0]
    GJStrength = parameterCombination[1]
    ligandGatingWeight = torch.DoubleTensor([parameterCombination[2]])

# Note that if analysisMode is 'sensitivity' then the parameters would be loaded from a file
if (analysisMode == 'sensitivity') or (analysisMode == 'sensitivityFieldVector'):  # parameters loaded from file
    parameters = dict()
    parameters['GJParameters'] = GJParameters
    parameters['fieldParameters'] = fieldParameters
    parameters['ligandParameters'] = ligandParameters
    parameters['GRNParameters'] = GRNParameters
    setGradient = True
    setGradientIter = clampParameters['clampEndIter'] + 1
    retainGradients = False
    circuit = cellularFieldNetwork(circuitDims,parameters=parameters,numSamples=numSamples)
elif analysisMode == 'robustness':  # parameters loaded from file
    parameters = dict()
    parameters['GJParameters'] = GJParameters
    parameters['fieldParameters'] = fieldParameters
    parameters['ligandParameters'] = ligandParameters
    parameters['ligandParameters']['vmemToLigandCurrentStrength'] = 0.0
    parameters['GRNParameters'] = GRNParameters
    setGradient = False
    setGradientIter = -1
    retainGradients = False
    circuit = cellularFieldNetwork(circuitDims,parameters=parameters,numSamples=numSamples)
    initialValues = defineInitialValues(circuit,randomize=randomizeInitialStates)
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
    if ('Sensitivity' in characteristicNames) or ('Hessian' in characteristicNames):
        setGradient = True
        setGradientIter = 1
        retainGradients = False
    else:
        setGradient = False
        setGradientIter = -1
        retainGradients = False
    circuit = cellularFieldNetwork(circuitDims,parameters=parameters,numSamples=numSamples)
    initialValues = defineInitialValues(circuit,randomize=randomizeInitialStates)  # randomizeInitialStates = False for 'Robustness' and True for the other characteristics

utils = utilities.utilities()

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
    Robustness = computeRobustness(circuit)
elif (analysisMode == 'fixBiasSweepWeightScreenGJ') or (analysisMode == 'sweepBiasWeightScreenGJFieldVector'):
    if 'Dimensionality' in characteristicNames:
        Dimensionality = computeDimensionality(circuit,ndims=3)
    if 'Information' in characteristicNames:
        if randomizeInitialStates:
            region = 'full'
        else:
            region = 'topLeftQuadrant'
        Information = computeInformationMeasures(circuit,region=region)
    if 'TSEComplexity' in characteristicNames:
        if randomizeInitialStates:
            region = 'full'
        else:
            region = 'topLeftQuadrant'
        TSEComplexity = computeTSEComplexity(circuit,region=region)
    if 'CellularFrequency' in characteristicNames:
        if randomizeInitialStates:
            region = 'full'
        else:
            region = 'topLeftQuadrant'
        CellularFrequency = computeCellularFrequency(circuit,region=region)
    if ('Robustness' in characteristicNames):  # permutationMode = permuteVmem
        Robustness = computeRobustness(circuit)
    if ('RobustnessGpol' in characteristicNames):  # permutationMode = permuteGpol
        RobustnessGpol = computeRobustness(circuit)
    if ('RobustnessSwapVmem' in characteristicNames):  # permutationMode = permuteVmem
        RobustnessSwapVmem = computeRobustness(circuit)
    if ('Persistence' in characteristicNames):  # permutationMode = swapClampVmem
        Persistence = computePersistence(circuit,referenceTimePoint=(numPerturbSimIters-900))
    if 'CorrelationDistance' in characteristicNames:
        if randomizeInitialStates:
            region = 'full'
        else:
            region = 'topLeftQuadrant'
        CorrelationDistance = computeCorrelationDistance(circuit,region=region,thresholdRank=2)
    if 'Correlation' in characteristicNames:
        if randomizeInitialStates:
            region = 'full'
        else:
            region = 'topLeftQuadrant'
        Correlation = computePearsonCorrelation(circuit,region=region)
    if 'Covariance' in characteristicNames:
        if randomizeInitialStates:
            region = 'full'
        else:
            region = 'topLeftQuadrant'
        Covariance = computeCovariance(circuit,region=region)
    if ('Sensitivity' in characteristicNames) and ('Hessian' in characteristicNames):
        timePoints = np.linspace(setGradientIter+1,numSimIters,numGradientTimePoints,dtype=np.int32)
        Sensitivity, Hessian = computeSensitivity(circuit,timePoints=timePoints,region=analysisRegion,order=2,
                                                  returnPreviousOrders=True)
    elif 'Sensitivity' in characteristicNames:
        timePoints = np.linspace(setGradientIter+1,numSimIters,numGradientTimePoints,dtype=np.int32)
        Sensitivity = computeSensitivity(circuit,timePoints=timePoints,region=analysisRegion,order=1,
                                         returnPreviousOrders=False)
    elif 'Hessian' in characteristicNames:
        timePoints = np.linspace(setGradientIter+1,numSimIters,numGradientTimePoints,dtype=np.int32)
        Hessian = computeSensitivity(circuit,timePoints=timePoints,region=analysisRegion,order=2,
                                     returnPreviousOrders=False)
elif analysisMode == 'fixBiasSweepWeightLigandGJ':
    if 'Sensitivity' in characteristicNames:
        timePoints = np.linspace(setGradientIter+1,numSimIters,numGradientTimePoints,dtype=np.int32)
        Sensitivity = computeSensitivity(circuit,timePoints=timePoints,region=analysisRegion,order=1)
    if 'Hessian' in characteristicNames:
        timePoints = np.linspace(setGradientIter+1,numSimIters,numGradientTimePoints,dtype=np.int32)
        Hessian = computeSensitivity(circuit,timePoints=timePoints,region=analysisRegion,order=2)
elif (analysisMode == 'sensitivity') or (analysisMode == 'sensitivityFieldVector'):
    timePoints = np.linspace(setGradientIter+1,numSimIters,numGradientTimePoints,dtype=np.int32)
    Sensitivity = computeSensitivity(circuit,timePoints=timePoints,region=analysisRegion)
elif analysisMode == 'robustness':
    Robustness = computeRobustness(circuit)

if analysisMode == 'fixScreenGJSweepWeightBias':
    Sfx = 'FixedScreenSizeGJ_'
elif analysisMode == 'fixWeightBiasSweepScreenGJ':
    Sfx = 'FixedWeightBias_'
elif analysisMode == 'fixBiasSweepWeightScreenGJ':
    if 'Covariance' in characteristicNames:
        Sfx = 'FixedBias_Covariance_'
    else:
        Sfx = 'FixedBias_'
elif analysisMode == 'sweepBiasWeightScreenGJFieldVector':
    Sfx = 'FixedNone_FieldVector_'
elif analysisMode == 'fixBiasSweepWeightLigandGJ':
    Sfx = 'FixedBias_Ligand_'
elif analysisMode == 'sensitivity':
    Sfx = 'Sensitivity_'
elif analysisMode == 'sensitivityFieldVector':
    Sfx = 'Sensitivity_FieldVector_'
elif analysisMode == 'robustness':
    Sfx = 'Robustness_'
if fileNumberVersion > 0:
    fileVersionSfx = '_V' + str(fileNumberVersion)
else:
    fileVersionSfx = ''
savefilename = './data/modelCharacteristics_' + Sfx + str(fileNumber) + fileVersionSfx + '.dat'

if os.path.isfile(savefilename):
    modelCharacteristics = torch.load(savefilename)
else:
    modelCharacteristics = dict()
    modelCharacteristics['latticeDims'] = circuitDims
    modelCharacteristics['GJParameters'] = dict()
    modelCharacteristics['fieldParameters'] = dict()
    modelCharacteristics['ligandParameters'] = dict()
    modelCharacteristics['GRNParameters'] = dict()
    modelCharacteristics['simParameters'] = dict()
    modelCharacteristics['analysisMode'] = analysisMode
    modelCharacteristics['characteristics'] = dict()

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


