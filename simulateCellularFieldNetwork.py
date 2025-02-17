import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import utilities
import matplotlib.pyplot as plt

circuitRows,circuitCols = latticeDims = (11,11)
hardCodeInitSingleCell = False
hardCodeInitTissue = False
fieldEnabled = True
ligandEnabled = True
fieldResolution = 1
fieldStrength = 1.0  # default value 1.0
fieldScreenSize = 4 # max: 2*max(latticeDims) - 1 relative to the cell at the corner of the lattice
fieldTransductionBias = torch.DoubleTensor([0.0005])  # 0.0214
fieldTransductionWeight = torch.DoubleTensor([1000.0])  # 9.4505
fieldTransductionGain = -1.0
fieldTransductionTimeConstant = torch.DoubleTensor([10.0])
fieldRangeSymmetric = False
fieldVector = True
# RelativePermittivity = 10**5
GapJunctionStrength = 0.05  # meaningful values in [0.05,1.0]
ligandGatingWeight = torch.DoubleTensor([0.5])  # default: 1.0
ligandGatingBias = torch.DoubleTensor([0.5])  # default: 0.0
ligandDiffusionStrength = torch.DoubleTensor([1.0])  # default: 1.0
vmemToLigandTransductionWeight = torch.DoubleTensor([1.0])  # default: 1.0
clampMode = 'learned'  # possible values: learned, field, fieldDome, fieldDomeTwoFoldSymmetry, tissueVmem, tissueDomeVmem, tissueLigand, tissueDomeLigand, tissueGpol, tissueDomeGpol, None
if clampMode is not None:
    if clampMode != 'learned':
        clampType = 'oscillatory'  # possible values: oscillatory, staticConstant, staticRandom
        clampValue = 1.0  # static clamp value when clampType = 'staticConstant'
        minClampAmplitude, maxClampAmplitude = -1000.0, 1000.0  # field: (-100.0,100.0); ligand: (0.0,1.0)
        minClampFrequency, maxClampFrequency = 100.0, 1000.0
        clampedCellsProp = 1.0
        if clampedCellsProp == 0.0:
            clampMode = None
        clampDurationProp = 0.1
perturbationMode = 'setLigand'  # possible values: setLigand, tissueDome, tissueDomePartial, None
numSamples = 1
numSimIters = 20000
RandomizeInitialIonChannelState = False
RandomizeInitialField = False
stochasticIonChannels = False
BlockGapJunctions = False
AmplifyGapJunctions = False

VmemBins = np.arange(-0.0, -0.1, -0.04)

fieldParameters = dict()
fieldParameters['fieldEnabled'] = fieldEnabled
fieldParameters['fieldResolution'] = fieldResolution
fieldParameters['fieldStrength'] = fieldStrength
fieldParameters['fieldAggregation'] = 'average'
fieldParameters['fieldScreenSize'] = fieldScreenSize
fieldParameters['fieldTransductionWeight'] = fieldTransductionWeight
fieldParameters['fieldTransductionBias'] = fieldTransductionBias
fieldParameters['fieldTransductionGain'] = fieldTransductionGain
fieldParameters['fieldTransductionTimeConstant'] = fieldTransductionTimeConstant
fieldParameters['fieldRangeSymmetric'] = fieldRangeSymmetric
fieldParameters['fieldVector'] = fieldVector
GJParameters = dict()
GJParameters['GJStrength'] = GapJunctionStrength
fieldParameters['GJParameters'] = GJParameters
ligandParameters = dict()
ligandParameters['ligandEnabled'] = ligandEnabled
ligandParameters['ligandGatingWeight'] = ligandGatingWeight
ligandParameters['ligandGatingBias'] = ligandGatingBias
ligandParameters['ligandDiffusionStrength'] = ligandDiffusionStrength
ligandParameters['vmemToLigandTransductionWeight'] = vmemToLigandTransductionWeight

modelparameters = dict()
modelparameters['fieldParameters'] = fieldParameters
modelparameters['GJParameters'] = GJParameters
modelparameters['GRNParameters'] = None
modelparameters['ligandParameters'] = ligandParameters

def defineInitValues():
    initialValues = dict()
    initVmem = list(chain([-9.2e-3] * numSamples))
    # initVmem = list(chain([-0.055] * numSamples))
    initialValues['Vmem'] = torch.repeat_interleave(torch.DoubleTensor(initVmem),numCells,0).view(numSamples,numCells,1)
    if RandomizeInitialField:
        initialValues['eV'] = torch.rand((numSamples,numFieldGridPoints,1),dtype=torch.float64)
    else:
        initialValues['eV'] = torch.zeros((numSamples,numFieldGridPoints,1),dtype=torch.float64)
    initialValues['ligandConc'] = torch.zeros((numSamples,numCells,1),dtype=torch.float64)
    initialValues['G_pol'] = dict()
    AllCells = list(range(numCells))
    initialValues['G_pol']['cells'] = [[AllCells]] * numSamples
    if RandomizeInitialIonChannelState:
        initialValues['G_pol']['values'] = [[torch.rand(numCells,dtype=torch.float64)*2] for _ in  range(numSamples)]  # covers a range of unistable and bistable values
    else:
        initialValues['G_pol']['values'] = [torch.DoubleTensor([1.0])] * numSamples  # 0.0=dep(-5mV);1.0=bistable;2.0=hyp(-55mV)
    initialValues['G_dep'] = dict()
    initialValues['G_dep']['cells'] = []
    initialValues['G_dep']['values'] = torch.DoubleTensor([])
    return initialValues

circuit = cellularFieldNetwork(latticeDims=latticeDims,parameters=modelparameters,numSamples=numSamples)

numCells = circuit.numCells
numFieldGridPoints = circuit.numFieldGridPoints

initialValues = defineInitValues()
circuit.initVariables(initialValues)
circuit.initParameters(initialValues)
# circuit.G_0 = GapJunctionStrength * circuit.G_ref
# circuit.relativePermittivity = RelativePermittivity

utils = utilities.utilities()
fieldDomeIndices = utils.computeDomeIndices(circuit,mode='field')

# block gap junctions by zeroing GJ current
if BlockGapJunctions:
    circuit.G_0 = 0.0
    circuit.G_res = 0.0
elif AmplifyGapJunctions:
    circuit.G_0 = 0.05 * circuit.G_ref
    circuit.G_res = 0.0

if hardCodeInitSingleCell:  # Hyperpolarized cell at the center of the tissue
    circuit.Vmem[:] = 0.0
    circuit.Vmem[0,60,0] = -0.06
elif hardCodeInitTissue:  # French flag pattern
    circuit.Vmem[:] = 0.0
    circuit.Vmem[0,0:44,0] = -0.005
    circuit.Vmem[0,44:77,0] = -0.03
    circuit.Vmem[0,77:,0] = -0.06

print("Initial Vmem:")
print(circuit.Vmem.view(numSamples,*latticeDims))

if clampMode == 'field':
    numTotalCells = circuit.numFieldGridPoints
    cellIndices = np.arange(numTotalCells)
elif clampMode == 'fieldDome':
    numTotalCells = len(fieldDomeIndices)
    cellIndices = fieldDomeIndices
elif clampMode == 'fieldDomeTwoFoldSymmetry':
    fieldDomeLeftHalfIndices = utils.computeDomeIndices(circuit,mode='field',region='leftHalf')
    numTotalCells = len(fieldDomeLeftHalfIndices)
    cellIndices = fieldDomeLeftHalfIndices
elif (clampMode == 'tissueDomeVmem') or (clampMode == 'tissueDomeLigand') or (clampMode == 'tissueDomeGpol'):
    tissueDomeIndices = utils.computeDomeIndices(circuit,mode='tissue')
    numTotalCells = len(tissueDomeIndices)
    cellIndices = tissueDomeIndices
elif (clampMode == 'tissueVmem') or (clampMode == 'tissueLigand') or (clampMode == 'tissueGpol'):
    numTotalCells = circuit.numCells
    cellIndices = np.arange(numTotalCells)
elif clampMode == 'tissueDomeLigandTwoFoldSymmetry':
    tissueDomeLeftHalfIndices = utils.computeDomeIndices(circuit,mode='tissue',region='leftHalf')
    numTotalCells = len(tissueDomeLeftHalfIndices)
    cellIndices = tissueDomeLeftHalfIndices

if clampMode == 'learned':  # retrieve clamp parameters from a file
    filenum = 1576  # weakly sensitive: 1294; strongly sensitive: 1576
    if fieldVector:
        Sfx = '_fieldVector'
    else:
        Sfx = ''
    parameterfilename = './data/bestModelParameters' + Sfx + '_' + str(filenum) + '.dat'  # 472 (fr=4); OLD: 483 (fieldRange=4); 759 (fieldRange=1); 825 (fieldRange=21)
    parameters = torch.load(parameterfilename)
elif clampMode != None:
    numClampPoints = int(clampedCellsProp*numTotalCells)
    clampPointIndices = np.array([np.random.choice(cellIndices,numClampPoints,replace=False)
                                             for _ in range(numSamples)]).reshape(numSamples,-1).tolist()
    sampleIndices = np.repeat(range(numSamples),numClampPoints).reshape(numSamples,numClampPoints).tolist()
    # clampIndices = (sampleIndices,clampPointIndices)
    clampStartIter, clampEndIter = 0, int(clampDurationProp * numSimIters)
    numClampIters = clampEndIter - clampStartIter + 1
    timeIndices = torch.linspace(0,0.5,numClampIters).view(-1,1)
    if clampType == 'oscillatory':
        clampValues = torch.zeros(numClampIters).view(numClampIters,1)
        for sample in range(numSamples):
            clampFrequencies = torch.rand(numClampPoints,dtype=torch.double)*(maxClampFrequency-minClampFrequency) + minClampFrequency
            clampPhases = torch.rand(numClampPoints,dtype=torch.double)*2*torch.pi
            if 'Symmetry' in clampMode:
                if 'FourFoldSymmetry' in clampMode:
                    if 'field' in clampMode:
                        verticalReflectedIndices, horizontalReflectedIndices, diagonalReflectedIndices = \
                            utils.computeSymmetricalIndices(circuit,clampPointIndices[sample],mode='field',symmetry='fourfold')
                    clampFrequenciesActual = torch.tile(clampFrequencies,(4,))
                    clampPhasesActual = torch.tile(clampPhases,(4,))
                    clampPointIndices[sample] = np.concatenate((clampPointIndices[sample],verticalReflectedIndices,horizontalReflectedIndices,
                                                                diagonalReflectedIndices))
                elif 'TwoFoldSymmetry' in clampMode:
                    if 'field' in clampMode:
                        verticalReflectedIndices = utils.computeSymmetricalIndices(circuit,clampPointIndices[sample],mode='field',symmetry='twofold')
                    elif 'tissue' in clampMode:
                        verticalReflectedIndices = utils.computeSymmetricalIndices(circuit,clampPointIndices[sample],mode='tissue',symmetry='twofold')
                    clampFrequenciesActual = torch.tile(clampFrequencies,(2,))
                    clampPhasesActual = torch.tile(clampPhases,(2,))
                    clampPointIndices[sample] = np.concatenate((clampPointIndices[sample],verticalReflectedIndices))
                _, uniqueClampPointIndices = np.unique(clampPointIndices[sample],return_index=True)  # first-occurrence indices
                clampPointIndices[sample] = clampPointIndices[sample][uniqueClampPointIndices]  # this will always be a sorted array
                numClampPoints = len(clampPointIndices[sample])
                sampleIndices[sample] = np.repeat(sample,numClampPoints)
                # clampIndices = (sampleIndices,clampPointIndices)
                clampValuesSample = torch.cos(timeIndices*clampFrequenciesActual + clampPhasesActual)
                clampValuesSample = ((clampValuesSample+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
                clampValuesSample = clampValuesSample[:,uniqueClampPointIndices]
            else:
                clampValuesSample = torch.cos(timeIndices*clampFrequencies + clampPhases)
                clampValuesSample = ((clampValuesSample+1)/2)*(maxClampAmplitude-minClampAmplitude)+minClampAmplitude
            clampValues = torch.hstack((clampValues,clampValuesSample))
        clampValues = clampValues[:,1:]  # first column is a junk buffer
        sampleIndices = np.concatenate(sampleIndices)
        clampPointIndices = np.concatenate(clampPointIndices)
    elif clampType == 'staticConstant':
        clampValues = (torch.ones(numSamples*numClampPoints*numClampIters,dtype=torch.double)*clampValue).view(numClampIters,numClampPoints)
    elif clampType == 'staticRandom':
        clampValues = (torch.rand(numSamples*numClampPoints*numClampIters,dtype=torch.double)*clampValue).view(numClampIters,numClampPoints)
    clampIndices = (sampleIndices,clampPointIndices)
else:
    clampParameters = None

if clampMode == 'learned':
    clampParameters = parameters['clampParameters']
elif clampMode != None:
    clampParameters = dict()
    clampParameters['clampMode'] = clampMode
    clampParameters['clampIndices'] = clampIndices
    clampParameters['clampValues'] = clampValues
    clampParameters['clampStartIter'] = clampStartIter
    clampParameters['clampEndIter'] = clampEndIter
else:
    clampParameters = None

if perturbationMode == 'setLigand':
    perturbation = dict()
    # indices = [0]
    indices = np.arange(circuit.numCells)
    perturbPointIndicesA = np.tile(indices,numSamples)
    perturbPointIndicesB = None
    # perturbValues = 1.0
    perturbValues = torch.tensor(np.random.rand(len(indices))).view(-1,1)
    perturbStartIter, perturbEndIter = 1000, 1005
    numPerturbPoints = len(perturbPointIndicesA)
    sampleIndices = np.repeat(range(numSamples),numPerturbPoints)  # assuming that there's only one sample in which the eye block is shifted
    perturbation['mode'] = perturbationMode
    perturbation['data'] = (sampleIndices,(perturbPointIndicesA,perturbPointIndicesB),perturbValues)
    perturbation['time'] = (perturbStartIter,perturbEndIter)
else:
    perturbation = None

# externalInputs = {'gene':None}
# circuit.simulate(externalInputs=externalInputs,clampParameters=clampParameters,perturbationParameters=perturbation,
# 				 numSimIters=numSimIters,stochasticIonChannels=False,setGradient=False,retainGradients=False,saveData=True)

def simulate(circuit,clampParameters=None,perturbation=None,numSimIters=1):
        numCells = circuit.numCells
        if circuit.GRNEnabled:
            numGenes = circuit.geneNetwork.numGenes
            numVariables = numGenes * numCells
            circuit.timeseriesGRN = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numGenes*numCells).view(numSimIters,circuit.numSamples,numGenes*numCells,1)
            circuit.timeseriesGRNExternalInputs = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numVariables).view(numSimIters,circuit.numSamples,numVariables,1)
        circuit.timeseriesVmem = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numCells).view(numSimIters,circuit.numSamples,numCells,1)
        # the below are recorded for debugging purpose only
        circuit.timeseriesGdep = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numCells).view(numSimIters,circuit.numSamples,numCells,1)
        circuit.timeseriesIncurrent = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numCells).view(numSimIters,circuit.numSamples,numCells,1)
        circuit.timeseriesOutcurrent = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numCells).view(numSimIters,circuit.numSamples,numCells,1)
        circuit.timeseriesGij = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numCells*numCells).view(numSimIters,circuit.numSamples,numCells,numCells)
        circuit.timeseriesGJcurrent = torch.DoubleTensor([-999]*numSimIters*circuit.numSamples*numCells).view(numSimIters,circuit.numSamples,numCells,1)
        if clampParameters is not None:
            clampMode = clampParameters['clampMode']
            clampIndices = clampParameters['clampIndices']
            clampValues = clampParameters['clampValues']
            clampStartIter =  clampParameters['clampStartIter']
            clampEndIter = clampParameters['clampEndIter']
            sampleIndices, clampPointIndices = clampIndices
            # Compute the field distance matrix consisting of the pairwise distances between the clamp points and extracellular coordinates
            # shape = (numSamples,numClampPoints,numFieldGridPoints)
            if 'field' in clampMode:
                circuit.fieldClampSampleIndices = sampleIndices
                circuit.fieldClampPointIndices1D = clampPointIndices
                circuit.numFieldClampPoints = int(len(circuit.fieldClampPointIndices1D)/circuit.numSamples)
                circuit.clampFieldPointCoordinates = (circuit.extracellularCoordinates[0][:,circuit.fieldClampPointIndices1D].view(circuit.numSamples,circuit.numFieldClampPoints),
                                                                    circuit.extracellularCoordinates[1][:,circuit.fieldClampPointIndices1D].view(circuit.numSamples,circuit.numFieldClampPoints))
                # NOTE: The setdiff would have to be done separately for each set of clamp points
                circuit.fieldClampPointIndices2D = circuit.fieldClampPointIndices1D.reshape(circuit.numSamples,circuit.numFieldClampPoints)
                circuit.freeFieldPointIndices1D = np.concatenate([np.setdiff1d(range(circuit.numFieldGridPoints),indices)
                                                                 for indices in circuit.fieldClampPointIndices2D])
                circuit.freeFieldPointCoordinates = (circuit.extracellularCoordinates[0][:,circuit.freeFieldPointIndices1D].view(circuit.numSamples,-1),
                                                  circuit.extracellularCoordinates[1][:,circuit.freeFieldPointIndices1D].view(circuit.numSamples,-1))  # shape = (numSamples,numFreeFieldPoints)
                circuit.fieldClampDistanceMatrix = (circuit.utils.computePairwiseDistances(circuit.clampFieldPointCoordinates,circuit.freeFieldPointCoordinates).double()
                                                 .view(circuit.numSamples,-1,circuit.numFieldClampPoints))
                circuit.numFreeFieldPoints = circuit.numFieldGridPoints - circuit.numFieldClampPoints
                circuit.fieldFreeSampleIndices = np.repeat(range(circuit.numSamples),circuit.numFreeFieldPoints)
            elif 'tissue' in clampMode:
                sampleIndices, clampPointIndices = clampIndices
        else:
            clampMode, sampleIndices, clampPointIndices, clampValues, clampStartIter, clampEndIter = None, None, None, None, 0, -1
        if perturbation is not None:
            perturbStartIter, perturbEndIter = perturbation['time']
        else:
            perturbStartIter, perturbEndIter = 0, -1
        for iter in range(numSimIters):
            if circuit.GRNEnabled:
                circuit.timeseriesGRN[iter] = circuit.geneNetwork.state
                circuit.timeseriesGRNExternalInputs[iter] = circuit.geneNetwork.tissueExternalInputs
            circuit.timeseriesVmem[iter] = circuit.Vmem
            # the below are recorded for debugging purpose only
            circuit.timeseriesGdep[iter] = circuit.G_dep
            # circuit.timeseriesIncurrent[iter] = circuit.InCurrent
            # circuit.timeseriesOutcurrent[iter] = circuit.OutCurrent
            circuit.timeseriesGij[iter] = circuit.G_ij
            circuit.timeseriesGJcurrent[iter] = circuit.GapJunctionCurrent
            if circuit.GRNEnabled:
                externalInputs = {'gene':circuit.geneNetwork.state}
            else:
                externalInputs = {'gene':None}
            circuit.simulate(externalInputs=externalInputs,numSimIters=1,stochasticIonChannels=False,
                                setGradient=False,retainGradients=False,saveData=False)  # shape = (numSamples,numGenes*numCells,1)
            if (iter >= perturbStartIter) and (iter <= perturbEndIter):
                circuit.perturb(perturbation=perturbation,currentIter=iter)
            if (iter >= clampStartIter) and (iter <= clampEndIter):
                if ('field' in clampMode) and circuit.fieldEnabled:
                    circuit.eV[sampleIndices,clampPointIndices,0] = clampValues[iter,:]  # clamped points act like field sources themselves
                    circuit.updateExtracellularVoltage(source='eVClamp')
                    circuit.updateIonChannelConductance(inputSource='field',stochasticIonChannels=False,fieldAggregation=circuit.fieldAggregation,perturbation=None)
                    if circuit.ligandEnabled:
                        circuit.updateLigandConcentration(source='Vmem')
                        circuit.updateLigandConcentration(source='ligand')
                        # circuit.updateIonChannelConductance(inputSource='ligand',stochasticIonChannels=stochasticIonChannels,perturbation=None)
                        circuit.updateFieldSensitivity(inputSource='ligand')
                    circuit.updateCurrent()
                    circuit.updateVmem()
                elif 'Vmem' in clampMode:
                    circuit.Vmem[sampleIndices,clampPointIndices,0] = clampValues[iter,:]
                elif ('Ligand' in clampMode) and circuit.ligandEnabled:
                    circuit.ligandConc[sampleIndices,clampPointIndices,0] = clampValues[iter,:]
                    circuit.updateLigandConcentration(source='ligand')
                    # circuit.updateIonChannelConductance(inputSource='ligand',stochasticIonChannels=stochasticIonChannels,perturbation=None)
                    circuit.updateFieldSensitivity(inputSource='ligand')
                    circuit.updateCurrent()
                    circuit.updateVmem()
                elif 'Gpol' in clampMode:
                    circuit.G_pol[sampleIndices,clampPointIndices,0] = clampValues[iter,:] * circuit.G_ref
                    circuit.updateCurrent()
                    circuit.updateVmem()

simulate(circuit,clampParameters=clampParameters,perturbation=perturbation,numSimIters=numSimIters)
print("\nFinal Vmem:")
np.set_printoptions(precision=2, suppress=True)  # suppresses scientific notation such as the suffix in 100e+02
print(circuit.Vmem.view(numSamples,*latticeDims))
# counts = [np.unique(np.digitize(circuit.Vmem.round(decimals=2)[i],VmemBins),return_counts=True)[1] for i in range(circuit.Vmem.shape[0])]
# print(*counts,sep='\n')
# counts = torch.unique(circuit.Vmem.round(decimals=2),return_counts=True)
# print("\nCounts of unique Vmems: ",counts)