import torch
import numpy as np
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import math
import dit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy
from scipy.stats import pearsonr
from multiSyncPy import synchrony_metrics as sm
from statsmodels.tsa.stattools import grangercausalitytests

# circuitDims = [(7,7),(10,10),(15,15)]
circuitDims = [(2,2)]
GapJunctionStrengths = np.linspace(0.05,1.0,1).round(2)
# circuitRows,circuitCols = 10,10
# circuitDims = (circuitRows,circuitCols)  # (rows,columns) of lattice
fieldResolution = 1
fieldStrength = 10.0
# GapJunctionStrength = 1.0
# numBoundingSquares = 4
eVBias = torch.DoubleTensor([0.0214])  # 0.0214
eVWeight = torch.DoubleTensor([9.4505])  # 9.4505
evTimeConstant = torch.DoubleTensor([10.0])
numSamples = 1
numSimIters = 10
correlationMode = None  # possible values: 'InterGlobal', 'InterLocal', 'Intra
setGradient = True
retainGradients = False

generataData = True
plotData = False
saveData = True

fieldParameters = (fieldResolution,fieldStrength,(eVBias,eVWeight,evTimeConstant))

def defineInitialValues(circuit):
    initialValues = dict()
    initVmem = torch.FloatTensor(list(chain([-9.2e-3] * numSamples)))
    initialValues['Vmem'] = torch.repeat_interleave(initVmem,circuit.numCells,0).double().view(numSamples,numCells,1)
    initialValues['eV'] = torch.zeros((numSamples,circuit.numExtracellularGridPoints,1),dtype=torch.float64)
    initialValues['G_pol'] = dict()
    initialValues['G_pol']['cells'] = [[[0]]] * numSamples
    initialValues['G_pol']['values'] = [torch.DoubleTensor([1.0])] * numSamples  # bistable
    initialValues['G_dep'] = dict()
    initialValues['G_dep']['cells'] = []
    initialValues['G_dep']['values'] = torch.DoubleTensor([])
    return initialValues

VmemBins = np.arange(-0.0, -0.1, -0.04)

def computeTotalCorrelations(circuit):
    vbin = 2 - np.digitize(circuit.timeseriesVmem[:,0,:,0],VmemBins)
    nr = math.ceil(circuitRows/2); nc = math.ceil(circuitCols/2); r=circuit.cell_radius
    topQuadrantCoords = ((circuit.cellularCoordinates[0] <= (r*(2*nr-1))) & (circuit.cellularCoordinates[1] <= (r*(2*nc-1))))[0]
    boundaryCoords = ((circuit.cellularCoordinates[0] == r) | (circuit.cellularCoordinates[1] == r))[0]
    softBoundaryCoords = ((circuit.cellularCoordinates[0] == (3*r)) | (circuit.cellularCoordinates[1] == (3*r)))[0]
    topQuadrantIdx = np.arange(circuit.numCells)[topQuadrantCoords]
    topQuadrantBoundaryIdx = np.arange(circuit.numCells)[boundaryCoords & topQuadrantCoords]
    topQuadrantSoftBoundaryIdx = np.arange(circuit.numCells)[softBoundaryCoords & topQuadrantCoords]
    topQuadrantSoftBoundaryIdx = np.setdiff1d(topQuadrantSoftBoundaryIdx,topQuadrantBoundaryIdx)
    topQuadrantBulkIdx = np.setdiff1d(topQuadrantIdx,topQuadrantBoundaryIdx)
    topQuadrantInnerBulkIdx = np.setdiff1d(topQuadrantBulkIdx,topQuadrantSoftBoundaryIdx)
    numTopQuadrantBoundaryCells = len(topQuadrantBoundaryIdx)
    numTopQuadrantInnerBulkCells = len(topQuadrantInnerBulkIdx)
    numTopQuadrantBulkCells = len(topQuadrantBulkIdx)
    # Boundary-InnerBulk
    fullstates = vbin[:,np.concatenate((topQuadrantBoundaryIdx,topQuadrantInnerBulkIdx))]
    uniquefullstates, countsfullstates = np.unique(fullstates,axis=0,return_counts=True)
    probsfullstates = countsfullstates / sum(countsfullstates)
    fullstatestr = [''.join(str(bit) for bit in state) for state in uniquefullstates]
    fulldistrdict = dict(zip(fullstatestr,probsfullstates))
    fulldistr = dit.Distribution(fulldistrdict)
    boibuTotalCorr = dit.multivariate.binding_information(fulldistr,[list(range(numTopQuadrantBoundaryCells)),list(range(numTopQuadrantBoundaryCells,numTopQuadrantInnerBulkCells+numTopQuadrantBoundaryCells))])
    # normTerm = dit.shannon.entropy(fulldistr,list(range(numTopQuadrantBoundaryCells))) + dit.shannon.entropy(fulldistr,list(range(numTopQuadrantBoundaryCells,numTopQuadrantInnerBulkCells+numTopQuadrantBoundaryCells)))
    normTerm = dit.multivariate.entropy(fulldistr,[list(range(numTopQuadrantBoundaryCells)),list(range(numTopQuadrantBoundaryCells,numTopQuadrantInnerBulkCells+numTopQuadrantBoundaryCells))])
    boibuTotalCorr = boibuTotalCorr / normTerm
    ## Boundary-Bulk
    fullstates = vbin[:,np.concatenate((topQuadrantBoundaryIdx,topQuadrantBulkIdx))]
    uniquefullstates, countsfullstates = np.unique(fullstates,axis=0,return_counts=True)
    probsfullstates = countsfullstates / sum(countsfullstates)
    fullstatestr = [''.join(str(bit) for bit in state) for state in uniquefullstates]
    fulldistrdict = dict(zip(fullstatestr,probsfullstates))
    fulldistr = dit.Distribution(fulldistrdict)
    bobuTotalCorr = dit.multivariate.binding_information(fulldistr,[list(range(numTopQuadrantBoundaryCells)),list(range(numTopQuadrantBoundaryCells,numTopQuadrantBulkCells+numTopQuadrantBoundaryCells))])
    # normTerm = dit.shannon.entropy(fulldistr,list(range(numTopQuadrantBoundaryCells))) + dit.shannon.entropy(fulldistr,list(range(numTopQuadrantBoundaryCells,numTopQuadrantBulkCells+numTopQuadrantBoundaryCells)))
    normTerm = dit.multivariate.entropy(fulldistr,[list(range(numTopQuadrantBoundaryCells)),list(range(numTopQuadrantBoundaryCells,numTopQuadrantBulkCells+numTopQuadrantBoundaryCells))])
    bobuTotalCorr = bobuTotalCorr / normTerm
    ## Boundary
    bostates = vbin[:,topQuadrantBoundaryIdx]
    uniquebostates, countsbostates = np.unique(bostates,axis=0,return_counts=True)
    probsbostates = countsbostates / sum(countsbostates)
    bostatestr = [''.join(str(bit) for bit in state) for state in uniquebostates]
    bodistrdict = dict(zip(bostatestr,probsbostates))
    bodistr = dit.Distribution(bodistrdict)
    boTotalCorr = dit.multivariate.binding_information(bodistr)
    boEntropySum = sum([dit.shannon.entropy(bodistr,[i]) for i in range(numTopQuadrantBoundaryCells)])
    normTerm = dit.shannon.entropy(bodistr)
    boTotalCorr = boTotalCorr / normTerm
    boEntropy = dit.shannon.entropy(bodistr)
    ## InnerBulk
    ibustates = vbin[:,topQuadrantInnerBulkIdx]
    uniqueibustates, countsibustates = np.unique(ibustates,axis=0,return_counts=True)
    probsibustates = countsibustates / sum(countsibustates)
    ibustatestr = [''.join(str(bit) for bit in state) for state in uniqueibustates]
    ibudistrdict = dict(zip(ibustatestr,probsibustates))
    ibudistr = dit.Distribution(ibudistrdict)
    ibuTotalCorr = dit.multivariate.binding_information(ibudistr)
    ibuEntropySum = sum([dit.shannon.entropy(ibudistr,[i]) for i in range(numTopQuadrantInnerBulkCells)])
    normTerm = dit.shannon.entropy(ibudistr)
    ibuTotalCorr = ibuTotalCorr / normTerm
    ibuEntropy = dit.shannon.entropy(ibudistr)
    ## Bulk
    bustates = vbin[:,topQuadrantBulkIdx]
    uniquebustates, countsbustates = np.unique(bustates,axis=0,return_counts=True)
    probsbustates = countsbustates / sum(countsbustates)
    bustatestr = [''.join(str(bit) for bit in state) for state in uniquebustates]
    budistrdict = dict(zip(bustatestr,probsbustates))
    budistr = dit.Distribution(budistrdict)
    buTotalCorr = dit.multivariate.binding_information(budistr)
    buEntropySum = sum([dit.shannon.entropy(budistr,[i]) for i in range(numTopQuadrantBulkCells)])
    normTerm = dit.shannon.entropy(budistr)
    buTotalCorr = buTotalCorr / normTerm
    buEntropy = dit.shannon.entropy(budistr)
    return ([boTotalCorr,buTotalCorr,ibuTotalCorr,bobuTotalCorr,boibuTotalCorr,
             boEntropy,ibuEntropy,buEntropy,boEntropySum,ibuEntropySum,buEntropySum])

def computeDimensionality(circuit):
    evData = circuit.timeserieseV[:,0,:,0]
    evData = StandardScaler().fit_transform(evData)
    pca = PCA(n_components=4)
    eVPCA = pca.fit_transform(evData)
    evPCAProps = pca.explained_variance_ratio_
    evCellWiseMeanData = (circuit.timeserieseV * circuit.fieldCellNeighborhoodBitmap).sum(2) / circuit.numFieldNeighbors
    evCellWiseMeanData = StandardScaler().fit_transform(evCellWiseMeanData[:,0,:])
    pca = PCA(n_components=4)
    eVCellWiseMeanPCA = pca.fit_transform(evCellWiseMeanData)
    eVCellWiseMeanPCAProps = pca.explained_variance_ratio_
    vmemData = circuit.timeseriesVmem[:,0,:,0]
    vmemData = StandardScaler().fit_transform(vmemData)
    pca = PCA(n_components=4)
    vmemPCA = pca.fit_transform(vmemData)
    vmemPCAProps = pca.explained_variance_ratio_
    return ([evPCAProps,eVCellWiseMeanPCAProps,vmemPCAProps])

def computeCorrelation(circuit,mode='global'):
    if mode == 'InterGlobal':
        evAvg = circuit.timeserieseV[:,0,:,0].mean(1)
        vmemAvg = circuit.timeseriesVmem[:,0,:,0].mean(1)
        corr = pearsonr(evAvg,vmemAvg)
        return corr.statistic
    elif mode == 'InterLocal':
        evCellWiseMean = (circuit.timeserieseV * circuit.fieldCellNeighborhoodBitmap).sum(2) / circuit.numFieldNeighbors
        evCellWiseMean = evCellWiseMean[:,0,:]
        vmem = circuit.timeseriesVmem[:,0,:,0]
        corr= [pearsonr(evCellWiseMean[:,cell],vmem[:,cell]).statistic for cell in range(circuit.numCells)]
        return [np.mean(corr),np.var(corr)]

def computeGrangerCausality(circuit,circuitDim,maxCausalLag=1):
    vmem = circuit.timeseriesVmem[:,0,:,0]
    eV = (circuit.timeserieseV[:,0,:,:] * circuit.fieldCellNeighborhoodBitmap).sum(1) / circuit.numFieldNeighbors  # cell-wise mean
    nr = (circuitDim[0]/2); nc = (circuitDim[1]/2); r = circuit.cell_radius
    topQuadrantCoords = ((circuit.cellularCoordinates[0] <= (r*(2*nr-1))) &
                         (circuit.cellularCoordinates[1] <= (r*(2*nc-1))))[0]
    FieldVariables = VmemVariables = np.arange(circuit.numCells)[topQuadrantCoords]
    numFieldVariables, numVmemVariables = len(FieldVariables), len(VmemVariables)
    numTestStats = 4  # hardcoded number of test stats returned by grangercausalitytests
    VmemToFieldCausalStrengths = np.zeros((numTestStats,maxCausalLag,numVmemVariables,numFieldVariables))
    VmemToFieldCausalPValues = np.zeros((numTestStats,maxCausalLag,numVmemVariables,numFieldVariables))
    FieldToVmemCausalStrengths = np.zeros((numTestStats,maxCausalLag,numFieldVariables,numVmemVariables))
    FieldToVmemCausalPValues = np.zeros((numTestStats,maxCausalLag,numFieldVariables,numVmemVariables))
    VmemToVmemCausalStrengths = np.zeros((numTestStats,maxCausalLag,numVmemVariables,numVmemVariables))
    VmemToVmemCausalPValues = np.zeros((numTestStats,maxCausalLag,numVmemVariables,numVmemVariables))
    # Vmem to Vmem
    for vmemVariableFromIdx in range(numVmemVariables):
        for vmemVariableToIdx in range(numVmemVariables):
            vmemVariableFrom = VmemVariables[vmemVariableFromIdx]
            vmemVariableTo = VmemVariables[vmemVariableToIdx]
            vmemValueFrom = vmem[:,vmemVariableFrom]
            vmemValueTo = vmem[:,vmemVariableTo]
            vmemToVmemData = np.vstack((vmemValueTo,vmemValueFrom)).transpose()  # for measuring causation of 2nd col on 1st col
            vmemToVmemCausality = grangercausalitytests(vmemToVmemData,maxCausalLag,verbose=False)
            for lag in range(1,maxCausalLag+1):
                print('Vmem-Vmem',GapJunctionStrength,numBoundingSquares,vmemVariableFrom,vmemVariableTo,lag+1)
                statsResultsVmemToVmem = vmemToVmemCausality[lag][0]
                stats = np.array([statsResultsVmemToVmem[test][0:2] for test in statsResultsVmemToVmem.keys()])
                VmemToVmemCausalStrengths[:,lag-1,vmemVariableFromIdx,vmemVariableToIdx] = stats[:,0]  # test statistic scores
                VmemToVmemCausalPValues[:,lag-1,vmemVariableFromIdx,vmemVariableToIdx] = stats[:,1]  # test statistic p-values
    # eV to Vmem and Vmem to eV
    for fieldVariableIdx in range(numFieldVariables):
        for vmemVariableIdx in range(numVmemVariables):
            fieldVariable = FieldVariables[fieldVariableIdx]
            vmemVariable = VmemVariables[vmemVariableIdx]
            fieldValue = eV[:,fieldVariable]
            vmemValue = vmem[:,vmemVariable]
            vmemToFieldData = np.vstack((fieldValue,vmemValue)).transpose()  # for measuring causation of 2nd col on 1st col
            fieldToVmemData = np.vstack((vmemValue,fieldValue)).transpose()  # for measuring causation of 2nd col on 1st col
            vmemToFieldCausality = grangercausalitytests(vmemToFieldData,maxCausalLag,verbose=False)
            fieldToVmemCausality = grangercausalitytests(fieldToVmemData,maxCausalLag,verbose=False)
            for lag in range(1,maxCausalLag+1):
                print('eV-Vmem',GapJunctionStrength,numBoundingSquares,fieldVariable,vmemVariable,lag+1)
                statsResultsFieldToVmem = fieldToVmemCausality[lag][0]
                stats = np.array([statsResultsFieldToVmem[test][0:2] for test in statsResultsFieldToVmem.keys()])
                FieldToVmemCausalStrengths[:,lag-1,fieldVariableIdx,vmemVariableIdx] = stats[:,0]  # test statistic scores
                FieldToVmemCausalPValues[:,lag-1,fieldVariableIdx,vmemVariableIdx] = stats[:,1]  # test statistic p-values
                statsResultsVmemToField = vmemToFieldCausality[lag][0]
                stats = np.array([statsResultsVmemToField[test][0:2] for test in statsResultsVmemToField.keys()])
                VmemToFieldCausalStrengths[:,lag-1,vmemVariableIdx,fieldVariableIdx] = stats[:,0]  # test statistic scores
                VmemToFieldCausalPValues[:,lag-1,vmemVariableIdx,fieldVariableIdx] = stats[:,1]  # test statistic p-values
    return (FieldToVmemCausalStrengths,FieldToVmemCausalPValues,VmemToFieldCausalStrengths,VmemToFieldCausalPValues,
            VmemToVmemCausalStrengths,VmemToVmemCausalPValues)

def computeSynchronicity(circuit,circuitDim):
    nr = (circuitDim[0]/2)
    nc = (circuitDim[1]/2)
    r = circuit.cell_radius
    topQuadrantCoords = ((circuit.cellularCoordinates[0] <= (r * (2 * nr - 1))) &
                         (circuit.cellularCoordinates[1] <= (r * (2 * nc - 1))))[0]
    topQuadrantIdx = np.arange(circuit.numCells)[topQuadrantCoords]
    boundaryCoords = ((circuit.cellularCoordinates[0] == r) | (circuit.cellularCoordinates[1] == r))[0]
    topQuadrantBoundaryIdx = np.arange(circuit.numCells)[boundaryCoords & topQuadrantCoords]
    topQuadrantBulkIdx = np.setdiff1d(topQuadrantIdx, topQuadrantBoundaryIdx)
    topQuadrantCoordsEV = ((circuit.extracellularCoordinates[0] <= (r * (2 * nr - 1))) &
                          (circuit.extracellularCoordinates[1] <= (r * (2 * nc - 1))))[0]
    topQuadrantIdxEV = np.arange(circuit.numExtracellularGridPoints)[topQuadrantCoordsEV]
    boundaryCoordsEV = ((circuit.extracellularCoordinates[0] == r) | (circuit.extracellularCoordinates[1] == r))[0]
    topQuadrantBoundaryIdxEV = np.arange(circuit.numExtracellularGridPoints)[boundaryCoordsEV & topQuadrantCoordsEV]
    topQuadrantBulkIdxEV = np.setdiff1d(topQuadrantIdxEV, topQuadrantBoundaryIdxEV)
    v = circuit.timeseriesVmem[:,0,np.concatenate((topQuadrantBulkIdx,topQuadrantBoundaryIdx)),0].t()
    ev = circuit.timeserieseV[:,0,np.concatenate((topQuadrantBulkIdxEV,topQuadrantBoundaryIdxEV)),0].t()
    vcoherence = sm.coherence_team(v)
    evcoherence = sm.coherence_team(ev)
    vangle = np.angle(scipy.signal.hilbert(v))
    evangle = np.angle(scipy.signal.hilbert(ev))
    vrho = sm.rho(vangle)[1]
    evrho = sm.rho(evangle)[1]
    return ([vcoherence,evcoherence,vrho,evrho])

def computeSensitivity(circuit,circuitDim):
    nr = (circuitDim[0]/2)
    nc = (circuitDim[1]/2)
    r = circuit.cell_radius
    topQuadrantCoords = ((circuit.cellularCoordinates[0] <= (r * (2 * nr - 1))) &
                         (circuit.cellularCoordinates[1] <= (r * (2 * nc - 1))))[0]
    topQuadrantIdx = np.arange(circuit.numCells)[topQuadrantCoords]
    boundaryCoords = ((circuit.cellularCoordinates[0] == r) | (circuit.cellularCoordinates[1] == r))[0]
    topQuadrantBoundaryIdx = np.arange(circuit.numCells)[boundaryCoords & topQuadrantCoords]
    topQuadrantBulkIdx = np.setdiff1d(topQuadrantIdx, topQuadrantBoundaryIdx)
    topQuadrantVmemVariables = np.concatenate((topQuadrantBoundaryIdx,topQuadrantBulkIdx))
    numTargetVmemVariables = len(topQuadrantVmemVariables)
    eVToVmemSensitivity = torch.zeros(circuit.numExtracellularGridPoints,numTargetVmemVariables)
    VmemToVemSensitivity = torch.zeros(circuit.numCells,numTargetVmemVariables)
    for variableIdx in range(numTargetVmemVariables):
        print("Computing sensitivity of variable: ", variableIdx)
        variable = topQuadrantVmemVariables[variableIdx]
        circuit.Vmem[0,variable,0].backward(retain_graph=True)
        eVToVmemSensitivity[:,variableIdx] = circuit.eVInit.grad.data[0,:,0]
        VmemToVemSensitivity[:,variableIdx] = circuit.VmemInit.grad.data[0,:,0]
        circuit.eVInit.grad.data.zero_()
        circuit.VmemInit.grad.data.zero_()
        circuit.GpolInit.grad.data.zero_()
    return([eVToVmemSensitivity,VmemToVemSensitivity])

if generataData:
    for circuitDim in circuitDims:
        # data = np.empty(14)
        # data = np.empty(6)
        data = dict()
        for GapJunctionStrength in GapJunctionStrengths:
            maxNumBoundingSquares = 2*max(circuitDim) - 1  # Max value of numBoundingSquares so the field will permeate the entire tissue = 2(l-1)+1, where l is the max of circuitDims
            # for numBoundingSquares in range(1,maxNumBoundingSquares+1,2):
            for numBoundingSquares in [4]:
                circuitRows, circuitCols = circuitDim
                numCells = circuitRows * circuitCols
                print('CircuitDims = ', circuitDim, 'GJStrength = ', GapJunctionStrength, "numBoundingSquares = ", numBoundingSquares)
                fieldParameters = (fieldResolution,fieldStrength,(eVBias,eVWeight,evTimeConstant))
                circuit = cellularFieldNetwork(circuitDim, GRNParameters=(None, None, None, None),
                                               fieldParameters=fieldParameters, numSamples=numSamples)
                initialValues = defineInitialValues(circuit)
                circuit.initVariables(initialValues)
                circuit.initParameters(initialValues)
                circuit.G_0 = GapJunctionStrength * circuit.G_ref
                externalInputs = {'gene':None}
                fieldScreenParameters = {'numBoundingSquares':numBoundingSquares}
                circuit.simulate(externalInputs=externalInputs,fieldEnabled=True,clampParameters=None,fieldScreenParameters=fieldScreenParameters,
                             perturbationParameters=None,numSimIters=numSimIters,stochasticIonChannels=False,
                             setGradient=setGradient,retainGradients=retainGradients,saveData=True)
                # InformationQuantities = computeTotalCorrelations(circuit)
                # PCAQuantities = np.concatenate(computeDimensionality(circuit))
                # FieldVoltageCorrelation = computeCorrelation(circuit,mode=correlationMode)
                # IntraFieldVoltageSynchronicity = computeSynchronicity(circuit,circuitDim)
                # grangerCausalityStats = computeGrangerCausality(circuit,circuitDim,maxCausalLag=5)
                causalSensitivities = computeSensitivity(circuit,circuitDim)
                # entry = np.array([GapJunctionStrength,numBoundingSquares])
                # entry = np.concatenate((entry,InformationQuantities,PCAQuantities))
                # entry = np.concatenate((entry,PCAQuantities))
                # entry = np.concatenate((entry,FieldVoltageCorrelation))
                # entry = np.concatenate((entry,IntraFieldVoltageSynchronicity))
                # data = np.vstack((data,entry))
                # data[(GapJunctionStrength,numBoundingSquares)] = grangerCausalityStats
                data[(GapJunctionStrength,numBoundingSquares)] = causalSensitivities
                duration = int(numSimIters/1000)
                if saveData:
                    fname = ('./data/CausalSensitivity' + str(duration) + 'K_' + str(circuitRows) + 'x' + str(circuitCols) + '.dat')
                    torch.save(data,fname)
        # data = data[1:]  # ignoring the first "empty" row
        # data[data!=data] = 0.0  # replacing NaNs with zeros
        duration = int(numSimIters/1000)
        if saveData:
            fname = ('./data/CausalSensitivity' + str(duration) + 'K_' + str(circuitRows) + 'x' + str(circuitCols) + '.dat')
            torch.save(data,fname)
# elif plotData:
#     duration = int(numSimIters / 1000)
#     fname = ('./data/VmemInformationMeasures_' + str(duration) + 'K_' + str(circuitRows) + 'x' + str(circuitCols) + '.dat')
#     data = torch.load(fname)

