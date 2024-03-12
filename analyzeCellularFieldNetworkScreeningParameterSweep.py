import torch
import numpy as np
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import math
import dit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

circuitDims = [(7,7),(10,10),(15,15)]
# circuitDims = [(3,3),(4,4)]
GapJunctionStrengths = np.linspace(0.05,1.0,7)
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
numSimIters = 50000

generataData = True
plotData = False

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
    buEntropySum = sum([dit.shannon.entropy(ibudistr,[i]) for i in range(numTopQuadrantBulkCells)])
    normTerm = dit.shannon.entropy(budistr)
    buTotalCorr = buTotalCorr / normTerm
    buEntropy = dit.shannon.entropy(budistr)
    return ([boTotalCorr,buTotalCorr,ibuTotalCorr,bobuTotalCorr,boibuTotalCorr,
             boEntropy,ibuEntropy,buEntropy,boEntropySum,ibuEntropySum,buEntropySum])

def computeDimensionality(circuit):
    evData = circuit.timeserieseV[:,0,:,0]
    evData = StandardScaler().fit_transform(evData)
    pca = PCA(n_components=2)
    eVPCA = pca.fit_transform(evData)
    evPCAProps = pca.explained_variance_ratio_
    evCellWiseMeanData = (circuit.timeserieseV * circuit.fieldCellNeighborhoodBitmap).sum(2) / circuit.numFieldNeighbors
    evCellWiseMeanData = StandardScaler().fit_transform(evCellWiseMeanData[:,0,:])
    pca = PCA(n_components=2)
    eVCellWiseMeanPCA = pca.fit_transform(evCellWiseMeanData)
    eVCellWiseMeanPCAProps = pca.explained_variance_ratio_
    vmemData = circuit.timeseriesVmem[:,0,:,0]
    vmemData = StandardScaler().fit_transform(vmemData)
    pca = PCA(n_components=2)
    vmemPCA = pca.fit_transform(vmemData)
    vmemPCAProps = pca.explained_variance_ratio_
    return ([evPCAProps,eVCellWiseMeanPCAProps,vmemPCAProps])

if generataData:
    for circuitDim in circuitDims:
        data = np.empty(16)
        for GapJunctionStrength in GapJunctionStrengths:
            maxNumBoundingSquares = 2*max(circuitDim) - 1  # Max value of numBoundingSquares so the field will permeate the entire tissue = 2(l-1)+1, where l is the max of circuitDims
            for numBoundingSquares in range(1,maxNumBoundingSquares+1,2):
                circuitRows, circuitCols = circuitDim
                numCells = circuitRows * circuitCols
                print('GJStrength = ', GapJunctionStrength, "numBoundingSquares = ",numBoundingSquares)
                fieldParameters = (fieldResolution,fieldStrength,(eVBias,eVWeight,evTimeConstant))
                circuit = cellularFieldNetwork(circuitDim, GRNParameters=(None, None, None, None),
                                               fieldParameters=fieldParameters, numSamples=numSamples)
                initialValues = defineInitialValues(circuit)
                circuit.initVariables(initialValues)
                circuit.initParameters(initialValues)
                circuit.G_0 = GapJunctionStrength * circuit.G_ref
                inputs = {'gene':None}
                fieldScreenParameters = {'numBoundingSquares':numBoundingSquares}
                circuit.simulate(inputs=inputs,fieldEnabled=True,fieldClampParameters=None,fieldScreenParameters=fieldScreenParameters,
                             perturbationParameters=None,numSimIters=numSimIters,stochasticIonChannels=False,saveData=True)
                InformationQuantities = computeTotalCorrelations(circuit)
                PCAQuantities = computeDimensionality(circuit)
                PCAQuantities = np.concatenate(PCAQuantities)
                entry = np.array([GapJunctionStrength,numBoundingSquares])
                entry = np.concatenate((entry,InformationQuantities,PCAQuantities))
                data = np.vstack((data,entry))
        data = data[1:]  # ignoring the first "empty" row
        data[data!=data] = 0.0  # replacing NaNs with zeros
        duration = int(numSimIters/1000)
        fname = ('./data/VmemInformationMeasures_' + str(duration) + 'K_' + str(circuitRows) + 'x' + str(circuitCols) + '.dat')
        torch.save(data,fname)
# elif plotData:
#     duration = int(numSimIters / 1000)
#     fname = ('./data/VmemInformationMeasures_' + str(duration) + 'K_' + str(circuitRows) + 'x' + str(circuitCols) + '.dat')
#     data = torch.load(fname)

