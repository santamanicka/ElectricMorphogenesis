import torch
import numpy as np
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
import math
import dit

circuitRows,circuitCols = 4,4
circuitDims = (circuitRows,circuitCols)  # (rows,columns) of lattice
fieldResolution = 1
fieldStrength = 10.0
# GapJunctionStrength = 1.0
maxNumBoundingSquares = 2*max(circuitDims) - 1  # Max value of numBoundingSquares so the field will permeate the entire tissue = 2(l-1)+1, where l is the max of circuitDims
# numBoundingSquares = 4
eVBias = torch.DoubleTensor([0.0214])  # 0.0214
eVWeight = torch.DoubleTensor([9.4505])  # 9.4505
evTimeConstant = torch.DoubleTensor([10.0])
numSamples = 1
numSimIters = 100
numCells = circuitRows * circuitCols

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
    # normTerm = sum([dit.shannon.entropy(bodistr,[i]) for i in range(numTopQuadrantBoundaryCells)])
    normTerm = dit.shannon.entropy(bodistr)
    boTotalCorr = boTotalCorr / normTerm
    ## InnerBulk
    ibustates = vbin[:,topQuadrantInnerBulkIdx]
    uniqueibustates, countsibustates = np.unique(ibustates,axis=0,return_counts=True)
    probsibustates = countsibustates / sum(countsibustates)
    ibustatestr = [''.join(str(bit) for bit in state) for state in uniqueibustates]
    ibudistrdict = dict(zip(ibustatestr,probsibustates))
    ibudistr = dit.Distribution(ibudistrdict)
    ibuTotalCorr = dit.multivariate.binding_information(ibudistr)
    # normTerm = sum([dit.shannon.entropy(ibudistr,[i]) for i in range(numTopQuadrantInnerBulkCells)])
    normTerm = dit.shannon.entropy(ibudistr)
    ibuTotalCorr = ibuTotalCorr / normTerm
    ## Bulk
    bustates = vbin[:,topQuadrantBulkIdx]
    uniquebustates, countsbustates = np.unique(bustates,axis=0,return_counts=True)
    probsbustates = countsbustates / sum(countsbustates)
    bustatestr = [''.join(str(bit) for bit in state) for state in uniquebustates]
    budistrdict = dict(zip(bustatestr,probsbustates))
    budistr = dit.Distribution(budistrdict)
    buTotalCorr = dit.multivariate.binding_information(budistr)
    # normTerm = sum([dit.shannon.entropy(ibudistr,[i]) for i in range(numTopQuadrantInnerBulkCells)])
    normTerm = dit.shannon.entropy(budistr)
    buTotalCorr = buTotalCorr / normTerm
    return ([boTotalCorr,buTotalCorr,ibuTotalCorr,bobuTotalCorr,boibuTotalCorr])

data = np.empty(7)
for GapJunctionStrength in [0.05,0.5,1.0]:
    for numBoundingSquares in range(1,maxNumBoundingSquares+1,2):
        print('GJStrength = ', GapJunctionStrength, "numBoundingSquares = ",numBoundingSquares)
        fieldParameters = (fieldResolution,fieldStrength,(eVBias,eVWeight,evTimeConstant))
        circuit = cellularFieldNetwork(circuitDims, GRNParameters=(None, None, None, None),
                                       fieldParameters=fieldParameters, numSamples=numSamples)
        initialValues = defineInitialValues(circuit)
        circuit.initVariables(initialValues)
        circuit.initParameters(initialValues)
        circuit.G_0 = GapJunctionStrength * circuit.G_ref
        inputs = {'gene':None}
        fieldScreenParameters = {'numBoundingSquares':numBoundingSquares}
        circuit.simulate(inputs=inputs,fieldEnabled=True,fieldClampParameters=None,fieldScreenParameters=fieldScreenParameters,
                     perturbationParameters=None,numSimIters=numSimIters,stochasticIonChannels=False,saveData=True)
        corrQuantities = computeTotalCorrelations(circuit)
        entry = np.array([GapJunctionStrength,numBoundingSquares])
        entry = np.concatenate((entry,corrQuantities))
        data = np.vstack((data,entry))

data = data[1:]  # ignoring the first "empty" row
data[data!=data] = 0.0  # replacing NaNs with zeros
duration = int(numSimIters/1000)
fname = ('./data/VmemTotalCorrelations_' + str(duration) + 'K_' + str(circuitRows) + 'x' + str(circuitCols) + '.dat')
torch.save(data,fname)

