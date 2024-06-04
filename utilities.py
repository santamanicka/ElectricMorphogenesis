import torch
import numpy as np
import math

class utilities():

    def __init__(self):
        pass

    # Computes a 2D lattice adjacency matrix
    # This procedure does not require cellular coordinates; it only needs the (row,column) indices
    def computeLatticeAdjacencyMatrix(self,latticeDims):
        numrows, numcols = latticeDims[0], latticeDims[1]
        rowIndices = np.repeat(np.arange(0, numrows), numcols)  # should be 'repeat' for correct (row-wise) ordering of cells
        colIndices = np.tile(np.arange(0, numcols), numrows)  # should be 'tile' for correct (row-wise) ordering of cells
        numCellsLayer = numrows * numcols
        cellIndices = np.arange(numCellsLayer)
        AdjMatrix = np.zeros((numCellsLayer, numCellsLayer), dtype=np.int8)
        for cell in cellIndices:
            r = rowIndices[cell]
            c = colIndices[cell]
            rowNeighbors = cellIndices[(rowIndices == r) & (np.abs(colIndices - c) == 1)]
            colNeighbors = cellIndices[(colIndices == c) & (np.abs(rowIndices - r) == 1)]
            AdjMatrix[cell, rowNeighbors] = 1
            AdjMatrix[cell, colNeighbors] = 1
        AdjMatrix = torch.tensor(AdjMatrix)
        return AdjMatrix

    # Computes the coordinates of the centers of the cells in a 2D lattice
    def computeCellularCoordinates(self,latticeDims,cell_radius):
        numrows, numcols = latticeDims[0], latticeDims[1]
        offset = 2 * cell_radius  # diameter-length spaces between adjacent cell-centers
        xcoords = torch.repeat_interleave(torch.arange(numrows),numcols) * offset
        ycoords = torch.tile(torch.arange(numcols),(numrows,)) * offset
        # Offset the entire coordinate to leave room for the field coordinates to begin at the corner of the square grid at (0,0)
        xcoords, ycoords = xcoords + cell_radius, ycoords + cell_radius
        return (xcoords,ycoords)

    # Computes the absolute coordinates of the points on the circumscribed square around the (circular) cells;
    # field coordinates start from (0,0) and increase positively in both directions in units of (2*cell_radius/resolution).
    # Here, resolution=1 covers only the corners of the squares; higher resolutions cover more points
    ## TODO: include option to make circular grid around each cell rather than square grid
    def computeExtracellularCoordinates(self,latticeDims,cell_radius=1,resolution=1):
        numrows, numcols = latticeDims[0], latticeDims[1]
        numrows_res, numcols_res = (numrows * resolution) + 1, (numcols * resolution) + 1
        offset = 2 * cell_radius / resolution
        resolution_per_row = np.pad(np.concatenate([[numcols_res],np.repeat(numcols+1,resolution-1)]),
                                    (0,numrows_res-resolution),'wrap')
        xcoords = torch.concatenate(list(map(lambda x: torch.repeat_interleave(torch.tensor(x[0]),x[1])*offset,
                                             zip(torch.arange(numcols_res),resolution_per_row))))
        ycoords = torch.concatenate(list(map(lambda x: torch.linspace(0,numcols_res-1,x,dtype=torch.int8)*offset,
                                             resolution_per_row)))
        return (xcoords,ycoords)

    # Computes the relative coordinates of the points on the circumscribed square around the (circular) cells;
    # field coordinates start from (0,0) and increase positively in both directions in units of one.
    # Also computes a 2d matrix containing the field point indices at their appropriate locations.
    def computeExtracellularIndexCoordinates(self,circuit):
        latticeDims = circuit.latticeDims
        extracellularXIndices = np.digitize(circuit.extracellularCoordinates[0],bins=np.unique(circuit.extracellularCoordinates[0]))-1
        extracellularYIndices = np.digitize(circuit.extracellularCoordinates[1],bins=np.unique(circuit.extracellularCoordinates[1]))-1
        nr = latticeDims[0]*circuit.fieldResolution+1; nc = latticeDims[1]*circuit.fieldResolution+1
        extracellularIndexGrid = np.ones((nr,nc)) * -1
        extracellularIndexGrid[extracellularXIndices,extracellularYIndices] = np.arange(circuit.numExtracellularGridPoints)
        return (extracellularXIndices,extracellularYIndices,extracellularIndexGrid)

    def computeDomeIndices(self,circuit,mode='field',region='full'):
        cellRadius = circuit.cell_radius
        if mode == 'field':
            coords = circuit.extracellularIndexCoordinates
            numIndices = circuit.numExtracellularGridPoints
            res = circuit.fieldResolution
            dims = (circuit.latticeDims[0]*res)+1, (circuit.latticeDims[1]*res)+1
        elif mode == 'tissue':
            coords = circuit.cellularCoordinates
            numIndices = circuit.numCells
            dims = circuit.latticeDims
        if region == 'full':
            numBoundRows = dims[0]
            numBoundCols = dims[1]
            xCoordMin, yCoordMin = coords[0].min(), coords[1].min()
            xCoordMax, yCoordMax = coords[0].max(), coords[1].max()
            boundaryCoords = (((coords[0] <= (cellRadius*(2*numBoundRows-1))) & (coords[1] == yCoordMin)) |  # left side
                              ((coords[0] <= (cellRadius*(2*numBoundRows-1))) & (coords[1] == yCoordMax)) |  # right side
                              ((coords[1] <= (cellRadius*(2*numBoundCols-1))) & (coords[0] == xCoordMin)) |  # top side
                              ((coords[1] <= (cellRadius*(2*numBoundCols-1))) & (coords[0] == xCoordMax)))[0]  # bottom side
        elif region == 'topLeftQuadrant':
            numBoundRows = math.ceil(dims[0]/2)
            numBoundCols = math.ceil(dims[1]/2)
            xCoordMin, yCoordMin = coords[0].min(), coords[1].min()
            boundaryCoords = (((coords[0] <= (cellRadius*(2*numBoundRows-1))) & (coords[1] == yCoordMin)) |  # left side
                              ((coords[1] <= (cellRadius*(2*numBoundCols-1))) & (coords[0] == xCoordMin)))[0]  # top side
        elif region == 'leftHalf':
            numBoundRows = dims[0] - 1
            numBoundCols = math.ceil(dims[1]/2) - 1
            boundaryCoords = (((coords[0] <= numBoundRows) & (coords[1] == 0)) |  # left side
                                ((coords[1] <= numBoundCols) & (coords[0] == 0)) |  # top side
                                ((coords[1] <= numBoundCols) & (coords[0] == numBoundRows)))[0]  # bottom side
        boundaryIndices = np.arange(numIndices)[boundaryCoords]
        return boundaryIndices.tolist()

    def computeCoreIndices(self,circuit,mode='field',numCoreSquares=1):
        cellRadius = circuit.cell_radius
        if mode == 'field':
            coords = circuit.extracellularCoordinates
            numIndices = circuit.numExtracellularGridPoints
            dims = circuit.latticeDims[0]+1, circuit.latticeDims[1]+1
            # boundDistance = ((numCoreSquares * 2) + 1) * cellRadius
        elif mode == 'tissue':
            coords = circuit.cellularCoordinates
            numIndices = circuit.numCells
            dims = circuit.latticeDims
            # boundDistance = numCoreSquares * 2 * cellRadius
        offset = 1 - (dims[0] % 2)  # offset is 1 for even dims and 0 for odd dims (assuming square lattice)
        boundDistance = (1 + offset) * numCoreSquares * cellRadius
        center = coords[0].mean(), coords[1].mean()
        offsetCoords = (coords[0]-center[0]).abs(), (coords[1]-center[1]).abs()
        padding = 0.01*cellRadius  # to accommodate numerical precision
        coreCoords = ((offsetCoords[0] <= (boundDistance+padding)) & (offsetCoords[1] <= (boundDistance+padding)))[0]
        coreIndices = np.arange(numIndices)[coreCoords]
        return coreIndices.tolist()

    def computeSymmetricalIndices(self,circuit,indices,mode='field',symmetry="fourfold"):
        if mode == 'field':
            res = circuit.fieldResolution
            dims = (circuit.latticeDims[0]*res)+1, (circuit.latticeDims[1]*res)+1
        elif mode == 'tissue':
            dims = circuit.latticeDims
        numRows, numCols = dims
        indices = np.array(indices)
        coords = [np.where(circuit.extracellularIndexGrid==ind) for ind in indices]
        coords = np.array([(coords[i][0].item(),coords[i][1].item()) for i in range(len(coords))])  # convert it into a 2d array
        verticalMirrorIndex = np.median(np.arange(numCols))  # reflects column indices
        verticalReflectionDists = np.abs(coords[:,1] - verticalMirrorIndex) * 2  # reflect y-coords off the vertical mirror axis
        coordsReflected = np.vstack((coords[:,0],(coords[:,1] + verticalReflectionDists))).astype(int).transpose()
        verticalReflectedIndices = circuit.extracellularIndexGrid[coordsReflected[:,0],coordsReflected[:,1]].astype(int)
        # horizontalMirrorIndex = np.median(np.arange(0,numLatticeIndices,numCols))  # reflects row indices
        # horizontalRemappedIndices = np.floor(indices/numCols).astype(int) * numCols
        # horizontalReflectionDists = np.abs(horizontalRemappedIndices - horizontalMirrorIndex) * 2
        # horizontalReflectedIndices = (indices + horizontalReflectionDists).astype(int)
        # if symmetry == 'fourfold':
        #     diagonalReflectedIndices = (indices + verticalReflectionDists + horizontalReflectionDists).astype(int)
        #     return ([verticalReflectedIndices,horizontalReflectedIndices,diagonalReflectedIndices])
        if symmetry == 'twofold':
            return (verticalReflectedIndices)

    # Compute the pairwise Euclidean distances between cellular and extracellular coordinates or between extracellular coordinates
    def computePairwiseDistances(self,coordinateSet1,coordinateSet2):
        xc1, yc1 = coordinateSet1
        xc2, yc2 = coordinateSet2
        pairWiseDistances = torch.sqrt((xc2[:,:,np.newaxis] - xc1[:,np.newaxis,:])**2 +(yc2[:,:,np.newaxis] - yc1[:,np.newaxis,:])**2)
        # NOTE: np.subtract.outer doesn't allow for broadcasting
        # pairWiseDistances = torch.sqrt(torch.FloatTensor((np.subtract.outer(xc2, xc1) ** 2 + np.subtract.outer(yc2, yc1) ** 2)))
        return pairWiseDistances

    # Compute a binary matrix with 1s marking the extracellular grid points that are within a given distance from a cell
    def defineFieldCellNeighborhoodMap(self,distanceMatrix,distanceThreshold):
        fieldNeighborhoodBitmap = (distanceMatrix <= distanceThreshold) * 1.0  # shape = (numSamples,numExtracellularGridPoints,numCells)
        # fieldNeighborhoodBitmap = 1 / (1 + torch.exp(100*(distanceMatrix - (distanceThreshold+0.01))))  # differentiable version of '<='
        return fieldNeighborhoodBitmap











