import torch
import numpy as np
import math

class utilities():

    def __init__(self):
        pass

    # Computes a 2D lattice adjacency matrix
    # This procedure does not require cellular coordinates; it only needs the (row,column) indices
    def computeLatticeAdjacencyMatrix(self,LatticeDims):
        numrows, numcols = LatticeDims[0], LatticeDims[1]
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
    def computeCellularCoordinates(self,LatticeDims,cell_radius):
        numrows, numcols = LatticeDims[0], LatticeDims[1]
        offset = 2 * cell_radius  # diameter-length spaces between adjacent cell-centers
        xcoords = np.repeat(np.arange(numrows),numcols) * offset
        ycoords = np.tile(np.arange(numcols),numrows) * offset
        # Offset the entire coordinate to leave room for the field coordinates to begin at the corner of the square grid at (0,0)
        xcoords, ycoords = xcoords + cell_radius, ycoords + cell_radius
        return (xcoords,ycoords)

    # Computes the coordinate of the points on the circumscribed square around the (circular) cells;
    # field coordinates start from (0,0) and increase positively in both directions.
    # Here, resolution=1 covers only the corners of the squares; higher resolutions cover more points
    ## TODO: include option to make circular grid around each cell rather than square grid
    def computeExtracellularCoordinates(self,LatticeDims,cell_radius=1,resolution=1):
        numrows, numcols = LatticeDims[0], LatticeDims[1]
        numrows_res, numcols_res = (numrows * resolution) + 1, (numcols * resolution) + 1
        offset = 2 * cell_radius / resolution
        resolution_per_row = np.pad(np.concatenate([[numcols_res],np.repeat(numcols+1,resolution-1)]),
                                    (0,numrows_res-resolution),'wrap')
        xcoords = torch.concatenate(list(map(lambda x: torch.repeat_interleave(torch.tensor(x[0]),x[1])*offset,
                                             zip(torch.arange(numcols_res),resolution_per_row))))
        ycoords = torch.concatenate(list(map(lambda x: torch.linspace(0,numcols_res-1,x,dtype=torch.int8)*offset,
                                             resolution_per_row)))
        return (xcoords,ycoords)

    def computeDomeIndices(self,circuit,mode='field',region='full'):
        cellRadius = circuit.cell_radius
        if mode == 'field':
            coords = circuit.extracellularCoordinates
            numIndices = circuit.numExtracellularGridPoints
            dims = circuit.LatticeDims[0]+1, circuit.LatticeDims[1]+1
        elif mode == 'tissue':
            coords = circuit.cellularCoordinates
            numIndices = circuit.numCells
            dims = circuit.LatticeDims
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
            numBoundRows = dims[0]
            numBoundCols = math.ceil(dims[1]/2)
            xCoordMin, yCoordMin = coords[0].min(), coords[1].min()
            xCoordMax, yCoordMax = coords[0].max(), coords[1].max()
            boundaryCoords = (((coords[0] <= (cellRadius*(2*numBoundRows-1))) & (coords[1] == yCoordMin)) |  # left side
                              ((coords[1] <= (cellRadius*(2*numBoundCols-1))) & (coords[0] == xCoordMin)) |  # top side
                              ((coords[1] <= (cellRadius*(2*numBoundCols-1))) & (coords[0] == xCoordMax)))[0]  # bottom side
        boundaryIndices = np.arange(numIndices)[boundaryCoords]
        return boundaryIndices.tolist()

    def computeSymmetricalIndices(self,circuit,indices,mode='field',symmetry="fourfold"):
        if mode == 'field':
            dims = circuit.LatticeDims[0]+1, circuit.LatticeDims[1]+1
        elif mode == 'tissue':
            dims = circuit.LatticeDims
        numRows, numCols = dims
        numLatticeIndices = numRows * numCols
        indices = np.array(indices)
        verticalMirrorIndex = np.median(np.arange(numCols))  # reflects column indices
        verticalRemappedIndices = indices % numCols
        verticalReflectionDists = np.abs(verticalRemappedIndices - verticalMirrorIndex) * 2
        verticalReflectedIndices = (indices + verticalReflectionDists).astype(int)
        horizontalMirrorIndex = np.median(np.arange(0,numLatticeIndices,numCols))  # reflects row indices
        horizontalRemappedIndices = np.floor(indices/numCols).astype(int) * numCols
        horizontalReflectionDists = np.abs(horizontalRemappedIndices - horizontalMirrorIndex) * 2
        horizontalReflectedIndices = (indices + horizontalReflectionDists).astype(int)
        if symmetry == 'fourfold':
            diagonalReflectedIndices = (indices + verticalReflectionDists + horizontalReflectionDists).astype(int)
            return ([verticalReflectedIndices,horizontalReflectedIndices,diagonalReflectedIndices])
        elif symmetry == 'twofold':
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
        # fieldNeighborhoodBitmap = (distanceMatrix <= distanceThreshold) * 1.0  # shape = (numSamples,numExtracellularGridPoints,numCells)
        fieldNeighborhoodBitmap = 1 / (1 + torch.exp(10*(distanceMatrix - distanceThreshold)))  # differentiable version of '<='
        return fieldNeighborhoodBitmap











