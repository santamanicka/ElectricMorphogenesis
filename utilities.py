import torch
import numpy as np

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

    def computeElectrodomeIndices(self,LatticeDims,resolution=1):
        numrows, numcols = LatticeDims[0], LatticeDims[1]
        numrows_res, numcols_res = (numrows * resolution) + 1, (numcols * resolution) + 1
        resolution_per_row_field = np.pad(np.concatenate([[numcols_res],np.repeat(numcols+1,resolution-1)]),
                                    (0,numrows_res-resolution),'wrap')
        resolution_per_row_dome = np.concatenate((np.array([numcols_res]),np.repeat(2,numrows_res-2),np.array([numcols_res])))
        resolution_per_row_field_cum = np.cumsum(np.concatenate(([0],resolution_per_row_field[:-1])))
        xindices_with_offset = torch.concatenate(list(map(lambda x: torch.repeat_interleave(torch.tensor(x[0]),x[1]),
                                             zip(resolution_per_row_field_cum,resolution_per_row_dome))))
        yindices = torch.concatenate(list(map(lambda x: torch.linspace(0,x[0]-1,x[1],dtype=torch.int8),
                                             zip(resolution_per_row_field,resolution_per_row_dome))))
        indices = xindices_with_offset + yindices
        indices = indices.tolist()
        return (indices)

    # Compute the pairwise Euclidean distances between cellular and extracellular coordinates
    def computeElectricFieldPairwiseDistances(self,cellularCoordinates,extracellularCoordinates):
        xc,yc = cellularCoordinates
        xec,yec = extracellularCoordinates
        # (m,n) matrix where m is the number of field coordinates and n the number of cell coordinates
        fieldDistances = torch.sqrt(torch.FloatTensor((np.subtract.outer(xec, xc) ** 2 + np.subtract.outer(yec, yc) ** 2)))
        return fieldDistances

    # Compute a binary matrix with 1s marking the extracellular grid points that are within a given distance from a cell
    def defineFieldNeighborhoodMap(self,fieldDistanceMatrix,distanceThreshold):
        fieldNeighborhoodBitmap = (fieldDistanceMatrix <= distanceThreshold) * 1.0  # shape = (numSamples,numExtracellularGridPoints,numCells)
        return fieldNeighborhoodBitmap











