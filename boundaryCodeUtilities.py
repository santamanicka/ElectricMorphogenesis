"""Shared helpers for the 11x11 band-hold boundary-code analyses (PolyPatterning_Sim.md, Section 12).

Kept apart from utilities.py, whose helpers act on a live circuit: everything here works on stored
checkpoints, lattice cell indices and Vmem arrays in millivolts. Lattice cells are indexed row-major,
index = row * latticeCols + col.

Clamp values are G_pol / G_ref, written directly during the hold by embryo.py (G_pol = value * G_ref). Training
allowed values in [-1, 1] (learnCellularFieldNetwork.py --clampAmplitudeRange), but the conductance equation keeps
G_pol within [0, 2 G_ref] (cellularFieldNetwork.py). Each held iteration runs the normal, clipped update and then
the clamp write followed by an extra current and Vmem update, so negative values act as negative conductances
throughout the hold and are clipped to 0 at release. Values above 1 are unreachable by training.
"""
import numpy as np
import torch
from scipy.ndimage import label as labelConnectedComponents

from embryo import model

latticeRows = latticeCols = 11
numCells = latticeRows * latticeCols
latticeCentre = (latticeCols - 1) / 2
hyperpolarizedThresholdMilliVolts = -34.6   # midway between the target's -60 mV features and -9.2 mV background
bandHoldFileNumbers = list(range(1600, 1984))
originalCohortLossMethods = ('correlation', 'globalsum')                        # Sim.md 12.4
newCohortLossMethods = ('facialFeatureOnly', 'facialFeatureBalanced')          # Sim.md 12.10, 12.11
lossMethods = originalCohortLossMethods + newCohortLossMethods


# ----------------------------------------------------------------------------------------- geometry
def rowColumnBlock(rowFractions, columnFractions):
    firstRow, lastRow = (round(fraction * latticeRows) for fraction in rowFractions)
    firstCol, lastCol = (round(fraction * latticeCols) for fraction in columnFractions)
    return [row * latticeCols + col for row in range(firstRow, lastRow) for col in range(firstCol, lastCol)]


def faceFeatureParts():
    """Eyes, nose and mouth, matching learnCellularFieldNetwork.py's faceFeatureIndices at 11x11."""
    return [rowColumnBlock((2 / 11, 4 / 11), (2 / 11, 4 / 11)),   # left eye
            rowColumnBlock((2 / 11, 4 / 11), (7 / 11, 9 / 11)),   # right eye
            rowColumnBlock((4 / 11, 7 / 11), (5 / 11, 6 / 11)),   # nose
            rowColumnBlock((8 / 11, 9 / 11), (4 / 11, 7 / 11))]   # mouth


featureParts = faceFeatureParts()
featureCellIndices = np.array(sorted(cell for part in featureParts for cell in part))
otherCellIndices = np.array([cell for cell in range(numCells) if cell not in set(featureCellIndices.tolist())])
interiorCellIndices = np.array([row * latticeCols + col for row in range(1, latticeRows - 1)
                                for col in range(1, latticeCols - 1)])
nonFeatureInteriorCellIndices = np.array([cell for cell in interiorCellIndices
                                          if cell not in set(featureCellIndices.tolist())])


def shellCells(shell):
    """Cells of concentric square shell `shell` (0 = the 40-cell boundary ring, 5 = the centre cell),
    in clockwise order starting at the shell's top-left corner."""
    low, high = shell, latticeCols - 1 - shell
    if low == high:
        return np.array([low * latticeCols + low])
    top = [(low, col) for col in range(low, high + 1)]
    right = [(row, high) for row in range(low + 1, high + 1)]
    bottom = [(high, col) for col in range(high - 1, low - 1, -1)]
    left = [(row, low) for row in range(high - 1, low, -1)]
    return np.array([row * latticeCols + col for row, col in top + right + bottom + left])


boundaryRingCells = shellCells(0)


def ringAngles(cells):
    """Angle of each cell about the lattice centre, measured clockwise from straight up (toward row 0)."""
    cells = np.asarray(cells)
    return np.arctan2((cells % latticeCols) - latticeCentre, latticeCentre - (cells // latticeCols))


def leftHalfRepresentatives(cellIndices):
    """Cells on or left of the vertical mirror axis, row-major: the independent values of a two-fold code."""
    cellIndices = np.asarray(cellIndices)
    rows, cols = cellIndices // latticeCols, cellIndices % latticeCols
    keep = cols <= (latticeCols - 1) // 2
    order = np.lexsort((cols[keep], rows[keep]))
    return cellIndices[keep][order]


# --------------------------------------------------------------------------------------- face scores
def featureRootMeanSquareError(vmem, target):
    return float(np.sqrt(np.mean((vmem[featureCellIndices] - target[featureCellIndices]) ** 2)))


def balancedRootMeanSquareError(vmem, target):
    """Sim.md 12.11's facialFeatureBalanced formula, used here as a score rather than a training loss."""
    otherError = float(np.sqrt(np.mean((vmem[otherCellIndices] - target[otherCellIndices]) ** 2)))
    return 0.5 * featureRootMeanSquareError(vmem, target) + 0.5 * otherError


def interiorDarkComponents(vmem):
    """Connected components of the hyperpolarised region, with the boundary ring excluded."""
    dark = (vmem < hyperpolarizedThresholdMilliVolts).reshape(latticeRows, latticeCols).copy()
    dark[0, :] = dark[-1, :] = dark[:, 0] = dark[:, -1] = False
    labels, numComponents = labelConnectedComponents(dark)
    return labels.reshape(-1), int(numComponents)


def structuralIntersectionOverUnion(vmem):
    """IoU of the hyperpolarised interior cells with the target feature cells. Asks which cells are dark,
    never how dark, and ignores the outline ring that essentially no seed produces."""
    darkInterior = (vmem < hyperpolarizedThresholdMilliVolts)[interiorCellIndices]
    targetInterior = np.isin(interiorCellIndices, featureCellIndices)
    union = np.logical_or(darkInterior, targetInterior).sum()
    return float(np.logical_and(darkInterior, targetInterior).sum() / union) if union else 0.0


def partSeparationScore(vmem):
    """coverage + separation - 2 * spurious (max 8). Coverage sums each part's dark fraction (0-4);
    separation counts covered parts sitting in their own connected component (0-4); spurious is the
    dark fraction of interior cells outside every part."""
    dark = vmem < hyperpolarizedThresholdMilliVolts
    componentLabels, _ = interiorDarkComponents(vmem)
    coverage = sum(sum(1 for cell in part if dark[cell]) / len(part) for part in featureParts)
    partComponents = [set(componentLabels[cell] for cell in part if componentLabels[cell] > 0)
                      for part in featureParts]
    coveredParts = [index for index in range(len(featureParts)) if partComponents[index]]
    separation = 0
    for index in coveredParts:
        otherComponents = set().union(*[partComponents[other] for other in coveredParts if other != index]) \
            if len(coveredParts) > 1 else set()
        if not (partComponents[index] & otherComponents):
            separation += 1
    spurious = sum(1 for cell in nonFeatureInteriorCellIndices if dark[cell]) / len(nonFeatureInteriorCellIndices)
    return coverage + separation - 2 * spurious, coverage, separation, spurious


# ------------------------------------------------------------------------------- checkpoints and replay
def loadCheckpoint(fileNumber):
    return torch.load(f'data/bestModelParameters_fieldVector_11x11_{fileNumber}.dat',
                      weights_only=False, map_location='cpu')


def targetVmemMilliVolts(checkpoint):
    return checkpoint['trainParameters']['targetVmem'].numpy().reshape(-1) * 1000.0


def checkpointMetadata(fileNumber, checkpoint):
    clamp = checkpoint['clampParameters']
    return dict(fileNumber=fileNumber,
                mechanism='Gpol+Vmem' if 'Vmem' in clamp['clampMode'] else 'Gpol-only',
                depth=1 if len(clamp['clampIndices'][1]) == 40 else 2,
                holdIterations=int(clamp['clampEndIter']) + 1,
                lossMethod=checkpoint['trainParameters']['lossMethod'])


def replay(parameters, clampParameters, onIteration):
    """Forward-simulate a checkpoint's model with the clamp held while iteration <= clampEndIter, then
    released. Same call sequence as compareFacialFeatureScore11x11.py, whose replays reproduce each
    checkpoint's stored training loss (Sim.md 12.7). onIteration(iteration, vmemMilliVolts) is called
    after every iteration."""
    torch.set_grad_enabled(False)
    parameters = dict(parameters)
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['ATPParameters'] = None
    numSamples = parameters['simParameters']['numSamples']
    system = model(parameters, numSamples)
    system.setExperimentalConditions((parameters['simParameters']['initialValues'], numSamples))
    circuit = system.electricNetwork
    clampEndIteration = int(clampParameters['clampEndIter'])
    for iteration in range(parameters['simParameters']['numSimIters']):
        activeClamp = clampParameters if iteration <= clampEndIteration else None
        system.simulate(clampParameters=activeClamp, numSimIters=1, outerIter=iteration, fieldModulation=False)
        onIteration(iteration, circuit.Vmem[0, :, 0].detach().numpy() * 1000.0)


def lateWindowMean(parameters, clampParameters, windowIterations=1000):
    """Replay and return the per-cell mean Vmem over the last `windowIterations` iterations."""
    numIterations = parameters['simParameters']['numSimIters']
    total = np.zeros(numCells)

    def accumulate(iteration, vmem):
        if iteration >= numIterations - windowIterations:
            total[:] += vmem
    replay(parameters, clampParameters, accumulate)
    return total / windowIterations


def ringClamp(referenceCheckpoint, ringValues, holdIterations):
    """Clamp parameters holding the 40 boundary-ring cells at `ringValues` (G_pol/G_ref, clockwise from the
    top-left corner) for `holdIterations`, using the reference checkpoint's clamp mode."""
    clamp = dict(referenceCheckpoint['clampParameters'])
    clamp['clampIndices'] = (np.zeros(len(boundaryRingCells), dtype=int), boundaryRingCells.copy())
    clamp['clampValues'] = torch.tensor(np.tile(ringValues, (holdIterations, 1)), dtype=torch.double)
    clamp['clampStartIter'], clamp['clampEndIter'] = 0, holdIterations - 1
    return clamp


# ------------------------------------------------------------------------------------ codes
def loadBandHoldCodes(fileNumbers=bandHoldFileNumbers):
    """Per checkpoint: its metadata, its full-lattice G_pol code field (zero off the band), its folded
    left-half G_pol values and, for Gpol+Vmem codes, its folded left-half Vmem values."""
    records = {}
    for fileNumber in fileNumbers:
        checkpoint = loadCheckpoint(fileNumber)
        clamp = checkpoint['clampParameters']
        cellIndices = clamp['clampIndices'][1]
        position = {cell: index for index, cell in enumerate(cellIndices)}
        representatives = leftHalfRepresentatives(cellIndices)
        field = np.zeros(numCells)
        field[cellIndices] = clamp['clampValues'][0].numpy()
        folded = {'Gpol': clamp['clampValues'][0].numpy()[[position[cell] for cell in representatives]]}
        if 'clampValuesVmem' in clamp:
            folded['Vmem'] = clamp['clampValuesVmem'][0].numpy()[[position[cell] for cell in representatives]]
        records[fileNumber] = dict(metadata=checkpointMetadata(fileNumber, checkpoint), field=field,
                                   folded=folded, representatives=representatives)
    return records


# ------------------------------------------------------------------------------------ statistics
def rankWithinGroups(values, groupLabels):
    """Ranks computed separately inside each group, for configuration-controlled correlations."""
    from scipy.stats import rankdata
    values, groupLabels = np.asarray(values), np.asarray(groupLabels)
    ranks = np.zeros(len(values))
    for group in np.unique(groupLabels):
        members = groupLabels == group
        ranks[members] = rankdata(values[members])
    return ranks


def circularHarmonicCoefficients(values, maxOrder):
    """Complex coefficients of orders 0..maxOrder for values in cyclic order."""
    count = len(values)
    positions = np.arange(count)
    return np.array([(values * np.exp(-2j * np.pi * order * positions / count)).sum() / count
                     for order in range(maxOrder + 1)])


def symmetricShare(fieldValues):
    """Energy remaining after averaging a lattice map over the 8 symmetries of the square, as a share of
    the map's energy: the part a perfectly uniform boundary could ever produce."""
    grid = np.asarray(fieldValues).reshape(latticeRows, latticeCols)
    images = []
    for quarterTurns in range(4):
        rotated = np.rot90(grid, quarterTurns)
        images += [rotated, rotated[:, ::-1]]
    symmetricPart = np.mean(images, axis=0)
    return float((symmetricPart ** 2).sum() / ((grid ** 2).sum() + 1e-12))


def shellRootMeanSquare(fieldValues):
    return [float(np.sqrt(np.mean(np.asarray(fieldValues)[shellCells(shell)] ** 2))) for shell in range(6)]


def participationRatio(explainedVariance):
    explainedVariance = np.asarray(explainedVariance)
    return float(explainedVariance.sum() ** 2 / (explainedVariance ** 2).sum())
