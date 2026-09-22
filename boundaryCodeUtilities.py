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
# Single-cell fixed points of the ion-channel model, in G_pol / G_ref (G_ref = 1 nS): below 0.802 one depolarised
# state (about -7.1 mV); from 0.802 to 1.439 bistable, stable at about -50.4 and -10.0 mV with a saddle at about
# -29.3 mV (V_th); above 1.439 one hyperpolarised state (about -52 to -53 mV).
singleCellBistableRange = (0.802, 1.439)
singleCellSaddleMilliVolts = -29.3
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


def replay(parameters, clampParameters, onIteration, passCircuit=False):
    """Forward-simulate a checkpoint's model with the clamp held while iteration <= clampEndIter, then
    released. Same call sequence as compareFacialFeatureScore11x11.py, whose replays reproduce each
    checkpoint's stored training loss (Sim.md 12.7). onIteration(iteration, vmemMilliVolts) is called
    after every iteration, or onIteration(iteration, vmemMilliVolts, circuit) when passCircuit is set."""
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
        if passCircuit:
            onIteration(iteration, circuit.Vmem[0, :, 0].detach().numpy() * 1000.0, circuit)
        else:
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


ringCodeReadoutKeys = ('endOfHoldVmem', 'endOfHoldGpol', 'windowMeanVmem', 'windowMeanGpol', 'windowStdVmem')


def ringCodeReadouts(parameters, referenceCheckpoint, ringValues, holdIterations, windowIterations=1000):
    """Replay `parameters` with the ring held at `ringValues` (G_pol / G_ref, which must lie in the physical range
    [0, 2]) for `holdIterations`. Returns Vmem and G_pol / G_ref at the last held iteration, their per-cell means over
    the last `windowIterations` iterations of the run, and the per-cell Vmem standard deviation over that window."""
    if ringValues.min() < 0 or ringValues.max() > 2:
        raise ValueError(f"held values leave the physical range [0, 2]: {ringValues.min():.3f} to {ringValues.max():.3f}")
    numIterations = parameters['simParameters']['numSimIters']
    readout = dict(windowMeanVmem=np.zeros(numCells), windowMeanGpol=np.zeros(numCells), windowSquaredVmem=np.zeros(numCells))

    def onIteration(iteration, vmem, circuit):
        conductance = circuit.G_pol[0, :, 0].detach().numpy() / circuit.G_ref
        if iteration == holdIterations - 1:
            readout['endOfHoldVmem'], readout['endOfHoldGpol'] = vmem.copy(), conductance.copy()
        if iteration >= numIterations - windowIterations:
            readout['windowMeanVmem'] += vmem
            readout['windowSquaredVmem'] += vmem ** 2
            readout['windowMeanGpol'] += conductance
    replay(parameters, ringClamp(referenceCheckpoint, ringValues, holdIterations), onIteration, passCircuit=True)
    readout['windowMeanVmem'] /= windowIterations
    readout['windowMeanGpol'] /= windowIterations
    readout['windowStdVmem'] = np.sqrt(np.maximum(readout.pop('windowSquaredVmem') / windowIterations - readout['windowMeanVmem'] ** 2, 0))
    return readout


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


# ------------------------------------------------------------------------------------ snapshots
def squareImages(square):
    """The 8 images of (a stack of) square lattices under the symmetries of the square."""
    turned = [np.rot90(square, turns, axes=(-2, -1)) for turns in range(4)]
    return turned + [each[..., ::-1] for each in turned]


class SnapshotLibrary:
    """Vmem snapshots (mV, one row per snapshot) searched for the one nearest a query, by RMS distance over all cells.
    `owner` labels each snapshot's run; `centre` is subtracted from everything to keep float32 distances accurate."""

    def __init__(self, snapshots, owner, centre):
        self.centre = np.float32(centre)
        self.snapshots = np.ascontiguousarray(snapshots - self.centre, dtype=np.float32)
        self.norms = (self.snapshots.astype(np.float64) ** 2).sum(1).astype(np.float32)
        self.owner = np.asarray(owner)

    def nearest(self, queries, skipOwner=None):
        """RMS distance from each query (in whichever of its 8 images lies nearest) to its nearest library snapshot,
        and that snapshot's index; snapshots of run `skipOwner` are left out."""
        best, where = np.full(len(queries), np.inf), np.zeros(len(queries), dtype=int)
        square = np.asarray(queries).reshape(-1, latticeRows, latticeCols)
        imageSet = [each.reshape(len(square), -1) for each in squareImages(square)]
        asymmetric = np.maximum(np.abs(square - np.rot90(square, 1, axes=(1, 2))).max((1, 2)), np.abs(square - square[:, :, ::-1]).max((1, 2))) > 0.01
        chunk = max(64, int(1.5e8 // len(self.owner)))
        for k, image in enumerate(imageSet):
            members = np.arange(len(square)) if k == 0 else np.where(asymmetric)[0]
            block = np.ascontiguousarray(image[members] - self.centre, dtype=np.float32)
            for start in range(0, len(block), chunk):
                part = block[start:start + chunk]
                squared = (part.astype(np.float64) ** 2).sum(1).astype(np.float32)[:, None] + self.norms[None] - 2 * part @ self.snapshots.T
                if skipOwner is not None:
                    squared[:, self.owner == skipOwner] = np.inf
                index = squared.argmin(1)
                exact = np.sqrt(((part.astype(np.float64) - self.snapshots[index]) ** 2).mean(1))
                rows = members[start:start + chunk]
                better = exact < best[rows]
                best[rows[better]], where[rows[better]] = exact[better], index[better]
        return best, where

    def snapshot(self, index):
        return self.snapshots[index] + self.centre


def interiorMotifs(course, canonical=False):
    """The motif (interior cells below the single-cell saddle) of each snapshot as bytes; with `canonical`, the smallest
    over its 8 images, so a motif and its rotations and reflections share one key."""
    dark = (np.asarray(course).reshape(-1, latticeRows, latticeCols) < singleCellSaddleMilliVolts)[:, 1:-1, 1:-1]
    keys = [np.packbits(image.reshape(len(dark), -1), axis=1) for image in (squareImages(dark) if canonical else [dark])]
    return [min(row[k].tobytes() for row in keys) for k in range(len(dark))]
