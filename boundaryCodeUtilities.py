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
import contextlib
import os

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


def interiorStripeParts():
    """The interior stripes (PolyPatterning_Design.md, pattern 6, the "French flag"): the 9 x 9 interior cut into three
    stripes of 3 columns x 9 rows, left flank, centre and right flank. Only the centre stripe is hyperpolarised in the
    target. As for the face, the ring carries the code only and is scored at the background value, so no ring cell is part
    of any stripe."""
    return [rowColumnBlock((1 / 11, 10 / 11), (firstColumn / 11, (firstColumn + 3) / 11)) for firstColumn in (1, 4, 7)]


stripeParts = interiorStripeParts()
leftFlankCellIndices, centreStripeCellIndices, rightFlankCellIndices = (np.array(part) for part in stripeParts)
flankCellIndices = np.concatenate([leftFlankCellIndices, rightFlankCellIndices])


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


def structuralIntersectionOverUnion(vmem, targetCellIndices=None):
    """IoU of the hyperpolarised interior cells with the target feature cells (the face's, unless `targetCellIndices` names
    others, e.g. centreStripeCellIndices). Asks which cells are dark, never how dark, and ignores the outline ring that
    essentially no seed produces."""
    darkInterior = (vmem < hyperpolarizedThresholdMilliVolts)[interiorCellIndices]
    targetInterior = np.isin(interiorCellIndices, featureCellIndices if targetCellIndices is None else targetCellIndices)
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


@contextlib.contextmanager
def fullDoublePrecision(enabled=None):
    """Build and run the model with torch's default dtype set to float64, so that every tensor the model creates is 64-bit.

    The state (Vmem, G_pol, eV, ...) is float64 already; what the default dtype leaves in float32 are the model's geometry
    constants: the cell coordinates, and so the distances and 1 / distance field kernel built from them (cellularFieldNetwork.py).
    `enabled` None reads the environment variable ELECTRICMORPHOGENESIS_FLOAT64 (1 or true), so any script that replays through
    this module can be run in 64-bit without editing it; False or True overrides the variable. The default dtype is restored on exit."""
    if enabled is None:
        enabled = os.environ.get('ELECTRICMORPHOGENESIS_FLOAT64', '').lower() in ('1', 'true')
    previous = torch.get_default_dtype()
    if enabled:
        torch.set_default_dtype(torch.float64)
    try:
        yield enabled
    finally:
        torch.set_default_dtype(previous)


def replay(parameters, clampParameters, onIteration, passCircuit=False, doublePrecision=None):
    """Forward-simulate a checkpoint's model with the clamp held while iteration <= clampEndIter, then
    released. Same call sequence as compareFacialFeatureScore11x11.py, whose replays reproduce each
    checkpoint's stored training loss (Sim.md 12.7). onIteration(iteration, vmemMilliVolts) is called
    after every iteration, or onIteration(iteration, vmemMilliVolts, circuit) when passCircuit is set."""
    torch.set_grad_enabled(False)
    parameters = dict(parameters)
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['ATPParameters'] = None
    numSamples = parameters['simParameters']['numSamples']
    with fullDoublePrecision(doublePrecision):
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


def ringHoldBatchReplay(referenceCheckpoint, ringValues, holdIterations, numIterations, onIteration, passConductance=False, doublePrecision=None):
    """Replay the reference checkpoint's model for several ring codes at once, one sample per code. Row k of
    `ringValues` (numCodes x 40, G_pol / G_ref) is held on sample k's ring for `holdIterations`, then released.
    Samples do not interact, so each follows the same trajectory as its own single-sample replay.
    onIteration(iteration, vmemMilliVolts) receives a numCodes x numCells torch tensor after every iteration; with passConductance it is called
    onIteration(iteration, vmemMilliVolts, conductance), conductance being each cell's G_pol / G_ref (numCodes x numCells).
    doublePrecision True builds the model with a float64 default dtype (see fullDoublePrecision); None follows ELECTRICMORPHOGENESIS_FLOAT64."""
    torch.set_grad_enabled(False)
    ringValues = np.asarray(ringValues, dtype=np.float64)
    numCodes = len(ringValues)
    if ringValues.min() < 0 or ringValues.max() > 2:
        raise ValueError(f"held values leave the physical range [0, 2]: {ringValues.min():.3f} to {ringValues.max():.3f}")
    parameters = dict(referenceCheckpoint)
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['ATPParameters'] = None
    initial = referenceCheckpoint['simParameters']['initialValues']
    batchInitial = {name: initial[name].repeat(numCodes, 1, 1) for name in ('Vmem', 'eV', 'ligandConc')}
    batchInitial['G_pol'] = dict(cells=[initial['G_pol']['cells'][0]] * numCodes, values=[initial['G_pol']['values'][0]] * numCodes)
    batchInitial['G_dep'] = initial['G_dep']
    clamp = dict(referenceCheckpoint['clampParameters'])
    clamp['clampIndices'] = (np.repeat(np.arange(numCodes), len(boundaryRingCells)), np.tile(boundaryRingCells, numCodes))
    clamp['clampValues'] = torch.tensor(np.tile(ringValues.reshape(1, -1), (holdIterations, 1)), dtype=torch.double)
    clamp['clampStartIter'], clamp['clampEndIter'] = 0, holdIterations - 1
    with fullDoublePrecision(doublePrecision):
        system = model(parameters, numCodes)
        system.setExperimentalConditions((batchInitial, numCodes))
        circuit = system.electricNetwork
        for iteration in range(numIterations):
            system.simulate(clampParameters=clamp if iteration < holdIterations else None, numSimIters=1, outerIter=iteration, fieldModulation=False)
            if passConductance:
                onIteration(iteration, circuit.Vmem[:, :, 0] * 1000.0, circuit.G_pol[:, :, 0] / circuit.G_ref)
            else:
                onIteration(iteration, circuit.Vmem[:, :, 0] * 1000.0)


def scoreRingCodesOverTime(referenceCheckpoint, ringValues, holdIterations, numIterations, targetMilliVolts, targetCellIndices, batchSize=64, onBatch=None, regions=None):
    """Replay many ring codes (rows of `ringValues`, G_pol / G_ref on the 40 ring cells, in [0, 2]) and read each against a target at every
    iteration from the release on: the balanced RMS over the target's feature cells (`targetCellIndices`, at `targetMilliVolts` per cell) and
    the other cells, and the structural overlap of the dark interior cells with the feature cells. Returns, per code: the best score and the
    iteration, overlap and stray dark interior cells and feature cells dark at it, the highest overlap at any iteration and where, and the
    longest unbroken run of iterations with overlap at or above 0.85. `regions` ({name: cell indices}, optional) adds, per region, the number of its
    cells dark at the best moment (`<name>DarkAtBest`) and the most dark at any iteration (`<name>MaxDark`). onBatch(numDone, results) is called after each batch."""
    reference = loadCheckpoint(referenceCheckpoint) if isinstance(referenceCheckpoint, int) else referenceCheckpoint
    target = torch.tensor(np.asarray(targetMilliVolts, dtype=np.float64))
    featureMask = torch.zeros(numCells, dtype=torch.bool)
    featureMask[torch.as_tensor(np.asarray(targetCellIndices))] = True
    interiorMask = torch.zeros(numCells, dtype=torch.bool)
    interiorMask[torch.as_tensor(np.asarray(interiorCellIndices))] = True
    numFeature = int(featureMask.sum())
    results = {key: [] for key in ('score', 'bestIteration', 'overlapAtBest', 'strayAtBest', 'featureDarkAtBest', 'maxOverlap', 'maxOverlapIteration', 'longestRunAbove0p85')}
    regions = regions or {}
    regionMasks = {}
    for name, cells in regions.items():
        regionMasks[name] = torch.zeros(numCells, dtype=torch.bool)
        regionMasks[name][torch.as_tensor(np.asarray(cells))] = True
        results[f'{name}DarkAtBest'], results[f'{name}MaxDark'] = [], []
    ringValues = np.asarray(ringValues, dtype=np.float64)
    for begin in range(0, len(ringValues), batchSize):
        batch = ringValues[begin:begin + batchSize]
        count = len(batch)
        best = dict(score=torch.full((count,), np.inf, dtype=torch.double), iteration=torch.zeros(count, dtype=torch.long),
                    overlap=torch.zeros(count, dtype=torch.double), stray=torch.zeros(count, dtype=torch.long), featureDark=torch.zeros(count, dtype=torch.long))
        maxOverlap = torch.zeros(count, dtype=torch.double)
        maxOverlapIteration = torch.zeros(count, dtype=torch.long)
        run, longest = torch.zeros(count, dtype=torch.long), torch.zeros(count, dtype=torch.long)
        regionBest = {name: torch.zeros(count, dtype=torch.long) for name in regionMasks}
        regionMax = {name: torch.zeros(count, dtype=torch.long) for name in regionMasks}

        def onIteration(iteration, vmem):
            if iteration < holdIterations:
                return
            dark = (vmem < hyperpolarizedThresholdMilliVolts) & interiorMask[None]
            featureDark, stray = (dark & featureMask[None]).sum(1), (dark & ~featureMask[None]).sum(1)
            overlap = featureDark.double() / (numFeature + stray).double()               # intersection over union: the union is the target plus the strays
            squared = (vmem - target) ** 2
            scores = squared[:, featureMask].mean(1).sqrt() * 0.5 + squared[:, ~featureMask].mean(1).sqrt() * 0.5
            better = scores < best['score']
            for key, value in (('score', scores), ('iteration', torch.full((count,), iteration)), ('overlap', overlap), ('stray', stray), ('featureDark', featureDark)):
                best[key] = torch.where(better, value, best[key])
            higher = overlap > maxOverlap
            maxOverlap.copy_(torch.where(higher, overlap, maxOverlap))
            maxOverlapIteration.copy_(torch.where(higher, torch.full((count,), iteration), maxOverlapIteration))
            run.copy_(torch.where(overlap >= 0.85, run + 1, torch.zeros_like(run)))
            longest.copy_(torch.maximum(longest, run))
            for name, mask in regionMasks.items():
                darkInRegion = (dark & mask[None]).sum(1)
                regionBest[name] = torch.where(better, darkInRegion, regionBest[name])
                regionMax[name] = torch.maximum(regionMax[name], darkInRegion)

        ringHoldBatchReplay(reference, batch, holdIterations, numIterations, onIteration)
        for name in regionMasks:
            results[f'{name}DarkAtBest'].extend(regionBest[name].tolist())
            results[f'{name}MaxDark'].extend(regionMax[name].tolist())
        for key, value in (('score', best['score']), ('bestIteration', best['iteration']), ('overlapAtBest', best['overlap']), ('strayAtBest', best['stray']),
                           ('featureDarkAtBest', best['featureDark']), ('maxOverlap', maxOverlap), ('maxOverlapIteration', maxOverlapIteration),
                           ('longestRunAbove0p85', longest)):
            results[key].extend(value.tolist())
        if onBatch:
            onBatch(begin + count, results)
    return results


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

def crossValidatedReadout(predictors, responses, seed=1, numFolds=5):
    """The best linear readout of `responses` from `predictors`, scored on held-out rows.

    Rank-one reduced-rank regression: on each training split the responses are regressed on the predictors (a
    four-column fit, so nothing ill-conditioned is inverted), the leading direction of the fitted values is taken as
    the readout's direction in response space, and that fixed direction is scored on the rows left out.

    This replaces scikit-learn's CCA, whose NIPALS fit is unstable when the responses are many and nearly collinear,
    as standardised mode amplitudes are: on this ensemble it moved between 0.39 and 0.64 with the fold seed alone, and
    rose when noise was added. The estimator here holds to within 0.02 across fold seeds, added noise and rounding.

    Returns the mean held-out correlation, the per-fold values, and the response direction fitted on everything.
    """
    responses = np.asarray(responses)
    responses = responses[:, responses.std(0) > 1e-9]
    responses = (responses - responses.mean(0)) / responses.std(0)
    predictors = np.asarray(predictors)
    assignment = np.random.default_rng(seed).permutation(len(predictors)) % numFolds

    def leadingDirection(rows):
        design = np.column_stack([np.ones(rows.sum()), predictors[rows]])
        fitted = design @ np.linalg.lstsq(design, responses[rows], rcond=None)[0]
        return np.linalg.svd(fitted - fitted.mean(0), full_matrices=False)[2][0]

    scores = []
    for fold in range(numFolds):
        train, test = assignment != fold, assignment == fold
        direction = leadingDirection(train)
        design = np.column_stack([np.ones(train.sum()), predictors[train]])
        weights = np.linalg.lstsq(design, responses[train] @ direction, rcond=None)[0]
        predicted = np.column_stack([np.ones(test.sum()), predictors[test]]) @ weights
        scores.append(abs(float(np.corrcoef(predicted, responses[test] @ direction)[0, 1])))
    everything = np.ones(len(predictors), dtype=bool)
    return float(np.mean(scores)), [round(value, 4) for value in scores], leadingDirection(everything)
