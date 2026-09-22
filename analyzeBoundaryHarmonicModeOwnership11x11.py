"""When does a spatial mode of the bulk stop belonging to an order of the ring code? (PolyPatterning_Sim.md, Section 12).

The predictions and the ownership test are fixed beforehand in --predictionsPath. An ensemble of codes spanning the
feasible region is replayed as in training; at each probe iteration every pattern is decomposed by a spatial Fourier
(2D cosine) transform, which assumes nothing about the tissue's mechanics, and each mode's amplitude is fitted from
the code's coefficients by a quadratic model. A mode belongs to an order when that fit explains at least half of the
mode's variance and the order holds at least half of the first-order variance.

--condition picks the run: baseline, heldThroughout (the ring is never released), fieldOff (the extracellular field's
feedback is disabled) or orders0to6 (a seven-coefficient code).

Writes data/boundaryHarmonicModeOwnership<checkpoint>Hold<hold><target><Condition>.json (never overwriting).
"""
import argparse
import copy
import itertools
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--trainedRunPath', type=str, default='data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5/order3_restart08.npz')
parser.add_argument('--predictionsPath', type=str, default='data/boundaryHarmonicModeOwnershipPredictions1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--condition', type=str, default='baseline', choices=('baseline', 'heldThroughout', 'fieldOff', 'orders0to6'))
parser.add_argument('--numCodes', type=int, default=1024)
parser.add_argument('--sampling', type=str, default='random', choices=('random', 'slice'))
parser.add_argument('--gridSize', type=int, default=32)
parser.add_argument('--sliceHalfWidth', type=float, default=0.6)
parser.add_argument('--interiorOnly', action='store_true', help='decompose only the 9x9 interior, so the clamped ring cannot make ownership trivial')
parser.add_argument('--storeAmplitudesAt', type=str, default='', help='comma-separated probe iterations whose raw mode amplitudes are written out')
parser.add_argument('--outputSuffix', type=str, default='')
parser.add_argument('--seed', type=int, default=17)
args = parser.parse_args()

predictions = json.load(open(args.predictionsPath))
run = dict(np.load(args.trainedRunPath))
outputPath = (f"data/boundaryHarmonicModeOwnership{int(run['referenceCheckpoint'])}Hold{int(run['holdIterations'])}"
              f"{run['targetName']}{args.condition[0].upper()}{args.condition[1:]}{args.outputSuffix}.json")
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
reference = boundary.loadCheckpoint(int(run['referenceCheckpoint']))
if args.condition == 'fieldOff':
    reference = copy.deepcopy(reference)
    reference['fieldParameters'] = dict(reference['fieldParameters'])
    reference['fieldParameters']['fieldEnabled'] = False
hold, numIterations = int(run['holdIterations']), int(run['numIterations'])
if args.condition == 'heldThroughout':
    hold = numIterations
trainedCode, trainedMoment = run['bestCoefficients'], int(run['bestIteration'])
numOrders = 7 if args.condition == 'orders0to6' else len(trainedCode)
target = np.asarray(run['target']).ravel()
angles = boundary.ringAngles(boundary.boundaryRingCells)
basis = np.cos(np.outer(angles, np.arange(numOrders)))
probeIterations = [1, 2, 4, 8, 16, 32, 64, 128, 200, 300, 301, 302, 305, 310, 320, 350, 400, 500, 600, 800,
                   1000, 1300, 1600, 1853, 2000, trainedMoment, 2500, 2999]

# ------------------------------------------------------------------------------- spatial Fourier basis
size = boundary.latticeRows - 2 if args.interiorOnly else boundary.latticeRows
decomposedCells = boundary.interiorCellIndices if args.interiorOnly else np.arange(boundary.numCells)
modePairs = [(rowFrequency, columnFrequency) for rowFrequency in range(size) for columnFrequency in range(size)]


def spatialMode(rowFrequency, columnFrequency):
    rows = np.cos(np.pi * rowFrequency * (np.arange(size) + 0.5) / size)
    columns = np.cos(np.pi * columnFrequency * (np.arange(size) + 0.5) / size)
    vector = np.outer(rows, columns).ravel()
    return vector / np.linalg.norm(vector)


modes = np.column_stack([spatialMode(*pair) for pair in modePairs])


def decompose(pattern):
    values = np.asarray(pattern)[decomposedCells]
    return modes.T @ (values - values.mean())
targetSpectrum = decompose(target) ** 2
targetSpectrum = targetSpectrum / targetSpectrum.sum()

# -------------------------------------------------------------------------------------------- ensemble
generator = np.random.default_rng(args.seed)
if args.sampling == 'slice':
    width, centre = args.sliceHalfWidth, (trainedCode[1], trainedCode[2])
    firstAxis = np.linspace(centre[0] - width, centre[0] + width, args.gridSize)
    secondAxis = np.linspace(centre[1] - width, centre[1] + width, args.gridSize)
    grid = np.array([[trainedCode[0], first, second, trainedCode[3]] for first in firstAxis for second in secondAxis])
    codes = np.array([row for row in grid if 0.02 <= (basis @ row).min() and (basis @ row).max() <= 1.98])
else:
    codes = []
    while len(codes) < args.numCodes:
        candidate = np.concatenate([generator.uniform(0.3, 1.3, 1), generator.uniform(-0.6, 0.6, numOrders - 1)])
        ringValues = basis @ candidate
        if ringValues.min() >= 0.02 and ringValues.max() <= 1.98:
            codes.append(candidate)
    codes = np.array(codes)
correlations = np.corrcoef(codes[:, codes.std(0) > 1e-9].T)
print(f"{args.condition}: {len(codes)} codes, {numOrders} orders; largest coefficient correlation "
      f"{np.abs(correlations - np.eye(numOrders)).max():.3f}", flush=True)

patterns = {}


def onIteration(iteration, vmem):
    if iteration in probeIterations:
        patterns[iteration] = vmem.numpy().copy()


boundary.ringHoldBatchReplay(reference, codes @ basis.T, hold, numIterations, onIteration)

# ---------------------------------------------------------------------------------- ownership by order
varying = codes.std(0) > 1e-9
standardized = np.zeros_like(codes)
standardized[:, varying] = (codes[:, varying] - codes[:, varying].mean(0)) / codes[:, varying].std(0)
pairsOfOrders = list(itertools.combinations(range(numOrders), 2))
columns = [np.ones(len(codes))]
columns += [standardized[:, order] for order in range(numOrders)]
columns += [standardized[:, order] ** 2 for order in range(numOrders)]
columns += [standardized[:, first] * standardized[:, second] for first, second in pairsOfOrders]
design = np.column_stack(columns)
linearSlice = slice(1, 1 + numOrders)
quadraticSlice = slice(1 + numOrders, 1 + 2 * numOrders)

storeAmplitudesAt = [int(value) for value in args.storeAmplitudesAt.split(',') if value]
result = dict(predictions=predictions, condition=args.condition, sampling=args.sampling, sliceHalfWidth=args.sliceHalfWidth, interiorOnly=bool(args.interiorOnly), codes=np.round(codes, 5).tolist(), amplitudes={}, numCodes=len(codes), numOrders=numOrders,
              hold=hold, trainedMoment=trainedMoment, modePairs=modePairs, probeIterations=probeIterations,
              targetSpectrum=np.round(targetSpectrum, 6).tolist(), moments={})
for iteration in probeIterations:
    amplitudes = np.array([decompose(pattern) for pattern in patterns[iteration]])
    variance = amplitudes.var(0)
    varianceShare = variance / (variance.sum() + 1e-12)
    rSquared, orderShares, interactionShares, owned, topOrders = [], [], [], [], []
    for index in range(len(modePairs)):
        response = amplitudes[:, index]
        if response.std() < 1e-12:
            rSquared.append(0.0), orderShares.append([0.0] * numOrders), interactionShares.append(0.0)
            owned.append(False), topOrders.append(-1)
            continue
        fit, *_ = np.linalg.lstsq(design, response, rcond=None)
        predicted = design @ fit
        rSquared.append(float(1 - (response - predicted).var() / response.var()))
        univariate = [float((fit[1 + order] * standardized[:, order] + fit[1 + numOrders + order] * standardized[:, order] ** 2).var())
                      for order in range(numOrders)]
        interaction = float((design[:, 1 + 2 * numOrders:] @ fit[1 + 2 * numOrders:]).var())
        shares = np.array(univariate) / (sum(univariate) + 1e-12)
        orderShares.append([float(value) for value in shares])
        interactionShares.append(interaction / (sum(univariate) + interaction + 1e-12))
        topOrders.append(int(np.argmax(shares)))
        owned.append(bool(rSquared[-1] >= 0.5 and shares.max() >= 0.5))
    owned, topOrders = np.array(owned), np.array(topOrders)
    result['moments'][str(iteration)] = dict(
        varianceShare=np.round(varianceShare, 6).tolist(), rSquared=np.round(rSquared, 4).tolist(),
        orderShares=np.round(orderShares, 4).tolist(), interactionShare=np.round(interactionShares, 4).tolist(),
        topOrder=topOrders.tolist(), owned=owned.tolist(),
        ownedCount=int(owned.sum()), ownedVarianceShare=round(float(varianceShare[owned].sum()), 4),
        ownedTargetShare=round(float(targetSpectrum[owned].sum()), 4),
        ownersPresent=sorted(set(int(order) for order in topOrders[owned])),
        participationRatio=round(boundary.participationRatio(varianceShare), 2),
        patternVariance=round(float(variance.sum()), 3))
    if iteration in storeAmplitudesAt:
        result['amplitudes'][str(iteration)] = np.round(amplitudes, 3).tolist()
    summary = result['moments'][str(iteration)]
    print(f"  iteration {iteration:>4}: owned {summary['ownedCount']:>3} modes, {summary['ownedVarianceShare']:.3f} of pattern variance, "
          f"{summary['ownedTargetShare']:.3f} of the target, owners {summary['ownersPresent']}, "
          f"participation {summary['participationRatio']}", flush=True)

json.dump(result, open(outputPath, 'w'))
print('wrote', outputPath, flush=True)
