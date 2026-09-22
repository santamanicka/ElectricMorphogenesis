"""Summarise the CMA-ES ring-code training runs of learnBoundaryHarmonics11x11.py (PolyPatterning_Sim.md, Section 12).

Per order N: every restart's best score and code, its start (library or random) and its score progression over the
generations. For the best code of each order, three checks by new simulation:
  long run      the code replayed for --longIterations, scored at every iteration after release: how often and for how
                long the tissue comes back near its best score, and where in the run the best moment falls;
  neighbours    --numNeighbours codes a small step away (Gaussian, --neighbourStep in scaled units, moved into the allowed
                set), scored as in training: how sharply the score depends on the code;
  face parts    feature RMS, other RMS, structural IoU and part score at the best moment (boundaryCodeUtilities).
Baselines, scored the same way: the tissue with no clamp, --numRandomCodes random allowed codes per order, and the best
stored sweep code among those rerun to seed training (the best --librarySeedCount by late-window mean).

Writes data/boundaryHarmonicTrainingSummary<checkpoint>Hold<hold><target>.json (never overwriting; --tag appends).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from scipy.optimize import linprog, minimize

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--trainingDirs', type=str, default='data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5,'
                    'data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5Population64', help='comma-separated; each folder is one round')
parser.add_argument('--longIterations', type=int, default=20000)
parser.add_argument('--numNeighbours', type=int, default=32)
parser.add_argument('--neighbourStep', type=float, default=0.01)
parser.add_argument('--numRandomCodes', type=int, default=64)
parser.add_argument('--nearMilliVolts', type=float, default=1.0, help='a moment counts as near the best within this margin')
parser.add_argument('--snapshotOffsets', type=str, default='-500,-100,-20,0,20,100,500')
parser.add_argument('--faceOverlap', type=float, default=0.9, help='structural IoU at or above which a moment counts as showing the face')
parser.add_argument('--tag', type=str, default='')
args = parser.parse_args()
args.snapshotOffsets = [int(offset) for offset in args.snapshotOffsets.split(',')]
startTime = time.time()


def log(message):
    print(f'[{time.time() - startTime:6.0f}s] {message}', flush=True)


runs = {}
trainingDirs = args.trainingDirs.split(',')
for round, directory in enumerate(trainingDirs):
    for path in sorted(glob.glob(f'{directory}/order*_restart*.npz')):
        run = dict(np.load(path))
        run['round'] = round
        runs.setdefault(int(run['maxOrder']), []).append(run)
orders = sorted(runs)
first = runs[orders[0]][0]
reference = boundary.loadCheckpoint(int(first['referenceCheckpoint']))
hold, numIterations, ceiling = int(first['holdIterations']), int(first['numIterations']), float(first['ceiling'])
target, targetName = first['target'], str(first['targetName'])
outputPath = f"data/boundaryHarmonicTrainingSummary{int(first['referenceCheckpoint'])}Hold{hold}{targetName}{args.tag}.json"
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
featureMask = np.isin(np.arange(boundary.numCells), boundary.featureCellIndices)
ringMask = np.isin(np.arange(boundary.numCells), boundary.boundaryRingCells)
scoreMasks = first['scoreGroupMasks'] if 'scoreGroupMasks' in first else np.array([featureMask, ~featureMask])
scoreGroups = str(first['scoreGroups']) if 'scoreGroups' in first else 'features,other'
scoreFrom = int(first['scoreFrom']) if 'scoreFrom' in first else hold
withOutline = 'outline' in scoreGroups.split(',')
targetTensor = torch.tensor(target, dtype=torch.double)
angles = boundary.ringAngles(boundary.boundaryRingCells)


def balancedScores(vmem):
    squared = (vmem - targetTensor) ** 2
    return sum(squared[:, mask].mean(1).sqrt() for mask in scoreMasks) / len(scoreMasks)


def ringDarkShare(vmem):
    return float((vmem[ringMask] < boundary.hyperpolarizedThresholdMilliVolts).mean())


def showsFace(overlap, ringShare):
    return (overlap >= args.faceOverlap) & ((ringShare >= args.faceOverlap) if withOutline else True)


def scoreCodes(ringValues, numRunIterations=numIterations, holdIterations=hold, keepTrace=False, captureIterations=()):
    """Best balanced RMS after release, its iteration and pattern, and optionally the whole score trace and the first
    code's snapshots at `captureIterations`."""
    numCodes = len(ringValues)
    best = dict(score=torch.full((numCodes,), np.inf, dtype=torch.double), iteration=torch.zeros(numCodes, dtype=torch.long),
                vmem=torch.zeros(numCodes, boundary.numCells, dtype=torch.double))
    trace = np.zeros((numCodes, numRunIterations - hold), dtype=np.float32) if keepTrace else None
    overlapTrace = np.zeros(numRunIterations - hold, dtype=np.float32) if keepTrace else None
    ringTrace = np.zeros(numRunIterations - hold, dtype=np.float32) if keepTrace else None
    captured = {}

    def onIteration(iteration, vmem):
        if iteration in captureIterations:
            captured[iteration] = vmem[0].numpy().copy()
        if iteration >= hold and keepTrace:
            trace[:, iteration - hold] = balancedScores(vmem).numpy()
            overlapTrace[iteration - hold] = boundary.structuralIntersectionOverUnion(vmem[0].numpy())
            ringTrace[iteration - hold] = ringDarkShare(vmem[0].numpy())
        if iteration >= scoreFrom:
            scores = balancedScores(vmem)
            better = scores < best['score']
            best['score'] = torch.where(better, scores, best['score'])
            best['iteration'][better] = iteration
            best['vmem'][better] = vmem[better]
    boundary.ringHoldBatchReplay(reference, np.asarray(ringValues), holdIterations, numRunIterations, onIteration)
    if captureIterations:
        return best['score'].numpy(), best['iteration'].numpy(), best['vmem'].numpy(), trace, captured, overlapTrace, ringTrace
    return best['score'].numpy(), best['iteration'].numpy(), best['vmem'].numpy(), trace


def facePartScores(vmem):
    return dict(featureRMS=boundary.featureRootMeanSquareError(vmem, target),
                otherRMS=float(np.sqrt(np.mean((vmem[~featureMask] - target[~featureMask]) ** 2))),
                groupRMS=[float(np.sqrt(np.mean((vmem[mask] - target[mask]) ** 2))) for mask in scoreMasks],
                outlineRMS=float(np.sqrt(np.mean((vmem[ringMask] - target[ringMask]) ** 2))), ringDark=ringDarkShare(vmem),
                structuralIoU=boundary.structuralIntersectionOverUnion(vmem),
                partScore=boundary.partSeparationScore(vmem)[0])


def polytope(numCoefficients):
    basis = np.cos(np.outer(angles, np.arange(numCoefficients)))
    return basis, np.vstack([basis, -basis]), np.r_[np.full(len(angles), ceiling), np.zeros(len(angles))]


def nearestAllowedCode(coefficients, basis, matrix, bound, halfRange):
    if (matrix @ coefficients <= bound + 1e-9).all():
        return coefficients
    result = minimize(lambda a: (((a - coefficients) / halfRange) ** 2).sum(), coefficients, method='SLSQP',
                      constraints=[dict(type='ineq', fun=lambda a: bound - matrix @ a, jac=lambda a: -matrix)], options=dict(ftol=1e-12))
    return result.x


def randomAllowedCodes(numCoefficients, count, generator, numSteps=500):
    basis, matrix, bound = polytope(numCoefficients)
    norms = np.linalg.norm(matrix, axis=1)
    point = linprog(np.r_[np.zeros(numCoefficients), -1], A_ub=np.c_[matrix, norms], b_ub=bound,
                    bounds=[(None, None)] * numCoefficients + [(0, None)]).x[:numCoefficients]
    codes = []
    for index in range(count * numSteps):
        direction = generator.standard_normal(numCoefficients)
        direction /= np.linalg.norm(direction)
        rate, slack = matrix @ direction, bound - matrix @ point
        upper = np.min(slack[rate > 1e-12] / rate[rate > 1e-12])
        lower = np.max(slack[rate < -1e-12] / rate[rate < -1e-12])
        point = point + generator.uniform(lower, upper) * direction
        if (index + 1) % numSteps == 0:
            codes.append(point.copy())
    return np.array(codes)


rounded = lambda values, digits=3: np.round(np.asarray(values, dtype=float), digits).tolist()
summary = dict(target=rounded(target, 2), targetName=targetName, scoreGroups=scoreGroups, scoreFrom=scoreFrom, withOutline=withOutline, faceOverlap=args.faceOverlap, hold=hold, numIterations=numIterations, ceiling=ceiling,
               featureCells=boundary.featureCellIndices.tolist(), orders={}, trainingDirs=trainingDirs, nearMilliVolts=args.nearMilliVolts)
generator = np.random.default_rng(7)

log('baselines: the tissue with no clamp, and the best stored sweep code')
free = scoreCodes(np.zeros((1, 40)), holdIterations=0)
summary['free'] = dict(score=float(free[0][0]), iteration=int(free[1][0]), vmem=rounded(free[2][0], 1), parts=facePartScores(free[2][0]))
sweepSeeds = [(float(run['seedScores'][index]), run['seedCoefficients'][index], int(run['seedIterations'][index]))
              for order in orders for run in runs[order] if 'seedSources' in run
              for index in np.flatnonzero(run['seedSources'] == 'sweep')]
if sweepSeeds:
    score, coefficients, iteration = min(sweepSeeds, key=lambda seed: seed[0])
    sweepBasis = polytope(len(coefficients))[0]
    sweepCheck = scoreCodes(np.clip(sweepBasis @ coefficients, 0, ceiling)[None])
    summary['bestSweepCode'] = dict(coefficients=rounded(coefficients, 4), score=score, iteration=iteration, rerunScore=float(sweepCheck[0][0]),
                                    ringValues=rounded(np.clip(sweepBasis @ coefficients, 0, ceiling)), vmem=rounded(sweepCheck[2][0], 1),
                                    parts=facePartScores(sweepCheck[2][0]))

for order in orders:
    orderRuns = sorted(runs[order], key=lambda run: (run['round'], int(run['restart'])))
    numCoefficients = order + 1
    basis, matrix, bound = polytope(numCoefficients)
    lowest, highest = orderRuns[0]['coefficientLowest'], orderRuns[0]['coefficientHighest']
    halfRange = (highest - lowest) / 2
    restarts = [dict(round=int(run['round']), populationSize=int(run['populationSize']), restart=int(run['restart']), startType=str(run['startType']), startCoefficients=rounded(run['startCoefficients'], 4),
                     bestScore=float(run['bestScore']), bestCoefficients=rounded(run['bestCoefficients'], 4), bestIteration=int(run['bestIteration']),
                     bestGeneration=int(run['bestGeneration']), numGenerations=len(run['bestScoreSoFar']), numEvaluations=int(run['numEvaluations']),
                     bestScoreSoFar=rounded(run['bestScoreSoFar']), generationBestScore=rounded(run['generationBestScore']),
                     generationMedianScore=rounded(run['generationMedianScore']), spread=rounded(run['spread'], 5),
                     startScore=float(run['evaluatedScore'][0].min()))
                for run in orderRuns]
    winner = orderRuns[int(np.argmin([run['bestScore'] for run in orderRuns]))]
    code = winner['bestCoefficients']
    log(f'order {order}: best {float(winner["bestScore"]):.3f} mV (restart {int(winner["restart"])}); long run, neighbours, random codes')
    ringValues = np.clip(basis @ code, 0, ceiling)
    offsets = [offset for offset in args.snapshotOffsets if hold <= int(winner['bestIteration']) + offset < args.longIterations]
    longScore, longIteration, longVmem, trace, captured, overlapTrace, ringTrace = scoreCodes(ringValues[None], numRunIterations=args.longIterations, keepTrace=True,
                                                                     captureIterations=[int(winner['bestIteration']) + offset for offset in offsets])
    trace = trace[0]
    face = showsFace(overlapTrace, ringTrace)
    near = trace <= float(winner['bestScore']) + args.nearMilliVolts
    visits = np.flatnonzero(np.diff(np.r_[0, near.astype(int)]) == 1)
    neighbours = np.array([nearestAllowedCode(code + halfRange * args.neighbourStep * generator.standard_normal(numCoefficients), basis, matrix, bound, halfRange)
                           for _ in range(args.numNeighbours)])
    neighbourScores = scoreCodes(np.clip(neighbours @ basis.T, 0, ceiling))[0]
    randomCodes = randomAllowedCodes(numCoefficients, args.numRandomCodes, generator)
    randomScores = scoreCodes(np.clip(randomCodes @ basis.T, 0, ceiling))[0]
    allScores = np.concatenate([run['evaluatedScore'].ravel() for run in orderRuns])
    allIterations = np.concatenate([run['evaluatedBestIteration'].ravel() for run in orderRuns])
    summary['orders'][order] = dict(
        restarts=restarts, coefficientLowest=rounded(lowest), coefficientHighest=rounded(highest),
        best=dict(round=int(winner['round']), populationSize=int(winner['populationSize']), restart=int(winner['restart']), score=float(winner['bestScore']), coefficients=rounded(code, 4), ringValues=rounded(ringValues),
                  iteration=int(winner['bestIteration']), vmem=rounded(winner['bestVmem'], 1), windowMeanVmem=rounded(winner['bestWindowMeanVmem'], 1),
                  parts=facePartScores(winner['bestVmem'].astype(np.float64))),
        longRun=dict(iterations=args.longIterations, bestScore=float(longScore[0]), bestIteration=int(longIteration[0]), bestVmem=rounded(longVmem[0], 1),
                     traceStride=10, trace=rounded(trace[::10], 2), nearFraction=float(near.mean()), numVisits=int(len(visits)),
                     nearFractionTrainingSpan=float(near[:numIterations - hold].mean()), traceMedian=float(np.median(trace)),
                     snapshots=[dict(offset=offset, iteration=int(winner['bestIteration']) + offset, vmem=rounded(captured[int(winner['bestIteration']) + offset], 1),
                                     score=float(trace[int(winner['bestIteration']) + offset - hold])) for offset in offsets],
                     overlapTrace=rounded(overlapTrace[::10], 2), ringTrace=rounded(ringTrace[::10], 2),
                     faceShapeIterations=int(face.sum()), faceShapeVisits=int((np.diff(np.r_[0, face.astype(int)]) == 1).sum()),
                     faceShapeSpan=[int(np.flatnonzero(face).min() + hold), int(np.flatnonzero(face).max() + hold)] if face.any() else None,
                     ringDarkIterations=int((ringTrace >= args.faceOverlap).sum()),
                     ringDarkSpan=[int(np.flatnonzero(ringTrace >= args.faceOverlap).min() + hold), int(np.flatnonzero(ringTrace >= args.faceOverlap).max() + hold)]
                     if (ringTrace >= args.faceOverlap).any() else None,
                     reproduced=bool(abs(float(trace[int(winner['bestIteration']) - hold]) - float(winner['bestScore'])) < 1e-6)),
        neighbours=dict(step=args.neighbourStep, scores=rounded(neighbourScores), median=float(np.median(neighbourScores))),
        randomCodes=dict(scores=rounded(randomScores), median=float(np.median(randomScores)), best=float(randomScores.min())),
        evaluated=dict(count=int(len(allScores)), histogram=np.histogram(allScores, bins=np.arange(0, 60.5, 0.5))[0].tolist(),
                       bestIterationHistogram=np.histogram(allIterations, bins=np.arange(hold, numIterations + 100, 100))[0].tolist(),
                       bestIterationHistogramTop=np.histogram(allIterations[allScores <= np.quantile(allScores, 0.05)], bins=np.arange(hold, numIterations + 100, 100))[0].tolist()),
        library=[dict(round=int(run['round']), sweepCodes=int(run['numSweepCodes']), trainedCodes=int(run['numTrainedCodes']))
                 for run in orderRuns if int(run['restart']) == 0],
        coefficientRanges=dict(lowest=rounded(lowest), highest=rounded(highest)))
    log(f'  long run: best {longScore[0]:.3f} at {longIteration[0]}, within {args.nearMilliVolts} mV {100 * near.mean():.2f}% of the time, '
        f'{len(visits)} visits; face shape at {int(face.sum())} iterations, ring dark at {int((ringTrace >= args.faceOverlap).sum())}; neighbours median {np.median(neighbourScores):.3f}; random median {np.median(randomScores):.3f}, best {randomScores.min():.3f}')

json.dump(summary, open(outputPath, 'w'), separators=(',', ':'))
log(f'wrote {outputPath}')
