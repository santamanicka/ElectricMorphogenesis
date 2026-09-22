"""Train a mirror-symmetric ring code, orders 0 to N, toward a target pattern (PolyPatterning_Sim.md, Section 12).

The code. During the hold, ring cell k (angle theta_k about the lattice centre, clockwise from straight up) is held at
    G_pol / G_ref = a_0 + sum_{n=1..N} a_n cos(n theta_k),
every term at phase 0 with a signed amplitude, which is exactly the set of codes mirror-symmetric about the vertical axis.
Written as B a, with B the 40 x (N+1) matrix of cos(n theta_k), the ceiling 0 <= B a <= --ceiling is 80 linear
inequalities, so the allowed codes form a convex polytope. Each coefficient's range over it comes from linear
programming, and the optimiser works on each coefficient scaled by half its range.

The score. The reference checkpoint's model and initial state are replayed with the ring held for --holdIterations and
then released, for --numIterations in all. Balanced RMS (0.5 x RMS over the feature cells + 0.5 x RMS over every other
cell) against the target is computed at every iteration from the release on, and the lowest value is the score.

The optimiser. CMA-ES (Hansen's (mu/mu_w, lambda) form with rank-one and rank-mu updates), one population per batched
simulation. A proposed code outside the polytope is simulated at its nearest allowed code (Euclidean, in scaled units),
and its fitness adds --penaltyWeight mV per scaled unit it had to move.

The start. Restart r starts either from the library (even r) or from a random allowed code drawn uniformly by
hit-and-run (odd r). The library holds every previously simulated code that is a phase-0 cosine series of order <= N:
the stored dial and boundary-harmonic sweeps (terms with sin(n phi) = 0 enter as a_n = G cos(n phi)) and every code evaluated by earlier
runs in --outputDir and --libraryDirs, each also with its top-to-bottom mirror image (a_n -> (-1)^n a_n, pattern flipped), which the
lattice's symmetry gives without simulation. Library codes are ranked by the lowest balanced RMS among their stored
patterns (the late-window mean, and for trained codes also the pattern at their best moment), the best --librarySeedCount are rerun and scored properly, and library restart r starts from the (r / 2)-th best.

Output: <outputDir>/order<N>_restart<rr>.npz with every evaluated code, its score, best moment and patterns, the
per-generation history and the best code. Existing files are never overwritten.
"""
import argparse
import glob
import os
import time

import numpy as np
import torch
from scipy.optimize import linprog, minimize

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--maxOrder', type=int, required=True, help='N: the code uses orders 0 to N')
parser.add_argument('--restart', type=int, required=True)
parser.add_argument('--referenceCheckpoint', type=int, default=1888)
parser.add_argument('--holdIterations', type=int, default=301)
parser.add_argument('--numIterations', type=int, default=3000)
parser.add_argument('--windowIterations', type=int, default=1000, help='late window whose mean is stored for the library')
parser.add_argument('--ceiling', type=float, default=1.3, help='every held value lies in [0, ceiling]')
parser.add_argument('--target', type=str, default='face', help="'face' or the path of a .npy array of 121 values in mV")
parser.add_argument('--featureMilliVolts', type=float, default=-60.0, help="face target: the 14 feature cells")
parser.add_argument('--backgroundMilliVolts', type=float, default=-5.0, help="face target: every other cell")
parser.add_argument('--targetName', type=str, default='FaceMinus60Minus5')
parser.add_argument('--populationSize', type=int, default=16)
parser.add_argument('--numGenerations', type=int, default=150)
parser.add_argument('--initialStep', type=float, default=0.3, help='CMA-ES starting step, in scaled units')
parser.add_argument('--penaltyWeight', type=float, default=10.0, help='mV per scaled unit a code is moved into the polytope')
parser.add_argument('--librarySeedCount', type=int, default=16)
parser.add_argument('--stopStep', type=float, default=1e-4, help='stop once every axis of the search is below this, in scaled units')
parser.add_argument('--outputDir', type=str, default=None)
parser.add_argument('--libraryDirs', type=str, default='', help='comma-separated folders of earlier training runs whose codes also join the library')
args = parser.parse_args()

numCoefficients = args.maxOrder + 1
outputDir = args.outputDir or f'data/boundaryHarmonicTraining{args.referenceCheckpoint}Hold{args.holdIterations}{args.targetName}'
os.makedirs(outputDir, exist_ok=True)
outputPath = f'{outputDir}/order{args.maxOrder}_restart{args.restart:02d}.npz'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
startTime = time.time()


def log(message):
    print(f'[{time.time() - startTime:7.0f}s] {message}', flush=True)


# ------------------------------------------------------------------------------------------ target and code
if args.target == 'face':
    target = np.full(boundary.numCells, args.backgroundMilliVolts)
    target[boundary.featureCellIndices] = args.featureMilliVolts
else:
    target = np.load(args.target).reshape(-1).astype(np.float64)
featureMask = np.isin(np.arange(boundary.numCells), boundary.featureCellIndices)
targetTensor = torch.tensor(target, dtype=torch.double)


def balancedScores(vmem):
    """Balanced RMS of each row of vmem (codes x cells, mV) against the target."""
    squared = (vmem - targetTensor) ** 2
    return 0.5 * squared[:, featureMask].mean(1).sqrt() + 0.5 * squared[:, ~featureMask].mean(1).sqrt()


angles = boundary.ringAngles(boundary.boundaryRingCells)
basis = np.cos(np.outer(angles, np.arange(numCoefficients)))                  # 40 x (N+1)
constraintMatrix = np.vstack([basis, -basis])
constraintBound = np.r_[np.full(len(angles), args.ceiling), np.zeros(len(angles))]
lowest, highest = np.zeros(numCoefficients), np.zeros(numCoefficients)
for order in range(numCoefficients):
    unit = np.eye(numCoefficients)[order]
    lowest[order] = linprog(unit, A_ub=constraintMatrix, b_ub=constraintBound, bounds=[(None, None)] * numCoefficients).fun
    highest[order] = -linprog(-unit, A_ub=constraintMatrix, b_ub=constraintBound, bounds=[(None, None)] * numCoefficients).fun
middle, halfRange = (lowest + highest) / 2, (highest - lowest) / 2
toCoefficients = lambda scaled: middle + halfRange * scaled
toScaled = lambda coefficients: (coefficients - middle) / halfRange
feasible = lambda coefficients, slack=1e-9: bool((constraintMatrix @ coefficients <= constraintBound + slack).all())
log(f'order {args.maxOrder}: coefficient ranges ' + ', '.join(f'a{order} [{lowest[order]:.3f}, {highest[order]:.3f}]' for order in range(numCoefficients)))


def nearestAllowed(scaled):
    """The allowed code nearest to `scaled`, in scaled units."""
    if feasible(toCoefficients(scaled)):
        return scaled.copy()
    scaledMatrix, scaledBound = constraintMatrix * halfRange, constraintBound - constraintMatrix @ middle
    result = minimize(lambda z: ((z - scaled) ** 2).sum(), np.zeros(numCoefficients), jac=lambda z: 2 * (z - scaled),
                      constraints=[dict(type='ineq', fun=lambda z: scaledBound - scaledMatrix @ z, jac=lambda z: -scaledMatrix)],
                      method='SLSQP', options=dict(ftol=1e-12, maxiter=500))
    return result.x


def randomAllowedCode(generator, numSteps=500):
    """A code drawn uniformly from the polytope by hit-and-run, started at its Chebyshev centre."""
    norms = np.linalg.norm(constraintMatrix, axis=1)
    centre = linprog(np.r_[np.zeros(numCoefficients), -1], A_ub=np.c_[constraintMatrix, norms], b_ub=constraintBound,
                     bounds=[(None, None)] * numCoefficients + [(0, None)]).x[:numCoefficients]
    point = centre
    for _ in range(numSteps):
        direction = generator.standard_normal(numCoefficients)
        direction /= np.linalg.norm(direction)
        rate, slack = constraintMatrix @ direction, constraintBound - constraintMatrix @ point
        upper = np.min(slack[rate > 1e-12] / rate[rate > 1e-12])
        lower = np.max(slack[rate < -1e-12] / rate[rate < -1e-12])
        point = point + generator.uniform(lower, upper) * direction
    return point


# ----------------------------------------------------------------------------------------------- simulation
reference = boundary.loadCheckpoint(args.referenceCheckpoint)
numEvaluations = 0


def evaluate(coefficientRows):
    """Simulate each code; return its score, best iteration, pattern at that iteration and late-window mean."""
    global numEvaluations
    ringValues = np.clip(np.asarray(coefficientRows) @ basis.T, 0, args.ceiling)
    numCodes = len(ringValues)
    best = dict(score=torch.full((numCodes,), np.inf, dtype=torch.double), iteration=torch.zeros(numCodes, dtype=torch.long),
                vmem=torch.zeros(numCodes, boundary.numCells, dtype=torch.double))
    windowSum = torch.zeros(numCodes, boundary.numCells, dtype=torch.double)

    def onIteration(iteration, vmem):
        if iteration >= args.holdIterations:
            scores = balancedScores(vmem)
            better = scores < best['score']
            best['score'] = torch.where(better, scores, best['score'])
            best['iteration'][better] = iteration
            best['vmem'][better] = vmem[better]
        if iteration >= args.numIterations - args.windowIterations:
            windowSum.add_(vmem)
    boundary.ringHoldBatchReplay(reference, ringValues, args.holdIterations, args.numIterations, onIteration)
    numEvaluations += numCodes
    return (best['score'].numpy(), best['iteration'].numpy(), best['vmem'].numpy().astype(np.float32),
            (windowSum / args.windowIterations).numpy().astype(np.float32))


# ----------------------------------------------------------------------------------------------- library
def verticalFlip(patterns):
    return patterns.reshape(-1, boundary.latticeRows, boundary.latticeCols)[:, ::-1].reshape(len(patterns), -1)


def sweepLibrary():
    """Codes from the stored dial and boundary-harmonic sweeps whose every harmonic term is at phase 0 or 180 degrees
    for its order. A sweep without ringValues is the dial alone, held uniformly."""
    coefficientRows, patterns = [], []
    paths = [f'data/boundaryDialSweep{args.referenceCheckpoint}Hold{args.holdIterations}.npz'] \
        + sorted(glob.glob(f'data/boundaryGradientLandscape{args.referenceCheckpoint}Hold{args.holdIterations}*.npz'))
    for path in filter(os.path.exists, paths):
        sweep = dict(np.load(path))
        numCodes = len(sweep['dialLevel'])
        combined = 'combinedOrders' in sweep and bool(sweep['combinedOrders'])
        order = int(sweep['harmonicOrder']) if 'harmonicOrder' in sweep else 1
        strength = sweep.get('gradientStrength', np.zeros(numCodes))
        directions = np.deg2rad(sweep.get('gradientDirection', np.zeros(numCodes)))
        secondStrength = sweep.get('secondOrderStrength', np.zeros(numCodes))
        storedRing = sweep.get('ringValues', np.repeat(sweep['dialLevel'][:, None], len(angles), 1))
        for index in range(numCodes):
            direction = directions[index]
            terms = [(1, strength[index]), (2, secondStrength[index])] if combined else [(order, strength[index])]
            if any(abs(np.sin(harmonic * direction)) > 1e-9 for harmonic, amplitude in terms if amplitude != 0):
                continue
            coefficients = np.zeros(max(3, numCoefficients))
            coefficients[0] = sweep['dialLevel'][index]
            for harmonic, amplitude in terms:
                coefficients[harmonic] += amplitude * np.cos(harmonic * direction)
            if np.abs(coefficients[numCoefficients:]).max(initial=0) > 1e-12:
                continue
            coefficients = coefficients[:numCoefficients]
            if np.abs(np.clip(basis @ coefficients, 0, None) - storedRing[index]).max() > 1e-9:
                raise ValueError(f'{path} run {index}: stored ring values do not match the phase-0 coefficients')
            coefficientRows.append(coefficients)
            patterns.append(sweep['windowMeanVmem'][index])
    return coefficientRows, patterns


def trainingLibrary():
    """Every code evaluated by earlier runs in outputDir and --libraryDirs whose orders above N are all zero."""
    coefficientRows, patterns = [], []
    directories = [outputDir] + [directory for directory in args.libraryDirs.split(',') if directory]
    for path in sorted(path for directory in directories for path in glob.glob(f'{directory}/order*_restart*.npz')):
        run = dict(np.load(path))
        rows = run['evaluatedCoefficients']
        rows = rows.reshape(-1, rows.shape[-1])
        if rows.shape[1] > numCoefficients and np.abs(rows[:, numCoefficients:]).max() > 1e-12:
            keep = np.abs(rows[:, numCoefficients:]).max(1) <= 1e-12
        else:
            keep = np.ones(len(rows), dtype=bool)
        padded = np.zeros((len(rows), numCoefficients))
        padded[:, :min(numCoefficients, rows.shape[1])] = rows[:, :numCoefficients]
        windows = run['evaluatedWindowMeanVmem'].reshape(len(rows), -1)
        bestMoments = run['evaluatedBestVmem'].reshape(len(rows), -1)
        for stored in (windows, bestMoments):
            coefficientRows.extend(padded[keep])
            patterns.extend(stored[keep])
    return coefficientRows, patterns


def seedingLibrary():
    sweepRows, sweepPatterns = sweepLibrary()
    trainedRows, trainedPatterns = trainingLibrary()
    coefficientRows = np.array(sweepRows + trainedRows).reshape(-1, numCoefficients)
    patterns = np.array(sweepPatterns + trainedPatterns, dtype=np.float64).reshape(-1, boundary.numCells)
    mirrorSigns = (-1.0) ** np.arange(numCoefficients)
    sources = np.array(['sweep'] * len(sweepRows) + ['trained'] * len(trainedRows))
    coefficientRows = np.vstack([coefficientRows, coefficientRows * mirrorSigns])
    patterns = np.vstack([patterns, verticalFlip(patterns)])
    sources = np.r_[sources, sources]
    proxies = balancedScores(torch.tensor(patterns)).numpy()
    distinct, groups = np.unique(np.round(coefficientRows, 9), axis=0, return_inverse=True)
    groups = groups.reshape(-1)
    bestProxy = np.full(len(distinct), np.inf)
    np.minimum.at(bestProxy, groups, proxies)
    firstRow = np.full(len(distinct), -1)
    for row in np.argsort(proxies)[::-1]:
        firstRow[groups[row]] = row
    log(f'library: {len(sweepRows)} sweep codes, {len(trainedRows) // 2} trained codes, {len(distinct)} distinct with mirror images')
    return coefficientRows[firstRow], bestProxy, sources[firstRow], len(sweepRows), len(trainedRows) // 2


# ------------------------------------------------------------------------------------------------ CMA-ES
class CovarianceMatrixAdaptation:
    """Hansen's (mu/mu_w, lambda)-CMA-ES, as in 'The CMA Evolution Strategy: A Tutorial' (2016)."""

    def __init__(self, mean, step, populationSize, generator):
        self.dimension, self.mean, self.step, self.generator = len(mean), np.array(mean, dtype=float), step, generator
        self.populationSize, self.parentCount = populationSize, populationSize // 2
        weights = np.log((populationSize + 1) / 2) - np.log(np.arange(1, self.parentCount + 1))
        self.weights = weights / weights.sum()
        self.effectiveParents = 1 / (self.weights ** 2).sum()
        n, mu = self.dimension, self.effectiveParents
        self.pathRate = (4 + mu / n) / (n + 4 + 2 * mu / n)
        self.stepPathRate = (mu + 2) / (n + mu + 5)
        self.rankOneRate = 2 / ((n + 1.3) ** 2 + mu)
        self.rankParentRate = min(1 - self.rankOneRate, 2 * (mu - 2 + 1 / mu) / ((n + 2) ** 2 + mu))
        self.stepDamping = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + self.stepPathRate
        self.expectedNorm = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        self.covariance, self.path, self.stepPath, self.generation = np.eye(n), np.zeros(n), np.zeros(n), 0
        self.decompose()

    def decompose(self):
        self.covariance = (self.covariance + self.covariance.T) / 2
        eigenvalues, self.axes = np.linalg.eigh(self.covariance)
        self.scales = np.sqrt(np.maximum(eigenvalues, 1e-20))

    def ask(self):
        self.normals = self.generator.standard_normal((self.populationSize, self.dimension))
        return self.mean + self.step * (self.normals * self.scales) @ self.axes.T

    def tell(self, candidates, fitness):
        order = np.argsort(fitness)[:self.parentCount]
        previousMean = self.mean
        self.mean = self.weights @ candidates[order]
        shift = (self.mean - previousMean) / self.step
        whitened = self.axes @ ((self.axes.T @ shift) / self.scales)
        n, mu = self.dimension, self.effectiveParents
        self.stepPath = (1 - self.stepPathRate) * self.stepPath + np.sqrt(self.stepPathRate * (2 - self.stepPathRate) * mu) * whitened
        self.generation += 1
        stepPathNorm = np.linalg.norm(self.stepPath)
        stalled = stepPathNorm / np.sqrt(1 - (1 - self.stepPathRate) ** (2 * self.generation)) < (1.4 + 2 / (n + 1)) * self.expectedNorm
        self.path = (1 - self.pathRate) * self.path + stalled * np.sqrt(self.pathRate * (2 - self.pathRate) * mu) * shift
        steps = (candidates[order] - previousMean) / self.step
        self.covariance = ((1 - self.rankOneRate - self.rankParentRate) * self.covariance
                           + self.rankOneRate * (np.outer(self.path, self.path) + (1 - stalled) * self.pathRate * (2 - self.pathRate) * self.covariance)
                           + self.rankParentRate * (self.weights[:, None] * steps).T @ steps)
        self.step *= np.exp((self.stepPathRate / self.stepDamping) * (stepPathNorm / self.expectedNorm - 1))
        self.decompose()

    def spread(self):
        return self.step * self.scales.max()


# ------------------------------------------------------------------------------------------------ run
generator = np.random.default_rng(1000 * args.maxOrder + args.restart)
libraryRows, proxyScores, librarySources, numSweepCodes, numTrainedCodes = seedingLibrary()
candidateOrder = [index for index in np.argsort(proxyScores) if feasible(libraryRows[index])][:args.librarySeedCount]
seedRows = libraryRows[candidateOrder]
seedScores, seedIterations, _, _ = evaluate(seedRows)
seedRanking = np.argsort(seedScores)
log(f'library seeds rerun: best {seedScores.min():.3f} mV, stored-pattern proxy of those {proxyScores[candidateOrder].min():.3f} mV')
if args.restart % 2 == 0:
    startType = 'library'
    startCoefficients = seedRows[seedRanking[(args.restart // 2) % len(seedRanking)]]
else:
    startType = 'random'
    startCoefficients = randomAllowedCode(generator)
log(f'restart {args.restart}: {startType} start ' + ', '.join(f'{value:.3f}' for value in startCoefficients))

search = CovarianceMatrixAdaptation(toScaled(startCoefficients), args.initialStep, args.populationSize, generator)
record = {key: [] for key in ('evaluatedCandidates', 'evaluatedCoefficients', 'evaluatedPenalty', 'evaluatedScore', 'evaluatedFitness',
                              'evaluatedBestIteration', 'evaluatedBestVmem', 'evaluatedWindowMeanVmem')}
history = {key: [] for key in ('generationBestScore', 'generationMedianScore', 'bestScoreSoFar', 'step', 'spread', 'meanCoefficients')}
bestSoFar = dict(score=np.inf)
for generation in range(args.numGenerations):
    candidates = search.ask()
    allowed = np.array([nearestAllowed(candidate) for candidate in candidates])
    penalty = np.linalg.norm(candidates - allowed, axis=1)
    coefficients = toCoefficients(allowed)
    scores, bestIterations, bestVmem, windowMean = evaluate(coefficients)
    fitness = scores + args.penaltyWeight * penalty
    search.tell(candidates, fitness)
    for key, value in zip(record, (candidates, coefficients, penalty, scores, fitness, bestIterations, bestVmem, windowMean)):
        record[key].append(value)
    best = int(np.argmin(scores))
    if scores[best] < bestSoFar['score']:
        bestSoFar = dict(score=float(scores[best]), coefficients=coefficients[best], iteration=int(bestIterations[best]),
                         vmem=bestVmem[best], windowMean=windowMean[best], generation=generation)
    for key, value in zip(history, (scores.min(), np.median(scores), bestSoFar['score'], search.step, search.spread(), toCoefficients(search.mean))):
        history[key].append(value)
    if generation % 10 == 0 or generation == args.numGenerations - 1:
        log(f'generation {generation}: best {scores.min():.3f}, median {np.median(scores):.3f}, best so far {bestSoFar["score"]:.3f} mV, '
            f'spread {search.spread():.4f}, mean ' + ', '.join(f'{value:.3f}' for value in toCoefficients(search.mean)))
    if search.spread() < args.stopStep:
        log(f'stopped at generation {generation}: search spread {search.spread():.2e} below {args.stopStep}')
        break

np.savez_compressed(
    outputPath, **{key: np.array(value) for key, value in record.items()}, **{key: np.array(value) for key, value in history.items()},
    bestScore=bestSoFar['score'], bestCoefficients=bestSoFar['coefficients'], bestRingValues=np.clip(basis @ bestSoFar['coefficients'], 0, args.ceiling),
    bestIteration=bestSoFar['iteration'], bestVmem=bestSoFar['vmem'], bestWindowMeanVmem=bestSoFar['windowMean'], bestGeneration=bestSoFar['generation'],
    startType=startType, startCoefficients=startCoefficients, seedCoefficients=seedRows, seedScores=seedScores, seedIterations=seedIterations,
    seedProxyScores=proxyScores[candidateOrder], seedSources=librarySources[candidateOrder], numSweepCodes=numSweepCodes, numTrainedCodes=numTrainedCodes,
    coefficientLowest=lowest, coefficientHighest=highest, ringAngles=angles, target=target, targetName=args.targetName,
    maxOrder=args.maxOrder, restart=args.restart, referenceCheckpoint=args.referenceCheckpoint, holdIterations=args.holdIterations,
    numIterations=args.numIterations, windowIterations=args.windowIterations, ceiling=args.ceiling, populationSize=args.populationSize,
    numGenerations=args.numGenerations, initialStep=args.initialStep, penaltyWeight=args.penaltyWeight, numEvaluations=numEvaluations,
    clampMode='tissueRingGpolHarmonic', libraryDirs=args.libraryDirs)
log(f'best {bestSoFar["score"]:.3f} mV at iteration {bestSoFar["iteration"]} (generation {bestSoFar["generation"]}), code '
    + ', '.join(f'a{order} {value:.4f}' for order, value in enumerate(bestSoFar['coefficients'])) + f'; wrote {outputPath}')
