"""How much does the model's numerical precision matter to the trained faces? (PolyPatterning_Sim.md, Section 12)

The model's state (Vmem, G_pol, eV, ...) is float64, but torch's default dtype is float32, so the geometry constants built
without an explicit dtype are float32 and promoted afterwards: the cell coordinates, hence the distances and the 1 / distance
field kernel (cellularFieldNetwork.py). boundaryCodeUtilities.fullDoublePrecision builds the whole model in float64 instead.
This script replays each order's best ring code under both models and asks four things.

  1. Do the two models agree? The largest difference in Vmem over the cells, at every iteration, and the first iteration at
     which it exceeds 0.1, 1 and 5 mV.
  2. Does the face survive? Each model's best balanced RMS after the release, its iteration, the structural IoU there,
     and each model's score at the iteration the code was trained to (the default model must reproduce the training score).
  3. How fast does any tiny difference grow? In the float64 model, each code's ring values are changed by +-epsilon per cell
     (epsilon from 1e-4 down to 1e-15, `--seeds` random sign patterns each), and the first iteration at which the tissue is
     0.1, 1 and 5 mV from the unchanged code's tissue is recorded. Rounding in float64 is about 1e-16 per operation, so the
     small epsilons say how far a float64 run can be trusted pointwise.
  4. Does the same hold when the code is not exactly the trained one? (Question 3 covers it: the perturbed codes are
     neighbours of the trained codes.)

Writes data/boundaryHarmonicPrecision<rest of the summary's name> (never overwriting); the file name gains Horizon<N> for
--numIterations other than 6000.
"""
import argparse
import json
import os
import time

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--numIterations', type=int, default=6000)
parser.add_argument('--epsilons', type=str, default='1e-4,1e-6,1e-8,1e-10,1e-12,1e-14,1e-15')
parser.add_argument('--seeds', type=int, default=4)
parser.add_argument('--stride', type=int, default=20, help='spacing of the stored divergence curve')
args = parser.parse_args()

summary = json.load(open(args.summaryPath))
outputPath = args.summaryPath.replace('boundaryHarmonicTrainingSummary', 'boundaryHarmonicPrecision')
if args.numIterations != 6000:
    outputPath = outputPath.replace('.json', f'Horizon{args.numIterations}.json')
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

reference = boundary.loadCheckpoint(1888)
hold, numIterations = int(summary['hold']), args.numIterations
target = np.asarray(summary['target'], dtype=float)
orderNames = sorted(summary['orders'], key=int)
angles = boundary.ringAngles(boundary.boundaryRingCells)
THRESHOLDS = (0.1, 1.0, 5.0)


def trainedCoefficients(name):
    """The training run's own full-precision coefficients (the summary rounds them)."""
    winner = summary['orders'][name]['best']
    path = f"{summary['trainingDirs'][winner['round']]}/order{name}_restart{winner['restart']:02d}.npz"
    return np.asarray(np.load(path)['bestCoefficients'], dtype=float)


ringValues = np.array([np.cos(np.outer(angles, np.arange(len(c)))) @ c for c in (trainedCoefficients(n) for n in orderNames)])
feature, other = boundary.featureCellIndices, boundary.otherCellIndices


def balancedScores(vmem):
    return (0.5 * np.sqrt(((vmem[:, feature] - target[feature]) ** 2).mean(1))
            + 0.5 * np.sqrt(((vmem[:, other] - target[other]) ** 2).mean(1)))


def replayAll(codes, doublePrecision, onIteration):
    started = time.time()
    boundary.ringHoldBatchReplay(reference, codes, hold, numIterations, lambda i, v: onIteration(i, v.numpy()), doublePrecision=doublePrecision)
    print(f"  {'float64' if doublePrecision else 'default'}, {len(codes)} codes, {numIterations} iterations: {time.time() - started:.0f} s", flush=True)


class Divergence:
    """Largest |Vmem difference| over the cells against a stored baseline, at every iteration, and the first iteration past each threshold."""
    def __init__(self, baseline, rows):
        self.baseline, self.rows = baseline, rows
        self.first = np.full((len(rows), len(THRESHOLDS)), -1, dtype=int)
        self.curve = np.zeros((len(rows), numIterations))

    def __call__(self, iteration, vmem):
        difference = np.abs(vmem[self.rows] - self.baseline[iteration]).max(1)
        self.curve[:, iteration] = difference
        for k, threshold in enumerate(THRESHOLDS):
            fresh = (difference > threshold) & (self.first[:, k] < 0)
            self.first[fresh, k] = iteration


# ------------------------------------------------------------------ 1 and 2: default against float64
print(f'{len(orderNames)} orders, hold {hold}, {numIterations} iterations', flush=True)
default = np.zeros((numIterations, len(orderNames), boundary.numCells))
replayAll(ringValues, False, lambda i, v: default.__setitem__(i, v))
double = np.zeros_like(default)
vsDefault = Divergence(default, list(range(len(orderNames))))


def storeDouble(iteration, vmem):
    double[iteration] = vmem
    # the baseline is indexed [iteration][code]; Divergence reads baseline[iteration] as (codes, cells)
    vsDefault(iteration, vmem)


replayAll(ringValues, True, storeDouble)

orders = {}
for k, name in enumerate(orderNames):
    best = summary['orders'][name]['best']
    entry = dict(order=int(name), trainedScore=round(float(best['score']), 3), trainedIteration=int(best['iteration']))
    for label, run in (('default', default), ('float64', double)):
        scores = balancedScores(run[:, k])
        scores[:hold] = np.inf
        where = int(np.argmin(scores))
        entry[label] = dict(bestScore=round(float(scores[where]), 3), bestIteration=where,
                            structuralIoU=round(boundary.structuralIntersectionOverUnion(run[where, k]), 3),
                            scoreAtTrainedIteration=round(float(balancedScores(run[best['iteration']][None, k])[0]), 3))
    entry['partingIteration'] = {str(t): int(vsDefault.first[k, j]) for j, t in enumerate(THRESHOLDS)}
    entry['largestDifferenceAtTrainedIteration'] = round(float(vsDefault.curve[k, best['iteration']]), 4)
    entry['divergence'] = [round(float(v), 6) for v in vsDefault.curve[k, ::args.stride]]
    orders[name] = entry
    print(f"  order {name}: trained {entry['trainedScore']} at {entry['trainedIteration']}; default {entry['default']['bestScore']} at "
          f"{entry['default']['bestIteration']}; float64 {entry['float64']['bestScore']} at {entry['float64']['bestIteration']} "
          f"(IoU {entry['float64']['structuralIoU']}); parts at {entry['partingIteration']}", flush=True)
del default

# ------------------------------------------------------------------ 3: the growth of tiny differences in float64
epsilons = [float(e) for e in args.epsilons.split(',')]
rng = np.random.default_rng(20251201)
perturbed, labels = [], []
for k in range(len(orderNames)):
    for epsilon in epsilons:
        for seed in range(args.seeds):
            signs = rng.choice([-1.0, 1.0], size=ringValues.shape[1])
            perturbed.append(np.clip(ringValues[k] + epsilon * signs, 0.0, 2.0))
            labels.append((k, epsilon, seed))
perturbed = np.array(perturbed)
rows = list(range(len(perturbed)))
baselineOfRow = np.array([k for k, _, _ in labels])


class LadderDivergence(Divergence):
    def __call__(self, iteration, vmem):
        difference = np.abs(vmem - self.baseline[iteration][baselineOfRow]).max(1)
        self.curve[:, iteration] = difference
        for j, threshold in enumerate(THRESHOLDS):
            fresh = (difference > threshold) & (self.first[:, j] < 0)
            self.first[fresh, j] = iteration


ladder = LadderDivergence(double, rows)
replayAll(perturbed, True, ladder)
byEpsilon = {}
for epsilon in epsilons:
    members = [r for r, (_, e, _) in enumerate(labels) if e == epsilon]
    byEpsilon[f'{epsilon:g}'] = dict(
        epsilon=epsilon,
        parting={str(t): [int(ladder.first[r, j]) for r in members] for j, t in enumerate(THRESHOLDS)},
        orderOfRow=[int(labels[r][0]) for r in members],
        largestDifferenceAtTrainedIteration=[round(float(ladder.curve[r, int(summary['orders'][orderNames[labels[r][0]]]['best']['iteration'])]), 5)
                                             for r in members])
    median = {t: np.median([v for v in byEpsilon[f'{epsilon:g}']['parting'][str(t)] if v >= 0] or [np.nan]) for t in THRESHOLDS}
    never = {t: sum(v < 0 for v in byEpsilon[f'{epsilon:g}']['parting'][str(t)]) for t in THRESHOLDS}
    print(f"  epsilon {epsilon:g}: median iteration at {THRESHOLDS[0]} / {THRESHOLDS[1]} / {THRESHOLDS[2]} mV = "
          f"{median[0.1]:.0f} / {median[1.0]:.0f} / {median[5.0]:.0f}  (never reached within {numIterations}: {never[0.1]} / {never[1.0]} / {never[5.0]} of {len(members)})", flush=True)

result = dict(hold=hold, numIterations=numIterations, stride=args.stride, thresholdsMilliVolts=list(THRESHOLDS),
              seeds=args.seeds, orders=orders, ladder=byEpsilon,
              floatingPointNote='float64 holds 53 bits: spacing 1.1e-16 near 1; float32 24 bits: 6e-8')
json.dump(result, open(outputPath, 'w'))
print('wrote', outputPath, os.path.getsize(outputPath) // 1024, 'KiB', flush=True)
