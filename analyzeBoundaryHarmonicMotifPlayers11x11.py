"""Frame-by-frame Vmem for the best code of every order, for the report's synchronous players
(PolyPatterning_Sim.md, Section 12).

Each order's best ring code from the training summary is replayed under the standard protocol: held for
`hold` iterations, then released. Every --stride iterations the whole 11x11 Vmem map is stored, along with
the face score at that moment and the number of cells that crossed the hyperpolarised threshold since the
previous stored frame. The last of those is what makes stasis visible: a run of frames with no crossings is
a stretch during which the discrete pattern does not change.

Writes data/boundaryHarmonicMotifPlayers<rest of the summary's name> (never overwriting).
"""
import argparse
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--stride', type=int, default=20)
args = parser.parse_args()

summary = json.load(open(args.summaryPath))
outputPath = args.summaryPath.replace('boundaryHarmonicTrainingSummary', 'boundaryHarmonicMotifPlayers')
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

reference = boundary.loadCheckpoint(1888)
hold, numIterations = int(summary['hold']), int(summary['numIterations'])
target = np.asarray(summary['target'], dtype=float)
orderNames = sorted(summary['orders'], key=int)
# the summary rounds its stored coefficients and ring values, and this face is sharp enough that the rounding
# costs a visible part of the score, so the codes come from the training run files at full precision
angles = boundary.ringAngles(boundary.boundaryRingCells)


def trainedCoefficients(name):
    winner = summary['orders'][name]['best']
    path = f"{summary['trainingDirs'][winner['round']]}/order{name}_restart{winner['restart']:02d}.npz"
    return np.asarray(np.load(path)['bestCoefficients'], dtype=float)


coefficients = {name: trainedCoefficients(name) for name in orderNames}
ringValues = np.array([np.cos(np.outer(angles, np.arange(len(coefficients[name])))) @ coefficients[name]
                       for name in orderNames])
# every order's own best moment joins the strided grid, so the marked frame is the face that was scored
# rather than whichever neighbouring frame the stride happens to land on
bestMoments = {int(summary['orders'][name]['best']['iteration']) for name in orderNames}
storedIterations = sorted(set(range(0, numIterations, args.stride)) | bestMoments)
print(f"{len(orderNames)} orders, hold {hold}, {numIterations} iterations, stride {args.stride}, "
      f"{len(storedIterations)} frames ({len(bestMoments - set(range(0, numIterations, args.stride)))} added for best moments)", flush=True)

feature, other = boundary.featureCellIndices, boundary.otherCellIndices
targetFeature, targetOther = target[feature], target[other]
frames = {name: [] for name in orderNames}
scores = {name: [] for name in orderNames}
crossings = {name: [] for name in orderNames}
times = []
previousDark = {'value': None}
storedIterationSet = set(storedIterations)


def onIteration(iteration, vmem):
    values = vmem.numpy()
    dark = values < boundary.hyperpolarizedThresholdMilliVolts
    if iteration in storedIterationSet:
        score = (0.5 * np.sqrt(((values[:, feature] - targetFeature) ** 2).mean(1))
                 + 0.5 * np.sqrt(((values[:, other] - targetOther) ** 2).mean(1)))
        changed = np.zeros(len(orderNames), dtype=int) if previousDark['value'] is None else (dark != previousDark['value']).sum(1)
        times.append(iteration)
        for index, name in enumerate(orderNames):
            frames[name].append([round(float(v), 1) for v in values[index]])
            scores[name].append(round(float(score[index]), 2))
            crossings[name].append(int(changed[index]))
        previousDark['value'] = dark.copy()


boundary.ringHoldBatchReplay(reference, ringValues, hold, numIterations, onIteration)

result = dict(hold=hold, numIterations=numIterations, stride=args.stride, times=times,
              target=[round(float(v), 1) for v in target], orders={})
for name in orderNames:
    best = summary['orders'][name]['best']
    result['orders'][name] = dict(
        order=int(name),
        coefficients=[round(float(v), 4) for v in coefficients[name]],
        trainedScore=round(float(best['score']), 2),
        trainedIteration=int(best['iteration']),
        vmem=frames[name], score=scores[name], crossings=crossings[name],
        bestStoredScore=round(float(min(scores[name])), 2),
        bestStoredTime=int(times[int(np.argmin(scores[name]))]),
        bestFrame=int(times.index(int(best['iteration']))))
    print(f"  order {name}: {len(best['coefficients'])} coefficients, trained score "
          f"{best['score']:.2f} at {best['iteration']}, best stored {result['orders'][name]['bestStoredScore']}", flush=True)

json.dump(result, open(outputPath, 'w'))
print('wrote', outputPath, os.path.getsize(outputPath) // 1024, 'KiB', flush=True)
