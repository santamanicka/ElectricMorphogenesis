"""Does the moment of measurement explain the code's lost grip? (PolyPatterning_Sim.md, Section 12).

A10 to A13 all read the ensemble at iteration 2,173, which is the trained code's own best moment and has no special
meaning for any other code. Two rivals are compared here on the same replay. Averaging each code's pattern over a
window removes the churn of single iterations. Taking each code at its own best moment removes the timing difference
between codes altogether, comparing them where each is closest to the face rather than where one particular code was.

For each of the three, the same measures as A10 and A13: how many interior modes a quadratic fit on the coefficients
explains half of, how much of the ensemble's variance belongs to a single order, and the held-out canonical
correlation of the best linear readout.

Writes data/boundaryHarmonicWhenMeasured<checkpoint>Hold<hold><target>.json (never overwriting).
"""
import argparse
import itertools
import json
import os

import numpy as np
import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--trainedRunPath', type=str, default='data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5/order3_restart08.npz')
parser.add_argument('--window', type=str, default='2000,2400')
parser.add_argument('--numCodes', type=int, default=1024)
parser.add_argument('--seed', type=int, default=17)
args = parser.parse_args()

run = dict(np.load(args.trainedRunPath))
outputPath = f"data/boundaryHarmonicWhenMeasured{int(run['referenceCheckpoint'])}Hold{int(run['holdIterations'])}{run['targetName']}.json"
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
reference = boundary.loadCheckpoint(int(run['referenceCheckpoint']))
hold, numIterations = int(run['holdIterations']), int(run['numIterations'])
trainedCode, trainedMoment = run['bestCoefficients'], int(run['bestIteration'])
windowStart, windowEnd = (int(value) for value in args.window.split(','))
target = np.asarray(run['target']).ravel()
featureMask = np.isin(np.arange(boundary.numCells), boundary.featureCellIndices)
angles = boundary.ringAngles(boundary.boundaryRingCells)
basis = np.cos(np.outer(angles, np.arange(len(trainedCode))))

generator = np.random.default_rng(args.seed)
codes = []
while len(codes) < args.numCodes:
    candidate = np.concatenate([generator.uniform(0.3, 1.3, 1), generator.uniform(-0.6, 0.6, len(trainedCode) - 1)])
    if (basis @ candidate).min() >= 0.02 and (basis @ candidate).max() <= 1.98:
        codes.append(candidate)
codes = np.array(codes)

gathered = dict(atMoment=None, windowSum=np.zeros((len(codes), boundary.numCells)), windowCount=0,
                bestScore=np.full(len(codes), np.inf), bestPattern=np.zeros((len(codes), boundary.numCells)),
                bestIteration=np.zeros(len(codes), dtype=int))


def onIteration(iteration, vmem):
    if iteration < hold:
        return
    values = vmem.numpy()
    if iteration == trainedMoment:
        gathered['atMoment'] = values.copy()
    if windowStart <= iteration <= windowEnd:
        gathered['windowSum'] += values
        gathered['windowCount'] += 1
    squared = (values - target) ** 2
    scores = 0.5 * np.sqrt(squared[:, featureMask].mean(1)) + 0.5 * np.sqrt(squared[:, ~featureMask].mean(1))
    better = scores < gathered['bestScore']
    gathered['bestScore'][better] = scores[better]
    gathered['bestPattern'][better] = values[better]
    gathered['bestIteration'][better] = iteration


print(f'{len(codes)} codes; replaying {numIterations} iterations', flush=True)
boundary.ringHoldBatchReplay(reference, codes @ basis.T, hold, numIterations, onIteration)

size = boundary.latticeRows - 2
interior = boundary.interiorCellIndices
modes = np.column_stack([np.outer(np.cos(np.pi * p * (np.arange(size) + 0.5) / size),
                                  np.cos(np.pi * q * (np.arange(size) + 0.5) / size)).ravel()
                         for p in range(size) for q in range(size)])
modes = modes / np.linalg.norm(modes, axis=0)
standardCodes = (codes - codes.mean(0)) / codes.std(0)
columns = [np.ones(len(codes))] + [standardCodes[:, k] for k in range(4)] + [standardCodes[:, k] ** 2 for k in range(4)]
columns += [standardCodes[:, a] * standardCodes[:, b] for a, b in itertools.combinations(range(4), 2)]
design = np.column_stack(columns)
folds = np.random.default_rng(1).permutation(len(codes)) % 5

result = dict(trainedMoment=trainedMoment, window=[windowStart, windowEnd], numCodes=len(codes), measured={})
for name, patterns in (('at the trained moment', gathered['atMoment']),
                       (f'averaged over {windowStart}-{windowEnd}', gathered['windowSum'] / gathered['windowCount']),
                       ('at each code\'s own best moment', gathered['bestPattern'])):
    amplitudes = np.array([modes.T @ (row[interior] - row[interior].mean()) for row in patterns])
    amplitudes = amplitudes[:, amplitudes.std(0) > 1e-9]
    variance = amplitudes.var(0)
    rSquared, owned = [], []
    for index in range(amplitudes.shape[1]):
        response = amplitudes[:, index]
        fit, *_ = np.linalg.lstsq(design, response, rcond=None)
        rSquared.append(1 - (response - design @ fit).var() / response.var())
        univariate = [float((fit[1 + k] * standardCodes[:, k] + fit[5 + k] * standardCodes[:, k] ** 2).var()) for k in range(4)]
        owned.append(rSquared[-1] >= 0.5 and max(univariate) / (sum(univariate) + 1e-12) >= 0.5)
    heldOut, _, _ = boundary.crossValidatedReadout(standardCodes, amplitudes, seed=1)
    owned = np.array(owned)
    result['measured'][name] = dict(predictableModes=int((np.array(rSquared) >= 0.5).sum()), numModes=int(len(rSquared)),
                                    meanRSquared=round(float(np.mean(rSquared)), 4),
                                    ownedModes=int(owned.sum()),
                                    ownedVarianceShare=round(float(variance[owned].sum() / variance.sum()), 4),
                                    heldOutCorrelation=round(heldOut, 4),
                                    participationRatio=round(boundary.participationRatio(variance / variance.sum()), 2))
    entry = result['measured'][name]
    print(f"{name}: {entry['predictableModes']} of {entry['numModes']} modes predictable, mean R2 {entry['meanRSquared']:.3f}, "
          f"{entry['ownedModes']} owned ({entry['ownedVarianceShare']:.3f} of variance), readout {entry['heldOutCorrelation']:.3f}", flush=True)
result['bestMoments'] = dict(median=int(np.median(gathered['bestIteration'])),
                             fifth=int(np.percentile(gathered['bestIteration'], 5)),
                             ninetyFifth=int(np.percentile(gathered['bestIteration'], 95)),
                             spanningTrainedMoment=int((np.abs(gathered['bestIteration'] - trainedMoment) <= 50).sum()))
print('own best moments: median', result['bestMoments']['median'], '5th-95th',
      result['bestMoments']['fifth'], '-', result['bestMoments']['ninetyFifth'], flush=True)
json.dump(result, open(outputPath, 'w'))
print('wrote', outputPath)
