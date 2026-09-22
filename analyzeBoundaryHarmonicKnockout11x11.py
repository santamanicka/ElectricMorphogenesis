"""Causal test of the roles given to the orders of a trained ring code (PolyPatterning_Sim.md, Section 12).

The predictions, measures and code are fixed beforehand in --predictionsPath. Every subset of the orders 1 to N of the
trained code is knocked out (its coefficients set to 0; a0 never), and for each knockout --numControls random changes to
orders 1 to N move the ring's held values by the same RMS amount. Held values are clipped to [0, 2]. Every code is
simulated as in training and measured at three moments: the trained code's best moment, the code's own best moment
(lowest balanced RMS from the release on) and averaged over --window. If the retraining runs of --retrainedDir exist (one
order held at 0, the others retrained by learnBoundaryHarmonics11x11.py --orders), their best codes are measured the same way.

Writes data/boundaryHarmonicKnockout<checkpoint>Hold<hold><target>.json (never overwriting).
"""
import argparse
import glob
import itertools
import json
import os

import numpy as np
import torch

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--trainedRunPath', type=str, default='data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5/order3_restart08.npz')
parser.add_argument('--predictionsPath', type=str, default='data/boundaryHarmonicKnockoutPredictions1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--retrainedDir', type=str, default='data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5Knockout')
parser.add_argument('--numControls', type=int, default=20)
parser.add_argument('--window', type=str, default='1700,2400')
parser.add_argument('--seed', type=int, default=11)
args = parser.parse_args()

run = dict(np.load(args.trainedRunPath))
outputPath = f"data/boundaryHarmonicKnockout{int(run['referenceCheckpoint'])}Hold{int(run['holdIterations'])}{run['targetName']}.json"
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
reference = boundary.loadCheckpoint(int(run['referenceCheckpoint']))
hold, numIterations = int(run['holdIterations']), int(run['numIterations'])
windowStart, windowEnd = (int(value) for value in args.window.split(','))
code, trainedMoment = run['bestCoefficients'], int(run['bestIteration'])
orders = np.arange(len(code))
angles = boundary.ringAngles(boundary.boundaryRingCells)
basis = np.cos(np.outer(angles, orders))
target = run['target']
featureMask = np.isin(np.arange(boundary.numCells), boundary.featureCellIndices)
targetTensor = torch.tensor(target, dtype=torch.double)

# ------------------------------------------------------------------------------------------------ measures
cell = lambda row, col: row * boundary.latticeCols + col
eyes, nose, mouth = boundary.featureParts[:2], boundary.featureParts[2], boundary.featureParts[3]
cheekCells = np.array([cell(row, col) for row in range(1, 10) for col in (1, 2, 8, 9) if cell(row, col) not in set(boundary.featureCellIndices.tolist())])
bridgeCells = np.array([cell(7, col) for col in (4, 5, 6)])
spuriousCells = boundary.nonFeatureInteriorCellIndices
measureNames = ('eyeCoverage', 'mouthCoverage', 'noseCoverage', 'eyeMinusMouth', 'cheekDark', 'bridgeDark', 'mouthSeparate', 'spuriousDark')


def measures(vmem):
    """The pre-registered measures of one pattern (121 cells, mV)."""
    dark = vmem < boundary.hyperpolarizedThresholdMilliVolts
    eyeCoverage = float(np.mean([dark[part].mean() for part in eyes]))
    mouthCoverage, noseCoverage = float(dark[mouth].mean()), float(dark[nose].mean())
    labels, _ = boundary.interiorDarkComponents(vmem)
    mouthRegions = set(labels[mouth][dark[mouth]].tolist()) - {0}
    noseRegions = set(labels[nose].tolist()) - {0}
    return dict(eyeCoverage=eyeCoverage, mouthCoverage=mouthCoverage, noseCoverage=noseCoverage, eyeMinusMouth=eyeCoverage - mouthCoverage,
                cheekDark=int(dark[cheekCells].sum()), bridgeDark=int(dark[bridgeCells].sum()),
                mouthSeparate=int(bool(mouthRegions) and not (mouthRegions & noseRegions)), spuriousDark=int(dark[spuriousCells].sum()))


def simulate(coefficientRows):
    """Balanced RMS own best moment, patterns at the trained and own best moments, and measures averaged over the window."""
    ringValues = np.clip(np.asarray(coefficientRows) @ basis.T, 0, 2)
    numCodes = len(ringValues)
    best = dict(score=np.full(numCodes, np.inf), iteration=np.zeros(numCodes, dtype=int), vmem=np.zeros((numCodes, boundary.numCells)))
    atTrained, windowSums = {}, np.zeros((numCodes, len(measureNames)))

    def onIteration(iteration, vmem):
        if iteration < hold:
            return
        squared = (vmem - targetTensor) ** 2
        scores = (0.5 * squared[:, featureMask].mean(1).sqrt() + 0.5 * squared[:, ~featureMask].mean(1).sqrt()).numpy()
        values = vmem.numpy()
        better = scores < best['score']
        best['score'][better], best['iteration'][better], best['vmem'][better] = scores[better], iteration, values[better]
        if iteration == trainedMoment:
            atTrained.update(vmem=values.copy(), score=scores.copy())
        if windowStart <= iteration <= windowEnd:
            windowSums[:] += [[measures(pattern)[name] for name in measureNames] for pattern in values]
    boundary.ringHoldBatchReplay(reference, ringValues, hold, numIterations, onIteration)
    windowMeans = windowSums / (windowEnd - windowStart + 1)
    return [dict(ringValues=np.round(ringValues[index], 4).tolist(),
                 ownBest=dict(score=float(best['score'][index]), iteration=int(best['iteration'][index]), vmem=np.round(best['vmem'][index], 1).tolist(),
                              measures=measures(best['vmem'][index])),
                 trainedMoment=dict(score=float(atTrained['score'][index]), vmem=np.round(atTrained['vmem'][index], 1).tolist(), measures=measures(atTrained['vmem'][index])),
                 window=dict(zip(measureNames, np.round(windowMeans[index], 4).tolist())))
            for index in range(numCodes)]


# ------------------------------------------------------------------------------------------------ knockouts and controls
generator = np.random.default_rng(args.seed)
knockouts = [subset for size in range(1, len(code)) for subset in itertools.combinations(range(1, len(code)), size)]
rows, labels = [code], [('trained', ())]
ringRMS = lambda change: float(np.sqrt(np.mean((basis @ change) ** 2)))
for subset in knockouts:
    knocked = code.copy()
    knocked[list(subset)] = 0
    rows.append(knocked)
    labels.append(('knockout', subset))
    size = ringRMS(knocked - code)
    for _ in range(args.numControls):
        direction = np.zeros(len(code))
        direction[1:] = generator.standard_normal(len(code) - 1)
        rows.append(code + direction * size / ringRMS(direction))
        labels.append(('control', subset))
print(f'{len(knockouts)} knockouts, {len(rows)} codes in all; trained moment {trainedMoment}', flush=True)
results = simulate(np.array(rows))
result = dict(predictions=json.load(open(args.predictionsPath)), code=code.tolist(), trainedMoment=trainedMoment, window=[windowStart, windowEnd],
              measureNames=measureNames, trained=results[0], knockouts=[])
for subset in knockouts:
    indices = [index for index, label in enumerate(labels) if label[1] == subset]
    knock = [index for index in indices if labels[index][0] == 'knockout'][0]
    controls = [index for index in indices if labels[index][0] == 'control']
    entry = dict(orders=list(subset), knockout=results[knock], ringChangeRMS=ringRMS(np.array(rows[knock]) - code),
                 controls={moment: {name: [results[index][moment]['measures'][name] if moment != 'window' else results[index]['window'][name] for index in controls]
                                    for name in measureNames} for moment in ('ownBest', 'trainedMoment', 'window')},
                 controlScores=[results[index]['ownBest']['score'] for index in controls])
    result['knockouts'].append(entry)
    knockMeasures = results[knock]['ownBest']['measures']
    print(f"a{subset} out: own best {results[knock]['ownBest']['score']:.2f} mV at {results[knock]['ownBest']['iteration']}; "
          + ', '.join(f'{name} {knockMeasures[name]:.2f}' for name in measureNames), flush=True)

# ------------------------------------------------------------------------------------------------ retrained codes
result['retrained'] = []
for path in sorted(glob.glob(f'{args.retrainedDir}/orders*_restart*.npz')):
    retrained = dict(np.load(path))
    result['retrained'].append(dict(orders=retrained['orders'].tolist(), restart=int(retrained['restart']), score=float(retrained['bestScore']),
                                    coefficients=retrained['bestCoefficients'].tolist(), file=path))
bestRetrained = {}
for entry in result['retrained']:
    key = tuple(entry['orders'])
    if key not in bestRetrained or entry['score'] < bestRetrained[key]['score']:
        bestRetrained[key] = entry
if bestRetrained:
    full = [np.array([dict(zip(entry['orders'], entry['coefficients'])).get(order, 0.0) for order in orders]) for entry in bestRetrained.values()]
    for entry, measured in zip(bestRetrained.values(), simulate(np.array(full))):
        entry.update(best=True, measured=measured)
        print(f"retrained with orders {entry['orders']}: {entry['score']:.2f} mV; " + ', '.join(f"{name} {measured['ownBest']['measures'][name]:.2f}" for name in measureNames), flush=True)
json.dump(result, open(outputPath, 'w'), separators=(',', ':'))
print(f'wrote {outputPath}')
