"""Are single spatial modes of the bulk pattern driven by single orders of the ring code? (PolyPatterning_Sim.md, Section 12).

Model-agnostic by design: the pattern is decomposed by a spatial Fourier (2D cosine) transform, the natural Fourier
basis for a grid that does not wrap around, and nothing about the tissue's mechanics enters the decomposition. An
ensemble of codes spanning the feasible region is simulated as in training, each pattern is decomposed, and every
mode's amplitude is regressed on the code's coefficients.

Fixed before the runs: a mode counts as belonging to an order when the ensemble regression explains at least half of
its variance (R2 >= 0.5) and one order accounts for at least half of the explained variance. The layer picture holds
descriptively if modes carrying a real share of the pattern's variance meet that test.

Writes data/boundaryHarmonicModes<checkpoint>Hold<hold><target>.json (never overwriting).
"""
import argparse
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--trainedRunPath', type=str, default='data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5/order3_restart08.npz')
parser.add_argument('--numCodes', type=int, default=256)
parser.add_argument('--seed', type=int, default=17)
args = parser.parse_args()

run = dict(np.load(args.trainedRunPath))
outputPath = f"data/boundaryHarmonicModes{int(run['referenceCheckpoint'])}Hold{int(run['holdIterations'])}{run['targetName']}.json"
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
reference = boundary.loadCheckpoint(int(run['referenceCheckpoint']))
hold, numIterations = int(run['holdIterations']), int(run['numIterations'])
code, trainedMoment = run['bestCoefficients'], int(run['bestIteration'])
target = np.asarray(run['target']).ravel()
angles = boundary.ringAngles(boundary.boundaryRingCells)
basis = np.cos(np.outer(angles, np.arange(len(code))))

# ------------------------------------------------------------------------------- spatial Fourier basis
size = boundary.latticeRows
modePairs = [(rowFrequency, columnFrequency) for rowFrequency in range(size) for columnFrequency in range(size)]


def spatialMode(rowFrequency, columnFrequency):
    rows = np.cos(np.pi * rowFrequency * (np.arange(size) + 0.5) / size)
    columns = np.cos(np.pi * columnFrequency * (np.arange(size) + 0.5) / size)
    vector = np.outer(rows, columns).ravel()
    return vector / np.linalg.norm(vector)


modes = np.column_stack([spatialMode(*pair) for pair in modePairs])
decompose = lambda pattern: modes.T @ (pattern - pattern.mean())

# ------------------------------------------------------------------------------------------- ensemble
generator = np.random.default_rng(args.seed)
rows = [code]
while len(rows) < args.numCodes:
    candidate = np.concatenate([generator.uniform(0.3, 1.3, 1), generator.uniform(-0.6, 0.6, len(code) - 1)])
    ringValues = basis @ candidate
    if ringValues.min() >= 0.02 and ringValues.max() <= 1.98:
        rows.append(candidate)
codes = np.array(rows)
ringValues = codes @ basis.T
print(f'{len(codes)} codes; ring values {ringValues.min():.3f} to {ringValues.max():.3f}', flush=True)

patterns = {}
moments = (hold - 1, trainedMoment)


def onIteration(iteration, vmem):
    if iteration in moments:
        patterns[iteration] = vmem.numpy().copy()


boundary.ringHoldBatchReplay(reference, ringValues, hold, numIterations, onIteration)

# ----------------------------------------------------------------------------------------- regression
result = dict(code=code.tolist(), trainedMoment=trainedMoment, numCodes=len(codes), modePairs=modePairs,
              criterion='R2 >= 0.5 and one order holding at least half of the explained variance',
              targetSpectrum=np.round(decompose(target) ** 2 / (decompose(target) ** 2).sum(), 5).tolist(), moments={})
design = np.column_stack([np.ones(len(codes)), (codes - codes.mean(0)) / codes.std(0)])
for moment in moments:
    amplitudes = np.array([decompose(pattern) for pattern in patterns[moment]])
    variance = amplitudes.var(0)
    share = variance / variance.sum()
    entries = []
    for index, pair in enumerate(modePairs):
        response = amplitudes[:, index]
        if response.std() < 1e-9:
            continue
        fit, *_ = np.linalg.lstsq(design, response, rcond=None)
        predicted = design @ fit
        rSquared = 1 - ((response - predicted).var() / response.var())
        contributions = np.abs(fit[1:]) / (np.abs(fit[1:]).sum() + 1e-12)
        entries.append(dict(mode=list(pair), varianceShare=round(float(share[index]), 5), rSquared=round(float(rSquared), 4),
                            orderShares=[round(float(value), 4) for value in contributions],
                            topOrder=int(np.argmax(contributions)), topShare=round(float(contributions.max()), 4),
                            belongsToOrder=bool(rSquared >= 0.5 and contributions.max() >= 0.5)))
    entries.sort(key=lambda entry: -entry['varianceShare'])
    result['moments'][str(moment)] = entries
    top = entries[:12]
    print(f'\nmoment {moment}: {sum(entry["belongsToOrder"] for entry in entries)} of {len(entries)} modes belong to a single order'
          f' ({sum(entry["varianceShare"] for entry in entries if entry["belongsToOrder"]):.3f} of the pattern variance)', flush=True)
    for entry in top:
        print('   mode %-7s variance %.3f  R2 %.2f  top order a%d holds %.2f  %s'
              % (str(tuple(entry['mode'])), entry['varianceShare'], entry['rSquared'], entry['topOrder'], entry['topShare'],
                 'belongs' if entry['belongsToOrder'] else ''), flush=True)

json.dump(result, open(outputPath, 'w'))
print('\nwrote', outputPath)
