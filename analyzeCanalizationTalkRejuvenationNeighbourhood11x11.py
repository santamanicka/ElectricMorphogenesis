"""Is the neighbourhood of the original code special on the aged tissue? (EXPLORATORY: not registered; it follows the observation that the order-4 face's best-overlap rejuvenation code lay within
0.06 of the original code, which itself reaches overlap 0.26 on the aged tissue.)

For each target, on the aged state at best moment + 300 (the search's own state, hold 301, window 301 to 2,999, overlap read at the iteration of lowest balanced-RMS score, as the figure captions do):
  - the original code itself,
  - `--nudges` copies of the original with Gaussian noise of each standard deviation in `--noise` added to every coefficient (the family's own ceiling applied when the ring values are made),
  - `--random` codes drawn uniformly from the allowed set of the family (the polytope of coefficients whose ring values stay within [0, ceiling]; hit-and-run, 500 steps from the Chebyshev centre).
Repeated over `--seeds` seeds. Reports, pooled over seeds, the median overlap, the highest overlap and the fraction at or above 0.6 and 0.7 for each group.

    python3 analyzeCanalizationTalkRejuvenationNeighbourhood11x11.py

Writes data/canalizationTalkRejuvenationNeighbourhood1888Hold301.json (never overwriting).
"""
import argparse
import json
import os

import numpy as np
import torch
from scipy.optimize import linprog

import boundaryCodeUtilities as boundary
from boundaryHarmonicStep import ringHoldBatchReplayFromState
from canalizationTalkCommon import RING_ANGLES, ringValuesOf

parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=int, default=3)
parser.add_argument('--random', type=int, default=128)
parser.add_argument('--nudges', type=int, default=64)
parser.add_argument('--noise', type=str, default='0.02,0.04')
parser.add_argument('--offset', type=int, default=300)
parser.add_argument('--outputPath', type=str, default='data/canalizationTalkRejuvenationNeighbourhood1888Hold301.json')
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')
ITERATIONS, WINDOW_START, HOLD = 3000, 301, 301
TARGETS = {'Stripe': boundary.centreStripeCellIndices, 'FaceOrder3': boundary.featureCellIndices, 'FaceOrder4': boundary.featureCellIndices}
noiseLevels = [float(value) for value in args.noise.split(',')]


def uniformCodes(generator, numCoefficients, ceiling, count, steps=500):
    """`count` coefficient vectors, each the end of a hit-and-run walk inside {c : 0 <= basis c <= ceiling}, started at the Chebyshev centre."""
    basis = np.cos(np.outer(RING_ANGLES, np.arange(numCoefficients)))
    constraints = np.vstack([basis, -basis])
    bounds = np.r_[np.full(len(basis), ceiling), np.zeros(len(basis))]
    norms = np.linalg.norm(constraints, axis=1)
    centre = linprog(np.r_[np.zeros(numCoefficients), -1.0], A_ub=np.c_[constraints, norms], b_ub=bounds, bounds=[(None, None)] * numCoefficients + [(0, None)]).x[:numCoefficients]
    codes = []
    for _ in range(count):
        point = centre.copy()
        for _ in range(steps):
            direction = generator.standard_normal(numCoefficients)
            direction /= np.linalg.norm(direction)
            rate, slack = constraints @ direction, bounds - constraints @ point
            high = np.min(slack[rate > 1e-12] / rate[rate > 1e-12])
            low = np.max(slack[rate < -1e-12] / rate[rate < -1e-12])
            point = point + generator.uniform(low, high) * direction
        codes.append(point)
    return np.array(codes)


def overlapsAt(label, state, codes):
    """Overlap (at the lowest-score iteration of the window) and that score, for every code, in one batched replay from the aged state."""
    cells = np.asarray(TARGETS[label])
    target = np.full(boundary.numCells, -5.0)
    target[cells] = -60.0
    targetTensor = torch.tensor(target)
    featureMask = torch.tensor(np.isin(np.arange(boundary.numCells), cells))
    ring = np.array([ringValuesOf(code, float(state['ceiling'])) for code in codes])
    best = dict(score=np.full(len(codes), np.inf), pattern=np.zeros((len(codes), boundary.numCells)))

    def onIteration(iteration, vmem):
        if iteration < WINDOW_START:
            return
        squared = (vmem - targetTensor) ** 2
        score = ((squared[:, featureMask].mean(1).sqrt() + squared[:, ~featureMask].mean(1).sqrt()) / 2).numpy()
        improved = score < best['score']
        best['score'][improved] = score[improved]
        best['pattern'][improved] = vmem.numpy()[improved]
    ringHoldBatchReplayFromState(ring, HOLD, ITERATIONS, onIteration, state['vmemVolts'], state['gpol'])
    return np.array([boundary.structuralIntersectionOverUnion(pattern, cells) for pattern in best['pattern']]), best['score']


def describe(overlaps):
    overlaps = np.asarray(overlaps)
    return dict(count=int(len(overlaps)), median=float(np.median(overlaps)), highest=float(overlaps.max()), fractionAtLeast060=float((overlaps >= 0.6).mean()), fractionAtLeast070=float((overlaps >= 0.7).mean()))


results = dict(note='EXPLORATORY; not registered. Overlap read at the lowest-score iteration of the window (301 to 2,999) after a 301-iteration hold on the aged state.', arguments=vars(args), targets={})
for label in TARGETS:
    state = dict(np.load(f'data/agedTissueState1888Hold301{label}After{args.offset}.npz', allow_pickle=True))
    original = np.asarray(state['coefficients'], float)
    groups = {'random': [], **{f'nudge{level:g}': [] for level in noiseLevels}}
    distances = {name: [] for name in groups}
    originalOverlap = None
    for seed in range(args.seeds):
        generator = np.random.default_rng(1000 + seed)
        codes, names = [original], ['original']
        random = uniformCodes(generator, len(original), float(state['ceiling']), args.random)
        codes += list(random); names += ['random'] * len(random)
        for level in noiseLevels:
            nudged = original + generator.normal(0.0, level, (args.nudges, len(original)))
            codes += list(nudged); names += [f'nudge{level:g}'] * len(nudged)
        overlaps, _ = overlapsAt(label, state, np.array(codes))
        originalOverlap = float(overlaps[0])
        for name, overlap, code in zip(names[1:], overlaps[1:], codes[1:]):
            groups[name].append(overlap)
            distances[name].append(float(np.abs(code - original).max()))
        print(f'{label} seed {seed}: original {overlaps[0]:.2f}; ' + '; '.join(f'{name} median {np.median([o for n, o in zip(names, overlaps) if n == name]):.2f}' for name in groups), flush=True)
    results['targets'][label] = dict(originalOverlap=originalOverlap, groups={name: dict(**describe(values), medianLargestCoefficientDistanceFromOriginal=float(np.median(distances[name]))) for name, values in groups.items()})
json.dump(results, open(args.outputPath, 'w'), indent=1)

print(f'\n{"target":11}{"group":>10}{"n":>6}{"median":>8}{"highest":>9}{">=0.6":>8}{">=0.7":>8}{"distance":>10}   (original on its own: overlap)')
for label, entry in results['targets'].items():
    for name, row in entry['groups'].items():
        print(f'{label:11}{name:>10}{row["count"]:6d}{row["median"]:8.2f}{row["highest"]:9.2f}{row["fractionAtLeast060"]:8.2f}{row["fractionAtLeast070"]:8.2f}{row["medianLargestCoefficientDistanceFromOriginal"]:10.2f}   ({entry["originalOverlap"]:.2f})')
print('wrote', args.outputPath)
