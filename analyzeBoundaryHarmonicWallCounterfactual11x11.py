"""Does a wall segment do on its own what its exact share of the gap says?

EXPLORATORY. Criteria: data/boundaryHarmonicWallCounterfactualPredictions1888Hold301FaceMinus60Minus5.json, committed
before any run here. The wall wiring (analyzeBoundaryHarmonicWallWiring11x11.py) splits the final selectivity gap by the
ring cell whose injection produced it. That is exact accounting on the actual trajectory pair, not a counterfactual: the
model is nonlinear and every term is carried by the Jacobian averaged along the actual pair. Here the model itself is
run with only part of the ring held (the trained code's values, iterations 0-300, with the clamp's tissue-wide extra
Vmem update as ever) against the same extraUpdateOnly baseline as the relay, and the selectivity difference at state
index 1766 is read off directly.

    python3 analyzeBoundaryHarmonicWallCounterfactual11x11.py --wiringPath data/boundaryHarmonicWallWiring...json
"""
import argparse
import json
import time

import numpy as np
import torch

import boundaryCodeUtilities as boundary
from boundaryHarmonicStep import Step

parser = argparse.ArgumentParser()
parser.add_argument('--wiringPath', type=str, default='data/boundaryHarmonicWallWiring1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--predictionsPath', type=str, default='data/boundaryHarmonicWallCounterfactualPredictions1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--outputPath', type=str, default='data/boundaryHarmonicWallCounterfactual1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--segmentCount', type=str, default='8', help='key of the wall segmentation whose segments are held')
args = parser.parse_args()
torch.set_grad_enabled(False)
started = time.time()

wiring = json.load(open(args.wiringPath))
segments = wiring['display'][args.segmentCount]['segmentCells']          # cell indices, one list per (mirror-symmetric) segment
shares = wiring['display'][args.segmentCount]['contribution']
ring = [int(c) for c in wiring['ringCells']]
fullGap = float(wiring['gap'])

summary = json.load(open(args.summaryPath))
hold = int(summary['hold'])
winner = summary['orders']['3']['best']
coefficients = np.asarray(np.load(
    f"{summary['trainingDirs'][winner['round']]}/order3_restart{winner['restart']:02d}.npz")['bestCoefficients'], float)
ringValues = np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(len(coefficients)))) @ coefficients
step = Step(ringCode=ringValues)
n, Gref, T = step.numCells, step.Gref, 1766
features = np.array(sorted(set(boundary.featureCellIndices.tolist())))
interior = np.array(boundary.interiorCellIndices)
background = np.array([c for c in interior if c not in set(features.tolist())])
fullMask = step.ringMask.clone()


def selectivityAt(G):
    return float((G[features].mean() - G[background].mean()) / Gref)


leftEye, rightEye, nose, mouth = (list(part) for part in boundary.featureParts)
destinationGroups = {'eyes': leftEye + rightEye, 'nose': nose, 'mouth': mouth, 'background': [int(c) for c in background]}
finalConductance = {}


def groupShares(G, reference):
    """Each destination group's part of the selectivity difference: the face groups add their mean conductance change
    over the fourteen face cells, the background subtracts its mean change over its sixty-seven."""
    change = (G - reference) / Gref
    return {name: float(change[cells].sum() / len(features)) if name != 'background' else float(-change[cells].sum() / len(background))
            for name, cells in destinationGroups.items()}


def run(held, keepTrajectory=False):
    """Selectivity at state T when only the cells in `held` are held; held=None is the baseline (no ring held, but the
    tissue-wide re-solve every hold iteration, as in the relay). With keepTrajectory the conductance at every state is
    kept in lastTrajectory[0]."""
    mask = torch.zeros(n, dtype=torch.double)
    if held:
        mask[list(held)] = 1.0
    step.ringMask = mask
    V, G = step.initialVmem.clone(), step.initialGpol.clone()
    trajectory = [G.numpy().copy()] if keepTrajectory else None
    for m in range(T):
        V, G = step(V, G, bool(held) and m < hold, m < hold)
        if keepTrajectory:
            trajectory.append(G.numpy().copy())
    step.ringMask = fullMask
    lastConductance[0] = G.clone()
    if keepTrajectory:
        lastTrajectory[0] = np.array(trajectory)
    return selectivityAt(G)


lastConductance, lastTrajectory = [None], [None]
baseline = run(None, keepTrajectory=True)
baselineConductance = lastConductance[0]
baselineTrajectory = lastTrajectory[0]
runs = {}
groupOutcomes = {}


timeCourses = {}
sampled = list(range(0, T, 5)) + [T]


def outcome(name, held, keepTrajectory=False):
    runs[name] = run(held, keepTrajectory) - baseline
    groupOutcomes[name] = groupShares(lastConductance[0], baselineConductance)
    if keepTrajectory:                                    # the selectivity difference, and each group's part of it, over time
        trajectory = lastTrajectory[0]
        record = dict(selectivity=[], **{group: [] for group in destinationGroups})
        for state in sampled:
            parts = groupShares(torch.as_tensor(trajectory[state]), torch.as_tensor(baselineTrajectory[state]))
            for group, value in parts.items():
                record[group].append(round(value, 6))
            record['selectivity'].append(round(sum(parts.values()), 6))
        timeCourses[name] = record
    return runs[name]


print(f'[{time.time() - started:4.0f}s] baseline selectivity {baseline:+.5f}', flush=True)
full = outcome('fullRing', ring, keepTrajectory=True)
print(f'[{time.time() - started:4.0f}s] full ring: {full:+.6f} (relay difference {fullGap:+.6f})', flush=True)

names = [f'segment{k}' for k in range(len(segments))]
for name, cells in zip(names, segments):
    alone = outcome(f'{name}Alone', cells)
    rest = outcome(f'{name}Complement', [c for c in ring if c not in set(cells)])
    print(f'[{time.time() - started:4.0f}s] {name} ({len(cells)} cells): alone {alone:+.4f}, everything else {rest:+.4f}, '
          f'exact share {shares[names.index(name)]:+.4f}', flush=True)

# the two-segment split: upper wall = the segments above the mouth-height side rows; lower wall = the rest
rowsOf = [sorted({c // 11 for c in cells}) for cells in segments]
lowerFlags = [min(r) >= 7 or (len(r) == 1 and r[0] == 10) for r in rowsOf]
upperCells = sorted({c for cells, low in zip(segments, lowerFlags) if not low for c in cells})
lowerCells = sorted({c for cells, low in zip(segments, lowerFlags) if low for c in cells})
upperShare = sum(s for s, low in zip(shares, lowerFlags) if not low)
lowerShare = sum(s for s, low in zip(shares, lowerFlags) if low)
upperAlone = outcome('upperWallAlone', upperCells, keepTrajectory=True)
lowerAlone = outcome('lowerWallAlone', lowerCells, keepTrajectory=True)
print(f'[{time.time() - started:4.0f}s] upper wall alone {upperAlone:+.4f} (share {upperShare:+.4f}); '
      f'lower wall alone {lowerAlone:+.4f} (share {lowerShare:+.4f})', flush=True)

cells = {}
for cell in ring:
    cells[str(cell)] = outcome(f'cell{cell}Alone', [cell])
print(f'[{time.time() - started:4.0f}s] single cells done', flush=True)

# every way of splitting the ring into two contiguous halves of twenty cells: is the top/bottom split special?
halfSplits = []
for offset in range(len(ring)):
    arc = [ring[(offset + k) % len(ring)] for k in range(20)]
    rest = [c for c in ring if c not in set(arc)]
    held, other = run(arc) - baseline, run(rest) - baseline
    halfSplits.append(dict(offset=offset, firstCell=arc[0], heldAlone=held, otherHeldAlone=other,
                           additivity=(held + other - full) / full))
print(f'[{time.time() - started:4.0f}s] half splits done: additivity index (sum of the two halves alone minus the full ring, '
      f'over the full ring) ranges {min(h["additivity"] for h in halfSplits):+.2f} to {max(h["additivity"] for h in halfSplits):+.2f}', flush=True)

# ------------------------------------------------------------------ verdicts
alone = np.array([runs[f'{name}Alone'] for name in names])
complement = np.array([runs[f'{name}Complement'] for name in names])
shares = np.array(shares)
V1 = dict(fullRing=full, relayDifference=fullGap, error=abs(full - fullGap), holds=bool(abs(full - fullGap) <= 1e-6))
C1 = dict(lowerAlone=lowerAlone, upperAlone=upperAlone, holds=bool(lowerAlone > 0.1 and upperAlone < lowerAlone))
C2 = dict(signsAgree=int((np.sign(alone) == np.sign(shares)).sum()), of=len(names), holds=bool((np.sign(alone) == np.sign(shares)).sum() >= 4))
withinFactor = [bool(a != 0 and s != 0 and 0.5 <= a / s <= 2.0) for a, s in zip(alone, shares)]
C3 = dict(sumOfAlone=float(alone.sum()), full=fullGap, sumRatio=float(alone.sum() / fullGap), withinFactorTwo=int(sum(withinFactor)),
          holds=bool(abs(alone.sum() - fullGap) <= 0.5 * fullGap and sum(withinFactor) >= 3))
additivityGap = np.abs(complement - (full - alone))
C4 = dict(complementMinusExpected=np.round(complement - (full - alone), 4).tolist(), nonAdditive=int((additivityGap > 0.25 * fullGap).sum()),
          holds=bool((additivityGap > 0.25 * fullGap).sum() >= 3))
verdicts = dict(V1=V1, C1=C1, C2=C2, C3=C3, C4=C4)
for key, verdict in verdicts.items():
    print(f"{key}: {'holds' if verdict['holds'] else 'FAILS'}", {k: v for k, v in verdict.items() if k != 'holds'}, flush=True)
perCell = np.array([cells[str(c)] for c in ring])
cellShares = np.array(wiring['contribution'])
correlation = float(np.corrcoef(perCell, cellShares)[0, 1])
print(f'single ring cells held alone vs their exact shares: correlation {correlation:+.3f}; sum of the forty alone {perCell.sum():+.4f}', flush=True)

result = dict(
    note='EXPLORATORY; criteria in the predictions file, registered before any run.', predictions=json.load(open(args.predictionsPath)),
    baselineSelectivity=baseline, segmentCount=args.segmentCount, segments=segments, exactShare=shares.tolist(),
    heldAlone=alone.tolist(), heldEverythingElse=complement.tolist(), fullRing=full,
    upperWall=dict(cells=upperCells, share=upperShare, alone=upperAlone), lowerWall=dict(cells=lowerCells, share=lowerShare, alone=lowerAlone),
    singleCells=dict(ring=ring, alone=perCell.tolist(), exactShare=cellShares.tolist(), correlation=correlation, sumOfAlone=float(perCell.sum())),
    halfSplits=halfSplits, sampledStates=sampled, timeCourses=timeCourses, runs=runs, groupOutcomes={k: v for k, v in groupOutcomes.items() if not k.startswith('cell')}, verdicts=verdicts)
json.dump(result, open(args.outputPath, 'w'))
print('wrote', args.outputPath, flush=True)
