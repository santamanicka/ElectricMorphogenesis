"""From each wall cell to the face, exactly, and how coarsely the wall can be described.

EXPLORATORY, not registered: added after the registered coarse-graining showed the relay cannot be closed on averaged
blocks, to ask what coarse-graining the books CAN do. The clamp writes only the ring, so the recursion
D(n+1) = Jbar(n) D(n) + src(n) splits by source cell into forty recursions, one per ring cell k, each driven only by that
cell's own injection src_k(n). Their sum is D, so each ring cell's contribution to the final selectivity gap is its own
term, and where that share is held at any moment is its own flux map. Nothing is fitted or averaged: every group below
is a sum of these exact terms.

Adds: the per-cell contribution to the gap (split into the conductance write and the Vmem side effect), the ring cell to
destination-group wiring at the end, the destination groups' holdings over time for every ring cell, and the mirror-
symmetric contiguous segmentation of the wall that keeps the most of the wiring's sign-and-route structure at each
number of segments (dynamic programming).

    python3 analyzeBoundaryHarmonicWallWiring11x11.py --jacobianPath <jacobians.npy> --relayPath <relayRegenerated.npz>
"""
import argparse
import json
import os
import time

import numpy as np

import boundaryCodeUtilities as boundary
import boundaryHarmonicCoarseGrain as coarse

parser = argparse.ArgumentParser()
parser.add_argument('--jacobianPath', type=str, required=True)
parser.add_argument('--relayPath', type=str, required=True)
parser.add_argument('--cachePath', type=str, default=None)
parser.add_argument('--outputPath', type=str, default='data/boundaryHarmonicWallWiring1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--maxDisplaySegments', type=int, default=14)
args = parser.parse_args()

RELEASE, TROUGH, PEAK = coarse.RELEASE, coarse.TROUGH, coarse.PEAK
n = coarse.NUM_CELLS
RING = boundary.boundaryRingCells
started = time.time()


def say(*parts):
    print(f'[{time.time() - started:6.0f}s]', *parts, flush=True)


relay, sweep, regenerated = coarse.loadRelay(args.jacobianPath, args.relayPath, args.cachePath, say=say)
difference, source, hold = relay.difference, relay.source, relay.hold
weights = relay.readouts[0]
gap = float(weights @ difference[PEAK])
adjoint = sweep['adjoint'][:, 0, :]                                   # (states, 2n) for the selectivity readout
reference = 1.0 / (len(boundary.featureCellIndices) * weights[n + boundary.featureCellIndices[0]])   # G_ref
total = np.stack([relay.total(step) for step in range(PEAK)])
say('gap', gap, 'G_ref', reference)
outsideRing = np.ones(2 * n, dtype=bool)
outsideRing[RING] = False
outsideRing[n + RING] = False
assert np.abs(source[:, outsideRing]).max() == 0.0, 'the clamp writes cells other than the ring'

# ------------------------------------------------------------------ destination groups
leftEye, rightEye, nose, mouth = (list(part) for part in boundary.featureParts)
featureSet = set(boundary.featureCellIndices.tolist())
background = [int(c) for c in boundary.interiorCellIndices if c not in featureSet]
groups = {'ring': [int(c) for c in RING], 'left eye': leftEye, 'right eye': rightEye, 'nose': nose, 'mouth': mouth,
          'background': background}
groupNames = list(groups)
member = np.zeros((n, len(groups)))
for column, cells in enumerate(groups.values()):
    member[cells, column] = 1.0
assert member.sum() == n and member.sum(1).max() == 1.0

# ------------------------------------------------------------------ one recursion per ring cell
sampled = np.array(list(range(0, PEAK, 5)) + [PEAK])
ringCount = len(RING)
holdings = np.zeros((ringCount, len(sampled), len(groups)))           # what each ring cell's share is held by, over time
cellFluxRelease = np.zeros((ringCount, n))
cellFluxTrough = np.zeros((ringCount, n))
cellFluxFinal = np.zeros((ringCount, n))
reconstructed = np.zeros_like(difference[:PEAK + 1])
for a, cell in enumerate(RING):
    mask = np.zeros(2 * n)
    mask[cell] = mask[n + cell] = 1.0
    y = np.zeros(2 * n)
    path = np.zeros((PEAK + 1, 2 * n))
    for step in range(PEAK):
        y = total[step] @ y
        if step < hold:
            y = y + source[step] * mask
        path[step + 1] = y
    reconstructed += path
    perCell = adjoint[:, :n] * path[:, :n] + adjoint[:, n:] * path[:, n:]          # (states, n)
    holdings[a] = perCell[sampled] @ member
    cellFluxRelease[a], cellFluxTrough[a], cellFluxFinal[a] = perCell[RELEASE], perCell[TROUGH], perCell[PEAK]
rebuildError = float(np.abs(reconstructed - difference[:PEAK + 1]).max(0) @ np.ones(2 * n) / np.abs(difference[:PEAK + 1]).max(0).sum())
readoutError = float(np.abs(reconstructed @ weights - difference[:PEAK + 1] @ weights).max() / abs(gap))
say(f'ring cells\' recursions sum back to D: worst readout error {readoutError:.2e}, state error {rebuildError:.2e}')

# ------------------------------------------------------------------ contributions and the wiring
contribution = holdings[:, -1, :].sum(1)                               # each ring cell's share of the final gap
injectionV = np.array([float(adjoint[1:hold + 1, cell] @ source[:, cell]) for cell in RING])
injectionG = np.array([float(adjoint[1:hold + 1, n + cell] @ source[:, n + cell]) for cell in RING])
say('contributions sum', contribution.sum(), 'from the adjoint', (injectionV + injectionG).sum(),
    '| conductance write', injectionG.sum(), 'Vmem side effect', injectionV.sum())
dent = difference[hold, n + RING] / reference                          # ring conductance, trained minus baseline, end of hold, G_ref
wiring = holdings[:, -1, 1:]                                           # destinations: left eye, right eye, nose, mouth, background
merged = np.stack([wiring[:, 0] + wiring[:, 1], wiring[:, 2], wiring[:, 3], wiring[:, 4]], axis=1)   # eyes, nose, mouth, background
mergedNames = ['eyes', 'nose', 'mouth', 'background']
singular = np.linalg.svd(wiring, compute_uv=False)
say('singular values of the 40 x 5 wiring', np.round(singular, 4), f'energy in the first {singular[0] ** 2 / (singular ** 2).sum():.3f}')

# ------------------------------------------------------------------ the mirror-symmetric segmentation of the wall
# the mirror axis runs through ring index 5 (top middle) and 25 (bottom middle); ring index a mirrors to (10 - a) mod 40
halfLength = 21
halfCells = [(5 + j, (5 - j) % ringCount) for j in range(halfLength)]
assert all(abs(merged[a] - merged[b]).max() < 1e-3 * np.abs(merged).max() for a, b in halfCells), 'the wiring is not mirror-symmetric'


def runValue(first, last):
    cells = sorted({c for j in range(first, last + 1) for c in halfCells[j]})
    return float(np.abs(merged[cells].sum(0)).sum()), cells


values = [[runValue(i, j)[0] if j >= i else 0.0 for j in range(halfLength)] for i in range(halfLength)]
whole = float(np.abs(merged).sum())
best = np.full((halfLength + 1, halfLength), -np.inf)                  # best[g][j]: g runs covering half cells 0..j
choice = np.zeros((halfLength + 1, halfLength), dtype=int)
for j in range(halfLength):
    best[1][j] = values[0][j]
for g in range(2, halfLength + 1):
    for j in range(g - 1, halfLength):
        for i in range(g - 1, j + 1):
            candidate = best[g - 1][i - 1] + values[i][j]
            if candidate > best[g][j]:
                best[g][j], choice[g][j] = candidate, i
segmentations = {}
for g in range(1, halfLength + 1):
    runs, j = [], halfLength - 1
    for level in range(g, 0, -1):
        i = 0 if level == 1 else choice[level][j]
        runs.append((i, j))
        j = i - 1
    runs.reverse()
    segments = [runValue(i, j)[1] for i, j in runs]
    count = 1 if g == 1 else 2 * g - 2
    segmentations[count] = dict(segments=segments, retained=float(best[g][halfLength - 1] / whole))
say('wall segments -> retained wiring:', {c: round(v['retained'], 3) for c, v in segmentations.items()})

# ------------------------------------------------------------------ what a chosen segmentation looks like
display = {}
for count, entry in segmentations.items():
    if count > args.maxDisplaySegments:
        continue
    segmentHoldings = np.stack([holdings[cells].sum(0) for cells in entry['segments']])       # (segments, states, groups)
    display[str(count)] = dict(
        retained=entry['retained'], segments=[[int(c) for c in cells] for cells in entry['segments']],
        segmentCells=[[int(RING[c]) for c in cells] for cells in entry['segments']],
        contribution=[round(float(contribution[cells].sum()), 6) for cells in entry['segments']],
        wiring=[np.round(wiring[cells].sum(0), 6).tolist() for cells in entry['segments']],
        holdings=[np.round(h, 6).tolist() for h in segmentHoldings],
        finalMap=[np.round(cellFluxFinal[cells].sum(0), 6).tolist() for cells in entry['segments']],
        troughMap=[np.round(cellFluxTrough[cells].sum(0), 6).tolist() for cells in entry['segments']],
        releaseMap=[np.round(cellFluxRelease[cells].sum(0), 6).tolist() for cells in entry['segments']])

result = dict(
    note='EXPLORATORY. An exact attribution of the selectivity gap to the ring cell whose injection produced it; not registered.',
    gap=gap, ringCells=[int(c) for c in RING], groupNames=groupNames, mergedNames=mergedNames, sampledStates=[int(s) for s in sampled],
    contribution=np.round(contribution, 6).tolist(), injectionConductance=np.round(injectionG, 6).tolist(),
    injectionVoltage=np.round(injectionV, 6).tolist(), dent=np.round(dent, 6).tolist(),
    wiring=np.round(wiring, 6).tolist(), singularValues=np.round(singular, 6).tolist(),
    columnTotals=dict(zip(groupNames[1:], np.round(wiring.sum(0), 6).tolist())),
    grossContribution=float(np.abs(contribution).sum()),
    rebuild=dict(readoutError=readoutError, stateError=rebuildError),
    retainedByCount={str(c): v['retained'] for c, v in segmentations.items()},
    segmentationByCount={str(c): [[int(x) for x in cells] for cells in v['segments']] for c, v in segmentations.items()},
    display=display)
json.dump(result, open(args.outputPath, 'w'))
say('wrote', args.outputPath, os.path.getsize(args.outputPath) // 1024, 'KB')
