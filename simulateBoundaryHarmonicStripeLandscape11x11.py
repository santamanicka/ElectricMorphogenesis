"""Where in the (a0, a2) plane of the ring code do the interior stripes form? EXPLORATORY: nothing here was predicted.

The stripes are left-right and top-bottom symmetric, and every best code found has no odd-order content, so the ring code
a0 + a2 cos(2 theta) (odd orders zero) is the natural plane to map. Each grid point is one code on the reference model, held for
the registered hold and released, scored exactly as in training (balanced RMS over the 27 centre-stripe cells and the other 94,
best iteration from the release to the end) and read at its best moment and over the whole run: the overlap of the dark cells
with the stripe, the number of stray dark cells, and the highest overlap at any iteration.

A point whose held values leave [0, ceiling] is clipped to it, as in training, and the clipped share is recorded (so a point's
ring values and the code's coefficients may differ at the edge of the plane).

    python3 simulateBoundaryHarmonicStripeLandscape11x11.py --ceiling 2 --a0Range=0,2,0.05 --a2Range=-1,1,0.05
    python3 simulateBoundaryHarmonicStripeLandscape11x11.py --levels --a0Range=1.40,1.60,0.005 --a2Range=0,2,0.05    # (T, S) plane

(a range that starts with a minus sign needs the = form, so that argparse does not read it as a flag)

Writes data/boundaryHarmonicStripeLandscape<checkpoint>Hold<hold><target>Order2Ceiling<ceiling>.json (never overwriting).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--referenceCheckpoint', type=int, default=1888)
parser.add_argument('--holdIterations', type=int, default=301)
parser.add_argument('--numIterations', type=int, default=3000)
parser.add_argument('--ceiling', type=float, default=2.0)
parser.add_argument('--a0Range', type=str, default='0,2,0.05', help='first,last,step of the dial a0')
parser.add_argument('--a2Range', type=str, default='-1,1,0.05', help='first,last,step of the oval a2')
parser.add_argument('--levels', action='store_true',
                    help='read the two ranges as the ring\'s level at the top and bottom, T = a0 + a2 (--a0Range), and at the sides, S = a0 - a2 '
                         '(--a2Range); the grid is then in (T, S) and the code is a0 = (T + S) / 2, a2 = (T - S) / 2')
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--featureMilliVolts', type=float, default=-60.0)
parser.add_argument('--backgroundMilliVolts', type=float, default=-5.0)
parser.add_argument('--targetName', type=str, default='StripesInteriorMinus60Minus5')
parser.add_argument('--outputPath', type=str, default=None)
args = parser.parse_args()
ceilingLabel = f'{args.ceiling:g}'.replace('.', 'p')
outputPath = args.outputPath or (f'data/boundaryHarmonicStripeLandscape{args.referenceCheckpoint}Hold{args.holdIterations}'
                                 f'{args.targetName}Order2Ceiling{ceilingLabel}.json')
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
started = time.time()

first, last, stepSize = (float(v) for v in args.a0Range.split(','))
dials = np.round(np.arange(first, last + stepSize / 2, stepSize), 6)
first, last, stepSize = (float(v) for v in args.a2Range.split(','))
ovals = np.round(np.arange(first, last + stepSize / 2, stepSize), 6)
angles = boundary.ringAngles(boundary.boundaryRingCells)
if args.levels:     # dials are the top/bottom level T and ovals the side level S; the code's own coefficients follow
    codes = np.array([((top + side) / 2, (top - side) / 2) for top in dials for side in ovals])
else:
    codes = np.array([(a0, a2) for a0 in dials for a2 in ovals])
rawRing = codes[:, [0]] + codes[:, [1]] * np.cos(2 * angles)[None, :]                    # a0 + a2 cos 2 theta, 40 values per code
clippedShare = ((rawRing < 0) | (rawRing > args.ceiling)).mean(1)
ringValues = np.clip(rawRing, 0, args.ceiling)
print(f'{len(codes)} codes ({len(dials)} dials x {len(ovals)} ovals), batches of {args.batchSize}', flush=True)

reference = boundary.loadCheckpoint(args.referenceCheckpoint)
target = np.full(boundary.numCells, args.backgroundMilliVolts)
target[boundary.centreStripeCellIndices] = args.featureMilliVolts
targetTensor = torch.tensor(target, dtype=torch.double)
stripeMask = torch.zeros(boundary.numCells, dtype=torch.bool)
stripeMask[boundary.centreStripeCellIndices] = True
interiorMask = torch.zeros(boundary.numCells, dtype=torch.bool)
interiorMask[boundary.interiorCellIndices] = True
numStripe = int(stripeMask.sum())


def balancedScores(vmem):
    squared = (vmem - targetTensor) ** 2
    return squared[:, stripeMask].mean(1).sqrt() * 0.5 + squared[:, ~stripeMask].mean(1).sqrt() * 0.5


records = {key: [] for key in ('score', 'bestIteration', 'overlapAtBest', 'strayAtBest', 'stripeDarkAtBest', 'maxOverlap', 'maxOverlapIteration',
                               'longestRunAbove0p85')}
for begin in range(0, len(codes), args.batchSize):
    batch = ringValues[begin:begin + args.batchSize]
    count = len(batch)
    best = dict(score=torch.full((count,), np.inf, dtype=torch.double), iteration=torch.zeros(count, dtype=torch.long),
                overlap=torch.zeros(count, dtype=torch.double), stray=torch.zeros(count, dtype=torch.long),
                stripeDark=torch.zeros(count, dtype=torch.long))
    maxOverlap = torch.zeros(count, dtype=torch.double)
    maxOverlapIteration = torch.zeros(count, dtype=torch.long)
    run = torch.zeros(count, dtype=torch.long)
    longest = torch.zeros(count, dtype=torch.long)

    def onIteration(iteration, vmem):
        if iteration < args.holdIterations:
            return
        dark = (vmem < boundary.hyperpolarizedThresholdMilliVolts) & interiorMask[None]
        stripeDark = (dark & stripeMask[None]).sum(1)
        stray = (dark & ~stripeMask[None]).sum(1)
        overlap = stripeDark.double() / (numStripe + stray).double()                 # = intersection / union, the union being target + strays
        scores = balancedScores(vmem)
        better = scores < best['score']
        for key, value in (('score', scores), ('iteration', torch.full((count,), iteration)), ('overlap', overlap), ('stray', stray), ('stripeDark', stripeDark)):
            best[key] = torch.where(better, value, best[key])
        higher = overlap > maxOverlap
        maxOverlap.copy_(torch.where(higher, overlap, maxOverlap))
        maxOverlapIteration.copy_(torch.where(higher, torch.full((count,), iteration), maxOverlapIteration))
        above = overlap >= 0.85
        run.copy_(torch.where(above, run + 1, torch.zeros_like(run)))
        longest.copy_(torch.maximum(longest, run))

    boundary.ringHoldBatchReplay(reference, batch, args.holdIterations, args.numIterations, onIteration)
    for key, value in (('score', best['score']), ('bestIteration', best['iteration']), ('overlapAtBest', best['overlap']), ('strayAtBest', best['stray']),
                       ('stripeDarkAtBest', best['stripeDark']), ('maxOverlap', maxOverlap), ('maxOverlapIteration', maxOverlapIteration),
                       ('longestRunAbove0p85', longest)):
        records[key].extend(value.tolist())
    print(f'[{time.time() - started:5.0f}s] {begin + count}/{len(codes)}  best overlap so far {max(records["overlapAtBest"]):.3f}, '
          f'highest at any iteration {max(records["maxOverlap"]):.3f}', flush=True)

result = dict(note='EXPLORATORY; no predictions registered.', referenceCheckpoint=args.referenceCheckpoint, hold=args.holdIterations,
              numIterations=args.numIterations, ceiling=args.ceiling, targetName=args.targetName, a0=dials.tolist(), a2=ovals.tolist(),
              gridIsLevels=bool(args.levels), codeA0=np.round(codes[:, 0], 6).tolist(), codeA2=np.round(codes[:, 1], 6).tolist(),
              clippedShare=np.round(clippedShare, 4).tolist(), ringMax=np.round(ringValues.max(1), 4).tolist(), ringMin=np.round(ringValues.min(1), 4).tolist(),
              **{key: [round(float(v), 5) for v in value] if key not in ('bestIteration', 'maxOverlapIteration', 'strayAtBest', 'stripeDarkAtBest',
                                                                           'longestRunAbove0p85') else [int(v) for v in value]
                 for key, value in records.items()})
json.dump(result, open(outputPath, 'w'), separators=(',', ':'))
print('wrote', outputPath, f'({time.time() - started:.0f}s)', flush=True)
