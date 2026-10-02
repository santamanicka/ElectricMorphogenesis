"""Are the stripe's two ends written independently, the upper by the ring's top and the lower by its bottom? EXPLORATORY.

A ring code of orders 0 to 2, a0 + a1 cos(theta) + a2 cos(2 theta), is fixed by three numbers: the ring's level at the top
(theta = 0), T_top = a0 + a1 + a2, at the bottom (theta = 180 degrees), T_bottom = a0 - a1 + a2, and at the sides (theta = 90 degrees),
S = a0 - a2. The stripe code has T_top and T_bottom both near 1.49 and S near 1.21, and a1 near zero. This scan holds S where it is and
moves the two ends separately:

    a0 = (T_top + T_bottom) / 4 + S / 2,    a1 = (T_top - T_bottom) / 2,    a2 = (T_top + T_bottom) / 4 - S / 2.

Each code is replayed and read as in training (balanced RMS, best moment, structural overlap), plus how many cells of the centre
stripe's upper part (rows 1-4), middle row (row 5) and lower part (rows 6-9) are dark at the best moment and at most at any iteration.
If the ends are independent, a code with only the top above the bistable edge makes the upper part and not the lower, and the reverse.

    python3 simulateBoundaryHarmonicStripeEndsScan11x11.py --levelRange=1.43,1.55,0.0025 --sideLevel 1.2127

Writes data/boundaryHarmonicStripeEndsScan<checkpoint>Hold<hold><target>Side<S>.json (never overwriting).
"""
import argparse
import json
import os
import time

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--referenceCheckpoint', type=int, default=1888)
parser.add_argument('--holdIterations', type=int, default=301)
parser.add_argument('--numIterations', type=int, default=3000)
parser.add_argument('--levelRange', type=str, default='1.43,1.55,0.0025', help='first,last,step of T_top and of T_bottom (use the = form)')
parser.add_argument('--sideLevel', type=float, default=1.2127)
parser.add_argument('--randomPoints', type=int, default=0,
                    help='instead of the grid, this many (T_top, T_bottom) pairs drawn uniformly at random from the level range (and S uniformly from --sideRange if given)')
parser.add_argument('--sideRange', type=str, default=None, help='low,high of S for the random points (default: --sideLevel for every point)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--ceiling', type=float, default=2.0)
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--targetName', type=str, default='StripesInteriorMinus60Minus5')
parser.add_argument('--outputPath', type=str, default=None)
args = parser.parse_args()
sideLabel = f'{args.sideLevel:g}'.replace('.', 'p')
randomLabel = f'Random{args.randomPoints}Seed{args.seed}' if args.randomPoints else ''
outputPath = args.outputPath or f'data/boundaryHarmonicStripeEndsScan{args.referenceCheckpoint}Hold{args.holdIterations}{args.targetName}Side{sideLabel}{randomLabel}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
started = time.time()
first, last, stepSize = (float(v) for v in args.levelRange.split(','))
levels = np.round(np.arange(first, last + stepSize / 2, stepSize), 6)
angles = boundary.ringAngles(boundary.boundaryRingCells)
if args.randomPoints:
    generator = np.random.default_rng(args.seed)
    pairs = generator.uniform(first, last, size=(args.randomPoints, 2))
    sideLevels = (generator.uniform(*(float(v) for v in args.sideRange.split(',')), size=args.randomPoints) if args.sideRange else np.full(args.randomPoints, args.sideLevel))
else:
    pairs = np.array([(top, bottom) for top in levels for bottom in levels])
    sideLevels = np.full(len(pairs), args.sideLevel)
a0 = (pairs[:, 0] + pairs[:, 1]) / 4 + sideLevels / 2
a1 = (pairs[:, 0] - pairs[:, 1]) / 2
a2 = (pairs[:, 0] + pairs[:, 1]) / 4 - sideLevels / 2
rawRing = a0[:, None] + a1[:, None] * np.cos(angles)[None, :] + a2[:, None] * np.cos(2 * angles)[None, :]
clippedShare = ((rawRing < 0) | (rawRing > args.ceiling)).mean(1)
ringValues = np.clip(rawRing, 0, args.ceiling)
print(f'{len(pairs)} codes ' + (f'drawn at random (seed {args.seed})' if args.randomPoints else f'({len(levels)} x {len(levels)})') + f', side level {args.sideLevel}', flush=True)

stripe = np.array(sorted(boundary.centreStripeCellIndices.tolist()))
stripeRows = stripe // boundary.latticeCols
regions = dict(upper=stripe[stripeRows <= 4], middle=stripe[stripeRows == 5], lower=stripe[stripeRows >= 6])
target = np.full(boundary.numCells, -5.0)
target[stripe] = -60.0
results = boundary.scoreRingCodesOverTime(args.referenceCheckpoint, ringValues, args.holdIterations, args.numIterations, target, stripe, batchSize=args.batchSize, regions=regions,
                                          onBatch=lambda done, r: print(f'[{time.time() - started:5.0f}s] {done}/{len(pairs)}  best overlap so far {max(r["overlapAtBest"]):.3f}', flush=True))
result = dict(note='EXPLORATORY; no predictions registered.', referenceCheckpoint=args.referenceCheckpoint, hold=args.holdIterations, numIterations=args.numIterations, ceiling=args.ceiling,
              sideLevel=args.sideLevel, levels=levels.tolist(), pairs=np.round(pairs, 6).tolist(), sideLevels=np.round(sideLevels, 6).tolist(), seed=args.seed, randomPoints=args.randomPoints, regionSizes={name: int(len(cells)) for name, cells in regions.items()}, a0=np.round(a0, 6).tolist(),
              a1=np.round(a1, 6).tolist(), a2=np.round(a2, 6).tolist(), clippedShare=np.round(clippedShare, 4).tolist(),
              **{key: [round(float(v), 5) for v in value] if key in ('score', 'overlapAtBest', 'maxOverlap') else [int(v) for v in value] for key, value in results.items()})
json.dump(result, open(outputPath, 'w'), separators=(',', ':'))
print('wrote', outputPath, f'({time.time() - started:.0f}s)', flush=True)
