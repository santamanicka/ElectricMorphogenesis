"""Do two bumps on the ring's top row and two on its bottom row, none on the left or right column, make the two flank stripes dark?
Registered in data/boundaryHarmonicDoubleStripesBumpPredictions1888Hold301DoubleStripesInteriorMinus60Minus5.json and its Amendment 1
(an exploratory screen with registered predictions B1-B3, B1p-B3p, C1, C2).

Stage 1 holds hand-designed ring profiles on the 40 ring cells, four groups with one level each:
  T  the flank ends: top row and bottom row, columns 1-3 and 7-9 (12 cells)       M  the middle: rows 0 and 10, columns 4-6 (6 cells)
  K  the corners: rows 0 and 10, columns 0 and 10 (4 cells)                       S  the sides: columns 0 and 10, rows 1-9 (18 cells)
and scores each as in training (301-iteration hold, release, 3000 iterations, balanced RMS against the flank target, overlap with the 54 flank cells).
The families: the registered grid (T x M with M <= T x S x K in {S, T}), the pure two-bump profiles of Amendment 1 (M = K = S = B < T) and eight uniform
rings as controls. Stage 2 fits the best profiles by cosine series of order N (contiguous 0..N for N = 2..20, even-only {0, 2, .., N} for N = 2, 4, .., 20) by least
squares on the 40 ring values, clips to [0, 2], and simulates the fitted codes; their coefficients and the ringing they leave on the sides are stored.

    python3 simulateBoundaryHarmonicDoubleStripesBumps11x11.py

Writes data/boundaryHarmonicDoubleStripesBumps<checkpoint>Hold<hold><targetName>.json (never overwriting).
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
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--featureMilliVolts', type=float, default=-60.0)
parser.add_argument('--backgroundMilliVolts', type=float, default=-5.0)
parser.add_argument('--targetName', type=str, default='DoubleStripesInteriorMinus60Minus5')
parser.add_argument('--numPureProfiles', type=int, default=8, help='stage 2: the pure two-bump profiles with the highest overlap')
parser.add_argument('--numWiderProfiles', type=int, default=4, help='stage 2: the other profiles of the registered grid with the highest overlap')
parser.add_argument('--outputPath', type=str, default=None)
args = parser.parse_args()
outputPath = args.outputPath or f'data/boundaryHarmonicDoubleStripesBumps{args.referenceCheckpoint}Hold{args.holdIterations}{args.targetName}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
started = time.time()

# ------------------------------------------------------------------------------------------------ the ring's four groups
ring = np.array(boundary.boundaryRingCells)
rows, columns = ring // boundary.latticeCols, ring % boundary.latticeCols
lastRow, lastColumn = boundary.latticeRows - 1, boundary.latticeCols - 1
onTopOrBottom = (rows == 0) | (rows == lastRow)
groups = dict(T=onTopOrBottom & np.isin(columns, [1, 2, 3, 7, 8, 9]), M=onTopOrBottom & np.isin(columns, [4, 5, 6]),
              K=onTopOrBottom & np.isin(columns, [0, lastColumn]), S=((columns == 0) | (columns == lastColumn)) & ~onTopOrBottom)
assert [int(groups[g].sum()) for g in 'TMKS'] == [12, 6, 4, 18] and (sum(groups[g].astype(int) for g in 'TMKS') == 1).all(), 'the four groups must tile the 40 ring cells'


def profile(top, middle, corner, side):
    values = np.zeros(len(ring))
    for name, level in (('T', top), ('M', middle), ('K', corner), ('S', side)):
        values[groups[name]] = level
    return values


tops = [1.30, 1.40, 1.43, 1.435, 1.44, 1.445, 1.45, 1.455, 1.46, 1.47, 1.48, 1.50, 1.55, 1.60, 1.80, 2.00]
middles = [0.8, 1.1, 1.25, 1.35, 1.40, 1.43, 1.44, 1.46, 1.50]
sides = [0.8, 1.0, 1.2, 1.3, 1.4, 1.46, 1.50]
bases = [0.8, 1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.40, 1.43, 1.44, 1.46, 1.50]
uniform = [0.8, 1.0, 1.2, 1.3, 1.4, 1.46, 1.5, 2.0]

codes, index = [], {}            # one entry per distinct ring profile; its families and levels


def add(family, top, middle, corner, side):
    values = np.round(profile(top, middle, corner, side), 6)
    key = tuple(values)
    if key not in index:
        index[key] = len(codes)
        codes.append(dict(values=values, families=set(), T=top, M=middle, K=corner, S=side))
    codes[index[key]]['families'].add(family)


for top in tops:
    for middle in sorted(set([m for m in middles if m <= top] + [top])):
        for side in sides:
            for corner in sorted({side, top}):
                add('wide', top, middle, corner, side)
    for base in bases:
        if base < top:
            add('pure', top, base, base, base)
for level in uniform:
    add('uniform', level, level, level, level)
print(f'stage 1: {len(codes)} distinct profiles ({sum("pure" in c["families"] for c in codes)} pure two-bump, {sum("wide" in c["families"] for c in codes)} in the registered grid, '
      f'{sum("uniform" in c["families"] for c in codes)} uniform)', flush=True)

target = np.full(boundary.numCells, args.backgroundMilliVolts)
target[boundary.flankCellIndices] = args.featureMilliVolts
regions = dict(centre=boundary.centreStripeCellIndices)


def simulate(ringValues):
    return boundary.scoreRingCodesOverTime(args.referenceCheckpoint, ringValues, args.holdIterations, args.numIterations, target, boundary.flankCellIndices,
                                           batchSize=args.batchSize, regions=regions,
                                           onBatch=lambda done, results: print(f'[{time.time() - started:5.0f}s] {done}/{len(ringValues)}  best overlap so far {max(results["overlapAtBest"]):.3f}', flush=True))


stage1 = simulate(np.array([c['values'] for c in codes]))
overlap, score = np.array(stage1['overlapAtBest']), np.array(stage1['score'])


def topOf(family, count, exclude=()):
    members = [i for i, c in enumerate(codes) if family in c['families'] and i not in exclude]
    return sorted(members, key=lambda i: (-overlap[i], score[i]))[:count]


pureChosen = topOf('pure', args.numPureProfiles)
widerChosen = topOf('wide', args.numWiderProfiles, exclude=set(pureChosen) | {i for i, c in enumerate(codes) if 'pure' in c['families']})
chosen = pureChosen + widerChosen
print(f'stage 2: cosine projections of {len(chosen)} profiles (pure {pureChosen}, wider {widerChosen})', flush=True)

# ------------------------------------------------------------------------------------------------ stage 2: cosine projections
angles = boundary.ringAngles(boundary.boundaryRingCells)
projections = []
for i in chosen:
    values = codes[i]['values']
    for kind, orderSets in (('contiguous', [list(range(n + 1)) for n in range(2, 21)]), ('evenOnly', [list(range(0, n + 1, 2)) for n in range(2, 21, 2)])):
        for orders in orderSets:
            basis = np.cos(np.outer(angles, orders))
            coefficients = np.linalg.lstsq(basis, values, rcond=None)[0]
            fitted = np.clip(basis @ coefficients, 0, 2)
            projections.append(dict(profile=int(i), kind=kind, maxOrder=int(orders[-1]), orders=orders, coefficients=np.round(coefficients, 5).tolist(),
                                    fitMaxError=float(np.abs(fitted - values).max()), sideRipple=float(np.abs(fitted - values)[groups['S']].max()),
                                    ringMin=float(fitted.min()), ringMax=float(fitted.max()), values=fitted))
stage2 = simulate(np.array([p['values'] for p in projections]))

# ------------------------------------------------------------------------------------------------ write
keys = ('score', 'bestIteration', 'overlapAtBest', 'strayAtBest', 'featureDarkAtBest', 'maxOverlap', 'maxOverlapIteration', 'longestRunAbove0p85', 'centreDarkAtBest', 'centreMaxDark')
asRecords = lambda results: {key: [round(float(v), 5) if key in ('score', 'overlapAtBest', 'maxOverlap') else int(v) for v in results[key]] for key in keys}
result = dict(note='EXPLORATORY SCREEN WITH REGISTERED PREDICTIONS: data/boundaryHarmonicDoubleStripesBumpPredictions1888Hold301DoubleStripesInteriorMinus60Minus5.json and Amendment 1.',
              referenceCheckpoint=args.referenceCheckpoint, hold=args.holdIterations, numIterations=args.numIterations, targetName=args.targetName,
              groups={g: np.flatnonzero(m).tolist() for g, m in groups.items()}, ringCells=ring.tolist(),
              stage1=dict(families=[sorted(c['families']) for c in codes], T=[c['T'] for c in codes], M=[c['M'] for c in codes], K=[c['K'] for c in codes],
                          S=[c['S'] for c in codes], ringValues=[c['values'].tolist() for c in codes], **asRecords(stage1)),
              chosenProfiles=dict(pure=pureChosen, wider=widerChosen),
              stage2=dict(profile=[p['profile'] for p in projections], kind=[p['kind'] for p in projections], maxOrder=[p['maxOrder'] for p in projections],
                          orders=[p['orders'] for p in projections], coefficients=[p['coefficients'] for p in projections],
                          fitMaxError=[round(p['fitMaxError'], 5) for p in projections], sideRipple=[round(p['sideRipple'], 5) for p in projections],
                          ringMin=[round(p['ringMin'], 4) for p in projections], ringMax=[round(p['ringMax'], 4) for p in projections], **asRecords(stage2)))
json.dump(result, open(outputPath, 'w'), separators=(',', ':'))
print('wrote', outputPath, f'({time.time() - started:.0f}s)', flush=True)
