"""The two families of codes that form the stripe in the (T, S) levels map. EXPLORATORY: nothing here was predicted.

In the levels map (simulateBoundaryHarmonicStripeLandscape11x11.py --levels) every code of orders 0 to 2 is a top/bottom level T = a0 + a2 and a
side level S = a0 - a2. Some of the codes that form the stripe (overlap >= 0.9) score far worse than the search's best although their stripe is
exact. This replays those codes on the reference model, finds each one's best moment as in training, and records the mean Vmem of the 40 ring
cells, of the interior flank cells and of the 27 stripe cells at that moment, to show what the registered score is counting against them.

    python3 analyzeBoundaryHarmonicStripeLevelsFamilies11x11.py

Writes data/boundaryHarmonicStripeLevelsFamilies1888Hold301StripesInteriorMinus60Minus5Order2Ceiling2.json (never overwriting).
"""
import argparse
import json
import os

import numpy as np
import torch

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301StripesInteriorMinus60Minus5Order2Ceiling2'
parser = argparse.ArgumentParser()
parser.add_argument('--landscapePath', type=str, default=f'data/boundaryHarmonicStripeLandscape{SUFFIX}LevelsTopBottomVersusSides.json')
parser.add_argument('--outputPath', type=str, default=f'data/boundaryHarmonicStripeLevelsFamilies{SUFFIX}.json')
parser.add_argument('--minimumOverlap', type=float, default=0.9)
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')
landscape = json.load(open(args.landscapePath))
overlap = np.array(landscape['overlapAtBest']).reshape(-1)
codeA0, codeA2 = np.array(landscape['codeA0']).reshape(-1), np.array(landscape['codeA2']).reshape(-1)
formed = np.flatnonzero(overlap >= args.minimumOverlap)
longestRun = np.array(landscape['longestRunAbove0p85']).reshape(-1)
angles = boundary.ringAngles(boundary.boundaryRingCells)
ringValues = np.clip(codeA0[formed, None] + codeA2[formed, None] * np.cos(2 * angles)[None, :], 0, 2)
stripe = np.array(sorted(boundary.centreStripeCellIndices.tolist()))
ringCells = np.array(sorted(boundary.boundaryRingCells))
flankInteriorCells = np.array(sorted(set(boundary.flankCellIndices.tolist()) - set(ringCells.tolist())))
target = torch.full((boundary.numCells,), -5.0, dtype=torch.double)
target[torch.as_tensor(stripe)] = -60.0
stripeMask = torch.zeros(boundary.numCells, dtype=torch.bool)
stripeMask[torch.as_tensor(stripe)] = True
best = torch.full((len(formed),), float('inf'), dtype=torch.double)
bestIteration = torch.zeros(len(formed), dtype=torch.long)
atBest = torch.zeros(len(formed), boundary.numCells, dtype=torch.double)
HOLD, ITERATIONS = 301, 3000


def onIteration(iteration, vmem):
    if iteration < HOLD:
        return
    squared = (vmem - target) ** 2
    scores = squared[:, stripeMask].mean(1).sqrt() * 0.5 + squared[:, ~stripeMask].mean(1).sqrt() * 0.5
    better = scores < best
    best.copy_(torch.where(better, scores, best))
    bestIteration.copy_(torch.where(better, torch.full_like(bestIteration, iteration), bestIteration))
    atBest.copy_(torch.where(better[:, None], vmem, atBest))


boundary.ringHoldBatchReplay(boundary.loadCheckpoint(1888), ringValues, HOLD, ITERATIONS, onIteration)
atBest = atBest.numpy()
codes = []
for position, index in enumerate(formed):
    row = atBest[position]
    codes.append(dict(topBottomLevel=round(float(codeA0[index] + codeA2[index]), 4), sideLevel=round(float(codeA0[index] - codeA2[index]), 4),
                      score=round(float(best[position]), 3), bestIteration=int(bestIteration[position]),
                      longestRunAbove0p85=int(longestRun[index]),
                      ringMeanMilliVolts=round(float(row[ringCells].mean()), 2), ringMinMilliVolts=round(float(row[ringCells].min()), 2),
                      flankInteriorMeanMilliVolts=round(float(row[flankInteriorCells].mean()), 2), stripeMeanMilliVolts=round(float(row[stripe].mean()), 2)))
json.dump(dict(note='EXPLORATORY: the codes of the levels map that form the stripe, replayed to read the ring, the flanks and the stripe at each best moment.',
               landscape=args.landscapePath, minimumOverlap=args.minimumOverlap, codes=codes), open(args.outputPath, 'w'), separators=(',', ':'))
print('wrote', args.outputPath, f'({len(codes)} codes)')
for code in codes:
    print(code)
