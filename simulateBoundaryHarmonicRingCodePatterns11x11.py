"""Does sliding a ring-code order change the interior's pattern smoothly? EXPLORATORY: nothing here was predicted.

The relay's slider and grid (the Relay Loop page) are a handful of stops each, because each stop is an exact decomposition of a whole run. The
pattern itself needs only the run, so it can be sampled densely. This replays ring codes on the reference model, held for the registered hold and
released, exactly as in training, and keeps for every code the interior's Vmem (mV, whole numbers) at a fixed readout iteration (the stripe code's
own, 504) and at the code's own best moment, with the score, the best moment and the stripe's overlap and strays at both.

Four families of codes, the third and fourth varying two orders at once:
  wideSlider  each order alone, over its whole allowed range (a0 in [0, 2], a1 and a2 in [-1, 1], --wideStops stops), the others at their trained values
  zoomSlider  each order alone, the trained value plus or minus --zoomHalfWidth in steps of --zoomStep: the window the stripe forms in is a few
              thousandths wide, so this is where a change of pattern shows
  mapA0A2     the plane a0 + a2 cos 2 theta (odd orders zero), the coarse map
  mapTopSide  the same plane read as the ring's level at the top and bottom, T = a0 + a2, and at the sides, S = a0 - a2, zoomed on the stripe's window

A held value outside [0, ceiling] is clipped to it, as in training; the clipped share of the ring is recorded.

    python3 simulateBoundaryHarmonicRingCodePatterns11x11.py            # the stripe code (default), about 4,000 codes
    python3 simulateBoundaryHarmonicRingCodePatterns11x11.py --target face --families wideSlider,zoomSlider --ceiling 2     # the face, for comparison

With --target face the same replays are read against the face (the eyes, nose and mouth, scored as in training), the orders are the face code's four,
and the readout is the face code's own best moment; the maps (which need three orders) are left out.

Writes data/boundaryHarmonicRingCodePatterns<checkpoint>Hold<hold><target>.json (never overwriting).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301StripesInteriorMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--referenceCheckpoint', type=int, default=1888)
parser.add_argument('--holdIterations', type=int, default=301)
parser.add_argument('--numIterations', type=int, default=3000)
parser.add_argument('--ceiling', type=float, default=2.0)
parser.add_argument('--target', type=str, default='stripesInterior', choices=('stripesInterior', 'face'))
parser.add_argument('--trainedRunPath', type=str, default=None, help='default: the stripe code, or the face code (order 3, restart 08)')
parser.add_argument('--readIteration', type=int, default=None, help='the recorded iteration every code is also read at (default: the trained code\'s own best moment, 504 for the stripe)')
parser.add_argument('--wideStops', type=int, default=201)
parser.add_argument('--zoomHalfWidth', type=float, default=0.06)
parser.add_argument('--zoomStep', type=float, default=0.001)
parser.add_argument('--a0Map', type=str, default='0.5,2.0,0.05', help='first,last,step of a0 in the coarse map')
parser.add_argument('--a2Map', type=str, default='=-0.6,0.6,0.05', help='first,last,step of a2 in the coarse map (a leading = lets it start with a minus)')
parser.add_argument('--topMap', type=str, default='1.44,1.54,0.002', help='first,last,step of T in the zoomed map')
parser.add_argument('--sideMap', type=str, default='1.0,1.45,0.01', help='first,last,step of S in the zoomed map')
parser.add_argument('--families', type=str, default='wideSlider,zoomSlider,mapA0A2,mapTopSide')
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--featureMilliVolts', type=float, default=-60.0)
parser.add_argument('--backgroundMilliVolts', type=float, default=-5.0)
parser.add_argument('--targetName', type=str, default=None)
parser.add_argument('--outputPath', type=str, default=None)
args = parser.parse_args()
args.trainedRunPath = args.trainedRunPath or (f'data/boundaryHarmonicTraining{SUFFIX}Ceiling2/order2_restart06.npz' if args.target == 'stripesInterior'
                                              else 'data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5/order3_restart08.npz')
args.targetName = args.targetName or ('StripesInteriorMinus60Minus5' if args.target == 'stripesInterior' else 'FaceMinus60Minus5')
outputPath = args.outputPath or f'data/boundaryHarmonicRingCodePatterns{args.referenceCheckpoint}Hold{args.holdIterations}{args.targetName}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
started = time.time()

run = np.load(args.trainedRunPath)
trained = np.asarray(run['bestCoefficients'], float)
boxLow, boxHigh = np.asarray(run['coefficientLowest'], float), np.asarray(run['coefficientHighest'], float)
NUM_ORDERS = len(trained)
args.readIteration = args.readIteration or int(run['bestIteration'])
angles = boundary.ringAngles(boundary.boundaryRingCells)
basis = np.cos(np.outer(angles, np.arange(len(trained))))


def rangeOf(spec):
    first, last, step = (float(v) for v in spec.lstrip('=').split(','))
    return np.round(np.arange(first, last + step / 2, step), 6)


families = {}
if 'wideSlider' in args.families:
    for order in range(NUM_ORDERS):
        for value in np.linspace(boxLow[order], boxHigh[order], args.wideStops):
            c = trained.copy(); c[order] = value
            families.setdefault('wideSlider', []).append((order, c))
if 'zoomSlider' in args.families:
    for order in range(NUM_ORDERS):
        for value in np.round(trained[order] + np.arange(-args.zoomHalfWidth, args.zoomHalfWidth + args.zoomStep / 2, args.zoomStep), 6):
            c = trained.copy(); c[order] = value
            families.setdefault('zoomSlider', []).append((order, c))
mapAxes = {}
if ('mapA0A2' in args.families or 'mapTopSide' in args.families) and NUM_ORDERS != 3:
    raise SystemExit('the maps vary a0 and a2 with a1 = 0 and need a three-order code')
if 'mapA0A2' in args.families:
    mapAxes['mapA0A2'] = (rangeOf(args.a0Map), rangeOf(args.a2Map))
    families['mapA0A2'] = [(-1, np.array([a0, 0.0, a2])) for a0 in mapAxes['mapA0A2'][0] for a2 in mapAxes['mapA0A2'][1]]
if 'mapTopSide' in args.families:
    mapAxes['mapTopSide'] = (rangeOf(args.topMap), rangeOf(args.sideMap))
    families['mapTopSide'] = [(-1, np.array([(t + s) / 2, 0.0, (t - s) / 2])) for t in mapAxes['mapTopSide'][0] for s in mapAxes['mapTopSide'][1]]
allCodes = [(name, order, c) for name, rows in families.items() for order, c in rows]
print(f'{len(allCodes)} codes: ' + ', '.join(f'{name} {len(rows)}' for name, rows in families.items()) + f'; batches of {args.batchSize}', flush=True)

rawRing = np.array([basis @ c for _, _, c in allCodes])
clippedShare = ((rawRing < 0) | (rawRing > args.ceiling)).mean(1)
ringValues = np.clip(rawRing, 0, args.ceiling)

reference = boundary.loadCheckpoint(args.referenceCheckpoint)
featureCells = boundary.centreStripeCellIndices if args.target == 'stripesInterior' else boundary.featureCellIndices
target = np.full(boundary.numCells, args.backgroundMilliVolts)
target[featureCells] = args.featureMilliVolts
targetTensor = torch.tensor(target, dtype=torch.double)
stripeMask = torch.zeros(boundary.numCells, dtype=torch.bool)                     # the target's feature cells (the stripe's 27, or the face's 14)
stripeMask[featureCells] = True
interiorMask = torch.zeros(boundary.numCells, dtype=torch.bool)
interiorMask[boundary.interiorCellIndices] = True
numStripe = int(stripeMask.sum())


def read(vmem):
    """(balanced score, overlap, strays) of each row of vmem, as in training."""
    squared = (vmem - targetTensor) ** 2
    scores = squared[:, stripeMask].mean(1).sqrt() * 0.5 + squared[:, ~stripeMask].mean(1).sqrt() * 0.5
    dark = (vmem < boundary.hyperpolarizedThresholdMilliVolts) & interiorMask[None]
    stripeDark, stray = (dark & stripeMask[None]).sum(1), (dark & ~stripeMask[None]).sum(1)
    return scores, stripeDark.double() / (numStripe + stray).double(), stray


columns = {key: [] for key in ('score', 'bestIteration', 'overlapAtBest', 'strayAtBest', 'vmemAtBest', 'scoreAtRead', 'overlapAtRead', 'strayAtRead', 'vmemAtRead')}
for begin in range(0, len(allCodes), args.batchSize):
    batch = ringValues[begin:begin + args.batchSize]
    count = len(batch)
    best = dict(score=torch.full((count,), np.inf, dtype=torch.double), iteration=torch.zeros(count, dtype=torch.long),
                overlap=torch.zeros(count, dtype=torch.double), stray=torch.zeros(count, dtype=torch.long),
                vmem=torch.zeros(count, boundary.numCells, dtype=torch.double))
    atRead = {}

    def onIteration(iteration, vmem):
        if iteration == args.readIteration:
            atRead['vmem'] = vmem.clone()
            atRead['score'], atRead['overlap'], atRead['stray'] = read(vmem)
        if iteration < args.holdIterations:
            return
        scores, overlap, stray = read(vmem)
        better = scores < best['score']
        best['vmem'] = torch.where(better[:, None], vmem, best['vmem'])
        for key, value in (('score', scores), ('iteration', torch.full((count,), iteration)), ('overlap', overlap), ('stray', stray)):
            best[key] = torch.where(better, value, best[key])

    boundary.ringHoldBatchReplay(reference, batch, args.holdIterations, args.numIterations, onIteration)
    whole = lambda t: [[int(round(v)) for v in row] for row in t.tolist()]
    for key, value in (('score', best['score']), ('bestIteration', best['iteration']), ('overlapAtBest', best['overlap']), ('strayAtBest', best['stray']),
                       ('scoreAtRead', atRead['score']), ('overlapAtRead', atRead['overlap']), ('strayAtRead', atRead['stray'])):
        columns[key].extend(value.tolist())
    columns['vmemAtBest'].extend(whole(best['vmem']))
    columns['vmemAtRead'].extend(whole(atRead['vmem']))
    print(f'[{time.time() - started:5.0f}s] {begin + count}/{len(allCodes)}', flush=True)

result = dict(note='EXPLORATORY; no predictions registered.', referenceCheckpoint=args.referenceCheckpoint, hold=args.holdIterations,
              numIterations=args.numIterations, ceiling=args.ceiling, targetName=args.targetName, target=args.target, readIteration=args.readIteration,
              featureCells=[int(c) for c in featureCells],
              trainedCoefficients=trained.tolist(), coefficientLowest=boxLow.tolist(), coefficientHighest=boxHigh.tolist(), families={})
at = 0
for name, rows in families.items():
    n = len(rows)
    block = dict(order=[int(o) for o, _ in rows], coefficients=[[round(float(x), 6) for x in c] for _, c in rows],
                 clippedShare=[round(float(v), 4) for v in clippedShare[at:at + n]],
                 ringMax=[round(float(v), 4) for v in ringValues[at:at + n].max(1)], ringMin=[round(float(v), 4) for v in ringValues[at:at + n].min(1)])
    for key, values in columns.items():
        part = values[at:at + n]
        block[key] = part if key.startswith('vmem') else [round(float(v), 5) if 'overlap' in key.lower() or 'score' in key.lower() else int(v) for v in part]
    if name in mapAxes:
        block['axes'] = [mapAxes[name][0].tolist(), mapAxes[name][1].tolist()]
    result['families'][name] = block
    at += n
json.dump(result, open(outputPath, 'w'), separators=(',', ':'))
print('wrote', outputPath, f'({os.path.getsize(outputPath) / 1e6:.1f} MB, {time.time() - started:.0f}s)', flush=True)
