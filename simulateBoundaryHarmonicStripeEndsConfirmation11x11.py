"""The registered confirmatory draw: are the stripe's two ends written independently?

Criteria: data/boundaryHarmonicStripeEndsPredictions1888Hold301StripesInteriorMinus60Minus5.json, committed before this was run.
Draws 250 ring codes in each of four strata (both levels inside the window W, the ring's level at the top only, at the bottom only, neither)
with numpy default_rng(--seed), replays them with the same batch replay and score as the exploratory scan, and adds the stripe code itself
(the registration's V2). The levels, the window and the guard are the registration's:

    T_top = a0 + a1 + a2, T_bottom = a0 - a1 + a2, S = a0 - a2 (1.2127);  W = [1.485, 1.495];  outside means outside [1.4825, 1.4975];
    both levels drawn in the box [1.43, 1.55]; a stratum draws its T_top array and then its T_bottom array, in the order BOTH, TOP ONLY,
    BOTTOM ONLY, NEITHER. 'Uniform outside' maps one uniform draw on the two outside intervals' total length back onto them.

    python3 simulateBoundaryHarmonicStripeEndsConfirmation11x11.py

Writes data/boundaryHarmonicStripeEndsConfirmation1888Hold301StripesInteriorMinus60Minus5.json (never overwriting); scored by
analyzeBoundaryHarmonicStripeEndsConfirmation11x11.py.
"""
import argparse
import json
import os
import time

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=20261002)
parser.add_argument('--perStratum', type=int, default=250)
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--outputPath', type=str, default='data/boundaryHarmonicStripeEndsConfirmation1888Hold301StripesInteriorMinus60Minus5.json')
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')
started = time.time()
BOX = (1.43, 1.55)
WINDOW = (1.485, 1.495)
GUARD = (1.4825, 1.4975)
SIDE = 1.2127
generator = np.random.default_rng(args.seed)


def uniformInside(count):
    return generator.uniform(WINDOW[0], WINDOW[1], count)


def uniformOutside(count):
    below, above = GUARD[0] - BOX[0], BOX[1] - GUARD[1]
    u = generator.uniform(0, below + above, count)
    return np.where(u < below, BOX[0] + u, GUARD[1] + (u - below))


strata = {}
for name, topInside, bottomInside in (('both', True, True), ('topOnly', True, False), ('bottomOnly', False, True), ('neither', False, False)):
    top = uniformInside(args.perStratum) if topInside else uniformOutside(args.perStratum)
    bottom = uniformInside(args.perStratum) if bottomInside else uniformOutside(args.perStratum)
    strata[name] = np.column_stack([top, bottom])
pairs = np.vstack(list(strata.values()))
labels = np.repeat(list(strata), args.perStratum)
a0 = (pairs[:, 0] + pairs[:, 1]) / 4 + SIDE / 2
a1 = (pairs[:, 0] - pairs[:, 1]) / 2
a2 = (pairs[:, 0] + pairs[:, 1]) / 4 - SIDE / 2
angles = boundary.ringAngles(boundary.boundaryRingCells)
ring = a0[:, None] + a1[:, None] * np.cos(angles)[None, :] + a2[:, None] * np.cos(2 * angles)[None, :]
clipped = float(((ring < 0) | (ring > 2)).mean())
ringValues = np.clip(ring, 0, 2)
stripeCodeRing = np.clip(1.3500107271948254 + 0.0012086756612862083 * np.cos(angles) + 0.13727922481966065 * np.cos(2 * angles), 0, 2)       # the stripe code, registered V2
ringValues = np.vstack([ringValues, stripeCodeRing])
print(f'{len(pairs)} codes in four strata of {args.perStratum} (clipped share {clipped:.4f}) plus the stripe code', flush=True)

stripe = np.array(sorted(boundary.centreStripeCellIndices.tolist()))
rows = stripe // boundary.latticeCols
regions = dict(upper=stripe[rows <= 4], middle=stripe[rows == 5], lower=stripe[rows >= 6])
target = np.full(boundary.numCells, -5.0)
target[stripe] = -60.0
results = boundary.scoreRingCodesOverTime(1888, ringValues, 301, 3000, target, stripe, batchSize=args.batchSize, regions=regions,
                                          onBatch=lambda done, r: print(f'[{time.time() - started:5.0f}s] {done}/{len(ringValues)}', flush=True))
result = dict(note='CONFIRMATORY draw; criteria registered in boundaryHarmonicStripeEndsPredictions...json before this was run.', seed=args.seed, perStratum=args.perStratum,
              side=SIDE, window=WINDOW, guard=GUARD, box=BOX, strata=list(strata), labels=labels.tolist() + ['stripeCode'],
              topLevel=np.round(np.append(pairs[:, 0], 1.3500107271948254 + 0.0012086756612862083 + 0.13727922481966065), 6).tolist(),
              bottomLevel=np.round(np.append(pairs[:, 1], 1.3500107271948254 - 0.0012086756612862083 + 0.13727922481966065), 6).tolist(), clippedShare=clipped,
              **{key: [round(float(v), 5) for v in value] if key in ('score', 'overlapAtBest', 'maxOverlap') else [int(v) for v in value] for key, value in results.items()})
json.dump(result, open(args.outputPath, 'w'), separators=(',', ':'))
print('wrote', args.outputPath, f'({time.time() - started:.0f}s)', flush=True)
