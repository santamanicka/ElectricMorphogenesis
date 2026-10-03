"""How robust is the pattern to jitter in the ring code, for the stripe and for the face under one protocol? EXPLORATORY: nothing here was predicted or registered.

The face's robustness test (analyzeBoundaryHarmonicCausalStoryRobustness11x11.py, section "jitter") multiplies each of the 40 held ring values by 1 + sigma * z,
z a fresh standard normal per cell, runs the hold and the release, and reads the interior at the code's own readout (state 2174 for the face code). This script
runs that same protocol, with the same generator, seed and draw order, for either target, so the face's stored numbers can be reproduced (the check) and the stripe code
can be put through the identical draws. Besides the fixed readout it keeps each draw's best overlap within +-window iterations of the readout, because a jittered code
may form the pattern a little earlier or later.

    python3 analyzeBoundaryHarmonicCodeJitter11x11.py --target face
    python3 analyzeBoundaryHarmonicCodeJitter11x11.py --target stripesInterior

Writes data/boundaryHarmonicCodeJitter<checkpoint>Hold<hold><target>.json (never overwriting).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import boundaryCodeUtilities as boundary
from boundaryHarmonicStep import Step

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, default='face', choices=('face', 'stripesInterior'))
parser.add_argument('--trainedRunPath', type=str, default=None, help='default: the face code (order 3, restart 08) or the stripe code (ceiling 2.0, order 2, restart 06)')
parser.add_argument('--readIteration', type=int, default=None, help='default: the trained code\'s best moment (2173 for the face, 504 for the stripe)')
parser.add_argument('--sigmas', type=str, default='0.0,0.01,0.03,0.10')
parser.add_argument('--draws', type=int, default=100)
parser.add_argument('--seed', type=int, default=20250926, help='the face robustness script\'s seed')
parser.add_argument('--window', type=int, default=100)
parser.add_argument('--holdIterations', type=int, default=301)
parser.add_argument('--outputPath', type=str, default=None)
args = parser.parse_args()
torch.set_grad_enabled(False)
started = time.time()

isFace = args.target == 'face'
targetName = 'FaceMinus60Minus5' if isFace else 'StripesInteriorMinus60Minus5'
args.trainedRunPath = args.trainedRunPath or ('data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5/order3_restart08.npz' if isFace
                                              else 'data/boundaryHarmonicTraining1888Hold301StripesInteriorMinus60Minus5Ceiling2/order2_restart06.npz')
args.outputPath = args.outputPath or f'data/boundaryHarmonicCodeJitter1888Hold{args.holdIterations}{targetName}.json'
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')

trained = np.load(args.trainedRunPath, allow_pickle=True)
coefficients = np.asarray(trained['bestCoefficients'], float)
readIteration = args.readIteration or int(trained['bestIteration'])
last = readIteration + 1                                                         # the face script reads state 2174 for its iteration 2173
ringValues = np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(len(coefficients)))) @ coefficients
step = Step(ringCode=ringValues)
n, Gref = step.numCells, step.Gref
ring = [int(c) for c in boundary.boundaryRingCells]
interior = np.array(boundary.interiorCellIndices)
targetCells = np.array(sorted(set((boundary.featureCellIndices if isFace else boundary.centreStripeCellIndices).tolist())))
strayCells = np.array([c for c in interior if c not in set(targetCells.tolist())])
parts = None
if isFace:
    leftEye, rightEye, nose, mouth = (list(part) for part in boundary.featureParts)
    parts = {'eyes': np.array(leftEye + rightEye), 'nose': np.array(nose), 'mouth': np.array(mouth)}
threshold = boundary.hyperpolarizedThresholdMilliVolts


def overlapOf(vMilli):
    return boundary.structuralIntersectionOverUnion(vMilli, None if isFace else boundary.centreStripeCellIndices)


def run(code):
    """The interior at the readout state and the best overlap within +-window of it, for a code held for the hold and then released."""
    step.ringMask = torch.zeros(n, dtype=torch.double)
    step.ringMask[step.ring] = 1.0
    step.ringCode = code
    V, G = step.initialVmem.clone(), step.initialGpol.clone()
    windowOverlaps = {}
    for m in range(last + args.window):
        V, G = step(V, G, m < args.holdIterations, m < args.holdIterations)
        if abs(m + 1 - last) <= args.window:
            windowOverlaps[m + 1] = overlapOf(V.numpy() * 1000.0)
        if m + 1 == last:
            atRead = V.numpy() * 1000.0
    dark = atRead < threshold
    record = dict(overlap=overlapOf(atRead), bestOverlapInWindow=max(windowOverlaps.values()), bestStateInWindow=int(max(windowOverlaps, key=windowOverlaps.get)) - 1,
                  targetDark=int(dark[targetCells].sum()), strayDark=int(dark[strayCells].sum()))
    if parts:
        record['dark'] = {name: int(dark[cells].sum()) for name, cells in parts.items()}
    return record


def say(*message):
    print(f'[{time.time() - started:5.0f}s]', *message, flush=True)


generator = np.random.default_rng(args.seed)
jitter = {}
for sigma in (float(s) for s in args.sigmas.split(',')):
    draws = []
    for _ in range(1 if sigma == 0.0 else args.draws):
        factors = 1.0 + sigma * generator.standard_normal(len(ring)) if sigma else np.ones(len(ring))
        code = torch.zeros(n, dtype=torch.double)
        code[step.ring] = torch.as_tensor(np.asarray(ringValues) * factors, dtype=torch.double) * Gref
        draws.append(run(code))
    jitter[str(sigma)] = draws
    say(f'sigma {sigma}: overlap >= 0.5 in {sum(d["overlap"] >= 0.5 for d in draws)} of {len(draws)}, >= 0.9 in {sum(d["overlap"] >= 0.9 for d in draws)}; '
        f'within +-{args.window}: >= 0.5 in {sum(d["bestOverlapInWindow"] >= 0.5 for d in draws)}, >= 0.9 in {sum(d["bestOverlapInWindow"] >= 0.9 for d in draws)}')

summary = {sigma: dict(draws=len(draws), medianOverlap=float(np.median([d['overlap'] for d in draws])),
                       atLeast0p5=int(sum(d['overlap'] >= 0.5 for d in draws)), atLeast0p9=int(sum(d['overlap'] >= 0.9 for d in draws)),
                       windowAtLeast0p5=int(sum(d['bestOverlapInWindow'] >= 0.5 for d in draws)), windowAtLeast0p9=int(sum(d['bestOverlapInWindow'] >= 0.9 for d in draws)),
                       medianTargetDark=float(np.median([d['targetDark'] for d in draws])), medianStrayDark=float(np.median([d['strayDark'] for d in draws])))
           for sigma, draws in jitter.items()}
json.dump(dict(note='EXPLORATORY: nothing here was predicted or registered. Same protocol, generator and draw order as the jitter section of analyzeBoundaryHarmonicCausalStoryRobustness11x11.py.',
               target=args.target, trainedRunPath=args.trainedRunPath, coefficients=coefficients.tolist(), readIteration=readIteration, readState=last, window=args.window,
               seed=args.seed, targetCells=len(targetCells), summary=summary, jitter=jitter), open(args.outputPath, 'w'))
say('wrote', args.outputPath)
