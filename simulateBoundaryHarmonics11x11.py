"""Synthetic boundary codes on one trained model: what does each circular-harmonic order of the boundary do
to the interior pattern? (PolyPatterning_Sim.md, Section 12)

The reference checkpoint's parameters, initial state, hold and run length are kept; only its boundary code is
replaced, on all 40 outer-ring cells. theta is each ring cell's angle about the lattice centre, measured
clockwise from straight up. Every run is read out as the per-cell mean Vmem over the last 1000 iterations.

Clamp values are G_pol / G_ref, whose physical range is [0, 2] (the conductance equation's limits; resting value
1). --sampling physical (the default) keeps every value inside that range without clipping. --sampling legacy
reproduces the first runs, which were sampled for the oscillatory-signal range [-1, 1] and so held some cells at
negative conductance (see boundaryCodeUtilities.py).

  --experiment firstOrderPairs   paired runs DC + G cos(theta - phi) and their G = 0 twins, phi ~ U(0, 2 pi).
                                   physical: DC ~ U(0.01, 1.99), then G log-uniform on [0.01, min(0.5, DC, 2 - DC)],
                                             so the whole code stays within [0, 2]
                                   legacy:   DC ~ U(-0.1, 0.65), G log-uniform on [0.01, 0.5], clipped to [-1, 1]
                                 plus a rotation grid at the reference model's own dial: G in {0.02, 0.05, 0.1, 0.2, 0.4}
                                 x phi every 45 degrees, and one G = 0 run. Under physical sampling each strength is
                                 capped at the largest value that stays in range at that dial (0.4 becomes 0.38 at 0.381).
  --experiment harmonicGrid      the dial alone at DC in {0, 0.2, reference DC, 0.55}; single harmonics
                                 reference DC + A cos(k theta) for k in {1,2,3,4,5,6,8,10} and A in {0.1,0.2,0.4};
                                 the k = 2 and k = 4 harmonics at A = 0.4 rotated by half a period; and the
                                 reference model's own trained code. Values outside the sampling scheme's range are
                                 clipped and the clipped fraction recorded; at the reference dial A = 0.4 dips below 0.
"""
import argparse
import time

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--experiment', choices=['harmonicGrid', 'firstOrderPairs'], required=True)
parser.add_argument('--sampling', choices=['physical', 'legacy'], default='physical')
parser.add_argument('--referenceCheckpoint', type=int, default=1888)
parser.add_argument('--numPairs', type=int, default=150)
parser.add_argument('--randomSeed', type=int, default=1)
parser.add_argument('--skipRotationGrid', action='store_true')
parser.add_argument('--outputPath', type=str, default=None)
args = parser.parse_args()

lowestValue, highestValue = (0.0, 2.0) if args.sampling == 'physical' else (-1.0, 1.0)
minimumStrength, maximumStrength = 0.01, 0.5
reference = boundary.loadCheckpoint(args.referenceCheckpoint)
holdIterations = int(reference['clampParameters']['clampEndIter']) + 1
referenceDialLevel = float(reference['clampParameters']['clampValues'][0].numpy().mean())
angles = boundary.ringAngles(boundary.boundaryRingCells)
suffix = '' if args.sampling == 'physical' else 'Legacy'
startTime = time.time()


def runRingCode(ringValues):
    clipped = float(np.mean((ringValues < lowestValue) | (ringValues > highestValue)))
    ringValues = np.clip(ringValues, lowestValue, highestValue)
    pattern = boundary.lateWindowMean(reference, boundary.ringClamp(reference, ringValues, holdIterations))
    return pattern, ringValues, clipped


if args.experiment == 'harmonicGrid':
    outputPath = args.outputPath or f'data/boundaryHarmonicGrid{suffix}{args.referenceCheckpoint}.npz'
    specifications = [(dialLevel, 0.0, 0, 0.0) for dialLevel in (0.0, 0.2, referenceDialLevel, 0.55)]
    specifications += [(referenceDialLevel, amplitude, order, 0.0)
                       for order in (1, 2, 3, 4, 5, 6, 8, 10) for amplitude in (0.1, 0.2, 0.4)]
    specifications += [(referenceDialLevel, 0.4, order, np.pi / (2 * order)) for order in (2, 4)]
    results = []
    for dialLevel, amplitude, order, phase in specifications:
        results.append(runRingCode(dialLevel + amplitude * np.cos(order * angles + phase)))
    trainedPattern = boundary.lateWindowMean(reference, reference['clampParameters'])
    np.savez_compressed(outputPath, dialLevel=np.array([spec[0] for spec in specifications]),
                        amplitude=np.array([spec[1] for spec in specifications]),
                        order=np.array([spec[2] for spec in specifications]),
                        phase=np.array([spec[3] for spec in specifications]),
                        patterns=np.stack([result[0] for result in results]),
                        ringValues=np.stack([result[1] for result in results]),
                        clippedFraction=np.array([result[2] for result in results]),
                        trainedCodePattern=trainedPattern, referenceDialLevel=referenceDialLevel,
                        referenceCheckpoint=args.referenceCheckpoint, ringCells=boundary.boundaryRingCells,
                        ringAngles=angles, sampling=args.sampling)

else:
    outputPath = args.outputPath or f'data/boundaryFirstOrderPairs{suffix}{args.referenceCheckpoint}.npz'
    randomGenerator = np.random.default_rng(args.randomSeed)
    pairs = dict(dialLevel=[], gradientStrength=[], gradientDirection=[], clippedFraction=[],
                 twinPatterns=[], gradientPatterns=[], codeMinimum=[], codeMaximum=[])
    for pairIndex in range(args.numPairs):
        if args.sampling == 'physical':
            dialLevel = randomGenerator.uniform(lowestValue + minimumStrength, highestValue - minimumStrength)
            strengthCeiling = min(maximumStrength, dialLevel - lowestValue, highestValue - dialLevel)
            gradientStrength = float(np.exp(randomGenerator.uniform(np.log(minimumStrength), np.log(strengthCeiling))))
        else:
            dialLevel = randomGenerator.uniform(-0.1, 0.65)
            gradientStrength = float(np.exp(randomGenerator.uniform(np.log(minimumStrength), np.log(maximumStrength))))
        gradientDirection = randomGenerator.uniform(0, 2 * np.pi)
        twinPattern, _, _ = runRingCode(dialLevel + 0 * angles)
        gradientPattern, heldValues, clipped = runRingCode(dialLevel + gradientStrength * np.cos(angles - gradientDirection))
        for key, value in zip(pairs, (dialLevel, gradientStrength, gradientDirection, clipped, twinPattern, gradientPattern,
                                      float(heldValues.min()), float(heldValues.max()))):
            pairs[key].append(value)
        if (pairIndex + 1) % 25 == 0:
            print(f"  {pairIndex + 1}/{args.numPairs} pairs  {time.time() - startTime:.0f}s", flush=True)
    grid = dict(gradientStrength=[], gradientDirection=[], patterns=[])
    gridBaseline = np.full(boundary.numCells, np.nan)
    if not args.skipRotationGrid:
        gridBaseline, _, _ = runRingCode(referenceDialLevel + 0 * angles)
        strengthCeiling = np.floor(100 * min(referenceDialLevel - lowestValue, highestValue - referenceDialLevel)) / 100
        for requestedStrength in (0.02, 0.05, 0.1, 0.2, 0.4):
            gradientStrength = min(requestedStrength, strengthCeiling) if args.sampling == 'physical' else requestedStrength
            for step in range(8):
                gradientDirection = step * np.pi / 4
                pattern, _, _ = runRingCode(referenceDialLevel + gradientStrength * np.cos(angles - gradientDirection))
                grid['gradientStrength'].append(gradientStrength)
                grid['gradientDirection'].append(gradientDirection)
                grid['patterns'].append(pattern)
    np.savez_compressed(outputPath, **{f'pair{key[0].upper()}{key[1:]}': np.array(value) for key, value in pairs.items()},
                        gridGradientStrength=np.array(grid['gradientStrength']),
                        gridGradientDirection=np.array(grid['gradientDirection']),
                        gridPatterns=np.array(grid['patterns']), gridBaselinePattern=gridBaseline,
                        referenceDialLevel=referenceDialLevel, referenceCheckpoint=args.referenceCheckpoint,
                        ringCells=boundary.boundaryRingCells, ringAngles=angles, randomSeed=args.randomSeed,
                        sampling=args.sampling)
print(f"wrote {outputPath} in {time.time() - startTime:.0f}s")
