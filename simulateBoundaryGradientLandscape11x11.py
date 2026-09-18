"""The boundary dial with a first- or second-order harmonic added, or both, kept below a ceiling on the ring
(PolyPatterning_Sim.md, Section 12).

Single harmonic (--harmonicOrder k, k = 1 or 2): the ring is held at code = DC + G cos(k (theta - phi)): the dial DC
(order 0) plus one harmonic of amplitude G. Order 1 is a gradient pointing along phi; order 2 raises the two opposite
sides facing phi and phi + 180 degrees and lowers the two sides between them, squeezing the ring into an oval. theta is
each ring cell's angle about the lattice centre, measured clockwise from straight up. Every code stays within
[0, --dialLimit]: DC - G >= 0 and DC + G <= --dialLimit, so DC runs from 0 to the limit and G from 0 to
min(DC, limit - DC), both on a grid of --gridStep. The grid step divides the dial sweep's own step, so each (DC, 0)
point is a run of the matching dial sweep. The reference checkpoint's parameters and initial state are kept; the hold
and readout follow simulateBoundaryDialLandscape11x11.py (--holdIterations, --alignReadoutToRelease).

The lattice and the dial are symmetric under the 8 symmetries of the square, which map a harmonic along phi onto the
same harmonic along phi + 90, -phi and 90 - phi degrees (for order 2, phi + 180 is the same code). So phi in [0, 45]
degrees stands for every direction: --gradientDirections 0,22.5,45 covers the axial and diagonal harmonics and the one
midway between. A few runs at other directions (--symmetryChecks) test that mapping directly.

Both orders (--combinedOrders): code = DC + G1 cos(theta) + G2 cos(2 theta), the gradient pointing straight up and the
oval's long axis vertical (G2 > 0) or horizontal (G2 < 0). Both terms are mirror-symmetric about the vertical axis, as
the target face is. DC and G1 >= 0 and signed G2 are multiples of --combinedStep, and every code whose held values all
lie in [0, --dialLimit] is run, including the single-order codes with G1 = 0 or G2 = 0 (G1 = G2 = 0 is left to the dial
sweep). Negative G1 gives the vertical mirror image, which --combinedChecks tests at a few codes.

Runs are split over --numTasks slurm array tasks (--taskIndex); each writes its share under
data/boundaryGradientLandscapeParts/, and --merge gathers them into data/boundaryGradientLandscape<checkpoint><hold><tag>.npz,
where the tag is empty for order 1, Order2 for order 2 and Orders12 for both.
"""
import argparse
import glob
import os
import time

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--referenceCheckpoint', type=int, default=1888)
parser.add_argument('--holdIterations', type=int, default=301)
parser.add_argument('--alignReadoutToRelease', action='store_true')
parser.add_argument('--windowIterations', type=int, default=1000)
parser.add_argument('--dialLimit', type=float, default=1.3, help='ceiling on every held value, DC + G')
parser.add_argument('--gridStep', type=float, default=0.02)
parser.add_argument('--gradientDirections', type=str, default='0,22.5,45', help='phi values in degrees')
parser.add_argument('--symmetryChecks', type=str, default='0.30:0.20,0.64:0.40,1.00:0.24,1.20:0.10',
                    help='DC:G points also run at phi = 90, 180, -22.5 and 67.5 degrees')
parser.add_argument('--harmonicOrder', type=int, default=1, choices=[1, 2])
parser.add_argument('--combinedOrders', action='store_true', help='orders 1 and 2 together, both mirror-symmetric about the vertical axis')
parser.add_argument('--combinedStep', type=float, default=0.04)
parser.add_argument('--combinedChecks', type=str, default='0.40:0.12:0.08,0.64:0.20:-0.24,0.80:0.16:0.20,1.00:0.12:-0.12',
                    help='DC:G1:G2 codes also run with G1 negated, the vertical mirror image')
parser.add_argument('--numTasks', type=int, default=1)
parser.add_argument('--taskIndex', type=int, default=0)
parser.add_argument('--merge', action='store_true')
args = parser.parse_args()

reference = boundary.loadCheckpoint(args.referenceCheckpoint)
referenceHoldIterations = int(reference['clampParameters']['clampEndIter']) + 1
extraHold = args.holdIterations - referenceHoldIterations
holdTag = '' if extraHold == 0 else f"Hold{args.holdIterations}{'Aligned' if args.alignReadoutToRelease else ''}"
numIterations = reference['simParameters']['numSimIters'] + (extraHold if args.alignReadoutToRelease else 0)
simulationParameters = dict(reference, simParameters=dict(reference['simParameters'], numSimIters=numIterations))
angles = boundary.ringAngles(boundary.boundaryRingCells)
codeTag = 'Orders12' if args.combinedOrders else ('Order2' if args.harmonicOrder == 2 else '')
partsDirectory = f'data/boundaryGradientLandscapeParts/{args.referenceCheckpoint}{holdTag}{codeTag}'
outputPath = f'data/boundaryGradientLandscape{args.referenceCheckpoint}{holdTag}{codeTag}.npz'


def ringCode(dialLevel, gradientStrength, direction, secondOrderStrength):
    """Held values on the 40 ring cells. For a single harmonic, gradientStrength is its amplitude and secondOrderStrength
    is unused; for both orders, gradientStrength is G1 and secondOrderStrength is G2, with both harmonics along phi = 0."""
    if args.combinedOrders:
        return dialLevel + gradientStrength * np.cos(angles) + secondOrderStrength * np.cos(2 * angles)
    return dialLevel + gradientStrength * np.cos(args.harmonicOrder * (angles - np.deg2rad(direction)))


def runList():
    """(DC, G or G1, phi in degrees, G2, is a symmetry check) for every run, in a fixed order. DC and the amplitudes are
    whole multiples of the grid step; the dial alone is left to the dial sweep."""
    runs = []
    if args.combinedOrders:
        numSteps = int(round(args.dialLimit / args.combinedStep))
        for dialStep in range(numSteps + 1):
            for firstStep in range(numSteps + 1):
                for secondStep in range(-numSteps, numSteps + 1):
                    if firstStep == 0 and secondStep == 0:
                        continue
                    code = (round(dialStep * args.combinedStep, 6), round(firstStep * args.combinedStep, 6), 0.0, round(secondStep * args.combinedStep, 6))
                    values = ringCode(code[0], code[1], 0.0, code[3])
                    if values.min() >= -1e-9 and values.max() <= args.dialLimit + 1e-9:
                        runs.append(code + (False,))
        for point in args.combinedChecks.split(','):
            dialLevel, firstStrength, secondStrength = (float(value) for value in point.split(':'))
            runs.append((dialLevel, -firstStrength, 0.0, secondStrength, True))
        return runs
    numSteps = int(round(args.dialLimit / args.gridStep))
    for direction in (float(value) for value in args.gradientDirections.split(',')):
        for dialStep in range(numSteps + 1):
            for gradientStep in range(1, min(dialStep, numSteps - dialStep) + 1):
                runs.append((round(dialStep * args.gridStep, 6), round(gradientStep * args.gridStep, 6), direction, 0.0, False))
    for point in args.symmetryChecks.split(','):
        dialLevel, gradientStrength = (float(value) for value in point.split(':'))
        for direction in (90.0, 180.0, -22.5, 67.5):
            runs.append((dialLevel, gradientStrength, direction, 0.0, True))
    return runs


if args.merge:
    parts = sorted(glob.glob(f'{partsDirectory}/task*.npz'))
    loaded = [np.load(path) for path in parts]
    merged = {key: np.concatenate([part[key] for part in loaded]) for key in loaded[0].files if key != 'runIndex'}
    order = np.argsort(np.concatenate([part['runIndex'] for part in loaded]))
    merged = {key: value[order] for key, value in merged.items()}
    expected = len(runList())
    if len(order) != expected:
        raise SystemExit(f"found {len(order)} of {expected} runs in {len(parts)} parts; not merging")
    np.savez_compressed(outputPath, **merged, ringAngles=angles, holdIterations=args.holdIterations, numIterations=numIterations,
                        windowStart=numIterations - args.windowIterations, dialLimit=args.dialLimit,
                        gridStep=args.combinedStep if args.combinedOrders else args.gridStep, harmonicOrder=args.harmonicOrder,
                        combinedOrders=args.combinedOrders, referenceCheckpoint=args.referenceCheckpoint)
    print(f"wrote {outputPath} ({expected} runs from {len(parts)} parts)")
else:
    startTime = time.time()
    runs = runList()
    mine = list(range(args.taskIndex, len(runs), args.numTasks))
    records = {key: [] for key in ('runIndex', 'dialLevel', 'gradientStrength', 'gradientDirection', 'secondOrderStrength', 'symmetryCheck',
                                   'ringValues') + boundary.ringCodeReadoutKeys}
    for count, runIndex in enumerate(mine):
        dialLevel, gradientStrength, direction, secondOrderStrength, symmetryCheck = runs[runIndex]
        ringValues = ringCode(dialLevel, gradientStrength, direction, secondOrderStrength)
        # rounding in the grid can leave the lowest cell a hair below zero
        ringValues = np.clip(ringValues, 0, None)
        readout = boundary.ringCodeReadouts(simulationParameters, reference, ringValues, args.holdIterations, args.windowIterations)
        for key, value in zip(('runIndex', 'dialLevel', 'gradientStrength', 'gradientDirection', 'secondOrderStrength', 'symmetryCheck', 'ringValues'),
                              (runIndex, dialLevel, gradientStrength, direction, secondOrderStrength, symmetryCheck, ringValues)):
            records[key].append(value)
        for key in boundary.ringCodeReadoutKeys:
            records[key].append(readout[key])
        if (count + 1) % 10 == 0:
            print(f"  {count + 1}/{len(mine)} runs  {time.time() - startTime:.0f}s", flush=True)
    os.makedirs(partsDirectory, exist_ok=True)
    partPath = f'{partsDirectory}/task{args.taskIndex:03d}.npz'
    np.savez_compressed(partPath, **{key: np.array(value) for key, value in records.items()})
    print(f"wrote {partPath} ({len(mine)} of {len(runs)} runs) in {time.time() - startTime:.0f}s")
