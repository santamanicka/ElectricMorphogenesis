"""When does a face run lose its left-right symmetry, and does the 64-bit model keep it longer?

Every ring code of the face is a cosine series in the angle about the lattice centre, measured from straight up, so it is
exactly mirror-symmetric (the left-right reflection takes the angle to its negative) and so is the tissue it acts on. In exact
arithmetic the run therefore stays mirror-symmetric for ever. In floating point, rounding differs between a cell and its
mirror image, the difference is amplified by the dynamics, and at some iteration the tissue is visibly lopsided: the
long-horizon players of "Training the face" lose their two-fold symmetry after a few thousand iterations.

This replays each order's best code for --numIterations iterations in two models and records, at every iteration, the largest
|Vmem - mirror image of Vmem| over the cells (mV):

  default    torch's default dtype: the geometry constants (coordinates, distances, the field kernel) are float32
  float64    the whole model in float64 (boundaryCodeUtilities.fullDoublePrecision)

and the first iteration at which the asymmetry exceeds 1e-6, 1e-3, 0.1, 1 and 5 mV. The ring values themselves are checked to be
mirror-symmetric to the last bit, so any asymmetry comes from the model's arithmetic. The question is whether 64 bits delay the
loss of symmetry for good or only for a while. Measured over 50,000 iterations: only for a while in four of the seven orders
(the asymmetry reaches 5 mV 2.6 to 7.5 times later in float64), and not at all within the run in the other three.

Writes data/boundaryHarmonicMirrorSymmetry<rest of the summary's name>Horizon<N>.json (never overwriting).
"""
import argparse
import json
import os
import time

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--numIterations', type=int, default=50000)
parser.add_argument('--stride', type=int, default=100, help='spacing of the stored curves')
args = parser.parse_args()

summary = json.load(open(args.summaryPath))
outputPath = args.summaryPath.replace('boundaryHarmonicTrainingSummary', 'boundaryHarmonicMirrorSymmetry').replace('.json', f'Horizon{args.numIterations}.json')
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

reference = boundary.loadCheckpoint(1888)
hold, numIterations = int(summary['hold']), args.numIterations
orderNames = sorted(summary['orders'], key=int)
THRESHOLDS = (1e-6, 1e-3, 0.1, 1.0, 5.0)
angles = boundary.ringAngles(boundary.boundaryRingCells)


def trainedCoefficients(name):
    """The training run's own full-precision coefficients (the summary rounds them), as in analyzeBoundaryHarmonicMotifPlayers11x11.py."""
    winner = summary['orders'][name]['best']
    path = f"{summary['trainingDirs'][winner['round']]}/order{name}_restart{winner['restart']:02d}.npz"
    return np.asarray(np.load(path)['bestCoefficients'], dtype=float)


ringValues = np.array([np.cos(np.outer(angles, np.arange(len(c)))) @ c for c in (trainedCoefficients(n) for n in orderNames)])

cells = np.arange(boundary.numCells)
mirrorOfCell = (cells // boundary.latticeCols) * boundary.latticeCols + (boundary.latticeCols - 1 - cells % boundary.latticeCols)
ringCells = list(boundary.boundaryRingCells)
mirrorOfRingPosition = np.array([ringCells.index(mirrorOfCell[cell]) for cell in ringCells])
print(f'{len(orderNames)} orders, hold {hold}, {numIterations} iterations; ring-value asymmetry of the cosine series '
      f'{np.abs(ringValues - ringValues[:, mirrorOfRingPosition]).max():.1e} (G_pol / G_ref)', flush=True)


def measureAsymmetry(doublePrecision):
    """The largest |Vmem - mirror image| over the cells, mV, for every iteration and code."""
    asymmetry = np.zeros((numIterations, len(ringValues)))

    def onIteration(iteration, vmem):
        vmem = vmem.numpy()
        asymmetry[iteration] = np.abs(vmem - vmem[:, mirrorOfCell]).max(1)

    started = time.time()
    boundary.ringHoldBatchReplay(reference, ringValues, hold, numIterations, onIteration, doublePrecision=doublePrecision)
    print(f"  {'float64' if doublePrecision else 'default'}, {len(ringValues)} codes, {numIterations} iterations: {time.time() - started:.0f} s", flush=True)
    return asymmetry


def firstPast(curve):
    """The first iteration at which each threshold is exceeded, or -1."""
    return {f'{t:g}': (int(np.argmax(curve > t)) if (curve > t).any() else -1) for t in THRESHOLDS}


result = dict(hold=hold, numIterations=numIterations, stride=args.stride, thresholdsMilliVolts=list(THRESHOLDS), arms={})
for label, doublePrecision in (('default', False), ('float64', True)):
    asymmetry = measureAsymmetry(doublePrecision)
    result['arms'][label] = {}
    for k, name in enumerate(orderNames):
        curve = asymmetry[:, k]
        result['arms'][label][name] = dict(first=firstPast(curve), curve=[float(f'{v:.4g}') for v in curve[::args.stride]],
                                           atEnd=round(float(curve[-1]), 3))
        print(f"  {label:8s} order {name}: first past {result['arms'][label][name]['first']}; at the end {curve[-1]:.2f} mV", flush=True)
json.dump(result, open(outputPath, 'w'))
print('wrote', outputPath, os.path.getsize(outputPath) // 1024, 'KiB', flush=True)
