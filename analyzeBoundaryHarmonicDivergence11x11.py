"""When a small change in the ring code takes effect: the distance between the tissue of a changed code and the tissue of the
trained code, over the whole run (PolyPatterning_Sim.md, Section 12).

For the best code of each code size in --sizes (full precision, from the training run files named in the summary), every
coefficient a_n in turn is changed by each of --steps (G_pol / G_ref) and the changed code is simulated alongside the
trained one. Every --stride iterations the RMS difference in Vmem over all 121 cells between each changed tissue and the
trained tissue is recorded. Held values are clipped only to the physical range [0, 2], as in
analyzeBoundaryHarmonicSensitivity11x11.py.

Writes data/boundaryHarmonicDivergence<rest of the summary's name> (never overwriting).
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--summaryPath', type=str, default='data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--sizes', type=str, default='3,4,5,6')
parser.add_argument('--steps', type=str, default='-0.02,-0.0025,0.0025,0.02')
parser.add_argument('--stride', type=int, default=10)
args = parser.parse_args()

summary = json.load(open(args.summaryPath))
outputPath = args.summaryPath.replace('boundaryHarmonicTrainingSummary', 'boundaryHarmonicDivergence')
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
steps = [float(step) for step in args.steps.split(',')]
run = dict(np.load(sorted(glob.glob(f"{summary['trainingDirs'][0]}/order*_restart*.npz"))[0]))
reference = boundary.loadCheckpoint(int(run['referenceCheckpoint']))
hold, numIterations = summary['hold'], summary['numIterations']
angles = boundary.ringAngles(boundary.boundaryRingCells)

result = dict(steps=steps, stride=args.stride, hold=hold, numIterations=numIterations, sourceSummary=args.summaryPath, orders={})
for size in args.sizes.split(','):
    winner = summary['orders'][size]['best']
    code = np.load(f"{summary['trainingDirs'][winner['round']]}/order{size}_restart{winner['restart']:02d}.npz")['bestCoefficients']
    basis = np.cos(np.outer(angles, np.arange(len(code))))
    labels, codes = [], [code]
    for order in range(len(code)):
        for step in steps:
            changed = code.copy()
            changed[order] += step
            codes.append(changed)
            labels.append((order, step))
    distances = np.zeros((len(labels), numIterations // args.stride + 1))

    def onIteration(iteration, vmem):
        if iteration % args.stride == 0:
            distances[:, iteration // args.stride] = torch.sqrt(((vmem[1:] - vmem[0]) ** 2).mean(1)).numpy()
    boundary.ringHoldBatchReplay(reference, np.clip(np.array(codes) @ basis.T, 0, 2), hold, numIterations, onIteration)
    result['orders'][size] = dict(bestIteration=winner['iteration'], rows=[dict(order=order, step=step, distance=[float(f'{value:.4g}') for value in distances[index]])
                                                                         for index, (order, step) in enumerate(labels)])
    print(f"orders 0-{size}: median distance at the trained best moment (iteration {winner['iteration']}): "
          + ', '.join(f"{step:+}: {np.median([distances[index, winner['iteration'] // args.stride] for index, (_, s) in enumerate(labels) if s == step]):.3g} mV" for step in steps), flush=True)
json.dump(result, open(outputPath, 'w'), separators=(',', ':'))
print(f'wrote {outputPath}')
