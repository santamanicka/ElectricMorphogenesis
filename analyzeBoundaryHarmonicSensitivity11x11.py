"""One coefficient at a time: how the best trained ring code of each size loses its score when a single order is changed
(PolyPatterning_Sim.md, Section 12).

For the best code of each code size in a training summary (analyzeBoundaryHarmonicTraining11x11.py), each coefficient a_n
in turn is changed by each of --steps (in G_pol / G_ref; a change of d in a_n moves ring cell k by d cos(n theta_k), so
by at most |d|), the others left as they are, and the code is scored as in training: the lowest balanced RMS from the
scoring start to the end of the run. The best codes are read at full precision from their training run files: the
summary's rounded coefficients do not reproduce their scores. Many best codes touch both 0 and the ceiling, so a changed
code may exceed the ceiling by up to the step; held values are clipped only to the physical range [0, 2].

Writes data/boundaryHarmonicSensitivity<rest of the summary's name> (never overwriting).
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
parser.add_argument('--steps', type=str, default='-0.02,-0.01,-0.005,-0.0025,0.0025,0.005,0.01,0.02',
                    help="comma-separated changes, or 'grid:<largest>:<spacing>' for every multiple of spacing up to largest, 0 left out")
parser.add_argument('--storePatterns', action='store_true',
                    help="also store each code's pattern and iteration at its best moment, and its pattern and score at the trained code's best moment")
parser.add_argument('--tag', type=str, default='')
args = parser.parse_args()

summary = json.load(open(args.summaryPath))
outputPath = args.summaryPath.replace('boundaryHarmonicTrainingSummary', 'boundaryHarmonicSensitivity').replace('.json', f'{args.tag}.json')
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
if args.steps.startswith('grid:'):
    largest, spacing = (float(value) for value in args.steps[5:].split(':'))
    count = int(round(largest / spacing))
    steps = [round(spacing * index, 10) for index in range(-count, count + 1) if index != 0]
else:
    steps = [float(step) for step in args.steps.split(',')]
run = dict(np.load(sorted(glob.glob(f"{summary['trainingDirs'][0]}/order*_restart*.npz"))[0]))
reference = boundary.loadCheckpoint(int(run['referenceCheckpoint']))
hold, numIterations, ceiling = summary['hold'], summary['numIterations'], summary['ceiling']
scoreFrom = summary.get('scoreFrom', hold)
target = torch.tensor(summary['target'], dtype=torch.double)
featureMask = np.isin(np.arange(boundary.numCells), boundary.featureCellIndices)
scoreMasks = run['scoreGroupMasks'] if 'scoreGroupMasks' in run else np.array([featureMask, ~featureMask])
angles = boundary.ringAngles(boundary.boundaryRingCells)


def scoreCodes(ringValues, withPatterns=False, captureIteration=None):
    best = torch.full((len(ringValues),), np.inf, dtype=torch.double)
    bestIteration = torch.zeros(len(ringValues), dtype=torch.long)
    bestVmem = torch.zeros(len(ringValues), boundary.numCells, dtype=torch.double)
    captured = {}

    def onIteration(iteration, vmem):
        if iteration >= scoreFrom:
            squared = (vmem - target) ** 2
            scores = sum(squared[:, mask].mean(1).sqrt() for mask in scoreMasks) / len(scoreMasks)
            if iteration == captureIteration:
                captured.update(vmem=vmem.numpy().copy(), score=scores.numpy().copy())
            better = scores < best
            best.copy_(torch.where(better, scores, best))
            bestIteration[better] = iteration
            bestVmem[better] = vmem[better]
    boundary.ringHoldBatchReplay(reference, ringValues, hold, numIterations, onIteration)
    if captureIteration is not None:
        return best.numpy(), bestIteration.numpy(), bestVmem.numpy(), captured
    return (best.numpy(), bestIteration.numpy(), bestVmem.numpy()) if withPatterns else best.numpy()


result = dict(steps=steps, sourceSummary=args.summaryPath, orders={})
for size, entry in sorted(summary['orders'].items(), key=lambda item: int(item[0])):
    winner = entry['best']
    code = np.load(f"{summary['trainingDirs'][winner['round']]}/order{size}_restart{winner['restart']:02d}.npz")['bestCoefficients']
    basis = np.cos(np.outer(angles, np.arange(len(code))))
    changed, labels = [], []
    for order in range(len(code)):
        for step in steps:
            candidate = code.copy()
            candidate[order] += step
            changed.append(np.clip(basis @ candidate, 0, 2))
            labels.append((order, step))
    scores, iterations, patterns, atTrained = scoreCodes(np.vstack([np.clip(basis @ code, 0, ceiling)] + changed), captureIteration=int(winner['iteration']))
    rounded = scoreCodes(np.clip(basis @ np.round(code, 4), 0, ceiling)[None])[0]
    extras = (lambda index: dict(iteration=int(iterations[index]), vmem=np.round(patterns[index], 1).tolist(),
                                 scoreAtTrainedMoment=round(float(atTrained['score'][index]), 3),
                                 vmemAtTrainedMoment=np.round(atTrained['vmem'][index], 1).tolist())) if args.storePatterns else (lambda index: {})
    result['orders'][size] = dict(bestScore=float(scores[0]), roundedCodeScore=float(rounded), coefficients=np.asarray(code).tolist(), **extras(0),
                                  rows=[dict(order=order, step=step, score=round(float(scores[index + 1]), 3), **extras(index + 1))
                                        for index, (order, step) in enumerate(labels)])
    print(f"orders 0-{size}: best {scores[0]:.3f} mV (coefficients rounded to 4 decimals: {rounded:.3f}); median change by order changed: "
          + ', '.join(f"a{order} {np.median([score - scores[0] for (o, _), score in zip(labels, scores[1:]) if o == order]):+.2f}"
                      for order in range(len(code)) if any(o == order for o, _ in labels)), flush=True)
json.dump(result, open(outputPath, 'w'), separators=(',', ':'))
print(f'wrote {outputPath}')
