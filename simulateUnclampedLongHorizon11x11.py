"""The frozen 11 x 11 tissue (checkpoint 1888) run completely free over a long horizon. EXPLORATORY: written for the talk's video of what the tissue does on its own, nothing here was predicted.

No boundary hold, no clamp, no tissue-wide extra update, no noise: the start is the parameter file's uniform state, so everything that follows is the model's own deterministic dynamics (the lattice's finite
edge is what breaks the uniformity). Parameters are the file's own: field screen size 4, transduction weight 1000, bias 0.0005, time constant 10, gap-junction strength 0.05, no ligands, float64, dt 0.01.

Writes the whole course (Vmem in mV and G_pol / G_ref of every cell after every iteration) and prints what the run does: whether it settles or becomes periodic, how many distinct patterns (sets of dark cells)
it visits and when each first appears, and how far it stays from left-right symmetry.

    python3 simulateUnclampedLongHorizon11x11.py --numIterations 20000          # about a minute; refuses to overwrite
"""
import argparse
import json
import os

import numpy as np

from canalizationTalkCommon import LATTICE, INTERIOR, RING_CELLS, THRESHOLD, Replayer

parser = argparse.ArgumentParser()
parser.add_argument('--numIterations', type=int, default=20000)
parser.add_argument('--outputPath', type=str, default='data/unclampedLongHorizon1888Uniform.npz')
parser.add_argument('--summaryPath', type=str, default='data/unclampedLongHorizon1888UniformSummary.json')
args = parser.parse_args()
for path in (args.outputPath, args.summaryPath):
    if os.path.exists(path):
        raise SystemExit(f'{path} exists; not overwriting')

replayer = Replayer()
# a hold of 0 iterations: the ring is never held and the extra update never runs, i.e. the tissue is completely free
vmem, conductance = replayer.run(np.zeros(len(RING_CELLS)), args.numIterations, keepEvery=1, holdIterations=0)
print(f'course {vmem.shape}: Vmem from {vmem.min():.1f} to {vmem.max():.1f} mV, G_pol / G_ref from {conductance.min():.3f} to {conductance.max():.3f}', flush=True)

# what the run does
interiorMean = vmem[:, INTERIOR].mean(1)
speed = np.abs(np.diff(vmem[:, INTERIOR], axis=0)).mean(1)                         # mean |change| of an interior cell's Vmem per iteration
blocks = [(start, min(start + 2000, len(speed))) for start in range(0, len(speed), 2000)]
print('mean interior speed (mV / iteration) per 2,000 iterations:', [round(float(speed[a:b].mean()), 3) for a, b in blocks], flush=True)

dark = vmem[:, INTERIOR] < THRESHOLD
patternIndex, firstSeen = {}, []
for iteration, row in enumerate(dark):
    key = row.tobytes()
    if key not in patternIndex:
        patternIndex[key] = len(patternIndex)
        firstSeen.append(iteration)
darkCount = dark.sum(1)
print(f'distinct dark-cell patterns visited: {len(patternIndex)} in {len(dark)} iterations; the last new one first appears at iteration {firstSeen[-1]}', flush=True)
for horizon in (1000, 2000, 5000, 10000, 20000):
    if horizon <= len(dark):
        print(f'   distinct patterns within the first {horizon:>6}: {sum(1 for t in firstSeen if t < horizon)}', flush=True)
print(f'dark interior cells: range {darkCount.min()} to {darkCount.max()}, never fewer than {darkCount[len(darkCount) // 10:].min()} after the first tenth of the run', flush=True)

# a repeat: the tissue returns close to an earlier state (a cycle) or stops moving (a fixed point)
sample = vmem[::10][:, INTERIOR]
returns = []
for index in range(61, len(sample)):                                                # compare with every state at least 600 iterations earlier
    distance = np.abs(sample[:index - 60] - sample[index]).max(1).min()
    returns.append(distance)
returns = np.array(returns)
print(f'closest return to any earlier state (max over cells, mV): lowest {returns.min():.3f}, median {np.median(returns):.3f}', flush=True)

mirror = np.array([row * LATTICE + (LATTICE - 1 - column) for row in range(LATTICE) for column in range(LATTICE)])
asymmetry = np.abs(vmem[:, INTERIOR] - vmem[:, mirror][:, INTERIOR]).max(1)
for iteration in (1000, 3000, 5000, 10000, 15000, 19999):
    if iteration < len(asymmetry):
        print(f'   left-right asymmetry (largest difference between mirror cells) at {iteration:>6}: {asymmetry[iteration]:.2e} mV', flush=True)

os.makedirs('data', exist_ok=True)
np.savez_compressed(args.outputPath, vmem=vmem.astype(np.float32), conductance=conductance.astype(np.float32))
json.dump(dict(numIterations=args.numIterations, distinctPatterns=len(patternIndex), firstSeenIteration=firstSeen, speedPerBlock=[float(speed[a:b].mean()) for a, b in blocks],
               closestReturnLowest=float(returns.min()), closestReturnMedian=float(np.median(returns)), asymmetryAt=dict((int(i), float(asymmetry[i])) for i in (1000, 3000, 5000, 10000, 15000, 19999) if i < len(asymmetry))),
          open(args.summaryPath, 'w'))
print('wrote', args.outputPath, 'and', args.summaryPath)
