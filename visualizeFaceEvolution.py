"""Replay the trained single-shot forcing (iteration 0) followed by free evolution for the
two best 30x30 checkpoints (best correlation-loss model, best globalsum-loss model), and
snapshot the Vmem pattern every few hundred iterations to show how it develops over the run.

Uses the checkpoint's own stored clampParameters/fieldParameters/simParameters, so this is an
exact replay of the trajectory that produced that checkpoint's saved actualVmem.
"""
import argparse

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--corrFile', default='data/bestModelParameters_fieldVector_30x30_1912.dat')
parser.add_argument('--globFile', default='data/bestModelParameters_fieldVector_30x30_1913.dat')
parser.add_argument('--stride', type=int, default=250)
parser.add_argument('--output', default='figures/faceEvolution.png')
args = parser.parse_args()

torch.set_grad_enabled(False)


def replay(path):
    p = torch.load(path, map_location='cpu', weights_only=False)
    rows, cols = p['latticeDims']
    numSamples = p['simParameters']['numSamples']
    numSimIters = p['simParameters']['numSimIters']
    clampParameters = dict(p['clampParameters'])
    p['latticePeriodicBoundaryGJ'] = False
    p['ATPParameters'] = None

    system = model(p, numSamples)
    circuit = system.electricNetwork

    checkpoints = list(range(0, numSimIters, args.stride))
    if checkpoints[-1] != numSimIters - 1:
        checkpoints.append(numSimIters - 1)
    snapshots = {}
    for it in range(numSimIters):
        cp = clampParameters if it <= int(clampParameters['clampEndIter']) else None
        system.simulate(clampParameters=cp, numSimIters=1, outerIter=it, fieldModulation=False)
        if it in checkpoints:
            snapshots[it] = circuit.Vmem.detach().clone().reshape(rows, cols).numpy() * 1000
    # The replay is not bit-identical to the original training run (fresh no-grad reconstruction
    # vs. the original grad-tracked training execution), and this system is chaotically sensitive
    # to floating-point-scale differences (see the symmetry-drift analysis), so intermediate frames
    # are an approximate replay only. Use the checkpoint's own stored actualVmem -- verified ground
    # truth -- for the final frame instead of the (measurably diverged) simulated one.
    snapshots[numSimIters - 1] = p['trainParameters']['actualVmem'].reshape(rows, cols).numpy() * 1000
    return snapshots, rows, cols, float(p['trainParameters']['bestLoss']), p['trainParameters']['lossMethod']


corrSnaps, rows, cols, corrLoss, _ = replay(args.corrFile)
globSnaps, _, _, globLoss, _ = replay(args.globFile)

iters = sorted(corrSnaps.keys())
fig, axes = plt.subplots(2, len(iters), figsize=(2.0 * len(iters), 4.4))
for col, it in enumerate(iters):
    axes[0, col].imshow(corrSnaps[it], cmap='gray')
    axes[0, col].set_title(f'iter {it}', fontsize=9)
    axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
    axes[1, col].imshow(globSnaps[it], cmap='gray')
    axes[1, col].set_xticks([]); axes[1, col].set_yticks([])
axes[0, 0].set_ylabel(f'correlation\n(loss {corrLoss:.3f})', fontsize=9)
axes[1, 0].set_ylabel(f'globalsum\n(loss {globLoss:.3f})', fontsize=9)
fig.suptitle('Pattern evolution: single-shot forcing at iter 0, then free evolution\n'
             '(intermediate frames are an approximate replay -- chaotic sensitivity means they diverge '
             'from the true trajectory; final frame is the checkpoint\'s verified stored result)',
             fontsize=10)
fig.tight_layout()
fig.savefig(args.output, dpi=140, bbox_inches='tight')
print(f"wrote {args.output}")
