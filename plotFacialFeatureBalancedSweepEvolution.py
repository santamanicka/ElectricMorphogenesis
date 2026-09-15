"""Visualize, for each of the eight facialFeatureBalanced-trained 11x11 band-hold configurations (T in
{100,300} x mechanism in {Gpol-only, Gpol+Vmem} x depth in {D1,D2}), the best-so-far-of-up-to-twelve-
seeds checkpoint's target, pattern evolution (t=1..2901, every 100 steps), and stored best face.

This sweep (files 1888-1983) may still be training when this is run -- bestLoss/actualVmem reflect
each checkpoint's best-found-so-far state, not necessarily a converged result, and a seed whose task
hasn't started yet (or hasn't found its first improvement) simply has no file and is skipped, with the
row label noting how many of the twelve seeds were actually available.

Same layout/method as plot11x11BandSweepEvolution.py / plotFacialFeatureOnlySweepEvolution.py -- each
row replays its own checkpoint's stored clampParameters from scratch, cheap at 11x11.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model

torch.set_grad_enabled(False)

fileNum = 1888
cases = []
for T in [100, 300]:
    for mech, mechLabel in [('Gpol', 'Gpol-only'), ('GpolVmem', 'Gpol+Vmem')]:
        for depth in [1, 2]:
            nums = list(range(fileNum, fileNum + 12))
            cases.append((f'{mechLabel}, D{depth}, T={T}', nums))
            fileNum += 12


def bestInGroup(nums):
    best = None
    nFound = 0
    for n in nums:
        f = f'data/bestModelParameters_fieldVector_11x11_{n}.dat'
        try:
            p = torch.load(f, map_location='cpu', weights_only=False)
        except FileNotFoundError:
            continue
        nFound += 1
        L = float(p['trainParameters']['bestLoss'])
        if best is None or L < best[0]:
            best = (L, n, p)
    return best, nFound


def replayEvolution(p, numSimIters, stride):
    p = dict(p)
    p['latticePeriodicBoundaryGJ'] = False
    p['ATPParameters'] = None
    system = model(p, p['simParameters']['numSamples'])
    system.setExperimentalConditions((p['simParameters']['initialValues'], p['simParameters']['numSamples']))
    circuit = system.electricNetwork
    clampParameters = dict(p['clampParameters'])
    clampEndIter = int(clampParameters['clampEndIter'])
    snaps = {}
    for it in range(numSimIters):
        cp = clampParameters if it <= clampEndIter else None
        system.simulate(clampParameters=cp, numSimIters=1, outerIter=it, fieldModulation=False)
        if it % stride == 1:
            snaps[it] = circuit.Vmem[0, :, 0].detach().numpy().copy() * 1000
    return snaps


stride = 100
snapIters = list(range(1, 3000, stride))  # 1, 101, ..., 2901
numCols = 1 + len(snapIters) + 1  # target + evolution + best face
rows, cols = 11, 11

fig, axes = plt.subplots(len(cases), numCols, figsize=(1.1 * numCols, 1.25 * len(cases)))

for row, (label, nums) in enumerate(cases):
    result, nFound = bestInGroup(nums)
    rowLabel = f'{label}\n({nFound}/12 seeds so far)'

    if result is None:
        for ax in axes[row, :]:
            ax.axis('off')
        axes[row, 0].set_ylabel(rowLabel, fontsize=8)
        print(f"{label}: no checkpoints available yet")
        continue

    L, n, p = result
    target = p['trainParameters']['targetVmem'].reshape(rows, cols).numpy() * 1000
    actual = p['trainParameters']['actualVmem'].reshape(rows, cols).numpy() * 1000
    numSimIters = p['simParameters']['numSimIters']

    axTarget = axes[row, 0]
    axTarget.imshow(target, cmap='gray')
    axTarget.set_ylabel(rowLabel, fontsize=8)
    if row == 0:
        axTarget.set_title('target', fontsize=8)
    axTarget.set_xticks([]); axTarget.set_yticks([])

    snaps = replayEvolution(p, numSimIters, stride)
    for col, it in enumerate(snapIters):
        ax = axes[row, col + 1]
        if it in snaps:
            ax.imshow(snaps[it].reshape(rows, cols), cmap='gray')
        else:
            ax.axis('off')
        if row == 0:
            ax.set_title(f't={it}', fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    axFinal = axes[row, -1]
    axFinal.imshow(actual, cmap='gray')
    axFinal.set_title(f'best face\nloss={L*1000:.2f}mV' if row == 0 else f'loss={L*1000:.2f}mV', fontsize=7)
    axFinal.set_xticks([]); axFinal.set_yticks([])

    print(f"{label}: file {n} (best of {nFound}/12 seeds so far), facialFeatureBalanced bestLoss={L*1000:.2f} mV")

fig.suptitle('11x11 facialFeatureBalanced sweep (best of twelve seeds/config): target, '
             'evolution (t=1..2901, every 100), best face', fontsize=12)
fig.tight_layout()
outPath = 'figures/facialFeatureBalancedSweepEvolution11x11.png'
fig.savefig(outPath, dpi=110, bbox_inches='tight')
print(f'\nwrote {outPath}')
