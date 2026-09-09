"""Visualize, for each of the 16 winning 11x11 band-hold checkpoints (T in {100,300} x mechanism in
{Gpol-only, Gpol+Vmem} x depth in {1,2} x lossMethod in {correlation, globalsum}): the target, the
pattern evolution from t=1 sampled every 100 steps out to the trained numSimIters, and the checkpoint's
own stored best face (actualVmem, the readout-scored state) in a final column.

Each group now pools both seed batches (files N..N+5 and N+96..N+101, 12 seeds total per config) and
picks the single best-loss checkpoint across all 12 to show.

Each row replays its own checkpoint's stored clampParameters via embryo.model.simulate() -- cheap at
11x11 (forward-only, no backprop), so nothing here is precomputed or approximated.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model

torch.set_grad_enabled(False)

fileNum = 1600
cases = []
for T in [100, 300]:
    for mech, mechLabel in [('Gpol', 'Gpol-only'), ('GpolVmem', 'Gpol+Vmem')]:
        for depth in [1, 2]:
            for loss, lossLabel in [('corr', 'correlation'), ('glob', 'globalsum')]:
                nums = list(range(fileNum, fileNum + 6)) + list(range(fileNum + 96, fileNum + 96 + 6))
                cases.append((f'{mechLabel}, D{depth}, T={T}, {lossLabel}', nums))
                fileNum += 6


def bestInGroup(nums):
    best = None
    for n in nums:
        f = f'data/bestModelParameters_fieldVector_11x11_{n}.dat'
        try:
            p = torch.load(f, map_location='cpu', weights_only=False)
        except FileNotFoundError:
            continue
        L = float(p['trainParameters']['bestLoss'])
        if best is None or L < best[0]:
            best = (L, n, p)
    return best


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
snapIters = list(range(1, 3000, stride))  # 1, 101, 201, ..., 2901
numCols = 1 + len(snapIters) + 1  # target + evolution + best face

rows, cols = 11, 11
fig, axes = plt.subplots(len(cases), numCols, figsize=(1.1 * numCols, 1.25 * len(cases)))

for row, (label, nums) in enumerate(cases):
    L, n, p = bestInGroup(nums)
    target = p['trainParameters']['targetVmem'].reshape(rows, cols).numpy() * 1000
    actual = p['trainParameters']['actualVmem'].reshape(rows, cols).numpy() * 1000
    numSimIters = p['simParameters']['numSimIters']

    axTarget = axes[row, 0]
    axTarget.imshow(target, cmap='gray')
    axTarget.set_ylabel(label, fontsize=7)
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
    axFinal.set_title(f'best face\nloss={L:.3f}' if row == 0 else f'loss={L:.3f}', fontsize=7)
    axFinal.set_xticks([]); axFinal.set_yticks([])

fig.suptitle('11x11 band-hold sweep (12 seeds/config, both batches pooled): target, evolution '
             '(t=1..2901, every 100), best face', fontsize=12)
fig.tight_layout()
outPath = 'figures/band11x11SweepEvolution.png'
fig.savefig(outPath, dpi=110, bbox_inches='tight')
print(f'wrote {outPath}')
