"""Visualize, for each of the eight facialFeatureOnly-trained 11x11 band-hold configurations (T in
{100,300} x mechanism in {Gpol-only, Gpol+Vmem} x depth in {D1,D2} -- no correlation/globalsum split
this time, since lossMethod='facialFeatureOnly' is the only loss used here), the best-of-twelve-seeds
checkpoint's target, pattern evolution (t=1..2901, every 100 steps), and stored best face.

Same layout/method as plot11x11BandSweepEvolution.py (Sec 12.5's sixteen-configuration figure) --
each row replays its own checkpoint's stored clampParameters from scratch, cheap at 11x11.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embryo import model

torch.set_grad_enabled(False)

fileNum = 1792
cases = []
for T in [100, 300]:
    for mech, mechLabel in [('Gpol', 'Gpol-only'), ('GpolVmem', 'Gpol+Vmem')]:
        for depth in [1, 2]:
            nums = list(range(fileNum, fileNum + 12))
            cases.append((f'{mechLabel}, D{depth}, T={T}', nums))
            fileNum += 12


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
snapIters = list(range(1, 3000, stride))  # 1, 101, ..., 2901
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
    axTarget.set_ylabel(label, fontsize=8)
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

    print(f"{label}: file {n}, facialFeatureOnly bestLoss={L*1000:.2f} mV")

fig.suptitle('11x11 facialFeatureOnly-trained sweep: target, evolution (t=1..2901, every 100), best face '
             '(best of twelve seeds/config)', fontsize=12)
fig.tight_layout()
outPath = 'figures/facialFeatureOnlySweepEvolution11x11.png'
fig.savefig(outPath, dpi=110, bbox_inches='tight')
print(f'\nwrote {outPath}')
