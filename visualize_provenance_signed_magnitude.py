"""
The signed-value counterpart to visualize_provenance_source_magnitude.py -- NOT to
visualize_provenance_signed_confidence.py. Those two prior scripts combined the exact signed-value
hue with two different opacity channels: confidence.py used the MARGIN between the winning and
runner-up group total (a modest, ratio-like quantity that was never going to look dramatic, margin
or share alike), while magnitude.py used the target's own raw |Vmem-V_th| (which directly inherits
the pattern's real drama). This script is the missing combination: exact signed hue + raw-magnitude
opacity, on the rigorously exact signed decomposition rather than the earlier post-hoc share version.

  python visualize_provenance_signed_magnitude.py
"""

import copy

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import utilities
from embryo import model

torch.set_grad_enabled(False)
V_TH_MV = -27.0

utils = utilities.utilities()
p = copy.deepcopy(torch.load('data/StigmergicModelParameters.dat', weights_only=False))
p['ATPParameters'] = None; p['latticePeriodicBoundaryGJ'] = False
m = model(p, 1)
boundaryIndices = set(int(i) for i in utils.computeDomeIndices(m.electricNetwork, mode='tissue'))

numRows, numCols = 11, 11
numCells = numRows * numCols
bulkIndices = np.array([i for i in range(numCells) if i not in boundaryIndices])

blocksData = np.load('data/provenanceBlocks.npz', allow_pickle=True)
blocks = blocksData['blocks'].item()
featureIndices = set(int(i) for name in blocks for i in blocks[name])

groups = {'north': [], 'south': [], 'west': [], 'east': [], 'corner': []}
for i in range(numCells):
    r, c = i // numCols, i % numCols
    if i not in boundaryIndices:
        continue
    isCorner = (r in (0, numRows - 1)) and (c in (0, numCols - 1))
    if isCorner:
        groups['corner'].append(i)
    elif r == 0:
        groups['north'].append(i)
    elif r == numRows - 1:
        groups['south'].append(i)
    elif c == 0:
        groups['west'].append(i)
    elif c == numCols - 1:
        groups['east'].append(i)
for name in groups:
    groups[name] = np.array(groups[name])
groupNames = list(groups.keys())
groupColor = {'north': '#4C72B0', 'south': '#C44E52', 'west': '#55A868',
             'east': '#8172B2', 'corner': '#937860'}

conditions = [('learned', 'data/provenanceSigned_learned.npz'),
             ('random (100-seed ensemble)', 'data/provenanceSigned_randomEnsemble100.npz')]

allDev = {}
allWinner = {}
for label, path in conditions:
    d = np.load(path, allow_pickle=True)
    key = [k for k in d.keys() if k.startswith('vVmem_step')][0]
    V = d[key]
    finalVmem = d['finalVmem']
    groupTotals = np.zeros((numCells, len(groupNames)))
    for gi, name in enumerate(groupNames):
        groupTotals[:, gi] = V[:, groups[name]].sum(axis=1)
    allWinner[label] = {int(t): groupNames[np.argmax(np.abs(groupTotals[int(t)]))] for t in bulkIndices}
    allDev[label] = {int(t): abs(finalVmem[int(t)] - V_TH_MV) for t in bulkIndices}

sharedMaxDev = max(v for cond in allDev.values() for v in cond.values())

for power, schemeName in [(1, 'magnitude'), (2, 'magnitudeSquared')]:
    fig, axes = plt.subplots(1, len(conditions), figsize=(6.4 * len(conditions), 5.6))
    scaleMax = sharedMaxDev ** power

    for ax, (label, path) in zip(axes, conditions):
        rgbaGrid = np.zeros((numRows, numCols, 4))
        rgbaGrid[..., :3] = 0.92; rgbaGrid[..., 3] = 1.0
        for t in bulkIndices:
            t = int(t)
            alpha = 0.06 + 0.94 * min((allDev[label][t] ** power) / scaleMax, 1.0)
            r, c = t // numCols, t % numCols
            rgbaGrid[r, c] = to_rgba(groupColor[allWinner[label][t]], alpha)

        ax.imshow(rgbaGrid, interpolation='nearest')
        for name in groupNames:
            rs, cs = groups[name] // numCols, groups[name] % numCols
            ax.scatter(cs, rs, s=90, facecolors=groupColor[name], edgecolors='black', linewidths=1.3,
                      marker='s', zorder=5)
        for t in featureIndices:
            r, c = t // numCols, t % numCols
            ax.add_patch(plt.Circle((c, r), 0.42, facecolor='none', edgecolor='black', linewidth=1.1,
                                    zorder=6))
        ax.set_xticks([]); ax.set_yticks([])
        meanDev = np.mean(list(allDev[label].values()))
        ax.set_title(f'{label}\nmean |Vmem-V_th| = {meanDev:.1f} mV (exact signed-value hue)',
                    fontsize=10.5)

    legendElems = [plt.Rectangle((0, 0), 1, 1, color=groupColor[n], label=f'{n} boundary') for n in groupNames]
    legendElems.append(plt.Line2D([0], [0], marker='o', color='none', markeredgecolor='black',
                                  markerfacecolor='none', markersize=10, label='facial-feature cell'))
    fig.legend(handles=legendElems, loc='lower center', ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'Signed-value, {schemeName}-weighted dominance map, all 81 bulk cells: hue = winning '
                f'boundary group\n(exact signed mV total), opacity = target\'s own |Vmem-V_th| '
                f'({schemeName}), SAME scale both panels', fontsize=10.3, y=1.08)
    outPath = f'figures/provenanceSignedMagnitude_{schemeName}.png'
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outPath}")

print(f"\nShared opacity scale max |Vmem-V_th| = {sharedMaxDev:.1f} mV")
for label in allDev:
    vals = list(allDev[label].values())
    print(f"{label}: mean {np.mean(vals):.1f} mV, min {min(vals):.1f}, max {max(vals):.1f}")
