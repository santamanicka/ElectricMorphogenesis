"""
Magnitude-weighted dominance map: same hue (winning boundary group, forced argmax over the 5 pooled
groups -- sharp and fully covering, as in visualize_provenance_source_regions_bulkFeatureOnly.py's
dominance map) but opacity now encodes |Vmem[target] - V_th|, the target cell's own realised
deviation from threshold, on a SHARED scale across both conditions (not independently normalised per
panel) so the two panels' brightness is directly comparable.

This is deliberately close to "recolouring the raw pattern by source identity": share-weighted hue
identifies WHICH boundary region is credited, and since a target-uniform rescaling factor cancels out
of a normalised blend (share x Vmem[target] and plain share give the identical hue -- the Vmem[target]
factor is the same for every source of a given target, so it cancels in the weighted average), the
only way magnitude actually enters the picture is as brightness, not as a hue-changing weight. Two
variants are produced: magnitude (|Vmem-V_th|) and its square, to see whether emphasising the largest
deviations further sharpens the visual gap.

Uses the already-computed data/provenance_learned.npz and data/provenance_randomEnsemble100.npz
(step 899) -- no new simulation needed.

  python visualize_provenance_source_magnitude_bulkFeatureOnly.py
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

blocksData = np.load('data/provenanceBlocks.npz', allow_pickle=True)
blocks = blocksData['blocks'].item()
featureIndices = np.concatenate([blocks[name] for name in blocks])
cellToBlock = {int(i): name for name in blocks for i in blocks[name]}
blockLetter = {'eye': 'E', 'nose': 'N', 'mouth': 'M'}

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

conditions = [('learned', 'data/provenance_learned.npz'),
             ('random (100-seed ensemble)', 'data/provenance_randomEnsemble100.npz')]

# Precompute deviations across BOTH conditions first, to set one shared opacity scale.
allDev = {}
allWinner = {}
for label, path in conditions:
    d = np.load(path, allow_pickle=True)
    P = np.asarray(d['pVmem_step899'])
    finalVmem = d['finalVmem']
    groupTotals = np.zeros((numCells, len(groupNames)))
    for gi, name in enumerate(groupNames):
        groupTotals[:, gi] = P[:, groups[name]].sum(axis=1)
    allWinner[label] = {int(t): groupNames[np.argmax(groupTotals[int(t)])] for t in featureIndices}
    allDev[label] = {int(t): abs(finalVmem[int(t)] - V_TH_MV) for t in featureIndices}

sharedMaxDev = max(v for cond in allDev.values() for v in cond.values())

for power, schemeName in [(1, 'magnitude'), (2, 'magnitudeSquared')]:
    fig, axes = plt.subplots(1, len(conditions), figsize=(6.4 * len(conditions), 5.6))
    scaleMax = sharedMaxDev ** power

    for ax, (label, path) in zip(axes, conditions):
        rgbaGrid = np.zeros((numRows, numCols, 4))
        rgbaGrid[..., :3] = 0.92; rgbaGrid[..., 3] = 1.0
        for t in featureIndices:
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
            t = int(t)
            r, c = t // numCols, t % numCols
            ax.annotate(f"{blockLetter[cellToBlock[t]]}\n{allDev[label][t]:.0f}", xy=(c, r), fontsize=6.5,
                       color='black', fontweight='bold', ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        meanDev = np.mean(list(allDev[label].values()))
        ax.set_title(f'{label}\nmean |Vmem-V_th| = {meanDev:.1f} mV', fontsize=11)

    legendElems = [plt.Rectangle((0, 0), 1, 1, color=groupColor[n], label=f'{n} boundary') for n in groupNames]
    fig.legend(handles=legendElems, loc='lower center', ncol=5, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'{schemeName}-weighted dominance map: hue = winning boundary group (forced argmax, '
                f'unchanged by\nthis reweighting), opacity = target\'s own |Vmem-V_th| ({schemeName}), '
                f'SAME scale both panels\n(letters + numbers show block and deviation in mV)',
                fontsize=10.3, y=1.07)
    outPath = f'figures/provenanceSourceMagnitude_bulkFeatureOnly_{schemeName}.png'
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {outPath}")

print(f"\nShared opacity scale max |Vmem-V_th| = {sharedMaxDev:.1f} mV")
for label in allDev:
    vals = list(allDev[label].values())
    print(f"{label}: mean {np.mean(vals):.1f} mV, min {min(vals):.1f}, max {max(vals):.1f}")
