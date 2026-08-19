"""
Full-bulk version of visualize_provenance_signed_confidence_bulkFeatureOnly.py: the same two
figures (confidence-weighted dominance map from exact signed group totals, and the sign-disagreement
map), but covering all 81 bulk cells instead of only the 14 facial-feature cells. Facial-feature
cells are outlined for reference.

  python visualize_provenance_signed_confidence.py
"""

import copy

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import utilities
from embryo import model

torch.set_grad_enabled(False)

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

MARGIN_SATURATE = 0.30
MIN_ALPHA = 0.12

# ==================================================================================================
# Figure 1: confidence-weighted dominance map, all 81 bulk cells
# ==================================================================================================
fig, axes = plt.subplots(1, len(conditions), figsize=(6.4 * len(conditions), 5.6))

allMargins = {}
for ax, (label, path) in zip(axes, conditions):
    d = np.load(path, allow_pickle=True)
    key = [k for k in d.keys() if k.startswith('vVmem_step')][0]
    V = d[key]

    groupTotals = np.zeros((numCells, len(groupNames)))
    for gi, name in enumerate(groupNames):
        groupTotals[:, gi] = V[:, groups[name]].sum(axis=1)

    rgbaGrid = np.zeros((numRows, numCols, 4))
    rgbaGrid[..., :3] = 0.92; rgbaGrid[..., 3] = 1.0
    margins = {}
    for t in bulkIndices:
        t = int(t)
        totals = np.abs(groupTotals[t])
        order = np.argsort(-totals)
        winner, runner = totals[order[0]], totals[order[1]]
        margin = (winner - runner) / (winner + runner)
        margins[t] = margin
        alpha = MIN_ALPHA + (1 - MIN_ALPHA) * min(margin / MARGIN_SATURATE, 1.0)
        r, c = t // numCols, t % numCols
        rgbaGrid[r, c] = to_rgba(groupColor[groupNames[order[0]]], alpha)
    allMargins[label] = margins

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
    meanMargin = np.mean(list(margins.values()))
    ax.set_title(f'{label}\nmean margin {meanMargin:.3f} (exact signed values, all 81 bulk cells)',
                fontsize=10.5)

legendElems = [plt.Rectangle((0, 0), 1, 1, color=groupColor[n], label=f'{n} boundary') for n in groupNames]
legendElems.append(plt.Line2D([0], [0], marker='o', color='none', markeredgecolor='black',
                              markerfacecolor='none', markersize=10, label='facial-feature cell'))
fig.legend(handles=legendElems, loc='lower center', ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Signed-value confidence-weighted dominance map, all 81 bulk cells: hue = winning '
            'boundary group\nby exact signed mV group total, opacity = margin over the runner-up '
            '(circles mark facial-feature cells)', fontsize=10.3, y=1.06)
plt.savefig('figures/provenanceSignedConfidence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceSignedConfidence.png")

meanL = np.mean(list(allMargins['learned'].values()))
meanE = np.mean(list(allMargins['random (100-seed ensemble)'].values()))
print(f"mean margin over all 81 bulk cells: learned {meanL:.3f}, ensemble {meanE:.3f}")

# ==================================================================================================
# Figure 2: sign-disagreement map, all 81 bulk cells
# ==================================================================================================
gridsByCond = {}
for label, path in conditions:
    d = np.load(path, allow_pickle=True)
    key = [k for k in d.keys() if k.startswith('vVmem_step')][0]
    V = d[key]
    finalVmem = d['finalVmem']

    disagreeMag = np.zeros(numCells)
    disagreeCount = np.zeros(numCells, dtype=int)
    for t in bulkIndices:
        t = int(t)
        row = V[t]
        targetSign = np.sign(finalVmem[t])
        opp = np.sign(row) != targetSign
        disagreeCount[t] = int(opp.sum())
        disagreeMag[t] = np.abs(row[opp]).sum()
    gridsByCond[label] = (disagreeMag.reshape(numRows, numCols), int(disagreeCount.sum()))

sharedVmax = max(g.max() for g, _ in gridsByCond.values())
sharedVmax = sharedVmax if sharedVmax > 0 else 1.0

fig2, axes2 = plt.subplots(1, len(conditions), figsize=(6.4 * len(conditions), 5.6))
for ax, (label, path) in zip(axes2, conditions):
    grid, totalPairs = gridsByCond[label]
    im = ax.imshow(grid, cmap='inferno', vmin=0, vmax=sharedVmax, interpolation='nearest')
    for t in featureIndices:
        t = int(t)
        r, c = t // numCols, t % numCols
        ax.add_patch(plt.Circle((c, r), 0.42, facecolor='none', edgecolor='cyan', linewidth=1.2))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{label}\n{totalPairs} opposite-sign (target,source) pairs of '
                f'{len(bulkIndices)*numCells} (81 bulk targets x 121 sources)', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='total opposing magnitude (mV)')

fig2.suptitle('Where contributors actually pull opposite to their target, all 81 bulk cells (cyan '
             'circles = facial-feature cells)', fontsize=10.7, y=1.05)
plt.savefig('figures/provenanceSignedDisagreement_allBulk.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceSignedDisagreement_allBulk.png")
