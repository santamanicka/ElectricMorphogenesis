"""
Bulk-feature-only version of visualize_provenance_source_regions.py: the same "which bulk cells does
each boundary region target" question, but candidates are restricted to the 14 facial-feature block
cells (eye: 8, nose: 3, mouth: 3) instead of all 81 bulk cells.

This is the accurate source-centric complement to visualize_provenance_blocks_top5_boundaryOnly.py /
visualize_provenance_blocks_boundaryOnly.py: those restricted the ORIGINAL target-centric analysis's
candidate SOURCES to boundary cells only; this restricts the source-centric analysis's candidate
TARGETS to the feature blocks only, so both halves of the pairing look at exactly the boundary<->
feature relationship and nothing else. It directly closes the loop with the very first block-level
result (learned eye's top-5 reaches both sides while the ensemble average's top-5 stays top-row only)
by showing, spatially, which boundary group actually dominates each individual eye/nose/mouth cell.

Produces two figures per condition (learned, 100-seed ensemble):
  1. a dominance map, restricted to the 14 feature cells -- since it is a forced choice among the 5
     pooled groups (exactly as in the unrestricted dominance map), every feature cell still gets a
     definite winner; non-feature cells are masked out
  2. a top-5 feature-cell table per boundary group, each entry labelled by which block it belongs to

Uses the already-computed data/provenance_learned.npz and data/provenance_randomEnsemble100.npz
(step 899) -- no new simulation needed.

  python visualize_provenance_source_regions_bulkFeatureOnly.py
"""

import copy

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap

import utilities
from embryo import model

torch.set_grad_enabled(False)
TOP_K = 5

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

# --- five geometric source groups: north/south/east/west edge, corners -------------------------
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

# ==================================================================================================
# Figure 1: dominance map restricted to the 14 feature cells
# ==================================================================================================
fig, axes = plt.subplots(1, len(conditions), figsize=(6.4 * len(conditions), 5.6))

dominanceByCond = {}
for ax, (label, path) in zip(axes, conditions):
    d = np.load(path, allow_pickle=True)
    P = np.asarray(d['pVmem_step899'])

    groupShare = np.zeros((numCells, len(groupNames)))
    for gi, name in enumerate(groupNames):
        groupShare[:, gi] = P[:, groups[name]].sum(axis=1)

    dominant = np.full(numCells, -1)
    dominant[featureIndices] = np.argmax(groupShare[featureIndices], axis=1)
    dominanceByCond[label] = dominant

    grid = dominant.reshape(numRows, numCols).astype(float)
    cmap = ListedColormap([groupColor[n] for n in groupNames] + ['0.92'])
    displayGrid = np.where(grid == -1, len(groupNames), grid)
    ax.imshow(displayGrid, cmap=cmap, vmin=0, vmax=len(groupNames), interpolation='nearest')

    for name in groupNames:
        rs, cs = groups[name] // numCols, groups[name] % numCols
        ax.scatter(cs, rs, s=90, facecolors=groupColor[name], edgecolors='black', linewidths=1.3,
                  marker='s', zorder=5)
    for i in featureIndices:
        r, c = i // numCols, i % numCols
        ax.annotate(blockLetter[cellToBlock[int(i)]], xy=(c, r), fontsize=7, color='white',
                   fontweight='bold', ha='center', va='center', zorder=6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{label}', fontsize=11)

legendElems = [plt.Rectangle((0, 0), 1, 1, color=groupColor[n], label=f'{n} boundary') for n in groupNames]
fig.legend(handles=legendElems, loc='lower center', ncol=5, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Which boundary region dominates each facial-feature cell (top-1 contributing group, t=899)\n'
            'letters mark block membership (E=eye, N=nose, M=mouth); all other bulk cells masked out',
            fontsize=10.5, y=1.05)
plt.savefig('figures/provenanceSourceRegions_bulkFeatureOnly.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceSourceRegions_bulkFeatureOnly.png")

for label in dominanceByCond:
    print(f"\n{label}: dominant group per feature cell")
    for name in blocks:
        winners = [groupNames[dominanceByCond[label][i]] for i in blocks[name]]
        print(f"  {name:<6}: {winners}")

agree = (dominanceByCond['learned'][featureIndices] == dominanceByCond['random (100-seed ensemble)'][featureIndices])
print(f"\nDominant-group agreement between learned and ensemble, feature cells: "
     f"{agree.sum()}/{len(featureIndices)} ({agree.mean()*100:.1f}%)")

# ==================================================================================================
# Figure 2: top-5 feature-cell targets per boundary group, learned vs. ensemble
# ==================================================================================================
PAD = 1
panelData = {}
rowBox = {}
table = []

for name in groupNames:
    srcRows, srcCols = groups[name] // numCols, groups[name] % numCols
    unionRows, unionCols = [srcRows], [srcCols]
    for colIdx, (label, path) in enumerate(conditions):
        d = np.load(path, allow_pickle=True)
        P = np.asarray(d['pVmem_step899'])
        colSum = P[:, groups[name]].sum(axis=1)
        order = featureIndices[np.argsort(-colSum[featureIndices])][:TOP_K]
        meanFeatureShare = colSum[featureIndices].mean()
        panelData[(name, colIdx)] = dict(order=order, colSum=colSum, meanFeatureShare=meanFeatureShare)
        table.append((name, label, [(int(t), float(colSum[t])) for t in order]))
        unionRows.append(order // numCols); unionCols.append(order % numCols)
    allRows, allCols = np.concatenate(unionRows), np.concatenate(unionCols)
    rMin, rMax = max(allRows.min() - PAD, 0), min(allRows.max() + PAD, numRows - 1)
    cMin, cMax = max(allCols.min() - PAD, 0), min(allCols.max() + PAD, numCols - 1)
    rowBox[name] = (rMin, rMax, cMin, cMax)

UNIT = 0.40
TITLE_H = 0.55
ROW_GAP = 0.30
COL_GAP = 0.5
MARGIN = 0.4
TOP_MARGIN = 0.55
BOTTOM_MARGIN = 0.55

rowHeightsIn = {n: (rowBox[n][1] - rowBox[n][0] + 1) * UNIT + TITLE_H for n in groupNames}
rowWidthsIn = {n: (rowBox[n][3] - rowBox[n][2] + 1) * UNIT for n in groupNames}
rowTotalWidthIn = {n: 2 * rowWidthsIn[n] + COL_GAP for n in groupNames}
figW = 2 * MARGIN + max(rowTotalWidthIn.values())
figH = TOP_MARGIN + BOTTOM_MARGIN + sum(rowHeightsIn.values()) + ROW_GAP * (len(groupNames) - 1)

fig2 = plt.figure(figsize=(figW, figH))
yCursor = figH - TOP_MARGIN
for name in groupNames:
    rMin, rMax, cMin, cMax = rowBox[name]
    rowH = rowHeightsIn[name]
    w = rowWidthsIn[name]
    yCursor -= rowH
    xCursor = (figW - rowTotalWidthIn[name]) / 2
    for colIdx, (label, path) in enumerate(conditions):
        info = panelData[(name, colIdx)]
        order, colSum, meanFeatureShare = info['order'], info['colSum'], info['meanFeatureShare']
        ax = fig2.add_axes([xCursor / figW, yCursor / figH, w / figW, (rowH - TITLE_H) / figH])
        xCursor += w + COL_GAP

        srcRows, srcCols = groups[name] // numCols, groups[name] % numCols
        ax.scatter(srcCols, srcRows, marker='s', s=140, color=groupColor[name],
                  edgecolors='black', linewidths=1.2, zorder=6)
        topShares = colSum[order]
        for rank, tgt in enumerate(order, 1):
            share = colSum[tgt]
            tr, tc = tgt // numCols, tgt % numCols
            size = 700 * share / max(topShares.max(), 1e-9) + 70
            ax.scatter([tc], [tr], s=size, color='#DD8452', alpha=0.85, edgecolors='0.2',
                      linewidths=0.8, zorder=5)
            ax.annotate(f"{rank}{blockLetter[cellToBlock[int(tgt)]]}", xy=(tc, tr), fontsize=6.5,
                       color='white', fontweight='bold', ha='center', va='center', zorder=7)
        ax.set_xlim(cMin - 0.6, cMax + 0.6); ax.set_ylim(rMax + 0.6, rMin - 0.6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('0.75')
        ax.set_title(f'{name}: {label}\nmean feature-cell share {meanFeatureShare*100:.2f}%', fontsize=8.5)
    yCursor -= ROW_GAP

legendElems = [plt.scatter([], [], marker='s', s=110, color='0.5', edgecolors='black',
                          linewidths=1.2, label='boundary group members'),
              plt.scatter([], [], color='#DD8452', s=90, label='top-5 feature-cell target '
                          '(rank + block letter shown)')]
fig2.legend(handles=legendElems, loc='lower center', ncol=1, fontsize=9,
           bbox_to_anchor=(0.5, 0.0), bbox_transform=fig2.transFigure)
fig2.suptitle(f'Top-{TOP_K} facial-feature-cell targets of each boundary group, learned vs. 100-seed '
             'ensemble (t=899)\n(E=eye, N=nose, M=mouth)', fontsize=10.5, y=0.998)
plt.savefig('figures/provenanceSourceRegionsTop5_bulkFeatureOnly.png', dpi=150)
plt.close()
print("Saved figures/provenanceSourceRegionsTop5_bulkFeatureOnly.png")

print(f"\n{'group':<8}{'condition':<28}{'mean feature share':>20}   top-5 targets (row,col,block,share%)")
for name, label, entries in table:
    d = np.load(dict(conditions)[label], allow_pickle=True)
    P = np.asarray(d['pVmem_step899'])
    meanFeatureShare = P[:, groups[name]].sum(axis=1)[featureIndices].mean()
    tgtStr = "; ".join(f"({t//numCols},{t%numCols}) {cellToBlock[t]} {s*100:.2f}%" for t, s in entries)
    print(f"{name:<8}{label:<28}{meanFeatureShare*100:>19.2f}%   {tgtStr}")
