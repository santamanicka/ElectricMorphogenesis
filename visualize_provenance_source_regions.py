"""
Source-centric complement to visualize_provenance_blocks_top5.py: instead of asking which cells a
target region owes its state to, ask which bulk cells a boundary region's provenance lands on.

There are 40 boundary cells versus 3 facial-feature blocks, so a per-cell top-5 (one panel per
source) would be forty panels wide. Instead the 40 boundary cells are pooled into five geometric
groups -- north/south/east/west edge (9 cells each) and corners (4 cells) -- the same degree-based
split (corner vs. edge) already used for the self-persistence analysis, extended with direction.
For a boundary group G and bulk target T, "how much of T's state traces to G" is the SUM of the
group's member columns in T's provenance row (a well-defined, correctly normalised quantity, since
a provenance row already sums to 1 across all sources; summing a subset of columns sums exactly that
subset's real share).

Produces two figures per condition (learned, 100-seed ensemble):
  1. a dominance map -- every bulk cell coloured by whichever of the 5 boundary groups contributes
     the most to it, which turns "40 top-5 lists" into one readable picture
  2. a top-5 bulk-cell table per boundary group (5 groups x 2 conditions), the direct numeric
     complement to visualize_provenance_blocks_top5.py's target-centric table

Uses the already-computed data/provenance_learned.npz and data/provenance_randomEnsemble100.npz
(step 899) -- no new simulation needed.

  python visualize_provenance_source_regions.py
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
bulkIndices = np.array([i for i in range(numCells) if i not in boundaryIndices])

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
    print(f"{name:>7}: {len(groups[name])} cells -> {list(groups[name])}")
assert sum(len(v) for v in groups.values()) == 40

groupColor = {'north': '#4C72B0', 'south': '#C44E52', 'west': '#55A868',
             'east': '#8172B2', 'corner': '#937860'}

conditions = [('learned', 'data/provenance_learned.npz'),
             ('random (100-seed ensemble)', 'data/provenance_randomEnsemble100.npz')]

# ==================================================================================================
# Figure 1: dominance map -- every bulk cell coloured by its top boundary-group contributor
# ==================================================================================================
groupNames = list(groups.keys())
fig, axes = plt.subplots(1, len(conditions), figsize=(6.4 * len(conditions), 5.6))

dominanceByCond = {}
for ax, (label, path) in zip(axes, conditions):
    d = np.load(path, allow_pickle=True)
    P = np.asarray(d['pVmem_step899'])

    groupShare = np.zeros((numCells, len(groupNames)))
    for gi, name in enumerate(groupNames):
        groupShare[:, gi] = P[:, groups[name]].sum(axis=1)

    dominant = np.full(numCells, -1)
    dominant[bulkIndices] = np.argmax(groupShare[bulkIndices], axis=1)
    dominanceByCond[label] = dominant

    grid = dominant.reshape(numRows, numCols).astype(float)
    cmap = ListedColormap([groupColor[n] for n in groupNames] + ['0.85'])
    displayGrid = np.where(grid == -1, len(groupNames), grid)
    ax.imshow(displayGrid, cmap=cmap, vmin=0, vmax=len(groupNames), interpolation='nearest')

    for name in groupNames:
        rs, cs = groups[name] // numCols, groups[name] % numCols
        ax.scatter(cs, rs, s=90, facecolors=groupColor[name], edgecolors='black', linewidths=1.3,
                  marker='s', zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{label}', fontsize=11)

legendElems = [plt.Rectangle((0, 0), 1, 1, color=groupColor[n], label=f'{n} boundary') for n in groupNames]
fig.legend(handles=legendElems, loc='lower center', ncol=5, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Which boundary region dominates each bulk cell (top-1 contributing group, t=899)\n'
            'squares mark boundary-cell group membership; interior cells shaded by dominant source',
            fontsize=10.5, y=1.03)
plt.savefig('figures/provenanceSourceDominanceMap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceSourceDominanceMap.png")

agree = (dominanceByCond['learned'][bulkIndices] == dominanceByCond['random (100-seed ensemble)'][bulkIndices])
print(f"\nDominant-group agreement between learned and ensemble, bulk cells: "
     f"{agree.sum()}/{len(bulkIndices)} ({agree.mean()*100:.1f}%)")
if (~agree).any():
    print("Cells where the dominant boundary group differs:")
    for i in bulkIndices[~agree]:
        r, c = i // numCols, i % numCols
        gl = groupNames[dominanceByCond['learned'][i]]
        ge = groupNames[dominanceByCond['random (100-seed ensemble)'][i]]
        print(f"  cell {i:>3} (row {r}, col {c}): learned={gl:<7} ensemble={ge:<7}")

# ==================================================================================================
# Figure 2: top-5 bulk targets per boundary group, learned vs. ensemble
# ==================================================================================================
PAD = 1
panelData = {}  # (name, colIdx) -> dict with order, colSum, meanBulkShare
rowBox = {}     # name -> (rMin, rMax, cMin, cMax), shared across both conditions in that row
table = []

for name in groupNames:
    srcRows, srcCols = groups[name] // numCols, groups[name] % numCols
    unionRows, unionCols = [srcRows], [srcCols]
    for colIdx, (label, path) in enumerate(conditions):
        d = np.load(path, allow_pickle=True)
        P = np.asarray(d['pVmem_step899'])
        colSum = P[:, groups[name]].sum(axis=1)
        order = bulkIndices[np.argsort(-colSum[bulkIndices])][:TOP_K]
        meanBulkShare = colSum[bulkIndices].mean()
        panelData[(name, colIdx)] = dict(order=order, colSum=colSum, meanBulkShare=meanBulkShare)
        table.append((name, label, [(int(t), float(colSum[t])) for t in order]))
        unionRows.append(order // numCols); unionCols.append(order % numCols)
    allRows, allCols = np.concatenate(unionRows), np.concatenate(unionCols)
    rMin, rMax = max(allRows.min() - PAD, 0), min(allRows.max() + PAD, numRows - 1)
    cMin, cMax = max(allCols.min() - PAD, 0), min(allCols.max() + PAD, numCols - 1)
    rowBox[name] = (rMin, rMax, cMin, cMax)

# --- lay out panels with axes sized to their actual content (no dead space) ---------------------
UNIT = 0.40          # inches per lattice cell
TITLE_H = 0.55        # inches reserved above each row for its titles
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
    xCursor = (figW - rowTotalWidthIn[name]) / 2  # centre this row's pair of panels
    for colIdx, (label, path) in enumerate(conditions):
        info = panelData[(name, colIdx)]
        order, colSum, meanBulkShare = info['order'], info['colSum'], info['meanBulkShare']
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
            ax.annotate(str(rank), xy=(tc, tr), fontsize=7, color='white', fontweight='bold',
                       ha='center', va='center', zorder=7)
        ax.set_xlim(cMin - 0.6, cMax + 0.6); ax.set_ylim(rMax + 0.6, rMin - 0.6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('0.75')
        ax.set_title(f'{name}: {label}\nmean bulk share {meanBulkShare*100:.2f}%', fontsize=8.5)
    yCursor -= ROW_GAP

legendElems = [plt.scatter([], [], marker='s', s=110, color='0.5', edgecolors='black',
                          linewidths=1.2, label='boundary group members'),
              plt.scatter([], [], color='#DD8452', s=90, label='top-5 bulk target (rank shown)')]
fig2.legend(handles=legendElems, loc='lower center', ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, 0.0), bbox_transform=fig2.transFigure)
fig2.suptitle(f'Top-{TOP_K} bulk targets of each boundary group, learned vs. 100-seed ensemble (t=899)',
             fontsize=10.5, y=0.995)
plt.savefig('figures/provenanceSourceRegionsTop5.png', dpi=150)
plt.close()
print("Saved figures/provenanceSourceRegionsTop5.png")

print(f"\n{'group':<8}{'condition':<28}{'mean bulk share':>17}   top-5 targets (row,col,share%)")
for name, label, entries in table:
    d = np.load(dict(conditions)[label], allow_pickle=True)
    P = np.asarray(d['pVmem_step899'])
    meanBulkShare = P[:, groups[name]].sum(axis=1)[bulkIndices].mean()
    tgtStr = "; ".join(f"({t//numCols},{t%numCols}) {s*100:.2f}%" for t, s in entries)
    print(f"{name:<8}{label:<28}{meanBulkShare*100:>16.2f}%   {tgtStr}")
