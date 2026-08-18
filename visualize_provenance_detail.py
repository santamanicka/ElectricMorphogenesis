"""
Higher-resolution view of the exact provenance matrix: not just the bulk-vs-boundary aggregate,
but who specifically contributes to each cell, and how concentrated or diffuse that is.

  1. Concentration map: what fraction of each cell's provenance its top-5 sources capture, over
     all 121 cells -- distinguishes cells whose state traces back to a handful of dominant sources
     from cells whose identity is smeared thinly across the whole tissue.
  2. Top-1 source type map: is each cell's single largest contributor itself, a boundary cell, or
     another bulk cell -- a categorical, at-a-glance summary of where the story comes from.
  3. Detailed top-5 panels for a curated set of cells (the facial features, plus a corner, an
     edge-boundary cell, and the centre for contrast): exact rank, location, and share, marked
     spatially so location and magnitude read together rather than needing a separate table.

  python visualize_provenance_detail.py --input data/provenance_learned.npz --step 899
"""

import argparse
import copy

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import utilities
from embryo import model

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='data/provenance_learned.npz')
parser.add_argument('--step', type=int, default=899)
parser.add_argument('--sourceDat', type=str, default='data/StigmergicModelParameters.dat')
parser.add_argument('--topK', type=int, default=5)
parser.add_argument('--outputPrefix', type=str, default='figures/provenanceDetail')
args = parser.parse_args()

d = np.load(args.input, allow_pickle=True)
P = d[f'pVmem_step{args.step}']
numRows, numCols = d['latticeDims']
numCells = numRows * numCols

utils = utilities.utilities()
p = copy.deepcopy(torch.load(args.sourceDat, weights_only=False))
p['ATPParameters'] = None
p['latticePeriodicBoundaryGJ'] = False
m = model(p, 1)
boundaryIndices = np.array(utils.computeDomeIndices(m.electricNetwork, mode='tissue'))
boundaryMask = np.zeros(numCells, bool); boundaryMask[boundaryIndices] = True

eyeIndices = np.array([24, 25, 35, 36, 29, 30, 40, 41])
noseIndices = np.array([49, 60, 71])
mouthIndices = np.array([92, 93, 94])

# --- Per-cell top-K contributors -----------------------------------------------------------
order = np.argsort(-P, axis=1)[:, :args.topK]           # (numCells, topK) source indices, ranked
topShares = np.take_along_axis(P, order, axis=1)        # (numCells, topK)
topKSum = topShares.sum(axis=1)                          # concentration: share captured by top-K

# --- Overview figure -------------------------------------------------------------------------
fig1, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

concGrid = topKSum.reshape(numRows, numCols)
im0 = axes[0].imshow(concGrid, cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
axes[0].set_xticks([]); axes[0].set_yticks([])
axes[0].set_title(f'Concentration: share held by top {args.topK} contributors', fontsize=10)
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label=f'sum of top-{args.topK} shares')

top1 = order[:, 0]
category = np.where(top1 == np.arange(numCells), 0, np.where(boundaryMask[top1], 1, 2))
catGrid = category.reshape(numRows, numCols)
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#4C72B0', '#C44E52', '#55A868'])   # self, boundary, other-bulk
im1 = axes[1].imshow(catGrid, cmap=cmap, vmin=-0.5, vmax=2.5, interpolation='nearest')
axes[1].set_xticks([]); axes[1].set_yticks([])
axes[1].set_title('Identity of the #1 contributor', fontsize=10)
cbar = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['self', 'boundary', 'other bulk'])

fig1.suptitle(f'Provenance resolution overview, step {args.step}', fontsize=11, y=1.03)
plt.tight_layout()
plt.savefig(f'{args.outputPrefix}_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {args.outputPrefix}_overview.png")

# --- Detail figure: top-K contributors for curated cells --------------------------------------
curated = {
    'eye (24)': 24, 'eye (30)': 30,
    'nose (60, centre)': 60,
    'mouth (93)': 93,
    'corner (0)': 0,
    'edge boundary (5)': 5,
    'deep interior (48)': 48,
}

numPanels = len(curated)
numColsFig = 4
numRowsFig = int(np.ceil(numPanels / numColsFig))
fig2 = plt.figure(figsize=(4.6 * numColsFig, 3.5 * numRowsFig))
outerGs = gridspec.GridSpec(numRowsFig, numColsFig, wspace=0.55, hspace=0.5)

typeColor = {'self': '#4C72B0', 'boundary': '#C44E52', 'bulk': '#55A868'}
for panelIdx, (label, target) in enumerate(curated.items()):
    cellGs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outerGs[panelIdx],
                                              width_ratios=[1.3, 1], wspace=0.15)
    axMap = fig2.add_subplot(cellGs[0])
    axText = fig2.add_subplot(cellGs[1])
    axText.axis('off')

    tr, tc = target // numCols, target % numCols
    selfShareVal = P[target, target]
    starColor = typeColor['self'] if order[target, 0] == target else 'black'
    axMap.scatter([tc], [tr], marker='*', s=420, color='white', edgecolors=typeColor['self'],
                 linewidths=2.2, zorder=6)
    axMap.annotate('self', xy=(tc, tr), xytext=(tc, tr - 0.62), fontsize=6.5, color=typeColor['self'],
                  ha='center')

    lines = []
    for rank in range(args.topK):
        src = order[target, rank]
        share = topShares[target, rank]
        isSelf = src == target
        srcType = 'self' if isSelf else ('boundary' if boundaryMask[src] else 'bulk')
        sr, sc = src // numCols, src % numCols
        if not isSelf:
            size = 900 * share + 90
            axMap.scatter([sc], [sr], s=size, color=typeColor[srcType], alpha=0.8, edgecolors='0.2',
                         linewidths=0.9, zorder=5)
            axMap.annotate(str(rank + 1), xy=(sc, sr), fontsize=8, color='white', fontweight='bold',
                          ha='center', va='center', zorder=7)
        lines.append((rank + 1, sr, sc, srcType, share))

    axMap.set_xlim(-0.7, numCols - 0.3); axMap.set_ylim(numRows - 0.3, -0.7)
    axMap.set_xticks([]); axMap.set_yticks([])
    for spine in axMap.spines.values():
        spine.set_edgecolor('0.75')
    axMap.set_title(label, fontsize=9.5)

    tableText = "\n".join(f"{rank}. ({sr:>2},{sc:>2}) {t:<8} {share*100:4.1f}%" for rank, sr, sc, t, share in lines)
    axText.text(0.0, 0.92, tableText, transform=axText.transAxes, fontsize=8.5, family='monospace',
               va='top', ha='left')

legendElems = [plt.scatter([], [], marker='*', s=200, color='white', edgecolors=typeColor['self'],
                          linewidths=2, label='target cell'),
              plt.scatter([], [], color=typeColor['boundary'], s=100, label='boundary source'),
              plt.scatter([], [], color=typeColor['bulk'], s=100, label='bulk source')]
fig2.legend(handles=legendElems, loc='lower center', ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
fig2.suptitle(f'Top-{args.topK} contributors by cell (marker size = share, number = rank; '
             f'self share shown in the adjoining list), step {args.step}', fontsize=11, y=1.01)
plt.savefig(f'{args.outputPrefix}_top{args.topK}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {args.outputPrefix}_top{args.topK}.png")

print(f"\nExact top-{args.topK} table:")
for label, target in curated.items():
    print(f"\n{label} (cell {target}):")
    for rank in range(args.topK):
        src = order[target, rank]
        share = topShares[target, rank]
        srcType = 'self' if src == target else ('boundary' if boundaryMask[src] else 'bulk')
        print(f"  #{rank+1}: cell {src:>3} (row {src//numCols}, col {src%numCols}, {srcType:<8}) — {share*100:.1f}%")
