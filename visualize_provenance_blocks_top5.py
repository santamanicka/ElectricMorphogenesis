"""
Top-5 contributors to whole facial-feature blocks (eye/nose/mouth, pooled across their member
cells), learned vs random, in the same rank-numbered-marker + exact-table style as
visualize_provenance_detail.py's per-cell version.

  python visualize_provenance_blocks_top5.py
"""

import copy

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import utilities
from embryo import model

torch.set_grad_enabled(False)
TOP_K = 5

d = np.load('data/provenanceBlocks.npz', allow_pickle=True)
blocks = d['blocks'].item()
numRows, numCols = d['latticeDims']
numCells = numRows * numCols

utils = utilities.utilities()
p = copy.deepcopy(torch.load('data/StigmergicModelParameters.dat', weights_only=False))
p['ATPParameters'] = None; p['latticePeriodicBoundaryGJ'] = False
m = model(p, 1)
boundaryIndices = set(int(i) for i in utils.computeDomeIndices(m.electricNetwork, mode='tissue'))

typeColor = {'block-member': '#4C72B0', 'boundary': '#C44E52', 'bulk': '#55A868'}
panels = [(name, label) for name in blocks for label in ('learned', 'random')]

fig = plt.figure(figsize=(4.6 * 2, 3.6 * 3))
outerGs = gridspec.GridSpec(3, 2, wspace=0.55, hspace=0.55)

for panelIdx, (name, label) in enumerate(panels):
    idxs = blocks[name]
    row = d[f'{name}_{label}'].copy()

    # Rank all non-block-member cells; block members are shown separately (they are the target).
    others = np.setdiff1d(np.arange(numCells), idxs)
    order = others[np.argsort(-row[others])][:TOP_K]
    topShares = row[order]

    cellGs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outerGs[panelIdx],
                                              width_ratios=[1.3, 1], wspace=0.15)
    axMap = fig.add_subplot(cellGs[0])
    axText = fig.add_subplot(cellGs[1])
    axText.axis('off')

    memberRows, memberCols = idxs // numCols, idxs % numCols
    axMap.scatter(memberCols, memberRows, marker='*', s=260, color='white',
                 edgecolors=typeColor['block-member'], linewidths=2.0, zorder=6)

    lines = []
    for rank, src in enumerate(order, 1):
        share = row[src]
        srcType = 'boundary' if src in boundaryIndices else 'bulk'
        sr, sc = src // numCols, src % numCols
        size = 900 * share / max(topShares.max(), 1e-9) + 90
        axMap.scatter([sc], [sr], s=size, color=typeColor[srcType], alpha=0.8, edgecolors='0.2',
                     linewidths=0.9, zorder=5)
        axMap.annotate(str(rank), xy=(sc, sr), fontsize=8, color='white', fontweight='bold',
                      ha='center', va='center', zorder=7)
        lines.append((rank, sr, sc, srcType, share))

    memberShare = row[idxs].sum()
    axMap.set_xlim(-0.7, numCols - 0.3); axMap.set_ylim(numRows - 0.3, -0.7)
    axMap.set_xticks([]); axMap.set_yticks([])
    for spine in axMap.spines.values():
        spine.set_edgecolor('0.75')
    axMap.set_title(f'{name}: {label}', fontsize=10)

    tableText = f"block members (self): {memberShare*100:4.1f}%\n" + "\n".join(
        f"{rank}. ({sr:>2},{sc:>2}) {t:<8} {share*100:4.1f}%" for rank, sr, sc, t, share in lines)
    axText.text(0.0, 0.92, tableText, transform=axText.transAxes, fontsize=8.5, family='monospace',
               va='top', ha='left')

legendElems = [plt.scatter([], [], marker='*', s=200, color='white', edgecolors=typeColor['block-member'],
                          linewidths=2, label='block member cell'),
              plt.scatter([], [], color=typeColor['boundary'], s=100, label='boundary source'),
              plt.scatter([], [], color=typeColor['bulk'], s=100, label='bulk source')]
fig.legend(handles=legendElems, loc='lower center', ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
fig.suptitle(f'Top-{TOP_K} contributors to whole facial-feature blocks, learned vs random '
            '(pooled across block member cells, excluding other block members from the ranking)',
            fontsize=10.5, y=1.0)
plt.savefig('figures/provenanceBlocksTop5.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceBlocksTop5.png")

print("\nExact top-5 table:")
for name, label in panels:
    idxs = blocks[name]
    row = d[f'{name}_{label}']
    others = np.setdiff1d(np.arange(numCells), idxs)
    order = others[np.argsort(-row[others])][:TOP_K]
    print(f"\n{name} ({label}), block members share (self) = {row[idxs].sum()*100:.1f}%:")
    for rank, src in enumerate(order, 1):
        srcType = 'boundary' if src in boundaryIndices else 'bulk'
        print(f"  #{rank}: cell {src:>3} (row {src//numCols}, col {src%numCols}, {srcType:<8}) "
             f"— {row[src]*100:.2f}%")
