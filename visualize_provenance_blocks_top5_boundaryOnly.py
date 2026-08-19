"""
Boundary-only version of visualize_provenance_blocks_top5.py: same top-5-contributors-to-each-
facial-feature-block analysis, but the ranking is restricted to boundary cells only (bulk cells
excluded from candidacy entirely, not merely outranked).

This is the correct "inward" complement to the source-centric analysis in
visualize_provenance_source_regions.py / visualize_provenance_source_spectrum.py, which only ever
considered boundary cells as sources. The original (unrestricted) provenanceBlocksTop5.png mixes
boundary and bulk candidates in one ranking, so a block whose leading contributors happen to be bulk
cells (nose) shows no boundary cells in its top-5 at all -- which answers "who contributes most" but
not "which boundary cells matter most", the question this script answers instead.

  python visualize_provenance_blocks_top5_boundaryOnly.py
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
boundaryIndices = np.array(sorted(int(i) for i in utils.computeDomeIndices(m.electricNetwork, mode='tissue')))

typeColor = {'block-member': '#4C72B0', 'boundary': '#C44E52'}
panels = [(name, label) for name in blocks for label in ('learned', 'random')]

fig = plt.figure(figsize=(4.6 * 2, 3.6 * 3))
outerGs = gridspec.GridSpec(3, 2, wspace=0.55, hspace=0.55)

for panelIdx, (name, label) in enumerate(panels):
    idxs = blocks[name]
    row = d[f'{name}_{label}'].copy()

    candidates = np.setdiff1d(boundaryIndices, idxs)   # boundary cells only; no block member is one
    order = candidates[np.argsort(-row[candidates])][:TOP_K]
    topShares = row[order]
    totalBoundaryShare = row[boundaryIndices].sum()

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
        sr, sc = src // numCols, src % numCols
        size = 900 * share / max(topShares.max(), 1e-9) + 90
        axMap.scatter([sc], [sr], s=size, color=typeColor['boundary'], alpha=0.8, edgecolors='0.2',
                     linewidths=0.9, zorder=5)
        axMap.annotate(str(rank), xy=(sc, sr), fontsize=8, color='white', fontweight='bold',
                      ha='center', va='center', zorder=7)
        lines.append((rank, sr, sc, share))

    axMap.set_xlim(-0.7, numCols - 0.3); axMap.set_ylim(numRows - 0.3, -0.7)
    axMap.set_xticks([]); axMap.set_yticks([])
    for spine in axMap.spines.values():
        spine.set_edgecolor('0.75')
    axMap.set_title(f'{name}: {label}', fontsize=10)

    tableText = (f"total boundary share: {totalBoundaryShare*100:4.1f}%\n"
                f"(top-5 of {len(candidates)} boundary cells)\n" +
                "\n".join(f"{rank}. ({sr:>2},{sc:>2})  {share*100:4.2f}%" for rank, sr, sc, share in lines))
    axText.text(0.0, 0.92, tableText, transform=axText.transAxes, fontsize=8.5, family='monospace',
               va='top', ha='left')

legendElems = [plt.scatter([], [], marker='*', s=200, color='white', edgecolors=typeColor['block-member'],
                          linewidths=2, label='block member cell'),
              plt.scatter([], [], color=typeColor['boundary'], s=100, label='boundary source (rank shown)')]
fig.legend(handles=legendElems, loc='lower center', ncol=2, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
fig.suptitle(f'Top-{TOP_K} BOUNDARY-ONLY contributors to whole facial-feature blocks, learned vs random\n'
            '(candidates restricted to the 40 boundary cells; bulk cells excluded from the ranking entirely)',
            fontsize=10.5, y=1.02)
plt.savefig('figures/provenanceBlocksTop5_boundaryOnly.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceBlocksTop5_boundaryOnly.png")

print("\nExact boundary-only top-5 table:")
for name, label in panels:
    idxs = blocks[name]
    row = d[f'{name}_{label}']
    candidates = np.setdiff1d(boundaryIndices, idxs)
    order = candidates[np.argsort(-row[candidates])][:TOP_K]
    totalBoundaryShare = row[boundaryIndices].sum()
    print(f"\n{name} ({label}), total boundary share = {totalBoundaryShare*100:.1f}%:")
    for rank, src in enumerate(order, 1):
        print(f"  #{rank}: cell {src:>3} (row {src//numCols}, col {src%numCols}) — {row[src]*100:.2f}%")
