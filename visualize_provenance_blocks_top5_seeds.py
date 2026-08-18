"""
Top-5 contributors to whole facial-feature blocks: learned vs 4 INDIVIDUAL random seeds (not
the 100-seed ensemble average), reusing the step-899 snapshots already computed for the
unfolding-over-time comparison. Individual seeds are the fair comparison here, since averaging
would smooth over exactly the seed-to-seed structure this is meant to show.

  python visualize_provenance_blocks_top5_seeds.py
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
STEP = 899

blocks = {
    'eye':   np.array([24, 25, 35, 36, 29, 30, 40, 41]),
    'nose':  np.array([49, 60, 71]),
    'mouth': np.array([92, 93, 94]),
}
numRows, numCols = 11, 11
numCells = numRows * numCols

utils = utilities.utilities()
p = copy.deepcopy(torch.load('data/StigmergicModelParameters.dat', weights_only=False))
p['ATPParameters'] = None; p['latticePeriodicBoundaryGJ'] = False
m = model(p, 1)
boundaryIndices = set(int(i) for i in utils.computeDomeIndices(m.electricNetwork, mode='tissue'))
typeColor = {'boundary': '#C44E52', 'bulk': '#55A868'}

conditions = [('learned', 'data/provenanceTimecourse_learned.npz')] + \
            [(f'seed {s}', f'data/provenanceTimecourse_seed{s}_random.npz') for s in range(1, 5)]

fig = plt.figure(figsize=(4.4 * len(conditions), 3.5 * len(blocks)))
outerGs = gridspec.GridSpec(len(blocks), len(conditions), wspace=0.5, hspace=0.55)

table = []
for rowIdx, (name, idxs) in enumerate(blocks.items()):
    for colIdx, (label, path) in enumerate(conditions):
        d = np.load(path, allow_pickle=True)
        row = np.asarray(d[f'pVmem_step{STEP}'])[idxs].mean(axis=0)
        others = np.setdiff1d(np.arange(numCells), idxs)
        order = others[np.argsort(-row[others])][:TOP_K]
        topShares = row[order]
        memberShare = row[idxs].sum()

        cellGs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outerGs[rowIdx, colIdx])
        ax = fig.add_subplot(cellGs[0])
        memberRows, memberCols = idxs // numCols, idxs % numCols
        ax.scatter(memberCols, memberRows, marker='*', s=200, color='white',
                  edgecolors='#4C72B0', linewidths=1.8, zorder=6)
        for rank, src in enumerate(order, 1):
            share = row[src]
            srcType = 'boundary' if src in boundaryIndices else 'bulk'
            sr, sc = src // numCols, src % numCols
            size = 700 * share / max(topShares.max(), 1e-9) + 70
            ax.scatter([sc], [sr], s=size, color=typeColor[srcType], alpha=0.8, edgecolors='0.2',
                      linewidths=0.8, zorder=5)
            ax.annotate(str(rank), xy=(sc, sr), fontsize=7, color='white', fontweight='bold',
                       ha='center', va='center', zorder=7)
        ax.set_xlim(-0.7, numCols - 0.3); ax.set_ylim(numRows - 0.3, -0.7)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('0.75')
        title = f'{name}: {label}' if rowIdx == 0 or colIdx == 0 else f'{label}'
        ax.set_title(f'{name}: {label}\nself {memberShare*100:.1f}%', fontsize=8.5)

        boundaryCount = sum(1 for s in order if s in boundaryIndices)
        table.append((name, label, boundaryCount, memberShare, topShares.sum()))

legendElems = [plt.scatter([], [], marker='*', s=160, color='white', edgecolors='#4C72B0',
                          linewidths=1.8, label='block member cell'),
              plt.scatter([], [], color=typeColor['boundary'], s=90, label='boundary source'),
              plt.scatter([], [], color=typeColor['bulk'], s=90, label='bulk source')]
fig.legend(handles=legendElems, loc='lower center', ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, -0.01))
fig.suptitle(f'Top-{TOP_K} contributors to facial-feature blocks: learned vs 4 individual random '
            f'seeds, step {STEP}', fontsize=11, y=1.0)
plt.savefig('figures/provenanceBlocksTop5_seeds.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceBlocksTop5_seeds.png")

print(f"\n{'block':<8}{'condition':<10}{'#boundary in top5':>20}{'block self %':>14}{'top5 sum %':>12}")
for name, label, bCount, memberShare, top5Sum in table:
    print(f"{name:<8}{label:<10}{bCount:>20}{memberShare*100:>14.1f}{top5Sum*100:>12.1f}")
