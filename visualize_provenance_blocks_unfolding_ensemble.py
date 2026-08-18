"""
Filmstrip of top-5 contributor maps for each facial-feature block, at several points along the
trajectory, learned vs the 100-seed random ensemble average -- shows whether the same cells stay
dominant over time (stable identity) or different cells cycle in and out (shifting identity).

Supersedes visualize_provenance_blocks_unfolding.py's single-random-seed comparison for the
main-text result; that version is kept separately in the Appendix as the individual-seed view.

  python visualize_provenance_blocks_unfolding_ensemble.py
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
TIME_POINTS = [50, 150, 300, 500, 700, 899]

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

ensembleData = np.load('data/provenanceTimecourse_randomEnsemble100.npz', allow_pickle=True)
numSeeds = int(ensembleData['numSeeds'])
learnedData = np.load('data/provenanceTimecourse_learned.npz', allow_pickle=True)

conditions = [('learned', 'learned'), (f'random ({numSeeds}-seed ensemble)', 'ensemble')]
rowSpecs = [(name, label) for name in blocks for label, _ in conditions]
fig = plt.figure(figsize=(2.6 * len(TIME_POINTS), 2.7 * len(rowSpecs)))
outerGs = gridspec.GridSpec(len(rowSpecs), len(TIME_POINTS), wspace=0.15, hspace=0.5)

rowIdx = 0
for name, idxs in blocks.items():
    for label, kind in conditions:
        for colIdx, t in enumerate(TIME_POINTS):
            if kind == 'learned':
                row = np.asarray(learnedData[f'pVmem_step{t}'])[idxs].mean(axis=0)
            else:
                row = np.asarray(ensembleData[f'{name}_pVmem_step{t}'])
            others = np.setdiff1d(np.arange(numCells), idxs)
            order = others[np.argsort(-row[others])][:TOP_K]
            topShares = row[order]

            ax = fig.add_subplot(outerGs[rowIdx, colIdx])
            memberRows, memberCols = idxs // numCols, idxs % numCols
            ax.scatter(memberCols, memberRows, marker='*', s=110, color='white',
                      edgecolors='#4C72B0', linewidths=1.4, zorder=6)
            for rank, src in enumerate(order, 1):
                share = row[src]
                srcType = 'boundary' if src in boundaryIndices else 'bulk'
                sr, sc = src // numCols, src % numCols
                size = 550 * share / max(topShares.max(), 1e-9) + 45
                ax.scatter([sc], [sr], s=size, color=typeColor[srcType], alpha=0.8,
                          edgecolors='0.2', linewidths=0.7, zorder=5)
                ax.annotate(str(rank), xy=(sc, sr), fontsize=6, color='white', fontweight='bold',
                          ha='center', va='center', zorder=7)
            ax.set_xlim(-0.7, numCols - 0.3); ax.set_ylim(numRows - 0.3, -0.7)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('0.8')
            if rowIdx == 0:
                ax.set_title(f't={t}', fontsize=9)
            if colIdx == 0:
                ax.set_ylabel(f'{name}\n{label}', fontsize=8.5)
        rowIdx += 1

legendElems = [plt.scatter([], [], marker='*', s=110, color='white', edgecolors='#4C72B0',
                          linewidths=1.4, label='block member cell'),
              plt.scatter([], [], color=typeColor['boundary'], s=70, label='boundary source'),
              plt.scatter([], [], color=typeColor['bulk'], s=70, label='bulk source')]
fig.legend(handles=legendElems, loc='lower center', ncol=3, fontsize=9.5, bbox_to_anchor=(0.5, -0.005))
fig.suptitle(f'Top-5 contributor maps over time: learned vs {numSeeds}-seed random ensemble, '
            'by facial-feature block', fontsize=11.5, y=0.995)
plt.savefig('figures/provenanceBlocksUnfoldingEnsemble.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceBlocksUnfoldingEnsemble.png")
