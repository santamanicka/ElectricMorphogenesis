"""
Compare pooled facial-feature-block provenance (whole eye/nose/mouth regions, not individual
cells) between the learned clamp and the 100-seed random ensemble, using the already-computed
full provenance matrices at step 899.

  python visualize_provenance_blocks.py
"""

import numpy as np
import matplotlib.pyplot as plt

d = np.load('data/provenanceBlocks.npz', allow_pickle=True)
blocks = d['blocks'].item()
numRows, numCols = d['latticeDims']
numCells = numRows * numCols

fig, axes = plt.subplots(3, 3, figsize=(11, 10.5))
for row, (name, idxs) in enumerate(blocks.items()):
    learned = d[f'{name}_learned'].reshape(numRows, numCols)
    random_ = d[f'{name}_random'].reshape(numRows, numCols)
    diff = learned - random_
    vabs = max(np.abs(learned).max(), np.abs(random_).max())
    dabs = np.abs(diff).max()

    for col, (grid, title, vlim, cmap) in enumerate([
        (learned, f'{name}: learned', vabs, 'viridis'),
        (random_, f'{name}: random (100-seed)', vabs, 'viridis'),
        (diff, f'{name}: learned - random', dabs, 'RdBu_r'),
    ]):
        ax = axes[row, col]
        vmin = 0 if cmap == 'viridis' else -vlim
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vlim, interpolation='nearest')
        r, c = idxs // numCols, idxs % numCols
        ax.scatter(c, r, s=40, facecolors='none', edgecolors='black' if cmap == 'viridis' else 'black',
                  linewidths=1.3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('figures/provenanceBlocks.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceBlocks.png")
