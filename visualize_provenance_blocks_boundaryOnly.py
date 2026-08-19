"""
Boundary-only version of visualize_provenance_blocks.py: the same learned/random/difference spatial
maps of each facial-feature block's incoming provenance, but with bulk (interior) cells masked out so
only the boundary ring's contribution values are shown. This is the correct "inward" complement to
the source-centric analysis (visualize_provenance_source_regions.py /
visualize_provenance_source_spectrum.py), which only ever considered boundary cells as sources --
the unrestricted provenanceBlocks.png colour-scales against the bulk cells too, which can wash out
real structure in the boundary ring itself (bulk values are often larger, e.g. for the nose).

  python visualize_provenance_blocks_boundaryOnly.py
"""

import copy

import numpy as np
import torch
import matplotlib.pyplot as plt

import utilities
from embryo import model

torch.set_grad_enabled(False)

d = np.load('data/provenanceBlocks.npz', allow_pickle=True)
blocks = d['blocks'].item()
numRows, numCols = d['latticeDims']
numCells = numRows * numCols

utils = utilities.utilities()
p = copy.deepcopy(torch.load('data/StigmergicModelParameters.dat', weights_only=False))
p['ATPParameters'] = None; p['latticePeriodicBoundaryGJ'] = False
m = model(p, 1)
boundaryIndices = np.array(sorted(int(i) for i in utils.computeDomeIndices(m.electricNetwork, mode='tissue')))
boundaryMask = np.zeros(numCells, dtype=bool)
boundaryMask[boundaryIndices] = True
boundaryMask2D = boundaryMask.reshape(numRows, numCols)

viridisMasked = plt.get_cmap('viridis').copy()
viridisMasked.set_bad('0.92')
rdbuMasked = plt.get_cmap('RdBu_r').copy()
rdbuMasked.set_bad('0.92')

fig, axes = plt.subplots(3, 3, figsize=(11, 10.5))
for row, (name, idxs) in enumerate(blocks.items()):
    learned = d[f'{name}_learned'].reshape(numRows, numCols)
    random_ = d[f'{name}_random'].reshape(numRows, numCols)
    diff = learned - random_

    learnedM = np.ma.masked_where(~boundaryMask2D, learned)
    randomM = np.ma.masked_where(~boundaryMask2D, random_)
    diffM = np.ma.masked_where(~boundaryMask2D, diff)

    vabs = max(learnedM.max(), randomM.max())
    dabs = np.abs(diffM).max()

    for col, (grid, title, vlim, cmap) in enumerate([
        (learnedM, f'{name}: learned (boundary only)', vabs, viridisMasked),
        (randomM, f'{name}: random (boundary only)', vabs, viridisMasked),
        (diffM, f'{name}: learned - random (boundary only)', dabs, rdbuMasked),
    ]):
        ax = axes[row, col]
        vmin = 0 if cmap is viridisMasked else -vlim
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vlim, interpolation='nearest')
        r, c = idxs // numCols, idxs % numCols
        ax.scatter(c, r, s=40, facecolors='none', edgecolors='black', linewidths=1.3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9.5)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('figures/provenanceBlocks_boundaryOnly.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceBlocks_boundaryOnly.png")
