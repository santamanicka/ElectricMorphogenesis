"""
Visualize the exact provenance matrix from measure_provenance_propagation.py: for every cell,
what share of its current state (excluding its own self-persistence) traces back to bulk cells
versus boundary cells, and specifically to the eye/nose/mouth cells versus generic skin.

  python visualize_provenance.py --input data/provenance_learned.npz --step 899
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
parser.add_argument('--outputPrefix', type=str, default='figures/provenance')
args = parser.parse_args()

d = np.load(args.input, allow_pickle=True)
P = d[f'pVmem_step{args.step}']   # (numCells, numCells): row i = provenance of cell i over sources
finalVmem = d['finalVmem']
numRows, numCols = d['latticeDims']
numCells = numRows * numCols
V_TH_MV = -27.0

utils = utilities.utilities()
p = copy.deepcopy(torch.load(args.sourceDat, weights_only=False))
p['ATPParameters'] = None
p['latticePeriodicBoundaryGJ'] = False
m = model(p, 1)
boundaryIndices = np.array(utils.computeDomeIndices(m.electricNetwork, mode='tissue'))
boundaryMask = np.zeros(numCells, bool); boundaryMask[boundaryIndices] = True
interiorMask = ~boundaryMask
interiorIndices = np.arange(numCells)[interiorMask]

eyeIndices = np.array([24, 25, 35, 36, 29, 30, 40, 41])
noseIndices = np.array([49, 60, 71])
mouthIndices = np.array([92, 93, 94])
featureSets = {'eye': eyeIndices, 'nose': noseIndices, 'mouth': mouthIndices}

# --- Bulk vs. boundary share per target cell, excluding self-attribution -----------------------
selfShare = np.diag(P)
otherMask = ~np.eye(numCells, dtype=bool)
boundaryShare = (P * boundaryMask[None, :] * otherMask).sum(1)
bulkShare = (P * interiorMask[None, :] * otherMask).sum(1)
otherTotal = boundaryShare + bulkShare
ratio = np.full(numCells, np.nan)
valid = otherTotal > 1e-12
ratio[valid] = np.log2((bulkShare[valid] + 1e-300) / (boundaryShare[valid] + 1e-300))

patternGrid = finalVmem.reshape(numRows, numCols)
ratioGrid = ratio.reshape(numRows, numCols)
selfGrid = selfShare.reshape(numRows, numCols)

fig = plt.figure(figsize=(15, 4.6))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.4)

ax0 = fig.add_subplot(gs[0])
vLimit = np.abs(patternGrid - V_TH_MV).max()
im0 = ax0.imshow(patternGrid, cmap='RdBu_r', vmin=V_TH_MV - vLimit, vmax=V_TH_MV + vLimit, interpolation='nearest')
ax0.set_xticks([]); ax0.set_yticks([])
ax0.set_title(f'Pattern at step {args.step}', fontsize=10)
plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label='Vmem (mV)')

ax1 = fig.add_subplot(gs[1])
vabs = np.nanmax(np.abs(ratioGrid))
im1 = ax1.imshow(ratioGrid, cmap='RdBu', vmin=-vabs, vmax=vabs, interpolation='nearest')
for label, idxSet in featureSets.items():
    r, c = idxSet // numCols, idxSet % numCols
    ax1.scatter(c, r, s=70, facecolors='none', edgecolors='black', linewidths=1.4)
ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_title('Bulk- vs. boundary-represented\n(blue=bulk, red=boundary; exact)', fontsize=10)
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='log2(bulk share / boundary share)')

ax2 = fig.add_subplot(gs[2])
im2 = ax2.imshow(selfGrid, cmap='viridis', interpolation='nearest')
ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_title('Self-persistence share\n(how much is still "itself")', fontsize=10)
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='diag(P): self share')

fig.suptitle(f'Exact provenance at step {args.step} (learned clamp)', fontsize=11, y=1.03)
plt.tight_layout()
plt.savefig(f'{args.outputPrefix}_step{args.step}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {args.outputPrefix}_step{args.step}.png")

# --- Depth-controlled eye/nose/mouth check, exact and full-coverage ----------------------------
rows = interiorIndices // numCols
colsArr = interiorIndices % numCols
depth = np.minimum(np.minimum(rows, numRows - 1 - rows), np.minimum(colsArr, numCols - 1 - colsArr))
boundaryShareInterior = boundaryShare[interiorIndices]

print(f"\nDepth-controlled boundary-share, exact (step {args.step}):")
print(f"{'feature':<10}{'depth':<8}{'boundary share (feature)':<28}{'boundary share (same-depth others)':<38}{'ratio'}")
for label, idxSet in featureSets.items():
    mask = np.isin(interiorIndices, idxSet)
    for d_ in np.unique(depth[mask]):
        atDepthFeature = mask & (depth == d_)
        atDepthOther = (~np.isin(interiorIndices, np.concatenate(list(featureSets.values())))) & (depth == d_)
        if atDepthOther.sum() == 0:
            continue
        featVal = boundaryShareInterior[atDepthFeature].mean()
        otherVal = boundaryShareInterior[atDepthOther].mean()
        print(f"{label:<10}{d_:<8}{featVal:<28.4f}{otherVal:<38.4f}{featVal/otherVal:.2f}")
