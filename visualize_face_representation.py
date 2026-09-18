"""
Visualize whether boundary cells preferentially represent facial features (eye/nose/mouth),
from the response operator already measured by measure_response_operator.py.

Three panels:
  1. Raw boundary sensitivity per interior cell (mean |response| across reliable boundary cells),
     which is dominated by depth-from-boundary -- cells near the edge respond more to any boundary
     cell, feature or not, simply by proximity.
  2. The same map divided by the mean response of OTHER cells at the same depth, which removes
     that proximity confound and isolates whatever is specific to being a facial feature.
  3. The depth-controlled ratio, summarised per feature.

  python visualize_face_representation.py --responseOperator data/responseOperator11x11_screen4.0_all.npz
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
parser.add_argument('--responseOperator', type=str, default='data/responseOperator11x11_screen4.0_all.npz')
parser.add_argument('--sourceDat', type=str, default='data/StigmergicModelParameters.dat')
parser.add_argument('--asymmetryThreshold', type=float, default=0.3,
                    help='exclude boundary cells whose opposite-perturbation asymmetry exceeds this')
parser.add_argument('--outputPrefix', type=str, default='figures/faceRepresentation')
args = parser.parse_args()

d = np.load(args.responseOperator, allow_pickle=True)
columns = d['columns'] * 1e-9   # response-operator units bug: stored per physical G_pol unit, not per G_ref
symmetry = d['symmetry']

numRows, numCols = 11, 11
numCells = numRows * numCols
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
faceIndices = np.concatenate(list(featureSets.values()))

rows = interiorIndices // numCols
colsArr = interiorIndices % numCols
depth = np.minimum(np.minimum(rows, numRows - 1 - rows), np.minimum(colsArr, numCols - 1 - colsArr))

reliable = symmetry < args.asymmetryThreshold
meanResp = np.abs(columns[reliable]).mean(axis=0)   # (numInterior,) mV per G_ref, averaged over reliable boundary cells

# Depth-controlled ratio: each interior cell's response divided by the mean response of every
# OTHER interior cell at the same depth (so a cell is never compared against itself).
depthRatio = np.zeros(len(interiorIndices))
for d_ in np.unique(depth):
    atDepth = depth == d_
    for i in np.where(atDepth)[0]:
        others = atDepth.copy()
        others[i] = False
        depthRatio[i] = meanResp[i] / meanResp[others].mean() if others.any() else np.nan

# --- Build full-grid arrays (NaN on boundary cells, which are not part of this readout) ---------
rawGrid = np.full(numCells, np.nan)
rawGrid[interiorIndices] = meanResp
rawGrid = rawGrid.reshape(numRows, numCols)

ratioGrid = np.full(numCells, np.nan)
ratioGrid[interiorIndices] = depthRatio
ratioGrid = ratioGrid.reshape(numRows, numCols)

markerStyle = {'eye': dict(marker='o', s=90, facecolors='none', edgecolors='black', linewidths=1.6),
              'nose': dict(marker='s', s=90, facecolors='none', edgecolors='black', linewidths=1.6),
              'mouth': dict(marker='^', s=100, facecolors='none', edgecolors='black', linewidths=1.6)}


def markFeatures(ax):
    for label, idxSet in featureSets.items():
        r, c = idxSet // numCols, idxSet % numCols
        ax.scatter(c, r, **markerStyle[label], label=label, zorder=5)


fig = plt.figure(figsize=(14, 4.6))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.45)

ax0 = fig.add_subplot(gs[0])
im0 = ax0.imshow(rawGrid, cmap='viridis', interpolation='nearest')
markFeatures(ax0)
ax0.set_xticks([]); ax0.set_yticks([])
ax0.set_title('Raw boundary sensitivity\n(dominated by depth from edge)', fontsize=10)
plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label='mean |response| (mV/G_ref)')

ax1 = fig.add_subplot(gs[1])
vabs = np.nanmax(np.abs(np.log2(ratioGrid[ratioGrid > 0])))
im1 = ax1.imshow(np.log2(ratioGrid), cmap='RdBu_r', vmin=-vabs, vmax=vabs, interpolation='nearest')
markFeatures(ax1)
ax1.set_xticks([]); ax1.set_yticks([])
ax1.set_title('Depth-controlled ratio\n(log2, vs. same-depth peers)', fontsize=10)
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='log2(response / same-depth mean)')
ax1.legend(handles=[plt.scatter([], [], **markerStyle[k], label=k) for k in featureSets],
          loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8, frameon=False)

ax2 = fig.add_subplot(gs[2])
barLabels, barValues, barColors = [], [], []
palette = {'eye': '#4C72B0', 'nose': '#DD8452', 'mouth': '#55A868'}
for label, idxSet in featureSets.items():
    mask = np.isin(interiorIndices, idxSet)
    for d_ in np.unique(depth[mask]):
        atDepthFeature = mask & (depth == d_)
        val = depthRatio[atDepthFeature].mean()
        if np.isnan(val):
            # The lattice's unique deepest cell (dead centre) has no same-depth peer to compare
            # against -- not a zero effect, an undefined one, so it is omitted rather than shown.
            continue
        barLabels.append(f'{label}\n(depth {d_})')
        barValues.append(val)
        barColors.append(palette[label])
bars = ax2.bar(range(len(barValues)), barValues, color=barColors, edgecolor='0.3', linewidth=0.8)
ax2.axhline(1.0, color='0.3', linestyle='--', linewidth=1, label='parity with same-depth peers')
ax2.set_xticks(range(len(barLabels))); ax2.set_xticklabels(barLabels, fontsize=8.5)
ax2.set_ylabel('response / same-depth mean', fontsize=9)
ax2.set_title('Depth-controlled elevation, by feature', fontsize=10)
ax2.set_ylim(0, max(barValues) * 1.28)
ax2.legend(fontsize=8, loc='upper left')
for spine in ['top', 'right']:
    ax2.spines[spine].set_visible(False)
for bar, val in zip(bars, barValues):
    ax2.annotate(f'{val:.2f}x', xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

fig.suptitle('Does the boundary preferentially represent facial features? '
            f'(learned clamp, {reliable.sum()}/{len(symmetry)} boundary cells with asym<{args.asymmetryThreshold})',
            fontsize=11, y=1.05)
plt.savefig(f'{args.outputPrefix}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {args.outputPrefix}.png")
