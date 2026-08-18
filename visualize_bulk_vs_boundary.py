"""
For every cell in the t=1000 pattern, is its state better explained by other bulk cells or by
the boundary? Blue = bulk-represented, red = boundary-represented, from the full response matrix
built by measure_bulk_vs_boundary_representation.py.

  python visualize_bulk_vs_boundary.py --input data/bulkVsBoundary_learned.npz
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='data/bulkVsBoundary_learned.npz')
parser.add_argument('--asymmetryThreshold', type=float, default=0.3,
                    help='exclude source cells whose opposite-perturbation asymmetry exceeds this')
parser.add_argument('--outputPrefix', type=str, default='figures/bulkVsBoundary')
args = parser.parse_args()

d = np.load(args.input, allow_pickle=True)
M = d['M']                      # (numCells, numCells) mV per G_ref, M[source, target]
asymmetry = d['asymmetry']
boundaryIndices = d['boundaryIndices']
interiorIndices = d['interiorIndices']
baseReadout = d['baseReadout']  # mV, the actual t=readoutIter pattern
numRows, numCols = d['latticeDims']
numCells = numRows * numCols
V_TH_MV = -27.0

reliable = asymmetry < args.asymmetryThreshold
reliableBoundary = np.array([s for s in boundaryIndices if reliable[s]])
reliableInterior = np.array([s for s in interiorIndices if reliable[s]])
print(f"reliable sources: {len(reliableBoundary)}/{len(boundaryIndices)} boundary, "
     f"{len(reliableInterior)}/{len(interiorIndices)} interior")

ratio = np.full(numCells, np.nan)
boundaryInfluence = np.full(numCells, np.nan)
bulkInfluence = np.full(numCells, np.nan)
for target in range(numCells):
    bInf = np.abs(M[reliableBoundary, target]).mean() if len(reliableBoundary) else np.nan
    interiorSources = reliableInterior[reliableInterior != target]
    iInf = np.abs(M[interiorSources, target]).mean() if len(interiorSources) else np.nan
    boundaryInfluence[target] = bInf
    bulkInfluence[target] = iInf
    if bInf > 0 and iInf > 0:
        ratio[target] = np.log2(iInf / bInf)

# Most-influential single source per target, for drilling down later.
reliableSourceIdx = np.where(reliable)[0]
bestSource = np.array([reliableSourceIdx[np.argmax(np.abs(M[reliableSourceIdx, t]))]
                       for t in range(numCells)])

patternGrid = baseReadout.reshape(numRows, numCols)
ratioGrid = ratio.reshape(numRows, numCols)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

vCentre = V_TH_MV
vLimit = np.abs(patternGrid - vCentre).max()
im0 = axes[0].imshow(patternGrid, cmap='RdBu_r', vmin=vCentre - vLimit, vmax=vCentre + vLimit,
                     interpolation='nearest')
axes[0].set_xticks([]); axes[0].set_yticks([])
axes[0].set_title(f'Pattern at t={int(d["readoutIter"])}', fontsize=10)
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label='Vmem (mV)')

vabs = np.nanmax(np.abs(ratioGrid))
im1 = axes[1].imshow(ratioGrid, cmap='RdBu', vmin=-vabs, vmax=vabs, interpolation='nearest')
axes[1].set_xticks([]); axes[1].set_yticks([])
axes[1].set_title('Bulk- vs. boundary-represented\n(blue=bulk, red=boundary)', fontsize=10)
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label='log2(bulk influence / boundary influence)')

fig.suptitle(f'Who represents each cell at t={int(d["readoutIter"])}?', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(f'{args.outputPrefix}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {args.outputPrefix}.png")

np.savez(f'{args.outputPrefix}.npz', ratio=ratio, boundaryInfluence=boundaryInfluence,
        bulkInfluence=bulkInfluence, bestSource=bestSource, latticeDims=(numRows, numCols))
print(f"Saved {args.outputPrefix}.npz (includes bestSource per target, for drilling into "
     f"'which cell specifically represents cell X')")
