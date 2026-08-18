"""
Standalone plot of the Vmem pattern a provenance run's trajectory actually produced, at the
step the provenance matrix was computed for -- the same data already saved by
measure_provenance_propagation.py, just shown on its own rather than as one panel of a
larger figure.

  python visualize_provenance_pattern.py --input data/provenance_random.npz --step 899
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='data/provenance_random.npz')
parser.add_argument('--step', type=int, default=899)
parser.add_argument('--outputPrefix', type=str, default=None)
args = parser.parse_args()

d = np.load(args.input, allow_pickle=True)
finalVmem = d['finalVmem']
numRows, numCols = d['latticeDims']
V_TH_MV = -27.0

grid = finalVmem.reshape(numRows, numCols)
vLimit = np.abs(grid - V_TH_MV).max()

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(grid, cmap='RdBu_r', vmin=V_TH_MV - vLimit, vmax=V_TH_MV + vLimit, interpolation='nearest')
ax.set_xticks([]); ax.set_yticks([])
condition = args.input.split('provenance_')[-1].replace('.npz', '')
ax.set_title(f'Vmem pattern, {condition} clamp, step {args.step}\n(white = V_th = {V_TH_MV:.0f} mV)', fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Vmem (mV)')
plt.tight_layout()

outputPrefix = args.outputPrefix or f'figures/provenancePattern_{condition}'
plt.savefig(f'{outputPrefix}_step{args.step}.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {outputPrefix}_step{args.step}.png")
print(f"range: [{grid.min():.1f}, {grid.max():.1f}] mV, mean {grid.mean():.1f} mV, std {grid.std():.1f} mV")
