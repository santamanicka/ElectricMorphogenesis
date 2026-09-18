"""Re-render releasedScaledFace_30x30.png with each snapshot on its own colour scale.

The first version pinned every panel to the target's own mV range, which washes out any frame whose
absolute range drifted away from the target (e.g. a fully hyperpolarized/depolarized interior) even
if it has strong internal structure. Reads the data releaseScaledPrepatternOn30x30.py already saved;
does not re-run the simulation.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_FILE = './data/releasedScaledFace_30x30.dat'
FIGURE = './figures/releasedScaledFace_30x30_perpanel.png'
SNAPSHOT_ITERS = [0, 50, 100, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000]

d = torch.load(DATA_FILE, weights_only=False)
storedIters, V, correlation = d['storedIters'], d['Vmem_mV'], d['correlation']
rows, cols = d['rows'], d['cols']
target = d['scaledTarget1000_mV']

fig = plt.figure(figsize=(2.05 * (len(SNAPSHOT_ITERS) + 1), 5.6))
grid = gridspec.GridSpec(2, len(SNAPSHOT_ITERS) + 1, height_ratios=[1, 1.2], hspace=0.55, wspace=0.08)

axTarget = fig.add_subplot(grid[0, 0])
axTarget.imshow(target, cmap='gray')
axTarget.set_title('target\n(11x11 iter1000,\nscaled)', fontsize=8)
axTarget.set_xticks([]); axTarget.set_yticks([])

for col, iteration in enumerate(SNAPSHOT_ITERS, start=1):
    row = int(np.searchsorted(storedIters, iteration))
    row = min(row, len(storedIters) - 1)
    frame = V[row].reshape(rows, cols)
    axis = fig.add_subplot(grid[0, col])
    axis.imshow(frame, cmap='gray', vmin=frame.min(), vmax=frame.max())
    axis.set_title(f'iter {storedIters[row]}\nr={correlation[row]:.2f}\n[{frame.min():.0f},{frame.max():.0f}] mV', fontsize=7)
    axis.set_xticks([]); axis.set_yticks([])

axCorr = fig.add_subplot(grid[1, :])
axCorr.plot(storedIters, correlation, color='steelblue')
axCorr.axhline(0, color='gray', linewidth=0.7)
axCorr.set_xlabel('iteration'); axCorr.set_ylabel('correlation to\n11x11 iter-1000 target')
axCorr.set_title('Pattern similarity over the unclamped 30x30 release', fontsize=10)

fig.suptitle('11x11 clamp-release prepattern, scaled to 30x30, released unclamped for 5000 iters\n'
             '(each snapshot on its own colour scale -- range shown per panel)', fontsize=11)
fig.savefig(FIGURE, dpi=140, bbox_inches='tight')
print(f'wrote {FIGURE}')
