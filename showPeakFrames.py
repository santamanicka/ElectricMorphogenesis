"""Show the frame at the global-max-correlation iteration for every releasedScaledFace_30x30 trial.

Most trials peak within the first few tens of iterations, at essentially the untouched injected
prepattern -- before free dynamics has had a chance to diverge, not because the tissue found its way
back to the target. A few (strength 0.25, 0.30, 0.1347) peak later (iter 400-700), which is a
genuinely evolved state, not just leftover initial condition.
"""
import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURE = './figures/peakFramesAllTrials.png'

files = sorted(glob.glob('./data/releasedScaledFace_30x30_*.dat')) + ['./data/releasedScaledFace_30x30.dat']
files = [f for f in files if os.path.exists(f)]
runs = []
for f in files:
    d = torch.load(f, weights_only=False)
    label = (os.path.basename(f).replace('releasedScaledFace_30x30_', '')
             .replace('releasedScaledFace_30x30.dat', 'screenNative4_strengthNative1').replace('.dat', ''))
    bestRow = int(np.argmax(d['correlation']))
    runs.append((label, d, bestRow))
runs.sort(key=lambda r: r[1]['storedIters'][r[2]])  # order by how early the peak occurs

target = runs[0][1]['scaledTarget1000_mV']
n = len(runs)
fig, axes = plt.subplots(1, n + 1, figsize=(2.2 * (n + 1), 2.6))
axes[0].imshow(target, cmap='gray')
axes[0].set_title('target', fontsize=8)
axes[0].set_xticks([]); axes[0].set_yticks([])

for ax, (label, d, bestRow) in zip(axes[1:], runs):
    rows, cols = d['rows'], d['cols']
    frame = d['Vmem_mV'][bestRow].reshape(rows, cols)
    iteration = d['storedIters'][bestRow]
    r = d['correlation'][bestRow]
    ax.imshow(frame, cmap='gray', vmin=frame.min(), vmax=frame.max())
    ax.set_title(f'{label}\niter {iteration}, r={r:.2f}', fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle('Frame at the global peak-correlation iteration, per trial (sorted by how early it peaks)', fontsize=10)
fig.tight_layout()
fig.savefig(FIGURE, dpi=150, bbox_inches='tight')
print(f'wrote {FIGURE}')
for label, d, bestRow in runs:
    print(f'  {label:30s} peak r={d["correlation"][bestRow]:.3f} at iteration {d["storedIters"][bestRow]}')
