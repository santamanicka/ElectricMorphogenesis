"""Summarize every releasedScaledFace_30x30_*.dat trial in one figure: final pattern + correlation
trace per run, so the screen/strength sweep can be read at a glance instead of one figure at a time.
"""
import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FIGURE = './figures/releasedScaledFaceComparison.png'

files = sorted(glob.glob('./data/releasedScaledFace_30x30_*.dat')) + ['./data/releasedScaledFace_30x30.dat']
files = [f for f in files if os.path.exists(f)]
runs = []
for f in files:
    d = torch.load(f, weights_only=False)
    label = os.path.basename(f).replace('releasedScaledFace_30x30_', '').replace('releasedScaledFace_30x30', 'screenNative4').replace('.dat', '')
    runs.append((label, d))

fig = plt.figure(figsize=(3.0 * len(runs), 6.4))
grid = gridspec.GridSpec(2, len(runs), height_ratios=[1, 1.3], hspace=0.5, wspace=0.15)

targetShown = False
for col, (label, d) in enumerate(runs):
    rows, cols = d['rows'], d['cols']
    finalFrame = d['Vmem_mV'][-1].reshape(rows, cols)
    axis = fig.add_subplot(grid[0, col])
    axis.imshow(finalFrame, cmap='gray', vmin=finalFrame.min(), vmax=finalFrame.max())
    axis.set_title(f'{label}\nfinal r={d["correlation"][-1]:.2f}', fontsize=8)
    axis.set_xticks([]); axis.set_yticks([])

axCorr = fig.add_subplot(grid[1, :])
for label, d in runs:
    axCorr.plot(d['storedIters'], d['correlation'], label=label, linewidth=1.3)
axCorr.axhline(0, color='gray', linewidth=0.7)
axCorr.set_xlabel('iteration'); axCorr.set_ylabel('correlation to\n11x11 iter-1000 target')
axCorr.legend(fontsize=7, ncol=3, loc='upper right')
axCorr.set_title('Pattern similarity across all screen/fieldStrength trials', fontsize=10)

fig.suptitle('11x11 clamp-release prepattern scaled to 30x30: outcome across every field-parameter trial', fontsize=12)
fig.savefig(FIGURE, dpi=140, bbox_inches='tight')
print(f'wrote {FIGURE}')
for label, d in runs:
    print(f'  {label:30s} final r={d["correlation"][-1]:+.3f}  peak r={d["correlation"].max():+.3f}')
