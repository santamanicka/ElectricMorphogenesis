"""Pull the peak-correlation frame out of a releasedScaledFace_30x30_*.dat trial and put it directly
beside the (scaled) target, so the resemblance the correlation number implies can be checked by eye.
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataFile', default='./data/releasedScaledFace_30x30_screen10.9091_strength0.3667.dat')
parser.add_argument('--output', default=None)
parser.add_argument('--skipEarly', type=int, default=200,
                     help='ignore stored frames before this iteration when picking the peak, since '
                          'the injected initial condition itself always starts near the target')
args = parser.parse_args()

d = torch.load(args.dataFile, weights_only=False)
storedIters, V, correlation = d['storedIters'], d['Vmem_mV'], d['correlation']
rows, cols = d['rows'], d['cols']
target = d['scaledTarget1000_mV']

eligible = storedIters >= args.skipEarly
bestRow = np.where(eligible)[0][np.argmax(correlation[eligible])]
bestIter, bestR = storedIters[bestRow], correlation[bestRow]
frame = V[bestRow].reshape(rows, cols)
print(f'  peak correlation after iter {args.skipEarly}: r={bestR:.3f} at iteration {bestIter}')

output = args.output or args.dataFile.replace('./data/', './figures/').replace('.dat', '_bestFrame.png')
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4))
axes[0].imshow(target, cmap='gray')
axes[0].set_title('target\n(11x11 iter1000, scaled)', fontsize=9)
axes[1].imshow(frame, cmap='gray', vmin=frame.min(), vmax=frame.max())
axes[1].set_title(f'iteration {bestIter}\nr={bestR:.3f}', fontsize=9)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(args.dataFile.split('/')[-1].replace('.dat', ''), fontsize=10)
fig.tight_layout()
fig.savefig(output, dpi=150, bbox_inches='tight')
print(f'  wrote {output}')
