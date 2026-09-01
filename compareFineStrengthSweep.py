"""Compare the fine fieldStrength sweep (0.25-0.50) around the 0.3667 run that showed a transient
regrowth toward face-like structure, all at the same reach-preserving screen size (10.909).
"""
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

FIGURE = './figures/fineStrengthSweepComparison.png'
STRENGTHS = [0.25, 0.30, 0.3667, 0.40, 0.45, 0.50]

runs = []
for s in STRENGTHS:
    tag = f'{s:g}'
    f = f'./data/releasedScaledFace_30x30_screen10.9091_strength{tag}.dat'
    d = torch.load(f, weights_only=False)
    runs.append((s, d))

fig = plt.figure(figsize=(3.0 * len(runs), 6.4))
grid = gridspec.GridSpec(2, len(runs), height_ratios=[1, 1.3], hspace=0.5, wspace=0.15)

for col, (s, d) in enumerate(runs):
    rows, cols = d['rows'], d['cols']
    finalFrame = d['Vmem_mV'][-1].reshape(rows, cols)
    axis = fig.add_subplot(grid[0, col])
    axis.imshow(finalFrame, cmap='gray', vmin=finalFrame.min(), vmax=finalFrame.max())
    peakPostTrough = d['correlation'][d['storedIters'] >= 2500].max()
    axis.set_title(f'strength={s:g}\nfinal r={d["correlation"][-1]:+.2f}, '
                    f'post-2500 peak r={peakPostTrough:+.2f}', fontsize=8)
    axis.set_xticks([]); axis.set_yticks([])

axCorr = fig.add_subplot(grid[1, :])
for s, d in runs:
    axCorr.plot(d['storedIters'], d['correlation'], label=f'strength={s:g}', linewidth=1.3)
axCorr.axhline(0, color='gray', linewidth=0.7)
axCorr.set_xlabel('iteration'); axCorr.set_ylabel('correlation to\n11x11 iter-1000 target')
axCorr.legend(fontsize=8, ncol=3, loc='lower right')
axCorr.set_title('Pattern similarity across the fine fieldStrength sweep (screen=10.909)', fontsize=10)

fig.suptitle('Fine fieldStrength sweep around the 0.3667 regrowth result', fontsize=12)
fig.savefig(FIGURE, dpi=140, bbox_inches='tight')
print(f'wrote {FIGURE}')
for s, d in runs:
    peakPostTrough = d['correlation'][d['storedIters'] >= 2500].max()
    print(f'  strength={s:5.3f}  final r={d["correlation"][-1]:+.3f}  post-2500 peak r={peakPostTrough:+.3f}')
