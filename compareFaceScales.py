"""Render the stored 11x11 face model beside the best 30x30 run, on matching terms.

The 11x11 model does produce the target: coarse and noisy, but the eyes, nose and mouth all sit
where they belong. The best 30x30 run so far does not -- it has a central dark oval near the nose
position, faint smudges where the eyes are, and no mouth. Both are shown against their own target
and in the same grayscale so the difference is a difference in the result rather than in the
rendering, since the scales, colour maps and targets would otherwise not be comparable.
"""
import glob, os, re
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(path):
    p = torch.load(path, map_location='cpu', weights_only=False)
    tp = p['trainParameters']
    rows, cols = p['latticeDims']
    return (tp['targetVmem'].detach().numpy().reshape(rows, cols) * 1000.0,
            tp['actualVmem'].detach().numpy().reshape(rows, cols) * 1000.0,
            float(tp['bestLoss']), rows, cols, int(p['simParameters']['numSimIters']))

# best 30x30 face run, excluding other targets' files and other lattice sizes
best = None
for f in sorted(glob.glob('data/bestModelParameters_fieldVector_*.dat')):
    tail = os.path.basename(f).split('fieldVector_', 1)[1].rsplit('.dat', 1)[0]
    if not re.fullmatch(r'(\d+x\d+_)?\d+', tail):
        continue
    p = torch.load(f, map_location='cpu', weights_only=False)
    if tuple(p.get('latticeDims') or ()) != (30, 30):
        continue
    L = float(p['trainParameters']['bestLoss'])
    if best is None or L < best[1]:
        best = (f, L, int(p['simParameters']['numSimIters']))

panels = [('data/StigmergicModelParameters.dat', 'stored 11x11 reference'),
          (best[0], 'best 30x30')]
fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.6))
for row, (path, label) in enumerate(panels):
    target, actual, loss, rows, cols, horizon = load(path)
    for col, (img, what) in enumerate([(target, 'target'), (actual, 'trained result')]):
        ax = axes[row][col]
        ax.imshow(img, cmap='gray', vmin=min(img.min(), -60), vmax=0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{label}, {horizon} iter horizon\n{what}'
                     + (f' (loss {loss:.4g})' if col else ''), fontsize=10)
fig.suptitle('11x11 resolves the features; 30x30 so far only the gross structure', fontsize=12)
fig.tight_layout()
fig.savefig('figures/faceScaleComparison.png', dpi=150)
print(f"  11x11 reference: data/StigmergicModelParameters.dat")
print(f"  best 30x30:      {best[0]} (loss {best[1]:.6g}, horizon {best[2]} iters)")
print("  wrote figures/faceScaleComparison.png")
