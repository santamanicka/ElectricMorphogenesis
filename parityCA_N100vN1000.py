"""Baseline N=100 (r=1, true Rule 90) vs. rescaled N=1000 (r=10, proportional), both variants,
at a scale large enough to actually eyeball the fine nested structure -- N=10 was too small to see
past wraparound clutter. Because N and r are scaled by the same factor here, the light-cone crossing
time N/2r is identical at both scales (50), so running both for the same number of steps is already a
fair comparison -- no matched-time bookkeeping needed, unlike the N=10-vs-N=100 sweep.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from generalizedParityCA import runParityCA, boxCountingDimension

FIGURE = './figures/parityCA_N100vN1000.png'
N_BASE, R_BASE = 100, 1
N_SCALED, R_SCALED = 1000, 10
NUM_STEPS = 60  # a bit past the crossing time (50) at both scales, for a little wraparound onset
theoretical = np.log(3) / np.log(2)

runs = [
    ('baseline\nN=100, r=1', N_BASE, R_BASE, 'fullNeighborhood'),
    ('(B) full-neighborhood\nN=1000, r=10', N_SCALED, R_SCALED, 'fullNeighborhood'),
    ('(A) two-tap\nN=1000, r=10', N_SCALED, R_SCALED, 'twoTap'),
]
results = []
for label, n, r, variant in runs:
    history = runParityCA(n, r, NUM_STEPS, variant=variant)
    dim, boxes, ns = boxCountingDimension(history)
    onFraction = history.mean()
    print(f"  {label.replace(chr(10), ' '):32s} onFraction={onFraction:.3f}  box-counting dim={dim:.3f}")
    results.append((label, n, history, dim, onFraction))
print(f"  theoretical Sierpinski dim: {theoretical:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
for ax, (label, n, history, dim, onFraction) in zip(axes, results):
    ax.imshow(history, cmap='gray_r', interpolation='nearest', aspect='auto')
    ax.set_title(f'{label}\nbox-counting dim={dim:.3f}, on={onFraction:.3f}', fontsize=10)
    ax.set_xlabel('cell'); ax.set_ylabel('iteration')
fig.suptitle(f'N=100 baseline vs N=1000 rescaled (r proportional), {NUM_STEPS} steps both '
             f'(theoretical Sierpinski dim = {theoretical:.3f})', fontsize=12)
fig.tight_layout()
fig.savefig(FIGURE, dpi=150, bbox_inches='tight')
print(f'  wrote {FIGURE}')
