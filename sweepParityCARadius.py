"""Sweep the (B) full-neighborhood-parity radius at fixed N=100, to check whether some r other than
the naively-proportional value (10, matching N=100/N=10=10x) recovers Sierpinski-like structure --
mirroring the bioelectric finding that naive proportional scaling of fieldScreenSize alone didn't
work and fieldStrength needed a non-obvious (and non-naive-direction) correction.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from generalizedParityCA import runParityCA, boxCountingDimension

FIGURE = './figures/parityCA_radiusSweep.png'
N = 100
NUM_STEPS = 40
RADII = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

baseHistory = runParityCA(10, 1, NUM_STEPS)
dimBase, _, _ = boxCountingDimension(baseHistory)
theoretical = np.log(3) / np.log(2)

results = []
for r in RADII:
    history = runParityCA(N, r, NUM_STEPS, variant='fullNeighborhood')
    dim, boxes, ns = boxCountingDimension(history)
    onFraction = history.mean()
    results.append((r, history, dim, onFraction))
    print(f"  r={r:2d}  onFraction={onFraction:.3f}  box-counting dim={dim:.3f}")

print(f"  baseline (N=10, r=1): onFraction={baseHistory.mean():.3f}  dim={dimBase:.3f}")
print(f"  theoretical Sierpinski dim: {theoretical:.3f}")

# ── figure: snapshot grid + dimension-vs-r curve ────────────────────────────
fig = plt.figure(figsize=(2.0 * (len(RADII) + 1), 6.4))
grid = gridspec.GridSpec(2, len(RADII) + 1, height_ratios=[1.6, 1], hspace=0.45, wspace=0.15)

axBase = fig.add_subplot(grid[0, 0])
axBase.imshow(baseHistory, cmap='gray_r', interpolation='nearest', aspect='auto')
axBase.set_title(f'baseline\nN=10,r=1\ndim={dimBase:.2f}', fontsize=8)
axBase.set_xticks([]); axBase.set_yticks([])

for col, (r, history, dim, onFraction) in enumerate(results, start=1):
    axis = fig.add_subplot(grid[0, col])
    axis.imshow(history, cmap='gray_r', interpolation='nearest', aspect='auto')
    axis.set_title(f'r={r}\ndim={dim:.2f}\non={onFraction:.2f}', fontsize=8)
    axis.set_xticks([]); axis.set_yticks([])

axCurve = fig.add_subplot(grid[1, :])
rs = [r for r, _, _, _ in results]
dims = [dim for _, _, dim, _ in results]
onFracs = [onFraction for _, _, _, onFraction in results]
axCurve.plot(rs, dims, 'o-', color='steelblue', label='box-counting dim (N=100, variant B)')
axCurve.axhline(dimBase, color='black', linestyle='--', linewidth=1, label=f'baseline dim ({dimBase:.2f})')
axCurve.axhline(theoretical, color='gray', linestyle=':', linewidth=1, label=f'theoretical Sierpinski ({theoretical:.2f})')
axCurve.axvline(10, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='naive proportional r=10')
axCurve.set_xlabel('radius r (N=100 fixed)'); axCurve.set_ylabel('box-counting dimension')
axCurve.legend(fontsize=8, loc='upper left')
axCurve.set_title('Dimension vs radius, full-neighborhood-parity variant (B), N=100 fixed', fontsize=10)

fig.suptitle(f'(B) full-neighborhood parity: radius sweep at N=100, {NUM_STEPS} timesteps', fontsize=12)
fig.savefig(FIGURE, dpi=140, bbox_inches='tight')
print(f'  wrote {FIGURE}')

# ── follow-up: matched crossing-time comparison (N/2r steps per r), not fixed absolute steps ──
print("\n  matched crossing-time (steps = ceil(N/(2r))) comparison:")
matchedResults = []
for r in RADII:
    steps = max(2, int(np.ceil(N / (2 * r))))
    history = runParityCA(N, r, steps, variant='fullNeighborhood')
    dim, boxes, ns = boxCountingDimension(history)
    onFraction = history.mean()
    matchedResults.append((r, steps, history, dim, onFraction))
    print(f"  r={r:2d}  steps={steps:3d}  onFraction={onFraction:.3f}  box-counting dim={dim:.3f}")

figM = plt.figure(figsize=(2.0 * (len(RADII) + 1), 6.4))
gridM = gridspec.GridSpec(2, len(RADII) + 1, height_ratios=[1.6, 1], hspace=0.45, wspace=0.15)

axBaseM = figM.add_subplot(gridM[0, 0])
axBaseM.imshow(baseHistory, cmap='gray_r', interpolation='nearest', aspect='auto')
axBaseM.set_title(f'baseline\nN=10,r=1,steps=40\ndim={dimBase:.2f}', fontsize=8)
axBaseM.set_xticks([]); axBaseM.set_yticks([])

for col, (r, steps, history, dim, onFraction) in enumerate(matchedResults, start=1):
    axis = figM.add_subplot(gridM[0, col])
    axis.imshow(history, cmap='gray_r', interpolation='nearest', aspect='auto')
    axis.set_title(f'r={r}, steps={steps}\ndim={dim:.2f}\non={onFraction:.2f}', fontsize=8)
    axis.set_xticks([]); axis.set_yticks([])

axCurveM = figM.add_subplot(gridM[1, :])
rsM = [r for r, _, _, _, _ in matchedResults]
dimsM = [dim for _, _, _, dim, _ in matchedResults]
axCurveM.plot(rsM, dimsM, 'o-', color='darkorange', label='box-counting dim (N=100, steps=N/2r, variant B)')
axCurveM.axhline(dimBase, color='black', linestyle='--', linewidth=1, label=f'baseline dim ({dimBase:.2f})')
axCurveM.axhline(theoretical, color='gray', linestyle=':', linewidth=1, label=f'theoretical Sierpinski ({theoretical:.2f})')
axCurveM.axvline(10, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='naive proportional r=10')
axCurveM.set_xlabel('radius r (N=100 fixed, steps = N/2r matched crossing time)')
axCurveM.set_ylabel('box-counting dimension')
axCurveM.legend(fontsize=8, loc='upper left')
axCurveM.set_title('Dimension vs radius, matched crossing-time comparison (steps = ceil(N/2r))', fontsize=10)

figM.suptitle('(B) full-neighborhood parity: radius sweep, each r run to its OWN light-cone crossing time', fontsize=12)
figM.savefig('./figures/parityCA_radiusSweep_matchedTime.png', dpi=140, bbox_inches='tight')
print('  wrote ./figures/parityCA_radiusSweep_matchedTime.png')
