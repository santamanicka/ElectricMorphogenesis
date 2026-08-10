"""Show the spatial course of unclamped evolution, not just its rate.

The existing plot reports rate of change and spatial spread over time, which says that screen 2
settles and the wider screens do not, but not what the tissue looks like while doing it. These are
the patterns themselves.

Screen 2 is the degenerate case: fieldScreenSize sets the action range, and with fieldRangeSymmetric
False the perception range is pinned at nearest neighbour, so screen 2 is the one setting where a
cell writes no further into the field than it reads from it. Everything above it broadcasts beyond
what it can perceive. That asymmetry is what the comparison is really about.

Autonomous means never clamped; released means clamped and then let go at iteration 100.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

frames = [0, 100, 300, 600, 1000, 1500, 2000, 2499]
series = [('data/unclampedEvolution.npz', 'screen2_autonomous', 'screen 2, autonomous'),
          ('data/unclampedEvolution.npz', 'screen2_released',   'screen 2, released at 100'),
          ('data/unclampedEvolution.npz', 'screen5_autonomous', 'screen 5, autonomous'),
          ('data/unclampedEvolution.npz', 'screen5_released',   'screen 5, released at 100')]

loaded = []
for path, key, label in series:
    values = np.load(path)[key]*1000.0
    loaded.append((label, values))
    rate = np.abs(np.diff(values, axis=0)).mean(axis=1)
    print(f"  {label:28s} spread at end {values[-1].std():6.3f} mV, "
          f"rate at end {rate[-1]:.4f} mV/iter, rate at 1000 {rate[1000]:.4f}")

allValues = np.concatenate([v.ravel() for _, v in loaded])
vmin, vmax = np.percentile(allValues, 0.5), np.percentile(allValues, 99.5)

fig, axes = plt.subplots(len(loaded), len(frames), figsize=(1.9*len(frames), 2.25*len(loaded)))
for row, (label, values) in enumerate(loaded):
    for col, iteration in enumerate(frames):
        image = values[min(iteration, len(values)-1)].reshape(30, 30)
        ax = axes[row][col]
        ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(f'iter {iteration}', fontsize=9)
        if col == 0:
            ax.set_ylabel(label, fontsize=8)
fig.suptitle('Unclamped evolution: screen 2 is the only setting whose action range does not exceed its perception range',
             fontsize=11)
fig.tight_layout()
fig.savefig('figures/unclampedPatternCourse.png', dpi=150)
print('  wrote figures/unclampedPatternCourse.png')
