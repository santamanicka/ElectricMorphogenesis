"""Does rescaling a nonlinear complex system's local rule and neighborhood, proportionally, rescale
its space-time pattern?

Toy model: a generalized outer-totalistic parity (XOR) elementary cellular automaton. At radius r=1
this is exactly Wolfram's Rule 90 -- new[i] = state[i-1] XOR state[i+1] -- which from a single seed
produces the Sierpinski triangle (cell (i,t) = C(t,i) mod 2, box-counting dimension log(3)/log(2)).
The generalization tested here ("full-neighborhood parity"): new[i] = XOR of all 2r neighbor cells
(self excluded, matching Rule 90's own outer-totalistic convention), for arbitrary radius r. Unlike a
threshold/AND-style rule, parity has no "k" to get wrong -- the only design choice is *which* cells
count as the neighborhood, not how many of them must agree.

Question: does running this same rule with radius scaled proportionally to lattice size (r=1 at N=10
-> r=10 at N=100) reproduce the same fractal structure at the larger scale, at the *same* number of
timesteps (testing the ballistic argument that N/r, the light-cone crossing time, is scale-invariant
when N and r are scaled together)?
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURE = './figures/generalizedParityCA_baseline_vs_rescaled.png'


def runParityCA(numCells, radius, numSteps, seedIndex=None, variant='fullNeighborhood', seedWidth=1):
    """Periodic boundary, single-seed initial condition by default (middle cell).

    variant='fullNeighborhood' (B): new[i] = parity of all 2*radius neighbor cells (self excluded).
    variant='twoTap' (A): new[i] = state[i-radius] XOR state[i+radius] -- only the two most distant
        cells matter, structurally identical to Rule 90's own two-tap form, just with the taps moved
        from distance 1 to distance `radius`. Coincides with 'fullNeighborhood' exactly at radius=1.
    seedWidth: number of consecutive cells set to 1, centred at seedIndex (default 1 = single seed).
    """
    if seedIndex is None:
        seedIndex = numCells // 2
    state = np.zeros(numCells, dtype=np.int8)
    seedStart = seedIndex - seedWidth // 2
    for offset in range(seedWidth):
        state[(seedStart + offset) % numCells] = 1
    history = np.zeros((numSteps + 1, numCells), dtype=np.int8)
    history[0] = state
    if variant == 'twoTap':
        offsets = [-radius, radius]
    else:
        offsets = [d for d in range(-radius, radius + 1) if d != 0]  # exclude self, matching Rule 90
    for t in range(1, numSteps + 1):
        total = np.zeros(numCells, dtype=np.int64)
        for d in offsets:
            total += np.roll(state, -d)  # np.roll(-d): state[i+d] lands at position i
        state = (total % 2).astype(np.int8)
        history[t] = state
    return history


def boxCountingDimension(binaryImage):
    """Standard box-counting estimate of the fractal dimension of the 'on' cells in a 2D binary
    array (rows=time, cols=space). Returns (dimension, box_sizes, counts) for inspection."""
    rows, cols = binaryImage.shape
    maxBox = min(rows, cols) // 2
    boxSizes = sorted(set(int(s) for s in np.geomspace(2, max(maxBox, 3), num=12).astype(int)))
    counts = []
    for box in boxSizes:
        nRows = rows // box
        nCols = cols // box
        if nRows < 2 or nCols < 2:
            continue
        trimmed = binaryImage[:nRows * box, :nCols * box]
        blocks = trimmed.reshape(nRows, box, nCols, box)
        occupied = blocks.any(axis=(1, 3))
        counts.append((box, int(occupied.sum())))
    boxes = np.array([c[0] for c in counts], dtype=float)
    ns = np.array([c[1] for c in counts], dtype=float)
    valid = ns > 0
    if valid.sum() < 2:
        return float('nan'), boxes, ns  # empty (or nearly empty) image: dimension undefined
    slope, intercept = np.polyfit(np.log(1.0 / boxes[valid]), np.log(ns[valid]), 1)
    return slope, boxes, ns


# ── baseline: N=10, r=1 (true Rule 90) ──────────────────────────────────────
N_BASE, R_BASE = 10, 1
# ── rescaled: N=100, r=10 (proportional: both x10) ──────────────────────────
N_SCALED, R_SCALED = 100, 10
NUM_STEPS = 40  # same step count at both scales -- the hypothesis under test

theoretical = np.log(3) / np.log(2)
runs = [
    ('baseline\nN=10, r=1', N_BASE, R_BASE, 'fullNeighborhood'),
    ('(B) full-neighborhood\nN=100, r=10', N_SCALED, R_SCALED, 'fullNeighborhood'),
    ('(A) two-tap\nN=100, r=10', N_SCALED, R_SCALED, 'twoTap'),
]
results = []
for label, n, r, variant in runs:
    history = runParityCA(n, r, NUM_STEPS, variant=variant)
    dim, boxes, ns = boxCountingDimension(history)
    print(f"  {label.replace(chr(10), ' '):28s} onFraction={history.mean():.3f}  box-counting dim={dim:.3f}")
    results.append((label, n, history, dim))
print(f"  theoretical Rule-90 Sierpinski dimension: log(3)/log(2) = {theoretical:.3f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [r[1] for r in results]})
for ax, (label, n, history, dim) in zip(axes, results):
    ax.imshow(history, cmap='gray_r', interpolation='nearest', aspect='auto')
    ax.set_title(f'{label}\nbox-counting dim={dim:.3f}', fontsize=10)
    ax.set_xlabel('cell'); ax.set_ylabel('iteration')
fig.suptitle(f'Generalized parity CA, same {NUM_STEPS} timesteps at all scales/variants '
             f'(theoretical Sierpinski dim = {theoretical:.3f})', fontsize=11)
fig.tight_layout()
fig.savefig(FIGURE, dpi=150, bbox_inches='tight')
print(f'  wrote {FIGURE}')
