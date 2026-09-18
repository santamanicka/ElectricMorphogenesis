"""The totalistic/threshold family (AND/OR/MAJORITY), the user's original 'n-input AND' example --
not tried until now (the earlier sub-investigation scoped to parity/XOR only). Unlike parity, threshold
rules have a tunable k that can be scaled proportionally (k/2r held constant) or left naive (k held at
its radius-1 absolute value) -- the direct CA analogue of the bioelectric naive-vs-proportional
fieldStrength lesson.

A single-point seed is useless here: AND requires ALL of a cell's neighbors on, so from one seed cell
no cell (at any radius) ever has all neighbors satisfied -- the lattice dies in one step, at every
scale, trivially. Threshold rules are conventionally studied from a random initial density instead, so
that's what's used here; comparison is by final dynamical behaviour (on-fraction trace, frozen vs
oscillating vs persistently active), not cell-by-cell match, since random ICs at different N never
align cell-by-cell regardless of whether the rule 'rescales' correctly.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIGURE = './figures/totalisticCA_sweep.png'
N_BASE, R_BASE = 100, 1
N_SCALED, R_SCALED = 1000, 10
NUM_STEPS = 60
INITIAL_DENSITY = 0.5
RNG_SEED = 0


def runThresholdCA(numCells, radius, threshold, numSteps, initialDensity=0.5, rngSeed=0):
    rng = np.random.default_rng(rngSeed)
    state = (rng.random(numCells) < initialDensity).astype(np.int8)
    history = np.zeros((numSteps + 1, numCells), dtype=np.int8)
    history[0] = state
    offsets = [d for d in range(-radius, radius + 1) if d != 0]  # self excluded, matching parity CA
    for t in range(1, numSteps + 1):
        total = np.zeros(numCells, dtype=np.int64)
        for d in offsets:
            total += np.roll(state, -d)
        state = (total >= threshold).astype(np.int8)
        history[t] = state
    return history


def classifyBehavior(history, tailLength=10):
    tail = history[-tailLength:]
    onFractions = tail.mean(axis=1)
    if np.all(tail == 0):
        return 'frozen-off'
    if np.all(tail == tail[0]):
        return 'frozen'
    period2 = np.array_equal(tail[-1], tail[-3]) if len(tail) >= 3 else False
    if period2 and not np.array_equal(tail[-1], tail[-2]):
        return '2-cycle'
    if onFractions.std() < 0.01:
        return f'settled ({onFractions[-1]:.2f})'
    return 'still dynamic'


configs = [
    ('AND, proportional\n(k=2r)', R_BASE, 2 * R_BASE, R_SCALED, 2 * R_SCALED),
    ('AND, naive\n(k=2 fixed)', R_BASE, 2 * R_BASE, R_SCALED, 2),
    ('OR\n(k=1, scale-free)', R_BASE, 1, R_SCALED, 1),
    ('MAJORITY, proportional\n(k=r+1)', R_BASE, R_BASE + 1, R_SCALED, R_SCALED + 1),
]

fig, axes = plt.subplots(2, len(configs), figsize=(5 * len(configs), 8))
print(f"  {'rule':32s} {'scale':10s} {'onFrac(final10)':16s} {'behavior'}")
for col, (label, rBase, kBase, rScaled, kScaled) in enumerate(configs):
    baseHist = runThresholdCA(N_BASE, rBase, kBase, NUM_STEPS, INITIAL_DENSITY, RNG_SEED)
    scaledHist = runThresholdCA(N_SCALED, rScaled, kScaled, NUM_STEPS, INITIAL_DENSITY, RNG_SEED)
    baseBehavior = classifyBehavior(baseHist)
    scaledBehavior = classifyBehavior(scaledHist)
    print(f"  {label.replace(chr(10),' '):32s} {'N=100,r='+str(rBase):10s} "
          f"{baseHist[-10:].mean():16.3f} {baseBehavior}")
    print(f"  {'':32s} {'N=1000,r='+str(rScaled):10s} "
          f"{scaledHist[-10:].mean():16.3f} {scaledBehavior}")

    axes[0, col].imshow(baseHist, cmap='gray_r', interpolation='nearest', aspect='auto')
    axes[0, col].set_title(f'{label}\nN=100, r={rBase}, k={kBase}\n{baseBehavior}', fontsize=9)
    axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
    axes[1, col].imshow(scaledHist, cmap='gray_r', interpolation='nearest', aspect='auto')
    axes[1, col].set_title(f'N=1000, r={rScaled}, k={kScaled}\n{scaledBehavior}', fontsize=9)
    axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

fig.suptitle(f'Totalistic/threshold family: baseline (top) vs rescaled (bottom), '
             f'random IC density={INITIAL_DENSITY}', fontsize=12)
fig.tight_layout()
fig.savefig(FIGURE, dpi=140, bbox_inches='tight')
print(f'  wrote {FIGURE}')
