"""AND and OR are not a chosen pair of extremes to sweep between -- at radius 1 (2 neighbors total) a
threshold rule can only be k=1 (OR) or k=2 (AND); there is no way to express a genuine intermediate
fraction at that radius. Testing an intermediate threshold rescaling honestly requires a baseline radius
wide enough to have intermediate values in the first place, so this uses r=5 (10 neighbors) -> r=50 (100
neighbors), the same 10x factor used throughout, rather than r=1 -> r=10.

Threshold is held as a fixed FRACTION f = k/(2r) of the neighborhood across both scales (the proportional
generalization); k is rounded to the nearest integer at each radius. Compared by dynamical behavior
(frozen-off / frozen-on / settled / still-dynamic), not cell-by-cell match, since independent random ICs
at different N never align cell-by-cell regardless of whether the rule rescales correctly.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from totalisticCA import runThresholdCA, classifyBehavior

FIGURE_SWEEP = './figures/totalisticCA_fractionSweep.png'
N_BASE, R_BASE = 100, 5
N_SCALED, R_SCALED = 1000, 50
NUM_STEPS = 80
DENSITY = 0.5
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEED_PAIRS = [(10, 20), (11, 21), (12, 22)]  # independent per-scale seeds, no prefix-sharing


def sweepFraction(f, seedBase, seedScaled):
    kBase = max(1, round(f * 2 * R_BASE))
    kScaled = max(1, round(f * 2 * R_SCALED))
    baseHist = runThresholdCA(N_BASE, R_BASE, kBase, NUM_STEPS, DENSITY, rngSeed=seedBase)
    scaledHist = runThresholdCA(N_SCALED, R_SCALED, kScaled, NUM_STEPS, DENSITY, rngSeed=seedScaled)
    return kBase, kScaled, baseHist, scaledHist


print(f"  {'f':>4} {'k_base':>7} {'k_scaled':>9}  base behavior (3 seeds){'':10}scaled behavior (3 seeds)")
for f in FRACTIONS:
    baseBehaviors, scaledBehaviors = [], []
    for seedBase, seedScaled in SEED_PAIRS:
        kBase, kScaled, b, s = sweepFraction(f, seedBase, seedScaled)
        baseBehaviors.append(classifyBehavior(b))
        scaledBehaviors.append(classifyBehavior(s))
    print(f"  {f:4.1f} {kBase:7d} {kScaled:9d}  {str(baseBehaviors):35s} {scaledBehaviors}")

# ── figure: representative space-time patterns across the transition ──
pickFractions = [0.1, 0.3, 0.5, 0.6]
fig, axes = plt.subplots(2, len(pickFractions), figsize=(5 * len(pickFractions), 8))
for col, f in enumerate(pickFractions):
    kBase, kScaled, b, s = sweepFraction(f, 10, 20)
    axes[0, col].imshow(b, cmap='gray_r', interpolation='nearest', aspect='auto')
    axes[0, col].set_title(f'f={f}, base r={R_BASE}, k={kBase}\nfinal onFrac={b[-1].mean():.2f}', fontsize=9)
    axes[1, col].imshow(s, cmap='gray_r', interpolation='nearest', aspect='auto')
    axes[1, col].set_title(f'f={f}, scaled r={R_SCALED}, k={kScaled}\nfinal onFrac={s[-1].mean():.2f}', fontsize=9)
fig.suptitle('Threshold fraction sweep: what the frozen states actually look like', fontsize=12)
fig.tight_layout()
fig.savefig(FIGURE_SWEEP, dpi=140, bbox_inches='tight')
print(f'  wrote {FIGURE_SWEEP}')
