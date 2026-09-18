"""Does scaling the seed proportionally to the lattice (1-cell seed at N=100 -> 10-cell seed block at
N=1000, matching how radius was scaled 1->10) change whether the rescaled rule reproduces the N=100
baseline? Previously only a single-point seed was tested at every scale, deliberately, to isolate the
rule/neighborhood question from initial-condition transplant questions -- this fills that gap in.

Prediction for (A) two-tap: the two-tap rule only couples cells exactly r apart, so it decomposes into
r non-interacting sublattices (indexed by i mod r). A single-point seed only ever activates ONE of
them (the degenerate result found earlier). A seedWidth=r block, if it spans one cell per residue class
mod r, should activate ALL r sublattices at once -- each evolving as its own independent copy of the
N=100 baseline, interleaved with stride r. If so, the full pattern should look like r superimposed
copies of the baseline rather than one isolated copy in a sea of zeros.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from generalizedParityCA import runParityCA, boxCountingDimension

FIGURE = './figures/parityCA_scaledSeed.png'
N_BASE, R_BASE = 100, 1
N_SCALED, R_SCALED = 1000, 10
NUM_STEPS = 60
BLOCK = N_SCALED // N_BASE  # 10
theoretical = np.log(3) / np.log(2)


def blockPool(image, block, rule='majority'):
    rows, cols = image.shape
    blocks = image.reshape(rows, cols // block, block)
    if rule == 'or':
        return blocks.any(axis=2).astype(np.int8)
    return (blocks.mean(axis=2) >= 0.5).astype(np.int8)


baseHistory = runParityCA(N_BASE, R_BASE, NUM_STEPS, seedWidth=1)
dimBase, _, _ = boxCountingDimension(baseHistory)
print(f"  baseline N=100,r=1,seedWidth=1: onFrac={baseHistory.mean():.3f} dim={dimBase:.3f}")

runs = [
    ('(A) two-tap\nseed=1 (old)', 'twoTap', 1),
    ('(A) two-tap\nseed=10 (scaled)', 'twoTap', BLOCK),
    ('(B) full-nbhd\nseed=1 (old)', 'fullNeighborhood', 1),
    ('(B) full-nbhd\nseed=10 (scaled)', 'fullNeighborhood', BLOCK),
]
results = []
for label, variant, seedWidth in runs:
    history = runParityCA(N_SCALED, R_SCALED, NUM_STEPS, variant=variant, seedWidth=seedWidth)
    dim, _, _ = boxCountingDimension(history)
    pooled = blockPool(history, BLOCK, 'majority')
    agree = (baseHistory == pooled).mean()
    corr = np.corrcoef(baseHistory.flatten(), pooled.flatten())[0, 1] if pooled.std() > 0 else float('nan')
    exactMatch = None
    if variant == 'twoTap':
        # check every residue class (not just the seed's own) for an exact copy of the baseline
        matches = [np.array_equal(baseHistory, np.roll(history[:, k::BLOCK], 0, axis=1))
                   for k in range(BLOCK)]
        exactMatch = sum(matches)
    print(f"  {label.replace(chr(10),' '):28s} onFrac={history.mean():.3f}  dim={dim:.3f}  "
          f"pooled-agree={agree:.3f}  pooled-corr={corr:.3f}"
          + (f"  sublattices-exact-matching-baseline={exactMatch}/{BLOCK}" if exactMatch is not None else ""))
    results.append((label, history, pooled, dim, agree, corr))

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for col, (label, history, pooled, dim, agree, corr) in enumerate(results):
    axes[0, col].imshow(history, cmap='gray_r', interpolation='nearest', aspect='auto')
    axes[0, col].set_title(f'{label}\nN=1000, dim={dim:.2f}', fontsize=9)
    axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
    axes[1, col].imshow(pooled, cmap='gray_r', interpolation='nearest', aspect='auto')
    axes[1, col].set_title(f'pooled to N=100\nagree={agree:.2f}, corr={corr:.2f}', fontsize=9)
    axes[1, col].set_xticks([]); axes[1, col].set_yticks([])

fig.suptitle(f'Effect of scaling the seed block (1 -> {BLOCK} cells) at N=1000, r=10 '
             f'-- baseline dim={dimBase:.2f}', fontsize=12)
fig.tight_layout()
fig.savefig(FIGURE, dpi=140, bbox_inches='tight')
print(f'  wrote {FIGURE}')
