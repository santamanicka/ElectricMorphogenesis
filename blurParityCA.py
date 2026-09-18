"""Literal test of the 'squint' intuition: 2D Gaussian blur (both space and time axes, matching how
squinting at the plotted image itself would blur it) of the raw (B) N=1000 pattern, re-thresholded
back to binary at a density matched to the baseline's own on-fraction (isolates whether blur recovers
the right *structure*, not just the right density). Compared against the baseline two ways: at full
N=1000 resolution (baseline nearest-neighbour-upsampled 10x) and pooled down to N=100 (consistent with
the earlier pooled/filtered comparisons).
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from generalizedParityCA import runParityCA, boxCountingDimension

FIGURE = './figures/parityCA_blurTest.png'
N_BASE, R_BASE = 100, 1
N_SCALED, R_SCALED = 1000, 10
NUM_STEPS = 60
BLOCK = N_SCALED // N_BASE


def blockPool(image, block, rule='majority'):
    rows, cols = image.shape
    blocks = image.reshape(rows, cols // block, block)
    if rule == 'or':
        return blocks.any(axis=2).astype(np.int8)
    return (blocks.mean(axis=2) >= 0.5).astype(np.int8)


def densityMatchedThreshold(continuous, targetOnFraction):
    cutoff = np.quantile(continuous, 1.0 - targetOnFraction)
    return (continuous >= cutoff).astype(np.int8)


baseHistory = runParityCA(N_BASE, R_BASE, NUM_STEPS, variant='fullNeighborhood')
scaledHistory = runParityCA(N_SCALED, R_SCALED, NUM_STEPS, variant='fullNeighborhood')
baseOnFraction = baseHistory.mean()
baseUpsampled = np.repeat(baseHistory, BLOCK, axis=1)  # nearest-neighbour, N=1000 resolution

rawPooled = blockPool(scaledHistory, BLOCK, 'majority')
rawCorrPooled = np.corrcoef(baseHistory.flatten(), rawPooled.flatten())[0, 1]
rawCorrFull = np.corrcoef(baseUpsampled.flatten(), scaledHistory.flatten())[0, 1]
print(f"  baseline onFraction={baseOnFraction:.3f}")
print(f"  no blur: pooled corr={rawCorrPooled:.3f}, full-res corr (vs upsampled baseline)={rawCorrFull:.3f}")

sigmas = [1, 2, 3, 5, 8, 10, 15, 20]
results = []
for sigma in sigmas:
    blurred = gaussian_filter(scaledHistory.astype(float), sigma=sigma, mode='wrap')
    thresholded = densityMatchedThreshold(blurred, baseOnFraction)
    fullCorr = np.corrcoef(baseUpsampled.flatten(), thresholded.flatten())[0, 1]
    pooled = blockPool(thresholded, BLOCK, 'majority')
    pooledCorr = (np.corrcoef(baseHistory.flatten(), pooled.flatten())[0, 1]
                  if pooled.std() > 0 else float('nan'))
    dimT, _, _ = boxCountingDimension(thresholded)
    results.append((sigma, thresholded, pooled, fullCorr, pooledCorr, dimT))
    print(f"  sigma={sigma:2d}  onFrac={thresholded.mean():.3f}  dim={dimT:.3f}  "
          f"full-res corr={fullCorr:.3f}  pooled corr={pooledCorr:.3f}")

best = max(results, key=lambda r: r[3])
print(f"  best by full-res correlation: sigma={best[0]} (corr={best[3]:.3f})")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes[0, 0].imshow(baseUpsampled, cmap='gray_r', interpolation='nearest', aspect='auto')
axes[0, 0].set_title('baseline (nearest-neighbour\nupsampled to N=1000)', fontsize=9)
axes[0, 1].imshow(scaledHistory, cmap='gray_r', interpolation='nearest', aspect='auto')
axes[0, 1].set_title(f'(B) raw, N=1000\nfull-res corr={rawCorrFull:.2f}', fontsize=9)
pickSigmas = [2, 5, best[0] if best[0] not in (2, 5) else 10, 15]
for col, sigma in zip([2, 3, 4], pickSigmas):
    entry = next(r for r in results if r[0] == sigma)
    axes[0, col].imshow(entry[1], cmap='gray_r', interpolation='nearest', aspect='auto')
    axes[0, col].set_title(f'blur sigma={sigma}, density-matched\nfull-res corr={entry[3]:.2f}', fontsize=9)
for ax in axes[0]:
    ax.set_xticks([]); ax.set_yticks([])
axes[0, 4].axis('off')

axes[1, 0].imshow(baseHistory, cmap='gray_r', interpolation='nearest', aspect='auto')
axes[1, 0].set_title('baseline N=100', fontsize=9)
for col, sigma in zip([1, 2, 3], pickSigmas[:3]):
    entry = next(r for r in results if r[0] == sigma)
    axes[1, col].imshow(entry[2], cmap='gray_r', interpolation='nearest', aspect='auto')
    axes[1, col].set_title(f'sigma={sigma}, pooled to N=100\npooled corr={entry[4]:.2f}', fontsize=9)
for ax in axes[1, :4]:
    ax.set_xticks([]); ax.set_yticks([])

axes[1, 4].plot([r[0] for r in results], [r[3] for r in results], 'o-', label='full-res corr')
axes[1, 4].plot([r[0] for r in results], [r[4] for r in results], 's-', label='pooled corr')
axes[1, 4].axhline(rawCorrFull, color='gray', linestyle='--', linewidth=1, label='no blur (full-res)')
axes[1, 4].axhline(rawCorrPooled, color='gray', linestyle=':', linewidth=1, label='no blur (pooled)')
axes[1, 4].set_xlabel('blur sigma'); axes[1, 4].set_ylabel('correlation to baseline')
axes[1, 4].legend(fontsize=7)

fig.suptitle('2D Gaussian blur + density-matched re-threshold of (B), vs. N=100 baseline', fontsize=12)
fig.tight_layout()
fig.savefig(FIGURE, dpi=140, bbox_inches='tight')
print(f'  wrote {FIGURE}')
