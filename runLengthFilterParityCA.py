"""Test whether a run-length threshold (morphological opening: zero out any maximal spatial run of
consecutive "on" cells shorter than `threshold`, periodic wraparound) recovers something closer to the
N=100 baseline than the earlier block-vote renormalization did, per-row (each timestep independently).

Unlike the earlier block transform (fixed-width 10-cell blocks, majority/OR vote -> downsamples
1000->100), this stays at N=1000 resolution and only removes short bursts. To compare against the true
N=100 baseline it is then block-pooled the same way as before, so any improvement is attributable to
the run-length filter itself, not a different pooling rule.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from generalizedParityCA import runParityCA, boxCountingDimension

FIGURE = './figures/parityCA_runLengthFilter.png'
N_BASE, R_BASE = 100, 1
N_SCALED, R_SCALED = 1000, 10
NUM_STEPS = 60
BLOCK = N_SCALED // N_BASE  # 10


def runLengthOpen(binaryImage, threshold):
    """Zero out any maximal run (periodic, per row) of consecutive 1s shorter than `threshold`."""
    out = np.zeros_like(binaryImage)
    numRows, numCols = binaryImage.shape
    for t in range(numRows):
        row = binaryImage[t]
        if not row.any():
            continue
        if row.all():
            out[t] = row
            continue
        # find run boundaries via periodic-aware run-length encoding
        idx = np.arange(numCols)
        diff = np.flatnonzero(row != np.roll(row, 1))
        # diff holds indices where a new run starts (periodic-aware since roll wraps)
        starts = diff
        runVals = row[starts]
        lengths = np.diff(np.concatenate([starts, [starts[0] + numCols]]))
        for s, val, length in zip(starts, runVals, lengths):
            if val == 1 and length >= threshold:
                positions = (s + np.arange(length)) % numCols
                out[t, positions] = 1
    return out


def blockPool(image, block, rule='majority'):
    rows, cols = image.shape
    blocks = image.reshape(rows, cols // block, block)
    if rule == 'or':
        return blocks.any(axis=2).astype(np.int8)
    return (blocks.mean(axis=2) >= 0.5).astype(np.int8)


baseHistory = runParityCA(N_BASE, R_BASE, NUM_STEPS, variant='fullNeighborhood')
scaledHistory = runParityCA(N_SCALED, R_SCALED, NUM_STEPS, variant='fullNeighborhood')
theoretical = np.log(3) / np.log(2)

dimBase, _, _ = boxCountingDimension(baseHistory)
print(f"  baseline N={N_BASE} r={R_BASE}: onFraction={baseHistory.mean():.3f} dim={dimBase:.3f}")

rawPooled = blockPool(scaledHistory, BLOCK, 'majority')
rawAgree = (baseHistory == rawPooled).mean()
rawCorr = np.corrcoef(baseHistory.flatten(), rawPooled.flatten())[0, 1]
print(f"  no filter, majority-pooled: onFraction={rawPooled.mean():.3f} "
      f"agree={rawAgree:.3f} corr={rawCorr:.3f}")

thresholds = [2, 3, 5, 8, 10, 15, 20, 30]
results = []
for T in thresholds:
    filtered = runLengthOpen(scaledHistory, T)
    dimF, _, _ = boxCountingDimension(filtered)
    pooled = blockPool(filtered, BLOCK, 'majority')
    agree = (baseHistory == pooled).mean()
    corr = np.corrcoef(baseHistory.flatten(), pooled.flatten())[0, 1] if pooled.std() > 0 else float('nan')
    onFrac = filtered.mean()
    results.append((T, filtered, pooled, dimF, onFrac, agree, corr))
    print(f"  threshold={T:3d}  onFraction={onFrac:.3f}  dim={dimF:.3f}  "
          f"pooled-agree={agree:.3f}  pooled-corr={corr:.3f}")

# ── figure: baseline | raw (B) | pooled raw | a few filtered examples (full-res) + their pooled compare
bestByCorr = max(results, key=lambda r: (r[6] if not np.isnan(r[6]) else -1))
print(f"  best by pooled-correlation: threshold={bestByCorr[0]}")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes[0, 0].imshow(baseHistory, cmap='gray_r', interpolation='nearest', aspect='auto')
axes[0, 0].set_title(f'baseline N=100\ndim={dimBase:.2f}', fontsize=9)
axes[0, 1].imshow(scaledHistory, cmap='gray_r', interpolation='nearest', aspect='auto')
axes[0, 1].set_title(f'(B) raw, N=1000\nonFrac={scaledHistory.mean():.2f}', fontsize=9)
axes[0, 2].imshow(rawPooled, cmap='gray_r', interpolation='nearest', aspect='auto')
axes[0, 2].set_title(f'(B) raw, majority-pooled\nagree={rawAgree:.2f}, corr={rawCorr:.2f}', fontsize=9)

pick = [2, 5, 10]
for col, T in zip([3, 4], [pick[0], pick[2]]):
    entry = next(r for r in results if r[0] == T)
    axes[0, col].imshow(entry[1], cmap='gray_r', interpolation='nearest', aspect='auto')
    axes[0, col].set_title(f'run-filtered N=1000, T={T}\nonFrac={entry[4]:.2f}', fontsize=9)

for ax in axes[0]:
    ax.set_xticks([]); ax.set_yticks([])

axes[1, 0].imshow(baseHistory, cmap='gray_r', interpolation='nearest', aspect='auto')
axes[1, 0].set_title('baseline N=100 (repeated)', fontsize=9)
for col, T in enumerate(pick, start=1):
    entry = next(r for r in results if r[0] == T)
    axes[1, col].imshow(entry[2], cmap='gray_r', interpolation='nearest', aspect='auto')
    axes[1, col].set_title(f'T={T} filtered, pooled to N=100\nagree={entry[5]:.2f}, corr={entry[6]:.2f}',
                            fontsize=9)
axes[1, 4].plot([r[0] for r in results], [r[6] for r in results], 'o-', color='steelblue')
axes[1, 4].axhline(rawCorr, color='gray', linestyle='--', linewidth=1, label='no filter')
axes[1, 4].set_xlabel('run-length threshold'); axes[1, 4].set_ylabel('pooled correlation to baseline')
axes[1, 4].legend(fontsize=8)
for ax in axes[1, :4]:
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle('Run-length opening (spatial, per-timestep) applied to (B) before block-pooling', fontsize=12)
fig.tight_layout()
fig.savefig(FIGURE, dpi=140, bbox_inches='tight')
print(f'  wrote {FIGURE}')
