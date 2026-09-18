"""Show the best boundary-clamp face for each of screen2, screen3, screen8 -- correlation-loss
and globalsum-loss side by side. Unlike the single-shot-sym class, these force only the tissue
boundary (fieldDomeTwoFoldSymmetry, oscillatory) for the first 100 iterations, then evolve freely.

Reads bestModelParameters_fieldVector_30x30_*.dat checkpoints directly -- each stores its own
targetVmem/actualVmem, so nothing is re-simulated here.
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

classes = [
    ('screen2', [1601,1602,1603,1604,1605,1606], [1701,1702,1703,1704,1705,1706]),
    ('screen3', [1801,1802,1803,1804,1805,1806], [1901,1902,1903,1904,1905,1906]),
    ('screen8', [1401,1402,1403,1404,1405,1406], [1501,1502,1503,1504,1505,1506]),
]

def bestInGroup(nums):
    best = None
    for n in nums:
        f = f"data/bestModelParameters_fieldVector_30x30_{n}.dat"
        try:
            p = torch.load(f, map_location='cpu', weights_only=False)
        except Exception:
            continue
        L = float(p['trainParameters']['bestLoss'])
        if best is None or L < best[0]:
            best = (L, n, p)
    return best

rows, cols = 30, 30
fig, axes = plt.subplots(len(classes), 3, figsize=(9.5, 3.2 * len(classes)))

for row, (label, corrNums, globNums) in enumerate(classes):
    c = bestInGroup(corrNums)
    g = bestInGroup(globNums)

    axTarget = axes[row, 0]
    if c is not None:
        target = c[2]['trainParameters']['targetVmem'].reshape(rows, cols).numpy() * 1000
        axTarget.imshow(target, cmap='gray')
    axTarget.set_ylabel(label, fontsize=11)
    axTarget.set_title('target' if row == 0 else '', fontsize=10)
    axTarget.set_xticks([]); axTarget.set_yticks([])

    axCorr = axes[row, 1]
    if c is not None:
        L, n, p = c
        actual = p['trainParameters']['actualVmem'].reshape(rows, cols).numpy() * 1000
        axCorr.imshow(actual, cmap='gray')
        axCorr.set_title(f'correlation\nfile {n}, loss {L:.3f}' if row == 0 else f'file {n}, loss {L:.3f}', fontsize=9)
    else:
        axCorr.set_title('no checkpoint', fontsize=9)
    axCorr.set_xticks([]); axCorr.set_yticks([])

    axGlob = axes[row, 2]
    if g is not None:
        L, n, p = g
        actual = p['trainParameters']['actualVmem'].reshape(rows, cols).numpy() * 1000
        axGlob.imshow(actual, cmap='gray')
        axGlob.set_title(f'globalsum\nfile {n}, loss {L:.3f}' if row == 0 else f'file {n}, loss {L:.3f}', fontsize=9)
    else:
        axGlob.set_title('no checkpoint', fontsize=9)
    axGlob.set_xticks([]); axGlob.set_yticks([])

fig.suptitle('Best boundary-clamp face by screen size (30x30)', fontsize=13)
fig.tight_layout()
fig.savefig('figures/bestFaceByBoundaryScreen.png', dpi=140, bbox_inches='tight')
print('wrote figures/bestFaceByBoundaryScreen.png')
