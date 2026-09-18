"""Show the best face found so far across all tracked 30x30 training classes, one panel
for the best correlation-loss result and one for the best globalsum-loss result.

Reads bestModelParameters_fieldVector_30x30_*.dat checkpoints directly -- each stores its
own targetVmem/actualVmem, so nothing is re-simulated here.
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

classes = [
    ('screen8',            [1401,1402,1403,1404,1405,1406], [1501,1502,1503,1504,1505,1506]),
    ('screen2',            [1601,1602,1603,1604,1605,1606], [1701,1702,1703,1704,1705,1706]),
    ('screen3',            [1801,1802,1803,1804,1805,1806], [1901,1902,1903,1904,1905,1906]),
    ('screen4 1-shot-sym', [1907,1908,1909,1910,1911,1912], [1913,1914,1915,1916,1917,1918]),
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

bestCorr = bestCorrLabel = None
bestGlob = bestGlobLabel = None
for label, corrNums, globNums in classes:
    c = bestInGroup(corrNums)
    g = bestInGroup(globNums)
    if c and (bestCorr is None or c[0] < bestCorr[0]):
        bestCorr, bestCorrLabel = c, label
    if g and (bestGlob is None or g[0] < bestGlob[0]):
        bestGlob, bestGlobLabel = g, label

rows, cols = 30, 30
fig, axes = plt.subplots(2, 2, figsize=(8, 8.5))

for col, (best, label, lossName) in enumerate([
    (bestCorr, bestCorrLabel, '1-r'),
    (bestGlob, bestGlobLabel, 'mV'),
]):
    L, n, p = best
    tp = p['trainParameters']
    target = tp['targetVmem'].reshape(rows, cols).numpy() * 1000
    actual = tp['actualVmem'].reshape(rows, cols).numpy() * 1000

    axTop = axes[0, col]
    axTop.imshow(target, cmap='gray')
    axTop.set_title(f"{tp['lossMethod']}\ntarget", fontsize=11)
    axTop.set_xticks([]); axTop.set_yticks([])

    axBot = axes[1, col]
    axBot.imshow(actual, cmap='gray')
    axBot.set_title(f"{label}\nfile {n}, loss {L:.3f} ({lossName})", fontsize=10)
    axBot.set_xticks([]); axBot.set_yticks([])

fig.suptitle('Best face so far across all classes (30x30, still running)', fontsize=13)
fig.tight_layout()
fig.savefig('figures/bestFaceSoFar.png', dpi=140, bbox_inches='tight')
print(f"correlation: best is file {bestCorr[1]} ({bestCorrLabel}), loss {bestCorr[0]:.4f}")
print(f"globalsum: best is file {bestGlob[1]} ({bestGlobLabel}), loss {bestGlob[0]:.4f}")
print("wrote figures/bestFaceSoFar.png")
