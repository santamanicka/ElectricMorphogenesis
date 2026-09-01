"""Show the best face so far for the single-shot (tissueGpolTwoFoldSymmetry) screen11/strength0.25
class -- files 1831-1836 (correlation) and 1837-1842 (globalsum).

Reads bestModelParameters_fieldVector_30x30_*.dat checkpoints directly -- each stores its own
targetVmem/actualVmem, so nothing is re-simulated here.
"""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

corrNums = [1831, 1832, 1833, 1834, 1835, 1836]
globNums = [1837, 1838, 1839, 1840, 1841, 1842]


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


c = bestInGroup(corrNums)
g = bestInGroup(globNums)
rows, cols = 30, 30
fig, axes = plt.subplots(2, 2, figsize=(8, 8.5))

for col, (best, lossName) in enumerate([(c, '1-r'), (g, 'mV')]):
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
    axBot.set_title(f"file {n}, loss {L:.3f} ({lossName})", fontsize=10)
    axBot.set_xticks([]); axBot.set_yticks([])

fig.suptitle('Best face so far -- single-shot, screen11, strength0.25 (still running)', fontsize=12)
fig.tight_layout()
outputPath = 'figures/bestFace_singleShot_screen11_strength025.png'
fig.savefig(outputPath, dpi=140, bbox_inches='tight')
print(f"correlation: best is file {c[1]}, loss {c[0]:.4f}")
print(f"globalsum: best is file {g[1]}, loss {g[0]:.4f}")
print(f"wrote {outputPath}")
