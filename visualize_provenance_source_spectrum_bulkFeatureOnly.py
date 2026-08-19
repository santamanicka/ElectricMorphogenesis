"""
Full-boundary-resolution version of visualize_provenance_source_regions_bulkFeatureOnly.py's top-5
table: instead of pooling the 40 boundary cells into 5 geometric groups, give every individual
boundary cell its own colour from the same continuous, cyclic perimeter spectrum used in
visualize_provenance_source_spectrum.py, and ask -- among ONLY the 14 facial-feature cells (not all
81 bulk cells) -- which one each boundary cell targets most. Each feature cell is then coloured by
the average spectrum colour of whichever boundary cell(s) name it as their single top-1 target among
those 14 candidates.

Restricting candidates to 14 rather than 81 targets changes the character of the result versus
visualize_provenance_source_spectrum.py: with so few candidates, most of the 40 boundary cells end up
naming ONE of the 14 feature cells as their best match even when their real influence on it is small,
so coverage is expected to be much higher (most/all feature cells claimed) and each claimed cell
typically averages many contributors -- which is itself the point: the blended colour shows the
CENTRE OF MASS of the stretch of perimeter that a given feature cell draws on, at full resolution,
without the 40-panel clutter a literal one-cell-per-source figure would produce.

Uses the already-computed data/provenance_learned.npz and data/provenance_randomEnsemble100.npz
(step 899) -- no new simulation needed.

  python visualize_provenance_source_spectrum_bulkFeatureOnly.py
"""

import copy

import numpy as np
import torch
import matplotlib.pyplot as plt

import utilities
from embryo import model

torch.set_grad_enabled(False)

utils = utilities.utilities()
p = copy.deepcopy(torch.load('data/StigmergicModelParameters.dat', weights_only=False))
p['ATPParameters'] = None; p['latticePeriodicBoundaryGJ'] = False
m = model(p, 1)
boundaryIndices = set(int(i) for i in utils.computeDomeIndices(m.electricNetwork, mode='tissue'))

numRows, numCols = 11, 11
numCells = numRows * numCols

blocksData = np.load('data/provenanceBlocks.npz', allow_pickle=True)
blocks = blocksData['blocks'].item()
featureIndices = np.concatenate([blocks[name] for name in blocks])
cellToBlock = {int(i): name for name in blocks for i in blocks[name]}
blockLetter = {'eye': 'E', 'nose': 'N', 'mouth': 'M'}

# --- order the 40 boundary cells once around the perimeter, clockwise from (0,0) ----------------
perimeter = []
perimeter += [(0, c) for c in range(numCols)]
perimeter += [(r, numCols - 1) for r in range(1, numRows)]
perimeter += [(numRows - 1, c) for c in range(numCols - 2, -1, -1)]
perimeter += [(r, 0) for r in range(numRows - 2, 0, -1)]
perimeter = [r * numCols + c for r, c in perimeter]
assert sorted(perimeter) == sorted(boundaryIndices) and len(perimeter) == 40

cmap = plt.get_cmap('hsv')
spectrumColor = {cellIdx: np.array(cmap(i / len(perimeter))[:3]) for i, cellIdx in enumerate(perimeter)}

conditions = [('learned', 'data/provenance_learned.npz'),
             ('random (100-seed ensemble)', 'data/provenance_randomEnsemble100.npz')]

fig, axes = plt.subplots(1, len(conditions), figsize=(7.2 * len(conditions), 6.4))

for ax, (label, path) in zip(axes, conditions):
    d = np.load(path, allow_pickle=True)
    P = np.asarray(d['pVmem_step899'])

    # top-1 target of each boundary cell, restricted to the 14 feature cells
    topTarget = {b: int(featureIndices[np.argmax(P[featureIndices, b])]) for b in perimeter}

    contributorsOf = {}
    for b, t in topTarget.items():
        contributorsOf.setdefault(t, []).append(b)

    rgbGrid = np.full((numRows, numCols, 3), 0.92)
    for t, srcList in contributorsOf.items():
        avgColor = np.mean([spectrumColor[b] for b in srcList], axis=0)
        r, c = t // numCols, t % numCols
        rgbGrid[r, c] = avgColor
    for b in perimeter:
        r, c = b // numCols, b % numCols
        rgbGrid[r, c] = spectrumColor[b]

    ax.imshow(rgbGrid, interpolation='nearest')
    for i in featureIndices:
        if int(i) not in contributorsOf:
            r, c = i // numCols, i % numCols
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor='none',
                                       edgecolor='0.4', linewidth=1.0, linestyle=':'))
    for t, srcList in contributorsOf.items():
        r, c = t // numCols, t % numCols
        label_ = blockLetter[cellToBlock[t]] + (f"×{len(srcList)}" if len(srcList) > 1 else "")
        ax.annotate(label_, xy=(c, r), fontsize=6.5, color='white', fontweight='bold',
                   ha='center', va='center')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('0.6')
    unclaimed = len(featureIndices) - len(contributorsOf)
    ax.set_title(f'{label}\n{len(contributorsOf)}/{len(featureIndices)} feature cells claimed as a '
                f'top-1 target ({unclaimed} unclaimed)', fontsize=10)

fig.suptitle('Full-resolution source map, restricted to the 14 facial-feature cells: each boundary\n'
            "cell's single top target among only eye/nose/mouth cells, coloured by a spectrum running "
            'once around the perimeter\n(labels = block + how many boundary cells share that target)',
            fontsize=11, y=1.14)

wheelAx = fig.add_axes([0.40, -0.16, 0.20, 0.20 * (7.2 * len(conditions)) / 6.4])
theta = np.linspace(0, 2 * np.pi, len(perimeter), endpoint=False) + np.pi / 2
wheelAx.scatter(np.cos(theta), -np.sin(theta), c=[spectrumColor[b] for b in perimeter], s=90,
               edgecolors='black', linewidths=0.6)
wheelAx.annotate('start\n(0,0)', xy=(np.cos(theta[0]), -np.sin(theta[0])), xytext=(0, 22),
                 textcoords='offset points', ha='center', fontsize=7.5)
wheelAx.set_xlim(-1.4, 1.4); wheelAx.set_ylim(-1.4, 1.4)
wheelAx.set_aspect('equal')
wheelAx.axis('off')
wheelAx.set_title('perimeter colour key\n(clockwise from top-left corner)', fontsize=8)

plt.savefig('figures/provenanceSourceSpectrumMap_bulkFeatureOnly.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceSourceSpectrumMap_bulkFeatureOnly.png")

for label, path in conditions:
    d = np.load(path, allow_pickle=True)
    P = np.asarray(d['pVmem_step899'])
    topTarget = {b: int(featureIndices[np.argmax(P[featureIndices, b])]) for b in perimeter}
    contributorsOf = {}
    for b, t in topTarget.items():
        contributorsOf.setdefault(t, []).append(b)
    print(f"\n{label}: {len(contributorsOf)}/{len(featureIndices)} feature cells claimed")
    for t, srcs in sorted(contributorsOf.items(), key=lambda kv: -len(kv[1])):
        tr, tc = t // numCols, t % numCols
        srcStr = ", ".join(f"({b//numCols},{b%numCols})" for b in srcs)
        print(f"  {cellToBlock[t]:<6} ({tr},{tc}) <- {len(srcs):>2} boundary cells: {srcStr}")
