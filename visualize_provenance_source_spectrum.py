"""
Full-resolution version of visualize_provenance_source_regions.py's dominance map: instead of
pooling the 40 boundary cells into 5 geometric groups, give every individual boundary cell its own
colour, from a continuous, cyclic spectrum that runs once around the perimeter (starting at cell
(0,0), clockwise, ending back at the same cell with the same colour -- a true wheel, not a gradient
with two mismatched ends). Every bulk cell is then coloured by the average colour of whichever
boundary cell(s) name it as their single top-1 target, which reads directly as "where on the
perimeter does this patch of tissue's identity come from" without needing a discrete legend.

This deliberately answers "which bulk cells do boundary cells best target" (a per-source argmax over
81 candidate targets), not "which boundary cell dominates each bulk cell" (a per-target argmax over
40 candidate sources, which forces full coverage and is what visualize_provenance_source_regions.py's
grouped dominance map already shows at 5-way resolution). Most of the interior is left unclaimed
here, correctly: a boundary cell's single best target is always its immediate neighbour, so no
individual boundary cell's top pick ever reaches deeper than the first ring in from the boundary.

Uses the already-computed data/provenance_learned.npz and data/provenance_randomEnsemble100.npz
(step 899) -- no new simulation needed.

  python visualize_provenance_source_spectrum.py
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
bulkIndices = np.array([i for i in range(numCells) if i not in boundaryIndices])

# --- order the 40 boundary cells once around the perimeter, clockwise from (0,0) ----------------
perimeter = []
perimeter += [(0, c) for c in range(numCols)]                          # top row, left -> right
perimeter += [(r, numCols - 1) for r in range(1, numRows)]              # right col, top -> bottom
perimeter += [(numRows - 1, c) for c in range(numCols - 2, -1, -1)]     # bottom row, right -> left
perimeter += [(r, 0) for r in range(numRows - 2, 0, -1)]                # left col, bottom -> top
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

    # top-1 bulk target of each boundary cell
    topTarget = {b: int(bulkIndices[np.argmax(P[bulkIndices, b])]) for b in perimeter}

    # each bulk cell's colour = average of the spectrum colours of boundary cells naming it #1
    contributorsOf = {}
    for b, t in topTarget.items():
        contributorsOf.setdefault(t, []).append(b)

    rgbGrid = np.full((numRows, numCols, 3), 0.92)  # unclaimed bulk cells: light grey
    for t, srcList in contributorsOf.items():
        avgColor = np.mean([spectrumColor[b] for b in srcList], axis=0)
        r, c = t // numCols, t % numCols
        rgbGrid[r, c] = avgColor
    for b in perimeter:
        r, c = b // numCols, b % numCols
        rgbGrid[r, c] = spectrumColor[b]

    ax.imshow(rgbGrid, interpolation='nearest')
    for t, srcList in contributorsOf.items():
        r, c = t // numCols, t % numCols
        if len(srcList) > 1:
            ax.annotate(str(len(srcList)), xy=(c, r), fontsize=6.5, color='black',
                       ha='center', va='center', fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('0.6')
    unclaimed = len(bulkIndices) - len(contributorsOf)
    ax.set_title(f'{label}\n{len(contributorsOf)}/{len(bulkIndices)} bulk cells claimed as a top-1 '
                f'target ({unclaimed} unclaimed)', fontsize=10)

fig.suptitle('Full-resolution source map: each boundary cell\'s single top bulk target, coloured by\n'
            'a spectrum running once around the perimeter (numbers = how many boundary cells share that target)',
            fontsize=11.5, y=1.1)

# --- perimeter colour wheel as a legend, drawn as a ring of the actual 40 boundary-cell colours --
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

plt.savefig('figures/provenanceSourceSpectrumMap.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceSourceSpectrumMap.png")

for label, path in conditions:
    d = np.load(path, allow_pickle=True)
    P = np.asarray(d['pVmem_step899'])
    topTarget = {b: int(bulkIndices[np.argmax(P[bulkIndices, b])]) for b in perimeter}
    contributorsOf = {}
    for b, t in topTarget.items():
        contributorsOf.setdefault(t, []).append(b)
    shared = {t: srcs for t, srcs in contributorsOf.items() if len(srcs) > 1}
    print(f"\n{label}: {len(contributorsOf)} distinct bulk cells claimed, "
         f"{len(shared)} claimed by more than one boundary cell")
    for t, srcs in sorted(shared.items(), key=lambda kv: -len(kv[1])):
        tr, tc = t // numCols, t % numCols
        srcStr = ", ".join(f"({b//numCols},{b%numCols})" for b in srcs)
        print(f"  target ({tr},{tc}) <- {len(srcs)} sources: {srcStr}")
