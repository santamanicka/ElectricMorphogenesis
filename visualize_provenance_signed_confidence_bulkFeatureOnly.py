"""
Signed-value version of visualize_provenance_source_confidence_bulkFeatureOnly.py: the same
confidence-weighted dominance map (hue = winning boundary group, opacity = margin over the
runner-up), but the group totals and margin are now computed from the EXACT, fully-propagated
signed-value decomposition (measure_provenance_propagation_signed.py /
measure_provenance_ensemble_signed.py) -- real summed mV, not magnitude-weighted shares -- which is
the more rigorous version of the same comparison. See Appendix 17 for why the two turn out to give
nearly the same answer (mean margin 0.292 vs. 0.212 here, against 0.282 vs. 0.204 for the
share-based version): sign disagreement between contributors is real but rare in this tissue (38 of
14,641 pairs under learned, 0 under the ensemble), so the two methods mostly agree.

  python visualize_provenance_signed_confidence_bulkFeatureOnly.py
"""

import copy

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

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

groups = {'north': [], 'south': [], 'west': [], 'east': [], 'corner': []}
for i in range(numCells):
    r, c = i // numCols, i % numCols
    if i not in boundaryIndices:
        continue
    isCorner = (r in (0, numRows - 1)) and (c in (0, numCols - 1))
    if isCorner:
        groups['corner'].append(i)
    elif r == 0:
        groups['north'].append(i)
    elif r == numRows - 1:
        groups['south'].append(i)
    elif c == 0:
        groups['west'].append(i)
    elif c == numCols - 1:
        groups['east'].append(i)
for name in groups:
    groups[name] = np.array(groups[name])
groupNames = list(groups.keys())
groupColor = {'north': '#4C72B0', 'south': '#C44E52', 'west': '#55A868',
             'east': '#8172B2', 'corner': '#937860'}

conditions = [('learned', 'data/provenanceSigned_learned.npz'),
             ('random (100-seed ensemble)', 'data/provenanceSigned_randomEnsemble100.npz')]

MARGIN_SATURATE = 0.30
MIN_ALPHA = 0.12

fig, axes = plt.subplots(1, len(conditions), figsize=(6.4 * len(conditions), 5.6))

allMargins = {}
for ax, (label, path) in zip(axes, conditions):
    d = np.load(path, allow_pickle=True)
    key = [k for k in d.keys() if k.startswith('vVmem_step')][0]
    V = d[key]

    groupTotals = np.zeros((numCells, len(groupNames)))
    for gi, name in enumerate(groupNames):
        groupTotals[:, gi] = V[:, groups[name]].sum(axis=1)

    rgbaGrid = np.zeros((numRows, numCols, 4))
    rgbaGrid[..., :3] = 0.92; rgbaGrid[..., 3] = 1.0
    margins = {}
    for t in featureIndices:
        t = int(t)
        totals = np.abs(groupTotals[t])
        order = np.argsort(-totals)
        winner, runner = totals[order[0]], totals[order[1]]
        margin = (winner - runner) / (winner + runner)
        margins[t] = margin
        alpha = MIN_ALPHA + (1 - MIN_ALPHA) * min(margin / MARGIN_SATURATE, 1.0)
        r, c = t // numCols, t % numCols
        rgbaGrid[r, c] = to_rgba(groupColor[groupNames[order[0]]], alpha)
    allMargins[label] = margins

    ax.imshow(rgbaGrid, interpolation='nearest')
    for name in groupNames:
        rs, cs = groups[name] // numCols, groups[name] % numCols
        ax.scatter(cs, rs, s=90, facecolors=groupColor[name], edgecolors='black', linewidths=1.3,
                  marker='s', zorder=5)
    for t in featureIndices:
        t = int(t)
        r, c = t // numCols, t % numCols
        ax.annotate(f"{blockLetter[cellToBlock[t]]}\n{margins[t]:.2f}", xy=(c, r), fontsize=6.5,
                   color='black', fontweight='bold', ha='center', va='center')
    ax.set_xticks([]); ax.set_yticks([])
    meanMargin = np.mean(list(margins.values()))
    ax.set_title(f'{label}\nmean margin {meanMargin:.3f} (exact signed values)', fontsize=11)

legendElems = [plt.Rectangle((0, 0), 1, 1, color=groupColor[n], label=f'{n} boundary') for n in groupNames]
fig.legend(handles=legendElems, loc='lower center', ncol=5, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Signed-value confidence-weighted dominance map: hue = winning boundary group by exact\n'
            'signed mV group total (not magnitude share), opacity = margin over the runner-up group',
            fontsize=10.5, y=1.05)
plt.savefig('figures/provenanceSignedConfidence_bulkFeatureOnly.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceSignedConfidence_bulkFeatureOnly.png")

print(f"\n{'cell':<12}{'learned margin':>16}{'ensemble margin':>18}")
for t in featureIndices:
    t = int(t)
    r, c = t // numCols, t % numCols
    ml = allMargins['learned'][t]
    me = allMargins['random (100-seed ensemble)'][t]
    print(f"{cellToBlock[t]:<6}({r},{c}){ml:>13.3f}{me:>18.3f}")
meanL = np.mean(list(allMargins['learned'].values()))
meanE = np.mean(list(allMargins['random (100-seed ensemble)'].values()))
print(f"\nmean margin: learned {meanL:.3f}, ensemble {meanE:.3f}")

# ==================================================================================================
# Figure 2: where does sign disagreement actually occur (the one genuinely new finding)?
# ==================================================================================================
gridsByCond = {}
for label, path in conditions:
    d = np.load(path, allow_pickle=True)
    key = [k for k in d.keys() if k.startswith('vVmem_step')][0]
    V = d[key]
    finalVmem = d['finalVmem']

    disagreeMag = np.zeros(numCells)
    disagreeCount = np.zeros(numCells, dtype=int)
    for t in range(numCells):
        row = V[t]
        targetSign = np.sign(finalVmem[t])
        opp = np.sign(row) != targetSign
        disagreeCount[t] = int(opp.sum())
        disagreeMag[t] = np.abs(row[opp]).sum()
    gridsByCond[label] = (disagreeMag.reshape(numRows, numCols), int(disagreeCount.sum()))

sharedVmax = max(g.max() for g, _ in gridsByCond.values())
sharedVmax = sharedVmax if sharedVmax > 0 else 1.0   # guard against an all-zero shared scale too

fig2, axes2 = plt.subplots(1, len(conditions), figsize=(6.4 * len(conditions), 5.6))
for ax, (label, path) in zip(axes2, conditions):
    grid, totalPairs = gridsByCond[label]
    im = ax.imshow(grid, cmap='inferno', vmin=0, vmax=sharedVmax, interpolation='nearest')
    for t in featureIndices:
        t = int(t)
        r, c = t // numCols, t % numCols
        ax.add_patch(plt.Circle((c, r), 0.42, facecolor='none', edgecolor='cyan', linewidth=1.2))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{label}\n{totalPairs} opposite-sign (target,source) pairs of {numCells*numCells}',
                fontsize=10.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='total opposing magnitude (mV)')

fig2.suptitle('Where contributors actually pull opposite to their target (cyan circles = facial-'
             'feature cells)\nEnsemble averaging eliminates this entirely; the learned clamp keeps a '
             'small amount', fontsize=10.5, y=1.05)
plt.savefig('figures/provenanceSignedDisagreement.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceSignedDisagreement.png")
