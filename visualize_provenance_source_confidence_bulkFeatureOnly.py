"""
Confidence-weighted dominance map: the plain dominance map (visualize_provenance_source_regions_
bulkFeatureOnly.py) colours each feature cell by its winning boundary group, but a forced argmax
throws away HOW CLOSE the contest was -- a group winning 20.5% to 15.4% and a group winning 19.1% to
18.5% both render as a flat, equally solid win. That turns out to be exactly why the plain dominance
map (and the top-5 blends) failed to discriminate cleanly between learned and random: the raw winner
identity is often nearly tied, and a forced choice hides that.

This script keeps the hue (which group wins) but modulates opacity by the MARGIN between the winner
and the runner-up group -- (winner - runner-up) / (winner + runner-up) -- so a decisive win renders
vivid and a near-tie renders washed out. The margin is computed from the exact same provenance
matrices already used everywhere else; no new simulation or method is needed, only a different
statistic drawn from the same numbers.

Mean margin across the 14 feature cells is a real, quantitative discriminator: 0.282 under the
learned clamp vs. 0.204 under the 100-seed ensemble -- the learned clamp commits to an asymmetric
answer noticeably more decisively, consistent with the ensemble average being a blend of 100
differently-oriented random clamps whose individual "wins" partially cancel.

  python visualize_provenance_source_confidence_bulkFeatureOnly.py
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

conditions = [('learned', 'data/provenance_learned.npz'),
             ('random (100-seed ensemble)', 'data/provenance_randomEnsemble100.npz')]

# margin -> opacity: a bare tie (margin 0) is nearly invisible, margin >= MARGIN_SATURATE is fully opaque
MARGIN_SATURATE = 0.30
MIN_ALPHA = 0.12

fig, axes = plt.subplots(1, len(conditions), figsize=(6.4 * len(conditions), 5.6))

allMargins = {}
for ax, (label, path) in zip(axes, conditions):
    d = np.load(path, allow_pickle=True)
    P = np.asarray(d['pVmem_step899'])

    groupTotals = np.zeros((numCells, len(groupNames)))
    for gi, name in enumerate(groupNames):
        groupTotals[:, gi] = P[:, groups[name]].sum(axis=1)

    rgbaGrid = np.zeros((numRows, numCols, 4))
    rgbaGrid[..., :3] = 0.92; rgbaGrid[..., 3] = 1.0
    margins = {}
    for t in featureIndices:
        t = int(t)
        totals = groupTotals[t]
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
    ax.set_title(f'{label}\nmean margin {meanMargin:.3f}', fontsize=11)

legendElems = [plt.Rectangle((0, 0), 1, 1, color=groupColor[n], label=f'{n} boundary') for n in groupNames]
fig.legend(handles=legendElems, loc='lower center', ncol=5, fontsize=9.5, bbox_to_anchor=(0.5, -0.02))
fig.suptitle('Confidence-weighted dominance map: hue = winning boundary group, opacity = margin over\n'
            'the runner-up group (washed out = near-tie, vivid = decisive) -- letters + numbers show '
            'block and margin', fontsize=10.5, y=1.05)
plt.savefig('figures/provenanceSourceConfidence_bulkFeatureOnly.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures/provenanceSourceConfidence_bulkFeatureOnly.png")

print(f"\n{'cell':<12}{'learned margin':>16}{'ensemble margin':>18}{'difference':>13}")
for t in featureIndices:
    t = int(t)
    r, c = t // numCols, t % numCols
    ml = allMargins['learned'][t]
    me = allMargins['random (100-seed ensemble)'][t]
    print(f"{cellToBlock[t]:<6}({r},{c}){ml:>13.3f}{me:>18.3f}{ml-me:>13.3f}")
meanL = np.mean(list(allMargins['learned'].values()))
meanE = np.mean(list(allMargins['random (100-seed ensemble)'].values()))
print(f"\nmean margin: learned {meanL:.3f}, ensemble {meanE:.3f}, difference {meanL-meanE:+.3f}")
