"""
True top-5 version of visualize_provenance_source_spectrum_bulkFeatureOnly.py, which only showed
each boundary cell's single #1 target (a top-1 map, despite the naming pattern). This version asks,
for every one of the 40 boundary cells, "which of the 14 feature cells are in your personal top-5?"
(ranked among only those 14 candidates), and blends every qualifying boundary cell's perimeter-
spectrum colour into each feature cell it names. That keeps full 40-cell resolution and a real top-5
criterion in a single picture per condition, instead of needing 40 individual panels.

The blend weight is a genuine design choice, not a fixed fact, so four schemes are produced side by
side:
  - linear:   weight = share            (the original attempt; direct provenance value)
  - quadratic: weight = share^2         (emphasises strong contributors more)
  - rankGeometric: weight = 0.5^(rank-1) (ignores magnitude entirely, decays by rank position --
    rank 1 counts 16x more than rank 5 -- so the blend stays close to the sharp top-1 map while still
    letting ranks 2-5 tint it, rather than blending ~15 near-equal contributors into a muddy average)
  - valueWeighted: weight = share * |finalVmem[source] - V_th| -- the first three schemes reweight
    only the STRUCTURAL share, which is largely a property of the fixed tissue geometry (nearby
    boundary cells qualify for a target's top-5 regardless of clamp) and so barely differs between
    conditions. This scheme injects the one thing that IS clamp-specific: how strongly polarised a
    given boundary cell's own realised state actually is. A source that the clamp drove hard (far
    from V_th) dominates the blend; a source near threshold contributes almost nothing regardless of
    its structural share. Under the learned clamp, boundary |Vmem-V_th| varies source-to-source
    (std 3.3 mV, range 12-24); under the ensemble it is smaller and closer to spatially uniform
    (std 2.6 mV, range 5-14), since averaging over random clamps flattens value variation the same
    way it flattens spatial asymmetry.

Uses the already-computed data/provenance_learned.npz and data/provenance_randomEnsemble100.npz
(step 899) -- no new simulation needed.

  python visualize_provenance_source_spectrum_bulkFeatureOnly_top5.py
"""

import copy

import numpy as np
import torch
import matplotlib.pyplot as plt

import utilities
from embryo import model

torch.set_grad_enabled(False)
TOP_K = 5

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

V_TH_MV = -27.0

weightSchemes = {
    'linear': lambda share, rank, valueDev: share,
    'quadratic': lambda share, rank, valueDev: share ** 2,
    'rankGeometric': lambda share, rank, valueDev: 0.5 ** (rank - 1),
    'valueWeighted': lambda share, rank, valueDev: share * valueDev,
}


def buildBlend(P, weightFn, finalVmem):
    valueDev = {b: abs(finalVmem[b] - V_TH_MV) for b in perimeter}
    top5Of = {}
    for b in perimeter:
        shares = P[featureIndices, b]
        order = np.argsort(-shares)[:TOP_K]
        top5Of[b] = list(zip(featureIndices[order], shares[order]))

    weightedColor = {int(t): np.zeros(3) for t in featureIndices}
    totalWeight = {int(t): 0.0 for t in featureIndices}
    contributorCount = {int(t): 0 for t in featureIndices}
    for b in perimeter:
        for rank, (t, share) in enumerate(top5Of[b], 1):
            w = weightFn(share, rank, valueDev[b])
            t = int(t)
            weightedColor[t] += w * spectrumColor[b]
            totalWeight[t] += w
            contributorCount[t] += 1
    return weightedColor, totalWeight, contributorCount


for schemeName, weightFn in weightSchemes.items():
    fig, axes = plt.subplots(1, len(conditions), figsize=(7.2 * len(conditions), 6.4))

    for ax, (label, path) in zip(axes, conditions):
        d = np.load(path, allow_pickle=True)
        P = np.asarray(d['pVmem_step899'])
        finalVmem = d['finalVmem']
        weightedColor, totalWeight, contributorCount = buildBlend(P, weightFn, finalVmem)

        rgbGrid = np.full((numRows, numCols, 3), 0.92)
        for t in featureIndices:
            t = int(t)
            if totalWeight[t] > 0:
                r, c = t // numCols, t % numCols
                rgbGrid[r, c] = weightedColor[t] / totalWeight[t]
        for b in perimeter:
            r, c = b // numCols, b % numCols
            rgbGrid[r, c] = spectrumColor[b]

        ax.imshow(rgbGrid, interpolation='nearest')
        for t in featureIndices:
            t = int(t)
            r, c = t // numCols, t % numCols
            ax.annotate(f"{blockLetter[cellToBlock[t]]}×{contributorCount[t]}", xy=(c, r), fontsize=6.5,
                       color='white', fontweight='bold', ha='center', va='center')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('0.6')
        ax.set_title(f'{label}', fontsize=11)

    fig.suptitle(f'Top-5 source map, weight scheme = {schemeName}: each feature cell coloured by the '
                f'{schemeName}-weighted\nblend of every boundary cell that has it in its own top-5 '
                '(of the 14 feature-cell candidates)\n(labels = block + contributor count)',
                fontsize=10.7, y=1.16)

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

    outPath = f'figures/provenanceSourceSpectrumMap_bulkFeatureOnly_top5_{schemeName}.png'
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {outPath}")
