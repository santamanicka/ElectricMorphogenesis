"""One figure for the re-canalization runs (EXPLORATORY; scores in data/canalizationTalkRecanalizationScores1888Hold301.json): for the stripe, the order-3 face and the order-4 face, the tissue at the end of
the first hold and at the end of each reapplied hold (ages 3000 and 5000), the frame of each branch closest to the target, and the peak match with the target against age.

    python3 buildCanalizationTalkRecanalizationFigure11x11.py

Writes figures/canalizationTalkRecanalization.png (never overwriting).
"""
import json
import os

import numpy as np

import canalizationTalkFamilies as fam
from canalizationTalkCommon import *

outputPath = 'figures/canalizationTalkRecanalization.png'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
scores = json.load(open('data/canalizationTalkRecanalizationScores1888Hold301.json'))
sources = {3: np.load('data/canalizationTalkRecanalizationTrajectories1888Hold301.npz', allow_pickle=True), 4: np.load('data/canalizationTalkRecanalizationTrajectories1888Hold301Order4.npz', allow_pickle=True)}
rows = [('stripe', 3, 'stripe', 'stripeTrained', TEAL), ('face order 3', 3, 'face', 'faceTrained', OCHRE), ('face order 4', 4, 'face', 'faceTrained', OCHRE)]
AGES = (3000, 5000)
interior = np.array(INTERIOR)


def unpack(array):
    return np.unpackbits(array, axis=-1)[..., :81].astype(bool)


def tissueOf(bits):
    vmem = np.full(LATTICE * LATTICE, -5.0)
    vmem[interior] = np.where(bits, -60.0, -5.0)
    return vmem


figure = plt.figure(figsize=(15, 8.6))
grid = figure.add_gridspec(3, 7, width_ratios=[1, 1, 1, 1, 1, 0.45, 2.1], hspace=0.32, wspace=0.12, left=0.04, right=0.975, top=0.86, bottom=0.11)
titles = ['first guidance\nend of the hold', f'reapplied at {AGES[0]}\nend of the hold', f'reapplied at {AGES[0]}\nclosest to the target', f'reapplied at {AGES[1]}\nend of the hold', f'reapplied at {AGES[1]}\nclosest to the target']
for r, (label, order, target, trainedName, colour) in enumerate(rows):
    store = sources[order]
    names = [str(n) for n in store['names']]
    held = [n for n in names if not n.startswith('noRing')]
    stride = int(store['stride'])
    cells = {'stripe': STRIPE_CELLS, 'face': FACE_CELLS}[target]
    original = unpack(store['original'][names.index(trainedName)])
    panels = [(original[300 // stride], 'First hold', 1.0)]
    for age in AGES:
        branch = unpack(store[f'branch{age}'][held.index(trainedName)])
        bars, faceBest, exact = fam.frameFlags(branch)
        best = int(np.argmax(exact[target]))
        panels += [(branch[300 // stride], f'age {age} hold end', exact[target][300 // stride]), (branch[best], f'age {age} best, tau {best * stride}', exact[target][best])]
    for c, (bits, caption, value) in enumerate(panels):
        axis = figure.add_subplot(grid[r, c])
        drawTissue(axis, tissueOf(bits), outlineCells=cells, outlineColour=colour, lineWidth=1.6)
        axis.set_xlabel(f'IoU {value:.2f}', fontsize=11, color=INK_2)
        if r == 0:
            axis.set_title(titles[c], fontsize=12, color=INK_2, pad=6)
        if c == 0:
            axis.set_ylabel(label, fontsize=17, color=colour, fontweight='bold', labelpad=10)
    axis = figure.add_subplot(grid[r, 6])
    table = scores['targets'][label]['table']
    ages = [1000, 3000, 5000, 10000, 20000]
    axis.axhline(0.9, color=GREEN, lw=1.2, ls=(0, (4, 3)))
    axis.text(20000, 0.92, 'returned (0.9)', ha='right', fontsize=10, color=GREEN)
    axis.plot(ages, [table['reapplied'][str(a)][trainedName]['peakExact'] for a in ages], 'o-', color=colour, lw=2.4, label='code reapplied')
    axis.plot(ages, [table['continued'][str(a)][trainedName]['peakExact'] for a in ages], 's--', color=INK_3, lw=1.6, ms=5, label='nothing applied')
    axis.plot([0], [table['ageZero'][trainedName]['peakExact']], '*', color=colour, ms=15, label='first guidance (age 0)')
    axis.set_xscale('symlog', linthresh=1000)
    axis.set_xticks([0, 1000, 3000, 5000, 10000, 20000]); axis.set_xticklabels(['0', '1k', '3k', '5k', '10k', '20k'], fontsize=10)
    axis.set_ylim(0, 1.08); axis.set_xlim(-200, 24000)
    axis.set_ylabel('peak match with target', fontsize=11)
    if r == 2:
        axis.set_xlabel('iterations of free running before the reapplication', fontsize=11)
    if r == 0:
        legendHandles = axis.get_legend_handles_labels()
figure.suptitle('A code held again on a tissue that has gone to the ocean does not bring its pattern back', fontsize=17, fontweight='bold', y=0.965)
figure.text(0.5, 0.925, 'tissue after the same 301-iteration hold: first from the uniform start, then from the state after 3,000 and 5,000 iterations of free running; outlined: the target', ha='center', fontsize=12, color=INK_2)
figure.legend(*legendHandles, loc='lower right', ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.975, 0.0))
save(figure, outputPath, False)
