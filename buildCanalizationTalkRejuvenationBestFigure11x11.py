"""The best patterns of the whole rejuvenation search so far (first and second pass; all ages, hold lengths, search sizes and families), for the stripe, the order-3 face and the order-4 face:
the target, the first training's best pattern (uniform start, for reference), and, over every restart of every variant, the pattern with the highest overlap and the one with the lowest score,
each captioned with its overlap, score and the variant it came from.

    python3 buildCanalizationTalkRejuvenationBestFigure11x11.py

Writes figures/canalizationTalkRejuvenationBest.png (never overwriting).
"""
import glob
import json
import os
import re

import numpy as np

import boundaryCodeUtilities as boundary
import canalizationTalkCommon as common
from canalizationTalkCommon import *

outputPath = 'figures/canalizationTalkRejuvenationBest.png'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
rows = [('Stripe', 'stripe', TEAL, boundary.centreStripeCellIndices, None), ('FaceOrder3', 'face order 3', OCHRE, boundary.featureCellIndices, 3), ('FaceOrder4', 'face order 4', OCHRE, boundary.featureCellIndices, 4)]
summary = json.load(open('data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json'))


def variantName(offset, suffix):
    extra = {'': '', 'Population64': ', population 64', 'LongHold1000': ', hold 1,000', 'Order4Family': ', orders 0-4'}[suffix]
    return f'+{offset}{extra}'


figure = plt.figure(figsize=(13.5, 8.6))
grid = figure.add_gridspec(3, 4, hspace=0.42, wspace=0.12, left=0.075, right=0.985, top=0.83, bottom=0.04)
titles = ['target', 'first training\n(uniform start)', 'highest overlap of\nany searched code', 'lowest score of\nany searched code']
for r, (label, name, colour, cells, order) in enumerate(rows):
    candidates = []
    for directory in sorted(glob.glob(f'data/boundaryHarmonicRejuvenationTraining1888Hold301{label}After*')):
        match = re.fullmatch(rf'.*{label}After(\d+)(.*)', directory)
        offset, suffix = int(match.group(1)), match.group(2)
        for path in sorted(glob.glob(f'{directory}/order*_restart*.npz')):
            run = np.load(path, allow_pickle=True)
            vmem = np.asarray(run['bestVmem'], float)
            candidates.append(dict(vmem=vmem, score=float(run['bestScore']), overlap=boundary.structuralIntersectionOverUnion(vmem, cells), variant=variantName(offset, suffix), restart=int(run['restart'])))
    if order is None:
        firstPath = common.STRIPE_RUN_PATH
    else:
        winner = summary['orders'][str(order)]['best']
        firstPath = f"{summary['trainingDirs'][winner['round']]}/order{order}_restart{winner['restart']:02d}.npz"
    first = np.load(firstPath, allow_pickle=True)
    firstVmem = np.asarray(first['bestVmem'], float)
    target = np.full(LATTICE * LATTICE, -5.0)
    target[cells] = -60.0
    byOverlap = max(candidates, key=lambda c: c['overlap'])
    byScore = min(candidates, key=lambda c: c['score'])
    panels = [(target, f'{len(candidates)} searched codes'),
              (firstVmem, f'overlap {boundary.structuralIntersectionOverUnion(firstVmem, cells):.2f}, {float(first["bestScore"]):.1f} mV'),
              (byOverlap['vmem'], f'overlap {byOverlap["overlap"]:.2f}, {byOverlap["score"]:.1f} mV\n{byOverlap["variant"]}, restart {byOverlap["restart"]}'),
              (byScore['vmem'], f'overlap {byScore["overlap"]:.2f}, {byScore["score"]:.1f} mV\n{byScore["variant"]}, restart {byScore["restart"]}')]
    for c, (vmem, caption) in enumerate(panels):
        axis = figure.add_subplot(grid[r, c])
        drawTissue(axis, vmem, outlineCells=cells, outlineColour=colour, lineWidth=1.6)
        axis.set_xlabel(caption, fontsize=10.5, color=INK_2)
        if r == 0:
            axis.set_title(titles[c], fontsize=12, color=INK_2, pad=6)
        if c == 0:
            axis.set_ylabel(name, fontsize=17, color=colour, fontweight='bold', labelpad=10)
figure.suptitle('The best patterns any searched code reaches on an aged tissue', fontsize=16, fontweight='bold', y=0.965)
figure.text(0.5, 0.915, 'over every restart of every variant (ages +100 to +3000 after the best moment, holds 301 and 1,000, populations 16 and 64, stripe orders 0-2 and 0-4); outlined: the target', ha='center', fontsize=11, color=INK_2)
save(figure, outputPath, False)
