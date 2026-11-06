"""The original code and the searched (rejuvenation) codes side by side (EXPLORATORY), for the stripe, the order-3 face and the order-4 face: each code painted on the 40 ring cells (held G_pol / G_ref,
the colour scale shared by every map) and the three profiles overlaid against the angle round the ring. The searched codes are those of figures/canalizationTalkRejuvenationBest.png: over every
restart of every variant, the one with the highest overlap and the one with the lowest score.

    python3 buildCanalizationTalkRejuvenationCodesFigure11x11.py

Writes figures/canalizationTalkRejuvenationCodes.png (never overwriting).
"""
import glob
import json
import os
import re

import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import boundaryCodeUtilities as boundary
from canalizationTalkCommon import *

outputPath = 'figures/canalizationTalkRejuvenationCodes.png'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
rows = [('Stripe', 'stripe', TEAL, boundary.centreStripeCellIndices), ('FaceOrder3', 'face order 3', OCHRE, boundary.featureCellIndices), ('FaceOrder4', 'face order 4', OCHRE, boundary.featureCellIndices)]
VMAX = 2.0
order = np.argsort(RING_ANGLES)
degrees = np.degrees(RING_ANGLES)[order]


def variantName(offset, suffix):
    extra = {'': '', 'Population64': ', population 64', 'LongHold1000': ', hold 1,000', 'Order4Family': ', orders 0-4'}[suffix]
    return f'+{offset}{extra}'


def coefficientText(coefficients):
    return ' '.join(f'{value:+.2f}' for value in coefficients)


figure = plt.figure(figsize=(14.5, 9.2))
grid = figure.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.8], hspace=0.5, wspace=0.12, left=0.07, right=0.985, top=0.84, bottom=0.11)
titles = ['original code\n(trained)', 'searched code with the\nhighest overlap', 'searched code with the\nlowest score', 'the three round the ring']
for r, (label, name, colour, cells) in enumerate(rows):
    state = np.load(f'data/agedTissueState1888Hold301{label}After300.npz', allow_pickle=True)
    ceiling = float(state['ceiling'])
    candidates = []
    for directory in sorted(glob.glob(f'data/boundaryHarmonicRejuvenationTraining1888Hold301{label}After*')):
        match = re.fullmatch(rf'.*{label}After(\d+)(.*)', directory)
        for path in sorted(glob.glob(f'{directory}/order*_restart*.npz')):
            run = np.load(path, allow_pickle=True)
            candidates.append(dict(coefficients=np.asarray(run['bestCoefficients'], float), score=float(run['bestScore']),
                                   overlap=boundary.structuralIntersectionOverUnion(np.asarray(run['bestVmem'], float), cells), variant=variantName(int(match.group(1)), match.group(2)), restart=int(run['restart'])))
    byOverlap, byScore = max(candidates, key=lambda c: c['overlap']), min(candidates, key=lambda c: c['score'])
    codes = [(np.asarray(state['coefficients'], float), 'original', f'{coefficientText(state["coefficients"])}', INK_2),
             (byOverlap['coefficients'], 'highest overlap', f'{coefficientText(byOverlap["coefficients"])}\noverlap {byOverlap["overlap"]:.2f}, {byOverlap["score"]:.1f} mV; {byOverlap["variant"]}', colour),
             (byScore['coefficients'], 'lowest score', f'{coefficientText(byScore["coefficients"])}\noverlap {byScore["overlap"]:.2f}, {byScore["score"]:.1f} mV; {byScore["variant"]}', DIAL)]
    profileAxis = figure.add_subplot(grid[r, 3])
    styles = [dict(ls='-', lw=2.6, color=INK_3), dict(ls='-', lw=2.4, color=colour), dict(ls='--', lw=2.2, color=DIAL)]
    for c, ((coefficients, short, caption, lineColour), style) in enumerate(zip(codes, styles)):
        ringValues = ringValuesOf(coefficients, ceiling)
        axis = figure.add_subplot(grid[r, c])
        drawRingCode(axis, ringValues, VMAX, markAboveEdge=False)
        axis.set_xlabel(caption, fontsize=9.5, color=INK_2)
        if r == 0:
            axis.set_title(titles[c], fontsize=12, color=INK_2, pad=6)
        if c == 0:
            axis.set_ylabel(name, fontsize=17, color=colour, fontweight='bold', labelpad=10)
        profileAxis.plot(degrees, ringValues[order], marker='o', ms=3.5, label=short, **style)
    profileAxis.axhline(ceiling, color=GREEN, lw=1.2, ls=(0, (4, 3)))
    profileAxis.axhline(BISTABLE_EDGE, color=ROSE, lw=1.0, ls=':')
    profileAxis.set_xlim(-180, 180)
    profileAxis.set_ylim(-0.05, 2.6)
    profileAxis.set_xticks([-180, -90, 0, 90, 180])
    profileAxis.set_ylabel('held G_pol / G_ref', fontsize=10.5)
    if r == 0:
        profileAxis.set_title(titles[3], fontsize=12, color=INK_2, pad=6)
        profileAxis.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=9.5, frameon=False)
    if r == 2:
        profileAxis.set_xlabel('angle round the ring (degrees; 0 = top, 180 = bottom)', fontsize=10.5)
    for spine in ('top', 'right'):
        profileAxis.spines[spine].set_visible(False)
figure.suptitle('The original code and the codes found on the aged tissue', fontsize=16, fontweight='bold', y=0.965)
figure.text(0.5, 0.915, 'each code painted on the 40 ring cells (interior blank); coefficients a0, a1, ... of the cosine series under each map; dashed green: the code\'s ceiling; dotted rose: upper edge of a lone cell\'s bistable window', ha='center', fontsize=11, color=INK_2)
colourbarAxis = figure.add_axes([0.25, 0.045, 0.30, 0.018])
figure.colorbar(ScalarMappable(norm=Normalize(0, VMAX), cmap=CODE_MAP), cax=colourbarAxis, orientation='horizontal').set_label('held G_pol / G_ref', fontsize=10.5)
save(figure, outputPath, False)
