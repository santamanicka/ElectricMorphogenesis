"""One figure for the second rejuvenation pass (scores in data/canalizationTalkRejuvenationExtensionScores1888Hold301.json; predictions in
data/canalizationTalkRejuvenationExtensionPredictions1888Hold301.json): for the stripe, the order-3 face and the order-4 face,
top row: the best score any searched code reaches (lowest mV) against the age at which the search starts (iterations after the best moment), with the first training's score, the best of 64 random codes
and the original code on the same aged state; bottom row: the highest overlap any restart reaches, with the 0.9 line. Variants at +300 (larger search, longer hold, wider stripe family) are marked.

    python3 buildCanalizationTalkRejuvenationExtensionFigure11x11.py

Writes figures/canalizationTalkRejuvenationExtension.png (never overwriting).
"""
import json
import os

import numpy as np

from canalizationTalkCommon import *

outputPath = 'figures/canalizationTalkRejuvenationExtension.png'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
results = json.load(open('data/canalizationTalkRejuvenationExtensionScores1888Hold301.json'))
rows = results['rows']
OFFSETS = {'A +100': 100, 'first pass +300': 300, 'A +600': 600, 'A +3000': 3000}
targets = [('Stripe', 'stripe', TEAL), ('FaceOrder3', 'face order 3', OCHRE), ('FaceOrder4', 'face order 4', OCHRE)]
markers = {'B population 64': ('D', 'larger search (population 64)'), 'C hold 1000': ('s', 'longer hold (1,000)'), 'D stripe orders 0-4': ('^', 'wider stripe family (orders 0 to 4)')}
figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True)
figure.subplots_adjust(left=0.085, right=0.985, top=0.82, bottom=0.17, hspace=0.12, wspace=0.22)
handles = {}
for c, (label, name, colour) in enumerate(targets):
    mine = [r for r in rows if r['target'] == label]
    aged = sorted([r for r in mine if r['variant'] in OFFSETS], key=lambda r: OFFSETS[r['variant']])
    x = [OFFSETS[r['variant']] for r in aged]
    top, bottom = axes[0, c], axes[1, c]
    handles['search'], = top.plot(x, [r['bestScore'] for r in aged], 'o-', color=colour, lw=2.4, ms=7, label='best searched code')
    handles['original'], = top.plot(x, [r['original']['score'] for r in aged], 's--', color=INK_3, lw=1.5, ms=5, label='original code, same state')
    handles['random'], = top.plot(x, [r['randomBest'] for r in aged], 'x:', color=INK_3, lw=1.3, ms=7, label='best of 64 random codes')
    first = json.load(open('data/canalizationTalkRejuvenationScores1888Hold301.json'))['targets'][label]['firstTrainingBestScore']
    handles['first'] = top.axhline(first, color=GREEN, lw=1.3, ls=(0, (4, 3)), label='first training (uniform start)')
    bottom.plot(x, [r['highestOverlap'] for r in aged], 'o-', color=colour, lw=2.4, ms=7)
    bottom.axhline(0.9, color=GREEN, lw=1.3, ls=(0, (4, 3)))
    if c == 0:
        bottom.text(3000, 0.92, 'target reached (0.9)', ha='right', fontsize=10, color=GREEN)
    for r in mine:
        if r['variant'] in markers:
            shape, text = markers[r['variant']]
            handles[r['variant']], = top.plot([300], [r['bestScore']], shape, color='white', mec=INK, mew=1.8, ms=9, label=text)
            bottom.plot([300], [r['highestOverlap']], shape, color='white', mec=INK, mew=1.8, ms=9)
    top.set_title(name, fontsize=17, color=colour, fontweight='bold')
    top.set_ylim(0, None)
    bottom.set_ylim(0, 1.05)
    bottom.set_xscale('symlog', linthresh=600)
    bottom.set_xticks([100, 300, 600, 1000, 3000]); bottom.set_xticklabels(['100', '300', '600', '1k', '3k'], fontsize=10)
    bottom.set_xlabel('iterations after the best moment', fontsize=11)
    if c == 0:
        top.set_ylabel('best score (mV, lower is better)', fontsize=10.5)
        bottom.set_ylabel('highest overlap with target', fontsize=10.5)
figure.suptitle('Searching for a code on older tissue, with a larger search, a longer hold or a wider family', fontsize=16, fontweight='bold', y=0.965)
figure.text(0.5, 0.91, '16 restarts per point (population 16; 64 in the larger search); open markers: variants at +300; the first pass is the +300 point', ha='center', fontsize=11.5, color=INK_2)
figure.legend(handles=list(handles.values()), loc='lower center', ncol=4, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
save(figure, outputPath, False)
