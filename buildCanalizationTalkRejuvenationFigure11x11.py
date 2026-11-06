"""One figure for the rejuvenation search (EXPLORATORY; scores in data/canalizationTalkRejuvenationScores1888Hold301.json): for each target, the tissue at the start of the search (best moment + 300),
the best pattern the original code reaches on that aged state, the best pattern any searched code reaches (lowest score), and the searched code with the highest overlap, each with its overlap.

    python3 buildCanalizationTalkRejuvenationFigure11x11.py

Writes figures/canalizationTalkRejuvenation.png (never overwriting).
"""
import glob
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary
from canalizationTalkCommon import *

outputPath = 'figures/canalizationTalkRejuvenation.png'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
scores = json.load(open('data/canalizationTalkRejuvenationScores1888Hold301.json'))
rows = [('Stripe', 'stripe', TEAL, boundary.centreStripeCellIndices), ('FaceOrder3', 'face order 3', OCHRE, boundary.featureCellIndices), ('FaceOrder4', 'face order 4', OCHRE, boundary.featureCellIndices)]
figure = plt.figure(figsize=(12.5, 8.4))
grid = figure.add_gridspec(3, 5, hspace=0.35, wspace=0.14, left=0.07, right=0.98, top=0.85, bottom=0.07)
titles = ['the tissue at the start\n(best moment + 300)', 'target', 'best of the original code\non the aged tissue', 'best of the search\n(lowest score)', 'search code with the\nhighest overlap']
for r, (label, name, colour, cells) in enumerate(rows):
    state = np.load(f'data/agedTissueState1888Hold301{label}After300.npz', allow_pickle=True)
    runs = [np.load(path, allow_pickle=True) for path in sorted(glob.glob(f'data/boundaryHarmonicRejuvenationTraining1888Hold301{label}After300/order*_restart*.npz'))]
    overlaps = [boundary.structuralIntersectionOverUnion(np.asarray(run['bestVmem'], float), cells) for run in runs]
    bestScore = int(np.argmin([float(run['bestScore']) for run in runs]))
    bestOverlap = int(np.argmax(overlaps))
    target = np.full(LATTICE * LATTICE, -5.0)
    target[cells] = -60.0
    original = scores['targets'][label]['originalOnAgedState']
    panels = [(state['vmemVolts'] * 1000.0, 'iteration %d' % int(state['age'])), (target, ''),
              (None, f'overlap {original["overlap"]:.2f}, {original["score"]:.1f} mV'),
              (np.asarray(runs[bestScore]['bestVmem'], float), f'overlap {overlaps[bestScore]:.2f}, {float(runs[bestScore]["bestScore"]):.1f} mV'),
              (np.asarray(runs[bestOverlap]['bestVmem'], float), f'overlap {overlaps[bestOverlap]:.2f}, {float(runs[bestOverlap]["bestScore"]):.1f} mV')]
    for c, (vmem, caption) in enumerate(panels):
        axis = figure.add_subplot(grid[r, c])
        if vmem is None:                                                               # the original code's best pattern on the aged state was not stored: replay it
            from boundaryHarmonicStep import ringHoldBatchReplayFromState
            import torch
            ring = np.clip(np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(len(state['coefficients'])))) @ np.asarray(state['coefficients'], float), 0, float(state['ceiling']))
            bestPattern, bestSquare = [None], [np.inf]
            targetTensor = torch.tensor(target)
            featureMask = np.isin(np.arange(LATTICE * LATTICE), cells)

            def onIteration(iteration, v):
                if iteration >= 301:
                    squared = (v[0] - targetTensor) ** 2
                    value = float((squared[featureMask].mean().sqrt() + squared[~featureMask].mean().sqrt()) / 2)
                    if value < bestSquare[0]:
                        bestSquare[0], bestPattern[0] = value, v[0].numpy().copy()
            ringHoldBatchReplayFromState(ring[None], 301, 3000, onIteration, state['vmemVolts'], state['gpol'])
            vmem = bestPattern[0]
        drawTissue(axis, vmem, outlineCells=cells, outlineColour=colour, lineWidth=1.6)
        axis.set_xlabel(caption, fontsize=10.5, color=INK_2)
        if r == 0:
            axis.set_title(titles[c], fontsize=11.5, color=INK_2, pad=6)
        if c == 0:
            axis.set_ylabel(name, fontsize=17, color=colour, fontweight='bold', labelpad=10)
figure.suptitle('A search for a code on the aged tissue gets part of the way, not to the target', fontsize=16, fontweight='bold', y=0.965)
figure.text(0.5, 0.915, 'tissue at the best pattern of each code after the code is held for 301 iterations on the aged state; outlined: the target; 16 searches per target', ha='center', fontsize=11.5, color=INK_2)
save(figure, outputPath, False)
