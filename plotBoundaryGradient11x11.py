"""Build the "Tilted Dial" report page (PolyPatterning_Sim.md, Section 12): the boundary dial with a first-order
gradient added on model 1888, every held value kept in [0, 1.3].

Step 1, the landscape: the dial alone below 1.3 at the 301-iteration hold, then code = DC + G cos(theta - phi) over the
whole allowed (DC, G) triangle for phi = 0, 22.5 and 45 degrees: how far each pattern moves from its G = 0 twin, where
the moves happen (the lines on which a ring cell's held value crosses a dial jump), which symmetry class they land in,
and what the gradient reaches that the dial alone does not.

Inputs: data/boundaryGradientLandscapeSummary1888Hold301.json (analyzeBoundaryGradientLandscape11x11.py) and
data/boundaryDialSweep1888Hold301.npz (simulateBoundaryDialLandscape11x11.py). The page's JSON keys are the ones
figures/boundaryGradientTemplate.html reads.
"""
import argparse
import json

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--summaryPath', type=str, default='data/boundaryGradientLandscapeSummary1888Hold301.json')
parser.add_argument('--sweepPath', type=str, default='data/boundaryDialSweep1888Hold301.npz')
parser.add_argument('--templatePath', type=str, default='figures/boundaryGradientTemplate.html')
parser.add_argument('--outputPath', type=str, default='figures/boundaryGradient.html')
args = parser.parse_args()

summary = json.load(open(args.summaryPath))
sweep = np.load(args.sweepPath)
limit, step = summary['dialLimit'], summary['gridStep']
dials = sweep['dialLevel']
onGridStep = np.isclose(np.round(dials / step) * step, dials) & (dials <= limit + 1e-9)

poolFrom = 12   # steps carrying this many ring cells across a jump, or more, are pooled
perDirection = {}
for key, entry in summary['perDirection'].items():
    classShare = np.array(entry['classShare'])
    steps = np.concatenate([np.array(entry['steps']['alongGradient']), np.array(entry['steps']['alongDial'])])
    cellsCrossing = np.minimum(steps[:, 3].astype(int), poolFrom)
    seamSteps = [dict(cells=int(cells), count=int((cellsCrossing == cells).sum()),
                      quartiles=np.percentile(steps[cellsCrossing == cells, 2], [25, 50, 75]).round(3).tolist())
                 for cells in np.unique(cellsCrossing)]
    perDirection[key] = dict(dial=entry['dial'], gradient=entry['gradient'], change=entry['changeAll'],
                             changeInterior=entry['changeInterior'], odd=classShare[:, 0].round(3).tolist(),
                             dialLike=classShare[:, 2].round(3).tolist(), nearest=entry['nearestDistance'],
                             crosses=[int(value) for value in entry['crossesJump']], churn=entry['churn'], pattern=entry['pattern'],
                             crossing=entry['crossing'], steps=entry['steps']['summary'], seams=entry['seams'], motifs=entry['motifs'],
                             classSummary=entry['classSummary'], seamSteps=seamSteps)

payload = dict(
    limit=limit, step=step, hold=summary['holdIterations'], windowStart=summary['windowStart'], directions=summary['directions'],
    jumpMilliVolts=summary['jumpMilliVolts'], changeMilliVolts=summary['changeMilliVolts'],
    bistable=list(boundary.singleCellBistableRange), saddle=boundary.singleCellSaddleMilliVolts,
    dial=dict(levels=dials[onGridStep].round(3).tolist(), patterns=sweep['windowMeanVmem'][onGridStep].round(1).tolist(),
              stepDials=summary['dialSweep']['dials'], steps=summary['dialSweep']['steps'], jumps=summary['dialSweep']['jumpDials'],
              churn=sweep['windowStdVmem'][onGridStep][:, boundary.interiorCellIndices].mean(1).round(3).tolist()),
    free=summary['free']['pattern'], perDirection=perDirection, poolFrom=poolFrom, motifs=summary['motifs'], dimension=summary['dimension'],
    patternSpace=summary['patternSpace'], symmetryChecks=summary['symmetryChecks'],
    mirrorResidual={key: summary[f'mirrorResidual{key}'] for key in ('0', '45')})

page = open(args.templatePath).read().replace('__DATA__', json.dumps(payload, separators=(',', ':')))
open(args.outputPath, 'w').write(page)
print(f"wrote {args.outputPath} ({len(page) / 1e6:.2f} MB)")
