"""Build the "Pinched Dial" report page (PolyPatterning_Sim.md, Section 12): the boundary dial with a second-order
harmonic added on model 1888, every held value kept in [0, 1.3], alone and together with the first-order gradient of
the "Tilted Dial" report.

Order 2 alone: code = DC + G2 cos(2 (theta - phi2)) over the whole allowed (DC, G2) triangle for phi2 = 0, 22.5 and 45
degrees. Both orders: code = DC + G1 cos(theta) + G2 cos(2 theta), mirror-symmetric about the vertical axis, with each
code classed by whether the order-2 term adds inflection points to the ring's profile (|G2| > G1 / 4).

Inputs: data/boundaryGradientLandscapeSummary1888Hold301Order2.json and ...Hold301.json
(analyzeBoundaryGradientLandscape11x11.py, for order 2 and, as the comparison, order 1),
data/boundaryOrderPairLandscapeSummary1888Hold301.json (analyzeBoundaryOrderPairLandscape11x11.py) and
data/boundaryDialSweep1888Hold301.npz. The page's JSON keys are the ones figures/boundarySecondOrderTemplate.html reads.
"""
import argparse
import json

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--firstOrderSummaryPath', type=str, default='data/boundaryGradientLandscapeSummary1888Hold301.json')
parser.add_argument('--secondOrderSummaryPath', type=str, default='data/boundaryGradientLandscapeSummary1888Hold301Order2.json')
parser.add_argument('--pairSummaryPath', type=str, default='data/boundaryOrderPairLandscapeSummary1888Hold301.json')
parser.add_argument('--sweepPath', type=str, default='data/boundaryDialSweep1888Hold301.npz')
parser.add_argument('--templatePath', type=str, default='figures/boundarySecondOrderTemplate.html')
parser.add_argument('--outputPath', type=str, default='figures/boundarySecondOrder.html')
args = parser.parse_args()

first, second, pair = (json.load(open(path)) for path in (args.firstOrderSummaryPath, args.secondOrderSummaryPath, args.pairSummaryPath))
sweep = np.load(args.sweepPath)
limit, step = second['dialLimit'], second['gridStep']
dials = sweep['dialLevel']
onGridStep = np.isclose(np.round(dials / step) * step, dials) & (dials <= limit + 1e-9)
poolFrom = 12


def pooledSeamSteps(entry):
    steps = np.concatenate([np.array(entry['steps']['alongGradient']), np.array(entry['steps']['alongDial'])])
    cells = np.minimum(steps[:, 3].astype(int), poolFrom)
    return [dict(cells=int(value), count=int((cells == value).sum()), quartiles=np.percentile(steps[cells == value, 2], [25, 50, 75]).round(3).tolist())
            for value in np.unique(cells)]


def harmonicPayload(summary):
    perDirection = {}
    for key, entry in summary['perDirection'].items():
        classShare = np.array(entry['classShare'])
        perDirection[key] = dict(dial=entry['dial'], gradient=entry['gradient'], change=entry['changeAll'],
                                 driven=classShare[:, summary.get('drivenClass', 0)].round(3).tolist(), dialLike=classShare[:, 2].round(3).tolist(),
                                 nearest=entry['nearestDistance'], crosses=[int(value) for value in entry['crossesJump']], pattern=entry['pattern'],
                                 crossing=entry['crossing'], steps=entry['steps']['summary'], seams=entry['seams'], motifs=entry['motifs'],
                                 classSummary=entry['classSummary'], seamSteps=pooledSeamSteps(entry))
    return perDirection


def comparison(summary):
    return dict(motifs=summary['motifs']['union'], motifsMerged=summary['motifs']['unionCanonical'], dimension=summary['dimension']['withGradientAll'],
                keepDialMotif=summary['motifs']['keepDialMotif'],
                perDirection={key: dict(medianChange=float(np.median(entry['changeAll'])), motifs=entry['motifs']['motifs'],
                                        dimension=summary['dimension'][f'withGradient{key}'],
                                        spanningChanged=entry['crossing']['bothTrue'] / (entry['crossing']['bothTrue'] + entry['crossing']['crossOnly']),
                                        quietChanged=entry['crossing']['changeOnly'] / (entry['crossing']['changeOnly'] + entry['crossing']['neither']))
                              for key, entry in summary['perDirection'].items()})


# two worked examples of whether the orders' changes add up: a code whose ring reaches no jump, and the most typical code of
# the commonest motif no single order makes
codes = pair['codes']
index = {(round(d / pair['step']), round(g1 / pair['step']), round(g2 / pair['step'])): row
         for row, (d, g1, g2) in enumerate(zip(codes['dial'], codes['first'], codes['second']))}
dialPatterns = {round(level / 0.01): pattern for level, pattern in zip(dials, sweep['windowMeanVmem'])}
interaction = dict(zip(pair['additivity']['rows'], pair['additivity']['interactionShare']))


def example(row, label):
    i, j, k = round(codes['dial'][row] / pair['step']), round(codes['first'][row] / pair['step']), round(codes['second'][row] / pair['step'])
    dialPattern = dialPatterns[round(codes['dial'][row] / 0.01)]
    alone1, alone2 = np.array(codes['pattern'][index[(i, j, 0)]]), np.array(codes['pattern'][index[(i, 0, k)]])
    predicted = alone1 + alone2 - dialPattern
    actual = np.array(codes['pattern'][row])
    return dict(label=label, dial=codes['dial'][row], first=codes['first'][row], second=codes['second'][row], dialPattern=dialPattern.round(1).tolist(),
                alone1=alone1.round(1).tolist(), alone2=alone2.round(1).tolist(), predicted=predicted.round(1).tolist(), actual=actual.round(1).tolist(),
                mismatch=float(np.sqrt(((predicted - actual) ** 2).mean())), change=codes['change'][row], interactionShare=float(interaction[row]))


quietCandidates = [row for row in pair['additivity']['rows'] if not codes['crosses'][row] and codes['kind'][row] == 'addsInflections'
                   and 0.8 < codes['change'][row] < 2.5 and interaction[row] < 0.05]
quietRow = max(quietCandidates, key=lambda row: codes['change'][row])
galleryExample = pair['newMotifGallery'][0]['example']
seamRow = index[(round(galleryExample['dial'] / pair['step']), round(galleryExample['first'] / pair['step']), round(galleryExample['second'] / pair['step']))]
examples = [example(quietRow, 'reaches no jump'), example(seamRow, 'past a jump')]
print(f"examples: quiet code {examples[0]['dial']}, {examples[0]['first']}, {examples[0]['second']} (mismatch {examples[0]['mismatch']:.2f} mV of change {examples[0]['change']:.2f}); "
      f"seam code {examples[1]['dial']}, {examples[1]['first']}, {examples[1]['second']} (mismatch {examples[1]['mismatch']:.2f} mV of change {examples[1]['change']:.2f})")

payload = dict(
    limit=limit, step=step, pairStep=pair['step'], hold=second['holdIterations'], windowStart=second['windowStart'], directions=second['directions'],
    jumpMilliVolts=second['jumpMilliVolts'], saddle=boundary.singleCellSaddleMilliVolts, poolFrom=poolFrom,
    dial=dict(levels=dials[onGridStep].round(3).tolist(), patterns=sweep['windowMeanVmem'][onGridStep].round(1).tolist(), jumps=second['dialSweep']['jumpDials'],
              stepDials=second['dialSweep']['dials'], steps=second['dialSweep']['steps']),
    free=second['free']['pattern'], order2=harmonicPayload(second),
    order2Motifs=second['motifs'], order2Dimension=second['dimension'], order2Checks=second['symmetryChecks'],
    compare=dict(order1=comparison(first), order2=comparison(second)),
    pair=dict(codes=codes, byKind=pair['byKind'], additivity=pair['additivity'], steps=pair['steps'], reach=pair['reach'], motifs=pair['motifs'],
              gallery=pair['newMotifGallery'], newMotifCodes=pair['newMotifCodes'], newMotifStats=pair['newMotifStats'], dimension=pair['dimension'], flipChecks=pair['flipChecks'],
              mirrorResidual=pair['mirrorResidual'], consistency=pair['singleGridConsistency'], inflection=pair['inflection'], dialSpread=pair['dialSpread']),
    examples=examples)
page = open(args.templatePath).read().replace('__DATA__', json.dumps(payload, separators=(',', ':')))
open(args.outputPath, 'w').write(page)
print(f"wrote {args.outputPath} ({len(page) / 1e6:.2f} MB)")
