"""Score the registered ends-independence test (E1-E5, V1, V2 of boundaryHarmonicStripeEndsPredictions...json) on the confirmatory draw.

Each criterion is evaluated as written and reported with its numbers and exact (Clopper-Pearson) 95% intervals; nothing else is folded in.

    python3 analyzeBoundaryHarmonicStripeEndsConfirmation11x11.py

Writes data/boundaryHarmonicStripeEndsConfirmationScoring1888Hold301StripesInteriorMinus60Minus5.json (never overwriting).
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import beta

SUFFIX = '1888Hold301StripesInteriorMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--drawPath', type=str, default=f'data/boundaryHarmonicStripeEndsConfirmation{SUFFIX}.json')
parser.add_argument('--predictionsPath', type=str, default=f'data/boundaryHarmonicStripeEndsPredictions{SUFFIX}.json')
parser.add_argument('--outputPath', type=str, default=f'data/boundaryHarmonicStripeEndsConfirmationScoring{SUFFIX}.json')
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')
draw = json.load(open(args.drawPath))
registration = json.load(open(args.predictionsPath))
labels = np.array(draw['labels'])
isDraw = labels != 'stripeCode'
upperFormed = np.array(draw['upperDarkAtBest']) >= 11
lowerFormed = np.array(draw['lowerDarkAtBest']) >= 11
stripeFormed = np.array(draw['overlapAtBest']) >= 0.9
top, bottom = np.array(draw['topLevel']), np.array(draw['bottomLevel'])


def interval(successes, total):
    low = beta.ppf(0.025, successes, total - successes + 1) if successes > 0 else 0.0
    high = beta.ppf(0.975, successes + 1, total - successes) if successes < total else 1.0
    return [round(float(low), 4), round(float(high), 4)]


def rate(mask, flags):
    count, total = int((flags & mask).sum()), int(mask.sum())
    return dict(count=count, of=total, rate=round(count / total, 4) if total else None, interval95=interval(count, total) if total else None)


inTop = np.isin(labels, ['both', 'topOnly'])
inBottom = np.isin(labels, ['both', 'bottomOnly'])
outTop = np.isin(labels, ['bottomOnly', 'neither'])
outBottom = np.isin(labels, ['topOnly', 'neither'])
verdicts = {}
a, b = rate(inTop, upperFormed), rate(outTop, upperFormed)
verdicts['E1-theTopWritesTheUpperHalf'] = dict(insideWindow=a, outsideWindow=b, holds=bool(a['rate'] >= 0.60 and b['rate'] <= 0.05))
a, b = rate(inBottom, lowerFormed), rate(outBottom, lowerFormed)
verdicts['E2-theBottomWritesTheLowerHalf'] = dict(insideWindow=a, outsideWindow=b, holds=bool(a['rate'] >= 0.60 and b['rate'] <= 0.05))
t, tl = rate(labels == 'topOnly', upperFormed), rate(labels == 'topOnly', lowerFormed)
bo, bu = rate(labels == 'bottomOnly', lowerFormed), rate(labels == 'bottomOnly', upperFormed)
verdicts['E3-theEndsAreWrittenWithoutEachOther'] = dict(topOnlyUpperFormed=t, topOnlyLowerFormed=tl, bottomOnlyLowerFormed=bo, bottomOnlyUpperFormed=bu,
                                                       holds=bool(t['rate'] >= 0.5 and tl['rate'] <= 0.05 and bo['rate'] >= 0.5 and bu['rate'] <= 0.05))
a, b = rate(labels == 'both', stripeFormed), rate(np.isin(labels, ['topOnly', 'bottomOnly', 'neither']), stripeFormed)
verdicts['E4-theStripeNeedsBothEnds'] = dict(bothStratum=a, otherStrata=b, holds=bool(a['rate'] >= 0.40 and b['rate'] <= 0.02))
above = (labels == 'neither') & (top > 1.4975) & (bottom > 1.4975)
c = rate(above, stripeFormed)
verdicts['E5-theWindowIsThin'] = dict(neitherWithBothLevelsAbove1p4975=c, holds=bool(c['of'] > 0 and c['count'] == 0))
# registered validity
perStratum = {name: int((labels == name).sum()) for name in draw['strata']}
outsideOk = lambda v: (v < 1.4825) | (v > 1.4975)
insideOk = lambda v: (v >= 1.485) & (v <= 1.495)
checks = dict(both=insideOk(top) & insideOk(bottom), topOnly=insideOk(top) & outsideOk(bottom), bottomOnly=outsideOk(top) & insideOk(bottom), neither=outsideOk(top) & outsideOk(bottom))
v1 = all(perStratum[name] == draw['perStratum'] and bool(checks[name][labels == name].all()) for name in draw['strata']) and abs(draw['side'] - 1.2127) < 1e-12
stripeCode = int(np.flatnonzero(labels == 'stripeCode')[0])
v2 = dict(overlap=draw['overlapAtBest'][stripeCode], bestMoment=draw['bestIteration'][stripeCode], holds=bool(draw['overlapAtBest'][stripeCode] >= 0.999 and draw['bestIteration'][stripeCode] == 504))
validity = dict(V1=dict(codesPerStratum=perStratum, holds=bool(v1)), V2=v2)
for key, verdict in {**validity, **verdicts}.items():
    print(f"{key}: {'holds' if verdict['holds'] else 'FAILS'}", {k: v for k, v in verdict.items() if k != 'holds'}, flush=True)
# descriptive: the strata's rates for every outcome
table = {name: dict(upperFormed=rate(labels == name, upperFormed), lowerFormed=rate(labels == name, lowerFormed), stripeFormed=rate(labels == name, stripeFormed)) for name in draw['strata']}
json.dump(dict(note='Each criterion as registered; the strata table is descriptive.', registration=args.predictionsPath, validity=validity, verdicts=verdicts, strataRates=table),
          open(args.outputPath, 'w'), separators=(',', ':'))
print('wrote', args.outputPath, flush=True)
