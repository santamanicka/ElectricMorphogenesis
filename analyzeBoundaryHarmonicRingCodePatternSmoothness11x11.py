"""Is the stripe's pattern smoother under a slid ring-code order than the face's? EXPLORATORY: nothing here was predicted or registered.

Reads two files written by simulateBoundaryHarmonicRingCodePatterns11x11.py, the stripe code's and the face code's (--target face), each a set of
dense one-order-at-a-time sliders (every other order at its trained value) and, for the stripe, two maps. For every slider, in its two ranges (the
whole allowed range in --wideStops equal steps; the trained value plus or minus a small amount in steps of 0.001), it asks how the interior's dark
cells change from one stop to the next, at the code's readout (the stripe code's best moment, 504; the face code's, 2173) and at each code's own best
moment:

  changedShare    the share of steps at which the set of dark interior cells (Vmem below -34.6 mV) changes at all
  meanChange      the mean number of cells that change (a cell going dark or light counts one)
  maxChange       the most that change in one step
  bigJumpShare    the share of steps at which 4 or more cells change
  distinctSets    how many different dark sets the slider passes through
  overlapStep     the mean and the largest change in overlap with the target between adjacent stops
  formedShare     the share of stops at which the target is formed (overlap 0.9 or more)
  trend           the Spearman correlation of the number of dark cells with the coefficient (1: more dark cells as the order goes up)

The two targets have different numbers of feature cells (the stripe's 27, the face's 14) and the orders have different allowed ranges (the stripe's
ceiling is 2.0, the face's 1.3), so the whole-range sliders have different step sizes; the zoomed sliders have the same absolute step (0.001) and are the
like-for-like comparison. A step is one stop to the next, so a metric of a smooth process is small and does not depend on where it is looked at.

Writes data/boundaryHarmonicRingCodePatternSmoothness<suffix>.json (never overwriting).

    python3 analyzeBoundaryHarmonicRingCodePatternSmoothness11x11.py
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import spearmanr

THRESHOLD = -34.6
INTERIOR = np.array([r * 11 + c for r in range(1, 10) for c in range(1, 10)])
parser = argparse.ArgumentParser()
parser.add_argument('--stripePath', type=str, default='data/boundaryHarmonicRingCodePatterns1888Hold301StripesInteriorMinus60Minus5.json')
parser.add_argument('--facePath', type=str, default='data/boundaryHarmonicRingCodePatterns1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--outputPath', type=str, default='data/boundaryHarmonicRingCodePatternSmoothness1888Hold301StripesAndFace.json')
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')


def darkSets(vmem):
    return np.array(vmem)[:, INTERIOR] < THRESHOLD                                # stops x interior cells


def describe(family, order, vmemKey, overlapKey):
    index = [i for i, o in enumerate(family['order']) if o == order]
    coefficient = np.array([family['coefficients'][i][order] for i in index])
    dark = darkSets(np.array(family[vmemKey])[index])
    overlap = np.array(family[overlapKey])[index]
    changed = (dark[1:] != dark[:-1]).sum(1)
    steps = len(changed)
    trend = spearmanr(coefficient, dark.sum(1)).statistic if dark.sum(1).std() > 0 else 0.0
    return dict(stops=len(index), first=float(coefficient[0]), last=float(coefficient[-1]), step=float(np.median(np.diff(coefficient))),
                changedShare=float((changed > 0).mean()), meanChange=float(changed.mean()), maxChange=int(changed.max()), bigJumpShare=float((changed >= 4).mean()),
                distinctSets=int(len({row.tobytes() for row in dark})), meanOverlapStep=float(np.abs(np.diff(overlap)).mean()), maxOverlapStep=float(np.abs(np.diff(overlap)).max()),
                formedShare=float((overlap >= 0.9).mean()), trend=float(trend), darkMin=int(dark.sum(1).min()), darkMax=int(dark.sum(1).max()))


result = dict(note='EXPLORATORY; no predictions registered.', threshold=THRESHOLD)
for name, path in (('stripe', args.stripePath), ('face', args.facePath)):
    data = json.load(open(path))
    result[name] = dict(trainedCoefficients=data['trainedCoefficients'], readIteration=data['readIteration'], targetCells=len(data['featureCells']) if 'featureCells' in data else 27, families={})        # (the stripe file was written before the key existed)
    for familyName in ('wideSlider', 'zoomSlider'):
        family = data['families'][familyName]
        result[name]['families'][familyName] = {
            str(order): dict(atRead=describe(family, order, 'vmemAtRead', 'overlapAtRead'), atBest=describe(family, order, 'vmemAtBest', 'overlapAtBest'))
            for order in range(len(data['trainedCoefficients']))}
json.dump(result, open(args.outputPath, 'w'), indent=1)
print(f'wrote {args.outputPath}\n')
header = f"{'':8s} {'range':6s} {'order':5s} {'step':>7s} {'changed':>8s} {'mean':>6s} {'max':>4s} {'>=4':>6s} {'sets':>5s} {'|dOv| mean':>10s} {'max':>5s} {'formed':>7s} {'trend':>6s}"
for view, label in (('atRead', 'at the readout'), ('atBest', 'at each code\'s best moment')):
    print(f'--- dark cells {label} ---\n' + header)
    for name in ('stripe', 'face'):
        for familyName, short in (('zoomSlider', 'zoom'), ('wideSlider', 'wide')):
            for order, entry in result[name]['families'][familyName].items():
                m = entry[view]
                print(f"{name:8s} {short:6s} {order:5s} {m['step']:7.4f} {m['changedShare']:8.2f} {m['meanChange']:6.2f} {m['maxChange']:4d} {m['bigJumpShare']:6.2f} {m['distinctSets']:5d} "
                      f"{m['meanOverlapStep']:10.3f} {m['maxOverlapStep']:5.2f} {m['formedShare']:7.2f} {m['trend']:6.2f}")
    print()
