"""Ring codes for the stripe's two-sided slider, its 5 x 5 grids and its curated single-code list: the stripes' counterpart of
buildBoundaryHarmonicRingCodeFiveLevelGrid11x11.py (which scales the face's trained coefficients by multipliers).

The stripe code has a1 = 0.0012, so a multiple of it is no change at all, and its window is thin (the stripe forms only while the ring's
level at the top and bottom, T = a0 + a2, sits within a few thousandths of 1.49), so multiples of a coefficient would step over everything
that matters. The stops here are therefore absolute offsets from the trained coefficient, closely spaced near it and widely spaced far away:

Slider: for each order (0, 1, 2), the trained coefficient plus an offset in {-0.6, -0.3, -0.15, -0.08, -0.04, -0.02, -0.01, 0, +0.01, +0.02,
        +0.04, +0.08, +0.15, +0.3, +0.6}, plus the order's knockout (coefficient 0) where that is not already a stop; the other two
        orders stay at their trained values. The ring values are clipped to [0, 2] G_pol / G_ref, the range G_pol can take.
Grid:   for each pair of orders, 5 x 5 cells, offsets {-0.3, -0.08, 0, +0.08, +0.3} on each axis independently, the third order at its
        trained value. A row or column at offset 0 is the other order's own slider point.
Curated: the trained code, each order knocked out alone and the two together, the uniform ring that forms the stripe while the ring is
        dark (a0 = 1.5, a2 = -0.05, found by the landscape map), and the four ring codes of the confirmation of the stripe's two ends
        (both ends in the window, only the top, only the bottom, neither), whose ring values are read from that registration's file.

A code is reused whenever another code in the list has the same clipped ring values (matched to 1e-4); only genuinely new codes are listed under
`variants`, ready for computeBoundaryHarmonicRelay11x11.py --ringCodeVariantsPath. The `slider` and `grid` placement maps say, for every
point and cell, which key it is. EXPLORATORY: nothing here was predicted.

Writes data/boundaryHarmonicRingCodeStripeLevels<suffix>.json (never overwriting).

    python3 buildBoundaryHarmonicRingCodeStripeLevels11x11.py
"""
import argparse
import itertools
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301StripesInteriorMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--trainedPath', type=str, default=f'data/boundaryHarmonicTraining{SUFFIX}Ceiling2/order2_restart06.npz', help='the stripe code (the run the gate was met on)')
parser.add_argument('--endsVariantsPath', type=str, default=f'data/boundaryHarmonicRingCodeStripeEnds{SUFFIX}.json')
parser.add_argument('--outputPath', type=str, default=f'data/boundaryHarmonicRingCodeStripeLevels{SUFFIX}.json')
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')

trainedRun = np.load(args.trainedPath)
trained = np.asarray(trainedRun['bestCoefficients'], dtype=float)
lowest, highest = np.asarray(trainedRun['coefficientLowest'], float), np.asarray(trainedRun['coefficientHighest'], float)    # the box the search was allowed
basis = np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(len(trained))))
NUM_ORDERS = len(trained)
SLIDER_OFFSETS = [-0.6, -0.3, -0.15, -0.08, -0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6]
GRID_OFFSETS = [-0.3, -0.08, 0.0, 0.08, 0.3]


def clippedRingValues(coefficients):
    raw = basis @ np.asarray(coefficients, float)
    return np.clip(raw, 0.0, 2.0), int(((raw < 0) | (raw > 2)).sum())


existing = {'trained': clippedRingValues(trained)[0]}
variants = []
tag = lambda offset: ('m' if offset < 0 else 'p' if offset > 0 else 'z') + f'{int(round(abs(offset) * 1000)):04d}'


def keyFor(coefficients, newKey, label, meta):
    """The key under which this code's relay lives: an existing one, or a new run (the same clipped code twice is one run)."""
    values, clipped = clippedRingValues(coefficients)
    for key, other in existing.items():
        if np.abs(values - other).max() < 1e-4:
            return key
    for other in variants:
        if np.abs(np.asarray(other['ringValues']) - values).max() < 1e-4:
            return other['key']
    variants.append(dict(key=newKey, label=label, ringValues=[round(float(v), 6) for v in values], cellsClipped=clipped,
                         coefficients=[round(float(c), 6) for c in coefficients], **meta))
    return newKey


def moved(changes):
    coefficients = trained.copy()
    for order, value in changes.items():
        coefficients[order] = value
    return coefficients


slider, sliderStops = {}, {}
for order in range(NUM_ORDERS):
    stops = {round(trained[order] + offset, 6): offset for offset in SLIDER_OFFSETS}
    if abs(trained[order]) >= 0.02:                                                      # the knockout, unless the trained coefficient is already ~0
        stops = {c: o for c, o in stops.items() if abs(c) >= 0.02 or abs(o) < 1e-9}      # and a ladder stop that close to it would only repeat it
        stops[0.0] = 0.0 - trained[order]
    stops = {c: o for c, o in stops.items() if lowest[order] - 1e-9 <= c <= highest[order] + 1e-9}
    slider[str(order)] = []
    for coefficient in sorted(stops):
        offset = stops[coefficient]
        isCentre = abs(offset) < 1e-9
        knockout = abs(coefficient) < 1e-9 and not isCentre
        key = 'trained' if isCentre else keyFor(moved({order: coefficient}), f'stripe_o{order}_{"ko" if knockout else tag(offset)}',
                                                 f'order {order} knocked out (coefficient 0)' if knockout else f'order {order} at {coefficient:+.4f}',
                                                 dict(kind='slider', order=order, offset=round(float(offset), 6)))
        slider[str(order)].append(dict(offset=round(float(offset), 6), coefficient=round(float(coefficient), 6), key=key, knockout=bool(knockout)))
    sliderStops[order] = {round(p['offset'], 6): p['key'] for p in slider[str(order)]}

grid = {}
for i, j in itertools.combinations(range(NUM_ORDERS), 2):
    rows = []
    for oi in GRID_OFFSETS:
        row = []
        for oj in GRID_OFFSETS:
            if oi == 0.0 and oj == 0.0:
                key = 'trained'
            elif oj == 0.0:
                key = sliderStops[i][oi]
            elif oi == 0.0:
                key = sliderStops[j][oj]
            else:
                key = keyFor(moved({i: trained[i] + oi, j: trained[j] + oj}), f'stripe_o{i}o{j}_r{tag(oi)}_c{tag(oj)}',
                             f'order {i} {oi:+g}, order {j} {oj:+g}', dict(kind='grid', orders=[i, j], offsets=[oi, oj]))
            row.append(key)
        rows.append(row)
    grid[f'{i}_{j}'] = rows

# curated single codes, the trained one first
curated = [dict(key='trained', label='Trained stripe code (a0 1.350, a1 0.001, a2 0.137)', kind='trained', coefficients=[round(float(c), 6) for c in trained])]
curatedSpecs = [('knockoutOrder0', 'Order 0 knocked out (a0 = 0)', {0: 0.0}), ('knockoutOrder1', 'Order 1 knocked out (a1 = 0; the trained a1 is 0.001)', {1: 0.0}),
                ('knockoutOrder2', 'Order 2 knocked out (a2 = 0): a uniform ring at 1.350', {2: 0.0}),
                ('knockoutOrders12', 'Orders 1 and 2 knocked out: the same uniform ring', {1: 0.0, 2: 0.0}),
                ('uniformRing1p5', 'Uniform ring at 1.5 (a0 1.5, a2 -0.05): the stripe forms only while the ring is dark', {0: 1.5, 1: 0.0, 2: -0.05})]
for key, label, changes in curatedSpecs:
    coefficients = moved(changes)
    found = keyFor(coefficients, key, label, dict(kind='curated'))
    curated.append(dict(key=found, label=label, kind='curated', coefficients=[round(float(c), 6) for c in coefficients]))
for v in json.load(open(args.endsVariantsPath))['variants']:        # the ends confirmation's own four codes: their ring values, as registered
    values = np.asarray(v['ringValues'], float)
    found = None
    for key, other in existing.items():
        if np.abs(values - other).max() < 1e-4:
            found = key
    for other in variants:
        if np.abs(np.asarray(other['ringValues']) - values).max() < 1e-4:
            found = other['key']
    if not found:
        variants.append(dict(key=v['key'], label=v.get('label', v['key']), ringValues=[round(float(x), 6) for x in values], cellsClipped=0,
                             coefficients=v.get('coefficients'), kind='ends', levels=v.get('levels')))
        found = v['key']
    curated.append(dict(key=found, label='Ends: ' + str(v.get('label', v['key'])), kind='ends', levels=v.get('levels')))

# the trained code is run too (as the first task), so every record comes from one code path and carries the same blocks
variants.insert(0, dict(key='trained', label='Trained stripe code', ringValues=[round(float(v), 6) for v in existing['trained']], cellsClipped=0,
                        coefficients=[round(float(c), 6) for c in trained], kind='trained'))
json.dump(dict(note='EXPLORATORY; no predictions registered.', trainedCoefficients=[round(float(c), 6) for c in trained], coefficientLowest=lowest.tolist(),
               coefficientHighest=highest.tolist(), sliderOffsets=SLIDER_OFFSETS, gridOffsets=GRID_OFFSETS, slider=slider, grid=grid, curated=curated, variants=variants),
          open(args.outputPath, 'w'), indent=1)
print(f'wrote {args.outputPath}: {len(variants)} codes to run (the trained code first) '
      f'({sum(v["kind"] == "slider" for v in variants)} slider, {sum(v["kind"] == "grid" for v in variants)} grid, '
      f'{sum(v["kind"] in ("curated", "ends") for v in variants)} curated); {sum(v["cellsClipped"] > 0 for v in variants)} of them have ring cells clipped')
