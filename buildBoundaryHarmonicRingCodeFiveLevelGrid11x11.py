"""Ring codes for the two-sided slider and the 5 x 5 grid: each order's coefficient moved from knocked out (0x the
trained value) through lower intermediate (0.5x), trained (1x) and higher intermediate (1.5x) to max (2x), with the
resulting ring values clipped to [0, 2] G_pol / G_ref -- the range G_pol can actually take (cellularFieldNetwork.py) --
so a held value is never a negative conductance and never past the ceiling. The earlier slider and grid runs
(buildBoundaryHarmonicRingCodeSliderGrid11x11.py) went only downward and did not clip; those whose ring values already
lay inside [0, 2] are identical here and are reused, and only the rest (the order-0 runs, which went negative) are run
again under new keys.

Slider: seven points per order, multiplier in {0, 0.25, 0.5, 0.75, 1, 1.5, 2}; 1 is the trained code.
Grid:   for each pair of orders (i, j), 5 x 5 cells, multiplier in {0, 0.5, 1, 1.5, 2} on each axis independently, the
        other two orders at their trained value. A row or column at 1x is that other order's own slider point.

A cell is reused whenever a code with the same clipped ring values already has a relay (matched to 1e-4); only genuinely
new codes are listed under `variants`, ready for computeBoundaryHarmonicRelay11x11.py --ringCodeVariantsPath. The
`slider` and `grid` placement maps say, for every point and cell, which key (new or reused) it is.

Writes data/boundaryHarmonicRingCodeFiveLevel1888Hold301FaceMinus60Minus5.json (never overwriting).

    python3 buildBoundaryHarmonicRingCodeFiveLevelGrid11x11.py
"""
import itertools
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301FaceMinus60Minus5'
trainedPath = 'data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5/order3_restart08.npz'
outputPath = f'data/boundaryHarmonicRingCodeFiveLevel{SUFFIX}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

trained = np.asarray(np.load(trainedPath)['bestCoefficients'], dtype=float)
basis = np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(len(trained))))
NUM_ORDERS = len(trained)
SLIDER_MULTIPLIERS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
GRID_MULTIPLIERS = [0.0, 0.5, 1.0, 1.5, 2.0]
GRID_NAMES = ['knockout', 'lower inter', 'trained', 'higher inter', 'max']


def clippedRingValues(multipliers):
    """multipliers: {order: multiple of its trained coefficient}; the rest stay trained."""
    coefficients = trained.copy()
    for order, m in multipliers.items():
        coefficients[order] = trained[order] * m
    raw = basis @ coefficients
    return np.clip(raw, 0.0, 2.0), int(((raw < 0) | (raw > 2)).sum())


# every code that already has a relay, by its ring values
existing = {'trained': clippedRingValues({})[0]}
for path in (f'data/boundaryHarmonicRingCodeVariants{SUFFIX}.json', f'data/boundaryHarmonicRingCodeSliderGrid{SUFFIX}.json'):
    for v in json.load(open(path))['variants']:
        values = np.asarray(v['ringValues'], float)
        if values.min() >= 0.0 and values.max() <= 2.0:            # an out-of-range code is not the clipped one: run it again
            existing[v['key']] = values


def findExisting(values):
    for key, other in existing.items():
        if np.abs(values - other).max() < 1e-4:
            return key
    return None


variants, keyOfCode = [], {}


def keyFor(multipliers, newKey, label, meta):
    """The key under which this code's relay lives: an existing one, or a new run (the same clipped code twice is one run)."""
    values, clipped = clippedRingValues(multipliers)
    found = findExisting(values)
    if found:
        return found
    for other in variants:
        if np.abs(np.asarray(other['ringValues']) - values).max() < 1e-4:
            return other['key']
    variants.append(dict(key=newKey, label=label, ringValues=[round(float(v), 6) for v in values], cellsClipped=clipped, **meta))
    return newKey


tag = lambda m: f'{int(round(m * 100)):03d}'
slider, grid = {}, {}
for order in range(NUM_ORDERS):
    slider[str(order)] = []
    for m in SLIDER_MULTIPLIERS:
        key = 'trained' if m == 1.0 else keyFor({order: m}, f'five_o{order}_m{tag(m)}', f'order {order} at {m:g}x',
                                              dict(kind='slider', order=order, multiplier=m))
        slider[str(order)].append(dict(multiplier=m, coefficient=round(float(trained[order] * m), 6), key=key))
for i, j in itertools.combinations(range(NUM_ORDERS), 2):
    rows = []
    for mi in GRID_MULTIPLIERS:
        row = []
        for mj in GRID_MULTIPLIERS:
            if mi == 1.0 and mj == 1.0:
                key = 'trained'
            elif mj == 1.0:
                key = next(p['key'] for p in slider[str(i)] if p['multiplier'] == mi)
            elif mi == 1.0:
                key = next(p['key'] for p in slider[str(j)] if p['multiplier'] == mj)
            else:
                key = keyFor({i: mi, j: mj}, f'five_o{i}o{j}_r{tag(mi)}_c{tag(mj)}', f'order {i} at {mi:g}x, order {j} at {mj:g}x',
                             dict(kind='grid', orders=[i, j], multipliers=[mi, mj]))
            row.append(key)
        rows.append(row)
    grid[f'{i}_{j}'] = rows

json.dump(dict(trainedCoefficients=[round(float(c), 6) for c in trained], sliderMultipliers=SLIDER_MULTIPLIERS,
               gridMultipliers=GRID_MULTIPLIERS, gridNames=GRID_NAMES, slider=slider, grid=grid, variants=variants),
          open(outputPath, 'w'), indent=1)
print(f'wrote {outputPath}: {len(variants)} codes to run '
      f'({sum(v["kind"] == "slider" for v in variants)} slider, {sum(v["kind"] == "grid" for v in variants)} grid); '
      f'{sum(v["cellsClipped"] > 0 for v in variants)} of them have ring cells clipped')
