"""Ring codes for the trained-to-knockout slider (one order at a time, coefficient(t) = trained*(1-t)) and the
two-order grid (two orders moved at once, the interior points a single-order slider can't reach).

Slider: 5 points per order (t = 0, 0.25, 0.5, 0.75, 1) on the straight line from the trained coefficient to 0.
t=0 is the trained code itself (no new run); t=1 for orders 1-3 is exactly analyzeBoundaryHarmonicKnockout11x11.py's
own knockout (reused, not recomputed here) -- t=1 for order 0 is new, since the knockout script never zeroes a0.

Grid: for each pair of orders (i, j), a 3x3 grid of t in {0, 0.5, 1} on each axis independently, all four other
orders held at their trained value throughout. Row 0 / column 0 (one order still at t=0, trained) is exactly that
OTHER order's own slider point, so only the four interior cells (both orders moved) are genuinely new -- the
corner a single-order view can never show.

Writes data/boundaryHarmonicRingCodeSliderGrid1888Hold301FaceMinus60Minus5.json (never overwriting): every new
ring code's key, its 40 ring values, and enough metadata to place it back on its slider or grid cell.

    python3 buildBoundaryHarmonicRingCodeSliderGrid11x11.py
"""
import itertools
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

trainedPath = 'data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5/order3_restart08.npz'
outputPath = 'data/boundaryHarmonicRingCodeSliderGrid1888Hold301FaceMinus60Minus5.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

trained = np.asarray(np.load(trainedPath)['bestCoefficients'], dtype=float)
angles = boundary.ringAngles(boundary.boundaryRingCells)
basis = np.cos(np.outer(angles, np.arange(len(trained))))
NUM_ORDERS = len(trained)


def ringValuesFor(coefficients):
    return basis @ coefficients


def coefficientsAt(levels):
    """levels: {order: t in [0,1]}, coefficient(t) = trained*(1-t); orders not given stay at their trained value."""
    c = trained.copy()
    for order, t in levels.items():
        c[order] = trained[order] * (1 - t)
    return c

variants = []

# ---------------------------------------------------------------- the slider: one order at a time, 5 points
SLIDER_T = [0.25, 0.5, 0.75, 1.0]                                  # t=0 is the trained code, not written here
for order in range(NUM_ORDERS):
    for t in SLIDER_T:
        if order != 0 and t == 1.0:
            continue                                                # orders 1-3 at t=1: reuse the existing knockout, not recomputed
        key = f'slider_o{order}_t{int(round(t * 100)):03d}'
        values = ringValuesFor(coefficientsAt({order: t}))
        variants.append(dict(key=key, kind='slider', order=order, t=t,
                              coefficient=round(float(trained[order] * (1 - t)), 6),
                              ringValues=[round(float(v), 6) for v in values],
                              label=f'order {order}, t={t:g} toward knockout'))

# ---------------------------------------------------------------- the grid: two orders at once, interior cells only
GRID_T = dict(half=0.5, knock=1.0)
for i, j in itertools.combinations(range(NUM_ORDERS), 2):
    for li, ti in GRID_T.items():
        for lj, tj in GRID_T.items():
            key = f'grid_o{i}o{j}_r{li}_c{lj}'
            values = ringValuesFor(coefficientsAt({i: ti, j: tj}))
            variants.append(dict(key=key, kind='grid', orders=[i, j], levels=[li, lj], t=[ti, tj],
                                  ringValues=[round(float(v), 6) for v in values],
                                  label=f'order {i} {li}, order {j} {lj}'))

json.dump(dict(trainedCoefficients=[round(float(c), 6) for c in trained], variants=variants), open(outputPath, 'w'), indent=1)
print(f'wrote {outputPath}: {len(variants)} new ring codes '
      f'({sum(1 for v in variants if v["kind"]=="slider")} slider, {sum(1 for v in variants if v["kind"]=="grid")} grid)')
for v in variants:
    print(' ', v['key'])
