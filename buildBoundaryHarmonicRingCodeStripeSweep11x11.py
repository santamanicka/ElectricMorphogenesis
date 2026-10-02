"""A larger set of ring codes for the stripe's steering lab: the stripes' counterpart of buildBoundaryHarmonicRingCodeSweep11x11.py. EXPLORATORY.

The Relay Loop page for the stripe estimates what the network does at a setting of the three orders (a0, a1, a2) by averaging the causal nets of the
nearest simulated ring codes, so the simulated codes must cover the space: widely, and finely around the trained code, because the stripe forms only
in a thin window (the ring's level at the top and bottom, T = a0 + a2, within a few thousandths of 1.49).

  global   scrambled Sobol points over the coefficient box (a0 in [0, 2], a1 and a2 in [-1, 1]), keeping only those whose ring values lie inside
           [0, 2] G_pol / G_ref (the range the conductance can take) with no clipping.
  local    a cloud around the trained code: 20 random directions at each of six distances (0.005, 0.01, 0.02, 0.05, 0.10, 0.20) in coefficient space,
           to test whether the net is smooth in a small neighbourhood.
  window   codes spread over the plane the stripe lives in: T uniform in [1.40, 1.58], S = a0 - a2 uniform in [0.9, 1.5], a1 small (normal, sd 0.02).

Every code's ring values are asserted to lie inside [0, 2]. `multipliers` holds the three coefficients, so the Relay Loop record extractor (which
reads that key) works unchanged. Output is ready for computeBoundaryHarmonicRelay11x11.py --ringCodeVariantsPath / --ringCodeKey.

Writes data/boundaryHarmonicRingCodeStripeSweep<suffix>.json (never overwriting).

    python3 buildBoundaryHarmonicRingCodeStripeSweep11x11.py
"""
import json
import os

import numpy as np
from scipy.stats import qmc

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301StripesInteriorMinus60Minus5'
trainedPath = f'data/boundaryHarmonicTraining{SUFFIX}Ceiling2/order2_restart06.npz'
outputPath = f'data/boundaryHarmonicRingCodeStripeSweep{SUFFIX}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

GLOBAL_CODES, WINDOW_CODES, LOCAL_RADII, LOCAL_PER_RADIUS = 160, 120, (0.005, 0.01, 0.02, 0.05, 0.10, 0.20), 20
SOBOL_SEED, LOCAL_SEED, WINDOW_SEED = 20261003, 20261004, 20261005
LOW, HIGH = 0.0, 2.0                                                     # the range of G_pol / G_ref

run = np.load(trainedPath)
trained = np.asarray(run['bestCoefficients'], dtype=float)
boxLow, boxHigh = np.asarray(run['coefficientLowest'], float), np.asarray(run['coefficientHighest'], float)
basis = np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(len(trained))))


def inRange(coefficients):
    values = basis @ np.asarray(coefficients)
    return values.min() >= LOW and values.max() <= HIGH and np.all(coefficients >= boxLow) and np.all(coefficients <= boxHigh)


def variant(key, kind, coefficients, **extra):
    values = basis @ np.asarray(coefficients)
    assert values.min() >= LOW and values.max() <= HIGH, (key, values.min(), values.max())
    return dict(key=key, kind=kind, multipliers=[round(float(c), 6) for c in coefficients], coefficients=[round(float(c), 6) for c in coefficients],
                ringValues=[round(float(v), 6) for v in values], ringMin=round(float(values.min()), 4), ringMax=round(float(values.max()), 4), **extra)


count = lambda variants, kind: sum(v['kind'] == kind for v in variants)
variants = []
sobol = qmc.Sobol(len(trained), scramble=True, seed=SOBOL_SEED)
while count(variants, 'sweepGlobal') < GLOBAL_CODES:
    for point in boxLow + sobol.random(64) * (boxHigh - boxLow):
        if inRange(point) and count(variants, 'sweepGlobal') < GLOBAL_CODES:
            variants.append(variant(f"sweepGlobal{count(variants, 'sweepGlobal'):03d}", 'sweepGlobal', point))
generator = np.random.default_rng(LOCAL_SEED)
for radius in LOCAL_RADII:
    for _ in range(LOCAL_PER_RADIUS):
        while True:
            direction = generator.normal(size=len(trained))
            point = trained + radius * direction / np.linalg.norm(direction)
            if inRange(point):
                break
        variants.append(variant(f"sweepLocal{count(variants, 'sweepLocal'):03d}", 'sweepLocal', point, radius=radius))
generator = np.random.default_rng(WINDOW_SEED)
while count(variants, 'sweepWindow') < WINDOW_CODES:
    top, side, tilt = generator.uniform(1.40, 1.58), generator.uniform(0.9, 1.5), generator.normal(0, 0.02)
    point = np.array([(top + side) / 2, tilt, (top - side) / 2])
    if inRange(point):
        variants.append(variant(f"sweepWindow{count(variants, 'sweepWindow'):03d}", 'sweepWindow', point))

json.dump(dict(trainedCoefficients=[round(float(c), 6) for c in trained], low=LOW, high=HIGH, variants=variants), open(outputPath, 'w'), indent=1)
coefficients = np.array([v['coefficients'] for v in variants])
print(f'wrote {outputPath}: {len(variants)} codes ({GLOBAL_CODES} global, {len(LOCAL_RADII) * LOCAL_PER_RADIUS} local, {WINDOW_CODES} window); '
      f'coefficients span {coefficients.min(0).round(2).tolist()} .. {coefficients.max(0).round(2).tolist()}; '
      f'ring values {min(v["ringMin"] for v in variants):.3f} .. {max(v["ringMax"] for v in variants):.3f}')
