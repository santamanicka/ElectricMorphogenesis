"""A larger set of ring codes to learn which edges of the causal net each region of the ring switches on: the four orders'
coefficients scaled by multipliers of the trained values, sampled to cover the space the Relay Loop page's slider and grid
cannot (they move one or two orders at a time on a coarse lattice).

  global   scrambled Sobol points over multipliers in [0, 2]^4, keeping only those whose ring values lie inside [0, 2]
           G_pol / G_ref -- the range the conductance can take -- with no clipping. About 40% of the cube is valid; with the other
           orders at their trained values order 0 can only range over about 0.81-1.88, and in combination it reaches 0.3-1.96
           (its knockout needs clipping, which this set avoids).
  local    a cloud around the trained code: 20 random directions at each of six distances (0.02, 0.05, 0.10, 0.15, 0.25, 0.35)
           from the trained multipliers (1, 1, 1, 1), to test whether the net is smooth in a small neighbourhood.

Every code's ring values are asserted to lie inside [0, 2]. Output is ready for computeBoundaryHarmonicRelay11x11.py
--ringCodeVariantsPath / --ringCodeKey.

Writes data/boundaryHarmonicRingCodeSweep1888Hold301FaceMinus60Minus5.json (never overwriting).

    python3 buildBoundaryHarmonicRingCodeSweep11x11.py
"""
import json
import os

import numpy as np
from scipy.stats import qmc

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301FaceMinus60Minus5'
trainedPath = f'data/boundaryHarmonicTraining{SUFFIX}/order3_restart08.npz'
outputPath = f'data/boundaryHarmonicRingCodeSweep{SUFFIX}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

GLOBAL_CODES, LOCAL_RADII, LOCAL_PER_RADIUS = 280, (0.02, 0.05, 0.10, 0.15, 0.25, 0.35), 20
SOBOL_SEED, LOCAL_SEED = 20251001, 20251002
LOW, HIGH = 0.0, 2.0                                                     # the range of G_pol / G_ref

trained = np.asarray(np.load(trainedPath)['bestCoefficients'], dtype=float)
basis = np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(len(trained))))


def ringValues(multipliers):
    return basis @ (np.asarray(multipliers) * trained)


def inRange(multipliers):
    values = ringValues(multipliers)
    return values.min() >= LOW and values.max() <= HIGH


def variant(key, kind, multipliers, **extra):
    values = ringValues(multipliers)
    assert values.min() >= LOW and values.max() <= HIGH, (key, values.min(), values.max())
    return dict(key=key, kind=kind, multipliers=[round(float(m), 6) for m in multipliers],
                coefficients=[round(float(c), 6) for c in np.asarray(multipliers) * trained],
                ringValues=[round(float(v), 6) for v in values], ringMin=round(float(values.min()), 4), ringMax=round(float(values.max()), 4), **extra)


variants = []
sobol = qmc.Sobol(4, scramble=True, seed=SOBOL_SEED)
while sum(v['kind'] == 'sweepGlobal' for v in variants) < GLOBAL_CODES:
    for point in sobol.random(64) * (HIGH - LOW):
        if inRange(point) and sum(v['kind'] == 'sweepGlobal' for v in variants) < GLOBAL_CODES:
            variants.append(variant(f"sweepGlobal{sum(v['kind'] == 'sweepGlobal' for v in variants):03d}", 'sweepGlobal', point))
generator = np.random.default_rng(LOCAL_SEED)
for radius in LOCAL_RADII:
    for _ in range(LOCAL_PER_RADIUS):
        while True:
            direction = generator.normal(size=4)
            point = 1.0 + radius * direction / np.linalg.norm(direction)
            if inRange(point):
                break
        variants.append(variant(f"sweepLocal{sum(v['kind'] == 'sweepLocal' for v in variants):03d}", 'sweepLocal', point, radius=radius))

json.dump(dict(trainedCoefficients=[round(float(c), 6) for c in trained], low=LOW, high=HIGH, variants=variants), open(outputPath, 'w'), indent=1)
multipliers = np.array([v['multipliers'] for v in variants])
print(f'wrote {outputPath}: {len(variants)} codes ({GLOBAL_CODES} global, {len(variants) - GLOBAL_CODES} local); '
      f'multipliers span {multipliers.min(0).round(2).tolist()} .. {multipliers.max(0).round(2).tolist()}; '
      f'ring values {min(v["ringMin"] for v in variants):.3f} .. {max(v["ringMax"] for v in variants):.3f}')
