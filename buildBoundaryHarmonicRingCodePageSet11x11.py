"""The five-order 'page set' of ring codes: the trained code, one-order sliders and two-order grids, as registered in Amendment 6 of
data/boundaryHarmonicOrder4ReplicationPredictions1888Hold301FaceMinus60Minus5.json (committed before this script existed).

The order-3 Relay Loop page showed 126 codes of these kinds (buildBoundaryHarmonicRingCodeFiveLevelGrid11x11.py); the order-4 set is
fixed to the same kinds at five orders:
  trained  the trained code
  slider   one order at a time, multiplier in {0, 0.25, 0.5, 0.75, 1.5, 2} on that coefficient, the other orders at their trained values
  grid     every pair of orders, multiplier in {0, 0.5, 1.5} on each axis independently, the other orders at their trained values
Ring values are clipped to [0, 2] G_pol / G_ref (a clipped code is kept and flagged `clipped`). A code is dropped when its clipped ring
values match an earlier page-set code or a sweep code to 1e-4; the number dropped is printed and stored.

Writes data/boundaryHarmonicRingCodePageSet1888Hold301FaceMinus60Minus5Order4.json (never overwriting), laid out like the sweep file so
computeBoundaryHarmonicRelay11x11.py --ringCodeVariantsPath / --ringCodeKey and extractRelayLoopSweepRecord11x11.py read it.

    BOUNDARY_FACE_ORDER=4 python3 buildBoundaryHarmonicRingCodePageSet11x11.py
"""
import itertools
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301FaceMinus60Minus5'
sweepPath = boundary.orderPath(f'data/boundaryHarmonicRingCodeSweep{SUFFIX}.json')
outputPath = boundary.orderPath(f'data/boundaryHarmonicRingCodePageSet{SUFFIX}.json')
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

SLIDER_MULTIPLIERS = [0.0, 0.25, 0.5, 0.75, 1.5, 2.0]
GRID_MULTIPLIERS = [0.0, 0.5, 1.5]
MATCH_TOLERANCE = 1e-4
trained = np.asarray(np.load(boundary.defaultFaceRunPath())['bestCoefficients'], dtype=float)
numOrders = len(trained)
basis = np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), np.arange(numOrders)))
sweep = json.load(open(sweepPath))
assert np.allclose(sweep['trainedCoefficients'], trained, atol=1e-6)
known = [np.array(v['ringValues']) for v in sweep['variants']]


def variant(key, kind, multipliers, **extra):
    raw = basis @ (np.asarray(multipliers) * trained)
    values = np.clip(raw, 0.0, 2.0)
    return dict(key=key, kind=kind, multipliers=[round(float(m), 6) for m in multipliers], coefficients=[round(float(m * c), 6) for m, c in zip(multipliers, trained)],
                ringValues=[round(float(v), 6) for v in values], ringMin=round(float(values.min()), 4), ringMax=round(float(values.max()), 4),
                clipped=bool(not np.array_equal(raw, values)), **extra)


candidates = [variant('pageTrained', 'pageTrained', np.ones(numOrders))]
for order in range(numOrders):
    for multiplier in SLIDER_MULTIPLIERS:
        at = np.ones(numOrders)
        at[order] = multiplier
        candidates.append(variant(f'pageSlider_o{order}_m{int(round(multiplier * 100)):03d}', 'pageSlider', at, order=order))
for i, j in itertools.combinations(range(numOrders), 2):
    for mi in GRID_MULTIPLIERS:
        for mj in GRID_MULTIPLIERS:
            at = np.ones(numOrders)
            at[i], at[j] = mi, mj
            candidates.append(variant(f'pageGrid_o{i}o{j}_m{int(round(mi * 100)):03d}_m{int(round(mj * 100)):03d}', 'pageGrid', at, orders=[i, j]))

variants, dropped = [], []
for candidate in candidates:
    values = np.array(candidate['ringValues'])
    if any(np.abs(values - other).max() <= MATCH_TOLERANCE for other in known):
        dropped.append(candidate['key'])
        continue
    known.append(values)
    variants.append(candidate)

json.dump(dict(registeredIn=f'boundaryHarmonicOrder4ReplicationPredictions{SUFFIX}.json (Amendment 6)', trainedCoefficients=[round(float(c), 6) for c in trained],
               sliderMultipliers=SLIDER_MULTIPLIERS, gridMultipliers=GRID_MULTIPLIERS, candidates=len(candidates), dropped=dropped, variants=variants),
          open(outputPath, 'w'), indent=1)
kinds = {kind: sum(v['kind'] == kind for v in variants) for kind in ('pageTrained', 'pageSlider', 'pageGrid')}
print(f'wrote {outputPath}: {len(variants)} codes of {len(candidates)} candidates ({kinds}); dropped {len(dropped)} duplicates: {dropped}; '
      f'{sum(v["clipped"] for v in variants)} clipped')
