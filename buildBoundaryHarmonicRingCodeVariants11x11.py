"""A curated set of steered and knocked-out ring codes, ready for computeBoundaryHarmonicRelay11x11.py --ringCodePath.

Eight variants: the largest valid step (+-0.30, walking inward only if that clips) on each of the trained order-3
code's four coefficients (recordBoundaryHarmonicRuns11x11.py --mode steering's own step list and clip rule), and
four knockouts already computed by analyzeBoundaryHarmonicKnockout11x11.py -- orders 1, 2, 3 alone and all three
together (order 0, the mean term, is never knocked out there, so it is not offered here either).

Writes data/boundaryHarmonicRingCodeVariants1888Hold301FaceMinus60Minus5.json (never overwriting).

    python3 buildBoundaryHarmonicRingCodeVariants11x11.py
"""
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

trainedPath = 'data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5/order3_restart08.npz'
knockoutPath = 'data/boundaryHarmonicKnockout1888Hold301FaceMinus60Minus5.json'
outputPath = 'data/boundaryHarmonicRingCodeVariants1888Hold301FaceMinus60Minus5.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

trained = np.asarray(np.load(trainedPath)['bestCoefficients'], dtype=float)
angles = boundary.ringAngles(boundary.boundaryRingCells)
basis = np.cos(np.outer(angles, np.arange(len(trained))))
STEPS = [0.30, -0.30, 0.20, -0.20, 0.10, -0.10, 0.05, -0.05]     # largest magnitude first, both signs, per order


def validRingValues(coefficients):
    values = basis @ coefficients
    return values if (values.min() >= 0.02 and values.max() <= 1.98) else None

variants = []
for order in range(len(trained)):
    for step in STEPS:
        candidate = trained.copy()
        candidate[order] += step
        values = validRingValues(candidate)
        if values is not None:
            variants.append(dict(key=f'steerOrder{order}', kind='steering', orders=[order], step=step,
                                  coefficients=[round(float(c), 6) for c in candidate],
                                  ringValues=[round(float(v), 6) for v in values],
                                  label=f'order {order} {"+" if step >= 0 else "−"}{abs(step):.2f}',
                                  description=f"the trained code with order {order}'s coefficient moved by "
                                              f"{'+' if step >= 0 else '-'}{abs(step):.2f}, the largest step that stays in [0.02, 1.98]"))
            break

knockoutData = json.load(open(knockoutPath))
wanted = {(1,): 'knockoutOrder1', (2,): 'knockoutOrder2', (3,): 'knockoutOrder3', (1, 2, 3): 'knockoutOrders123'}
for entry in knockoutData['knockouts']:
    orders = tuple(entry['orders'])
    if orders in wanted:
        variants.append(dict(key=wanted[orders], kind='knockout', orders=list(orders),
                              ringValues=[round(float(v), 6) for v in entry['knockout']['ringValues']],
                              label='order' + ('s ' if len(orders) > 1 else ' ') + ', '.join(str(o) for o in orders) + ' knocked out',
                              description=f"the trained code with order{'s' if len(orders) > 1 else ''} "
                                          f"{', '.join(str(o) for o in orders)} set to 0, the rest left at their trained values"))

assert len(variants) == 8, [v['key'] for v in variants]
json.dump(dict(trainedCoefficients=[round(float(c), 6) for c in trained], variants=variants), open(outputPath, 'w'), indent=1)
print(f'wrote {outputPath}: ' + ', '.join(v['key'] for v in variants))
