"""Ring codes at chosen levels at the top, the bottom and the sides, ready for computeBoundaryHarmonicRelay11x11.py --ringCodeVariantsPath.

A code of orders 0 to 2 is fixed by T_top = a0 + a1 + a2, T_bottom = a0 - a1 + a2 and S = a0 - a2 (see simulateBoundaryHarmonicStripeEndsScan11x11.py).
The default four illustrate the ends of the stripe at the sides' level of the stripe code: both levels in the window where the stripe forms,
only the top, only the bottom, neither. Each variant records its key, kind, levels (in `multipliers`, the field the relay record reader expects),
coefficients and ring values.

    python3 buildBoundaryHarmonicStripeEndsVariants11x11.py

Writes data/boundaryHarmonicRingCodeStripeEnds1888Hold301StripesInteriorMinus60Minus5.json (never overwriting).
"""
import argparse
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301StripesInteriorMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--sideLevel', type=float, default=1.2127)
parser.add_argument('--windowLevel', type=float, default=1.4875, help='a level inside the window')
parser.add_argument('--outsideLevel', type=float, default=1.44, help='a level outside it, just above the bistable edge')
parser.add_argument('--outputPath', type=str, default=f'data/boundaryHarmonicRingCodeStripeEnds{SUFFIX}.json')
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')
angles = boundary.ringAngles(boundary.boundaryRingCells)
variants = []
for key, top, bottom in (('endsBoth', args.windowLevel, args.windowLevel), ('endsTopOnly', args.windowLevel, args.outsideLevel),
                         ('endsBottomOnly', args.outsideLevel, args.windowLevel), ('endsNeither', args.outsideLevel, args.outsideLevel)):
    a0, a1, a2 = (top + bottom) / 4 + args.sideLevel / 2, (top - bottom) / 2, (top + bottom) / 4 - args.sideLevel / 2
    ring = a0 + a1 * np.cos(angles) + a2 * np.cos(2 * angles)
    assert ring.min() >= 0 and ring.max() <= 2, (key, ring.min(), ring.max())
    variants.append(dict(key=key, kind='ends', multipliers=[top, bottom, args.sideLevel], levels=dict(top=top, bottom=bottom, sides=args.sideLevel),
                         coefficients=[a0, a1, a2], ringValues=ring.tolist()))
json.dump(dict(note='ring codes at chosen levels at the top, the bottom and the sides', variants=variants), open(args.outputPath, 'w'))
print('wrote', args.outputPath, [(v['key'], v['levels']) for v in variants])
