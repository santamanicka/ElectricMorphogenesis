"""The tissue state a code's own run leaves --offset iterations after its best moment, for the rejuvenation search (data/canalizationTalkRejuvenationPredictions1888Hold301.json).

The trained code of the target (stripe, or the face of BOUNDARY_FACE_ORDER) is held for the standard 301 iterations and released, in 64-bit with the pure step, and the state (voltage in
volts, absolute polarising conductance) after best moment + offset iterations is stored with the code, so that learnBoundaryHarmonics11x11.py --startState can search new codes
from it. At the best moment itself the pattern is formed; a few hundred iterations later it is fading.

    python3 makeAgedTissueState11x11.py --target stripe
    BOUNDARY_FACE_ORDER=4 python3 makeAgedTissueState11x11.py --target face --offset 300

Writes data/agedTissueState1888Hold301<Stripe|FaceOrder3|FaceOrder4>After<offset>.npz (never overwriting).
"""
import argparse
import os

import numpy as np
import torch

import boundaryCodeUtilities as boundary
import canalizationTalkCommon as common
from canalizationTalkCommon import HOLD, RING_CELLS, TARGETS, ringValuesOf

parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True, choices=('stripe', 'face'))
parser.add_argument('--offset', type=int, default=300, help='iterations after the best moment')
args = parser.parse_args()
label = 'Stripe' if args.target == 'stripe' else f'FaceOrder{boundary.faceOrder}'
outputPath = f'data/agedTissueState1888Hold301{label}After{args.offset}.npz'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
target = TARGETS[args.target]
age = target['readIteration'] + args.offset
with boundary.fullDoublePrecision(True):
    step = common.Replayer().step
    step.ringMask = torch.zeros(step.numCells, dtype=torch.double)
    step.ringMask[torch.as_tensor(RING_CELLS, dtype=torch.long)] = 1.0
    code = torch.zeros(step.numCells, dtype=torch.double)
    code[step.ring] = torch.as_tensor(ringValuesOf(target['coefficients'], target['ceiling']), dtype=torch.double) * step.Gref
    step.ringCode = code
    vmem, gpol = step.initialVmem.clone(), step.initialGpol.clone()
    for iteration in range(age):
        vmem, gpol = step(vmem, gpol, iteration < HOLD, iteration < HOLD)
dark = int((vmem.numpy()[np.array(boundary.interiorCellIndices)] * 1000.0 < common.THRESHOLD).sum())
np.savez_compressed(outputPath, vmemVolts=vmem.numpy(), gpol=gpol.numpy(), age=age, bestMoment=target['readIteration'], offset=args.offset, coefficients=target['coefficients'],
                    orders=np.arange(len(target['coefficients'])), target=args.target, faceOrder=boundary.faceOrder, ceiling=target['ceiling'])
print(f'wrote {outputPath}: {args.target} state after {age} iterations (best moment {target["readIteration"]} + {args.offset}); {dark} dark interior cells; '
      f'mean G_pol / G_ref {float(gpol.numpy().mean() / step.Gref):.2f}')
