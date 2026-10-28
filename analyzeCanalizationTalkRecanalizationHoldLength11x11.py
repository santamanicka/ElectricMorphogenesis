"""Does a longer reapplied hold re-guide the aged tissue? EXPLORATORY, added after the registered re-canalization runs failed to bring the stripe or the face back with a hold of 301 iterations
(data/canalizationTalkRecanalizationScores1888Hold301.json); not part of the registration.

The first hold starts from the uniform state, every polarising conductance at G_ref. An aged tissue starts with the conductances spread over 0 to 1.7, and a hold of 301 iterations does not
level them (analyzeCanalizationTalkRecanalization11x11.py's scores and the diagnostic in the commit message). Here the trained code of each class is reapplied at the primary ages for
--holds iterations instead of 301, released, and run for another --after iterations, in 64-bit. Reported: the peak exact IoU with the target at any time from the start of the reapplied
hold (the stripe is written inside the hold) and after its release, the first time it reaches 0.9, the stripe-family and face-family frame counts after the release, and the mean G_pol / G_ref
of the interior at the end of the hold.

    python3 analyzeCanalizationTalkRecanalizationHoldLength11x11.py
    BOUNDARY_FACE_ORDER=4 python3 analyzeCanalizationTalkRecanalizationHoldLength11x11.py     # the order-4 face only

Writes data/canalizationTalkRecanalizationHoldLength1888Hold301.json (Order4-tagged for order 4; never overwriting).
"""
import argparse
import json
import os

import numpy as np
import torch

import boundaryCodeUtilities as boundary
import canalizationTalkCommon as common
import canalizationTalkFamilies as fam
from canalizationTalkCommon import INTERIOR, RING_CELLS, TARGETS, THRESHOLD, HOLD, ringValuesOf, tagged

parser = argparse.ArgumentParser()
parser.add_argument('--ages', type=str, default='3000,5000')
parser.add_argument('--holds', type=str, default='301,600,1000,2000,4000')
parser.add_argument('--after', type=int, default=3300)
parser.add_argument('--outputPath', type=str, default=tagged('data/canalizationTalkRecanalizationHoldLength1888Hold301.json'))
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')
ages, holds = [int(a) for a in args.ages.split(',')], [int(h) for h in args.holds.split(',')]
interior = np.array(INTERIOR)
classes = ('stripe', 'face') if boundary.faceOrder == 3 else ('face',)
results = {}
with boundary.fullDoublePrecision(True):
    step = common.Replayer().step
    for key in classes:
        step.ringMask = torch.zeros(step.numCells, dtype=torch.double)
        step.ringMask[torch.as_tensor(RING_CELLS, dtype=torch.long)] = 1.0
        code = torch.zeros(step.numCells, dtype=torch.double)
        code[step.ring] = torch.as_tensor(ringValuesOf(TARGETS[key]['coefficients'], 2.0), dtype=torch.double) * step.Gref
        step.ringCode = code
        vmem, gpol = step.initialVmem.clone(), step.initialGpol.clone()
        snapshots = {}
        for iteration in range(max(ages)):
            if iteration in ages:
                snapshots[iteration] = (vmem.clone(), gpol.clone())
            vmem, gpol = step(vmem, gpol, iteration < HOLD, iteration < HOLD)
        snapshots[max(ages)] = (vmem, gpol)
        for age in ages:
            for hold in holds:
                v, g = (snapshots[age][0].clone(), snapshots[age][1].clone())
                bits = []
                for k in range(hold + args.after):
                    v, g = step(v, g, k < hold, k < hold)
                    if k % 5 == 0:
                        bits.append(v.numpy()[interior] * 1000.0 < THRESHOLD)
                    if k == hold - 1:
                        meanG = float(g.numpy()[interior].mean() / step.Gref)
                bits = np.array(bits)
                bars, faceBest, exact = fam.frameFlags(bits)
                tau = np.arange(len(bits)) * 5
                reach = np.where(exact[key] >= 0.9)[0]
                after = tau >= hold + 1
                results[f'{key}|age {age}|hold {hold}'] = dict(
                    peakExact=float(exact[key].max()), peakExactAfterRelease=float(exact[key][after].max()), firstReaches0p9=int(tau[reach[0]]) if len(reach) else None,
                    stripeFamilyFramesAfterRelease=int((bars[after] > 0).sum()), faceFamilyFramesAfterRelease=int((faceBest[after] >= 0.8).sum()), meanGAtEndOfHold=meanG)
                print(f'{key:7} age {age:5} hold {hold:5}: peak exact {results[f"{key}|age {age}|hold {hold}"]["peakExact"]:.2f}  after release {results[f"{key}|age {age}|hold {hold}"]["peakExactAfterRelease"]:.2f}  '
                      f'first >= 0.9 at {results[f"{key}|age {age}|hold {hold}"]["firstReaches0p9"]}  mean G at end of hold {meanG:.2f}', flush=True)
json.dump(dict(note='EXPLORATORY; not registered. See the module docstring.', faceOrder=boundary.faceOrder, ages=ages, holds=holds, afterRelease=args.after, results=results), open(args.outputPath, 'w'), indent=1)
print('wrote', args.outputPath)
