"""Re-canalization: reapply a code to a tissue that has already run free. EXPLORATORY; protocol, measures and predictions are in
data/canalizationTalkRecanalizationPredictions1888Hold301.json (committed before this script existed).

For each code (the trained code of each class and its 24 symmetry-matched controls, the very codes of analyzeCanalizationTalkFamilyVisits11x11.py), in 64-bit:
  original  the standard run (code held for iterations 0-300, then released) to max(ages) + --branchIterations
  branches  at each age T the state after T iterations is copied and the code is held again for iterations T..T+300, then released, for --branchIterations iterations
            (the existing step with ringHeld and extraUpdate switched on for the hold; the model is not changed)
Stored: the interior dark set (below -34.6 mV) every --stride iterations of the original run and of every branch, packed. The unreapplied continuation at age T is the original run's
frames from T on, so it is not run again. BOUNDARY_FACE_ORDER=3 simulates the stripe and the face, BOUNDARY_FACE_ORDER=4 only the face (as the family analysis does).

    python3 simulateCanalizationTalkRecanalization11x11.py
    BOUNDARY_FACE_ORDER=4 python3 simulateCanalizationTalkRecanalization11x11.py

Writes data/canalizationTalkRecanalizationTrajectories1888Hold301.npz (Order4-tagged for order 4; never overwriting); analyzeCanalizationTalkRecanalization11x11.py reads it.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import boundaryCodeUtilities as boundary
import canalizationTalkCommon as common
from canalizationTalkCommon import INTERIOR, RING_CELLS, THRESHOLD, TARGETS, HOLD, ringValuesOf, tagged

parser = argparse.ArgumentParser()
parser.add_argument('--ages', type=str, default='1000,3000,5000,10000,20000')
parser.add_argument('--branchIterations', type=int, default=3300)
parser.add_argument('--stride', type=int, default=5)
parser.add_argument('--controls', type=int, default=24)
parser.add_argument('--seed', type=int, default=20261004)
parser.add_argument('--outputPath', type=str, default=tagged('data/canalizationTalkRecanalizationTrajectories1888Hold301.npz'))
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')
ages = [int(a) for a in args.ages.split(',')]
assert all(a % args.stride == 0 for a in ages)
interior = np.array(INTERIOR)
started = time.time()


def say(*message):
    print(f'[{time.time() - started:5.0f}s]', *message, flush=True)


def buildCodes():
    """The trained codes and the symmetry-matched controls, drawn exactly as buildCodes() of analyzeCanalizationTalkFamilyVisits11x11.py (same seed, same draws;
    for another face order the stripe controls' draws are skipped there, so they are skipped here)."""
    rng = np.random.default_rng(args.seed)
    codes = []
    classes = ('stripe', 'face') if boundary.faceOrder == 3 else ('face',)
    for key in classes:
        codes.append(dict(name=f'{key}Trained', group=f'{key}Trained', coefficients=TARGETS[key]['coefficients'].copy(), held=True))
    stripeSweep = json.load(open('data/relayLoopSweepNets1888Hold301StripesInteriorMinus60Minus5.json'))['codes']
    faceSweep = (json.load(open('data/relayLoopSweepNets1888Hold301FaceMinus60Minus5.json'))['codes'] if boundary.faceOrder == 3
                 else {v['key']: v for v in json.load(open(boundary.orderPath('data/boundaryHarmonicRingCodeSweep1888Hold301FaceMinus60Minus5.json')))['variants']})
    stripePool = [np.array([c['multipliers'][0], 0.0, c['multipliers'][2]]) for c in stripeSweep.values() if c['kind'] == 'sweepGlobal']
    facePool = [np.array(c['multipliers']) * TARGETS['face']['coefficients'] for c in faceSweep.values() if c['kind'] == 'sweepGlobal']
    for group, pool in (('stripeControl', stripePool), ('faceControl', facePool)):
        if boundary.faceOrder != 3 and group == 'stripeControl':
            continue
        for number, index in enumerate(rng.choice(len(pool), args.controls, replace=False)):
            codes.append(dict(name=f'{group}{number:02d}', group=group, coefficients=pool[index], held=True))
    codes.append(dict(name='noRing', group='noRing', coefficients=TARGETS['face']['coefficients'] * 0, held=False))
    return codes


def darkBits(vmem):
    return np.packbits(vmem.numpy()[interior] * 1000.0 < THRESHOLD)


def simulate(step, ringValues, held):
    """(original run's packed frames, {age: branch frames}) for one code. Frame k of the original is recorded iteration k * stride."""
    step.ringMask = torch.zeros(step.numCells, dtype=torch.double)
    step.ringMask[torch.as_tensor(RING_CELLS, dtype=torch.long)] = 1.0 if held else 0.0
    code = torch.zeros(step.numCells, dtype=torch.double)
    code[step.ring] = torch.as_tensor(ringValues, dtype=torch.double) * step.Gref
    step.ringCode = code
    vmem, gpol = step.initialVmem.clone(), step.initialGpol.clone()
    total = max(ages) + args.branchIterations
    original, snapshots = [], {}
    for iteration in range(total):
        if iteration in ages:
            snapshots[iteration] = (vmem.clone(), gpol.clone())
        vmem, gpol = step(vmem, gpol, iteration < HOLD and held, iteration < HOLD and held)
        if iteration % args.stride == 0:
            original.append(darkBits(vmem))
    branches = {}
    if held:
        for age, (vmem, gpol) in snapshots.items():
            frames = []
            for k in range(args.branchIterations):
                vmem, gpol = step(vmem, gpol, k < HOLD, k < HOLD)
                if k % args.stride == 0:
                    frames.append(darkBits(vmem))
            branches[age] = np.array(frames)
    return np.array(original), branches


codes = buildCodes()
names, groups, originals = [], [], []
branchStore = {age: [] for age in ages}
with boundary.fullDoublePrecision(True):
    replayer = common.Replayer()
    for number, code in enumerate(codes):
        original, branches = simulate(replayer.step, ringValuesOf(code['coefficients'], 2.0), code['held'])
        names.append(code['name'])
        groups.append(code['group'])
        originals.append(original)
        for age in branches:
            branchStore[age].append(branches[age])
        say(f'{number + 1}/{len(codes)} {code["name"]}')
arrays = {f'branch{age}': np.array(branchStore[age]) for age in ages}                     # held codes only (the no-ring run has no branches), in the order of `names`
np.savez_compressed(args.outputPath, names=np.array(names), groups=np.array(groups), original=np.array(originals), ages=np.array(ages), stride=args.stride,
                    branchIterations=args.branchIterations, hold=HOLD, faceOrder=boundary.faceOrder, **arrays)
say('wrote', args.outputPath)
