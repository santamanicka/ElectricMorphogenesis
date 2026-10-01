"""The tissue's Vmem at the trained model's best moment -- recorded iteration 2173, where the Latching Switch report scores the
orders 0-3 run (its `bestIteration`, marked "scored") -- for the trained code and every steered or knocked-out code the Relay Loop page
shows, read out of the raw relay files (their trainedState is the whole (Vmem, G_pol) trajectory), so nothing is re-simulated. This is the
latch's end, which the page's phase snapshots (flood 301, clear 585, write 1765) stop short of. Vmem is stored in volts, written here in
millivolts, one decimal.

Writes data/boundaryHarmonicRelayBestMomentVmem1888Hold301FaceMinus60Minus5.json (never overwriting).

    python3 extractBoundaryHarmonicRelayBestMomentVmem11x11.py
"""
import json
import os

import numpy as np

import boundaryHarmonicCoarseGrain as coarse

SUFFIX = '1888Hold301FaceMinus60Minus5'
outputPath = f'data/boundaryHarmonicRelayBestMomentVmem{SUFFIX}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

ITERATION = 2173                                  # the report's bestIteration for the orders 0-3 code
STATE = ITERATION + 1                             # a state is a recorded iteration + 1
NUM_CELLS = coarse.NUM_CELLS

placement = json.load(open(f'data/boundaryHarmonicRingCodeFiveLevel{SUFFIX}.json'))
pageKeys = {'trained'} | {p['key'] for points in placement['slider'].values() for p in points} | {k for rows in placement['grid'].values() for row in rows for k in row}
variants = json.load(open(f'data/boundaryHarmonicRingCodeVariants{SUFFIX}.json'))['variants']
pageKeys |= {v['key'] for v in variants}
paths = {'trained': f'data/boundaryHarmonicRingOnlyRelay{SUFFIX}Raw.npz'}
paths.update({v['key']: f"data/boundaryHarmonicRelayVariant_{v['key']}{SUFFIX}Raw.npz" for v in variants})
paths.update({v['key']: f"data/boundaryHarmonicRelaySliderGrid_{v['key']}{SUFFIX}Raw.npz" for v in json.load(open(f'data/boundaryHarmonicRingCodeSliderGrid{SUFFIX}.json'))['variants']})
paths.update({v['key']: f"data/boundaryHarmonicRelayFiveLevel_{v['key']}{SUFFIX}Raw.npz" for v in placement['variants']})

snapshots = {}
for key in sorted(pageKeys):
    state = np.load(paths[key])['trainedState']
    assert state.shape[1] == 2 * NUM_CELLS and state.shape[0] > STATE, (key, state.shape)
    snapshots[key] = [round(float(v) * 1000.0, 1) for v in state[STATE, :NUM_CELLS]]
json.dump(dict(iteration=ITERATION, state=STATE, snapshots=snapshots), open(outputPath, 'w'), separators=(',', ':'))
darkest = min(min(v) for v in snapshots.values())
print(f'wrote {outputPath}: {len(snapshots)} ring codes x {NUM_CELLS} cells at iteration {ITERATION}; darkest cell anywhere {darkest:.1f} mV')
