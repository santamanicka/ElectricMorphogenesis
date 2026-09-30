"""The tissue's own Vmem at the end of each phase (flood / clear / write), for the trained code and every steered or
knocked-out variant, read out of the raw relay files computeBoundaryHarmonicRelay11x11.py already wrote -- its
trainedState is the whole (Vmem, G_pol) trajectory of the ring code that run decomposed, so nothing is re-simulated.

Snapshots are at the same state indices the relay pipeline already uses for the phase ends (boundaryHarmonic
CoarseGrain.RELEASE / TROUGH / PEAK: recorded iterations 301, 585, 1765); Vmem is stored in volts, written here in
millivolts, one decimal.

Writes data/boundaryHarmonicRelayVmemSnapshots1888Hold301FaceMinus60Minus5.json (never overwriting).

    python3 extractBoundaryHarmonicRelayVmemSnapshots11x11.py
"""
import argparse
import json
import os

import numpy as np

import boundaryHarmonicCoarseGrain as coarse

parser = argparse.ArgumentParser()
parser.add_argument('--fiveLevel', action='store_true',
                    help='only the two-sided slider / 5x5 grid codes (buildBoundaryHarmonicRingCodeFiveLevelGrid11x11.py)')
args = parser.parse_args()

SUFFIX = '1888Hold301FaceMinus60Minus5'
outputPath = f"data/boundaryHarmonicRelayVmemSnapshots{'FiveLevel' if args.fiveLevel else ''}{SUFFIX}.json"
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

STATES = dict(flood=coarse.RELEASE, clear=coarse.TROUGH, write=coarse.PEAK)
NUM_CELLS = coarse.NUM_CELLS

paths = {}
if args.fiveLevel:
    for variant in json.load(open(f'data/boundaryHarmonicRingCodeFiveLevel{SUFFIX}.json'))['variants']:
        paths[variant['key']] = f"data/boundaryHarmonicRelayFiveLevel_{variant['key']}{SUFFIX}Raw.npz"
else:
    paths['trained'] = f'data/boundaryHarmonicRingOnlyRelay{SUFFIX}Raw.npz'
    for variant in json.load(open(f'data/boundaryHarmonicRingCodeVariants{SUFFIX}.json'))['variants']:
        paths[variant['key']] = f"data/boundaryHarmonicRelayVariant_{variant['key']}{SUFFIX}Raw.npz"
    for variant in json.load(open(f'data/boundaryHarmonicRingCodeSliderGrid{SUFFIX}.json'))['variants']:
        paths[variant['key']] = f"data/boundaryHarmonicRelaySliderGrid_{variant['key']}{SUFFIX}Raw.npz"

snapshots = {}
for key, path in paths.items():
    state = np.load(path)['trainedState']
    assert state.shape[1] == 2 * NUM_CELLS, (key, state.shape)
    snapshots[key] = {phase: [round(float(v) * 1000.0, 1) for v in state[index, :NUM_CELLS]] for phase, index in STATES.items()}
    lo, hi = min(min(v) for v in snapshots[key].values()), max(max(v) for v in snapshots[key].values())
    print(f'{key:28s} Vmem {lo:6.1f} .. {hi:6.1f} mV')

json.dump(dict(states=STATES, snapshots=snapshots), open(outputPath, 'w'), separators=(',', ':'))
print(f'wrote {outputPath}: {len(snapshots)} ring codes x {len(STATES)} phases x {NUM_CELLS} cells')
