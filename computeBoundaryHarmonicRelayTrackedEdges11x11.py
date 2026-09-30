"""The trained code's own eleven hand-picked transfers (Relay Loop artifact, TRAINED_EDGES), read off any other
ring code's raw relay -- a fixed lens for direct comparison, as opposed to computeBoundaryHarmonicRelayVariant
PhaseSummary11x11.py's top-3-per-phase, which picks different pairs for every code.

Each of the eleven is a specific (sender, receiver, phase) triple. For a code where the net flow between that
pair has reversed (receiver and sender swapped, relative to the trained code), the sender/receiver in the output
are swapped too and the value is still positive -- a viewer comparing across codes sees the same pair's traffic
and which way it is currently running, not a value that goes negative and back for no visible reason.

Writes data/boundaryHarmonicRelayTrackedEdges_<variantKey>1888Hold301FaceMinus60Minus5.json (never overwriting).

    python3 computeBoundaryHarmonicRelayTrackedEdges11x11.py --relayPath <variantRaw.npz> --variantKey steerOrder0
"""
import argparse
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary
import boundaryHarmonicCoarseGrain as coarse

parser = argparse.ArgumentParser()
parser.add_argument('--relayPath', type=str, required=True)
parser.add_argument('--variantKey', type=str, required=True)
args = parser.parse_args()

outputPath = f'data/boundaryHarmonicRelayTrackedEdges_{args.variantKey}1888Hold301FaceMinus60Minus5.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

# the trained code's own eleven, exactly as hand-picked for the Relay Loop artifact's TRAINED_EDGES
TRACKED = [
    ('ring top', 'background top-left', 'flood'),
    ('mouth', 'ring bottom', 'flood'),
    ('background top-left', 'eyes', 'clear'),
    ('background top-left', 'ring left', 'clear'),
    ('background top-left', 'nose', 'clear'),
    ('ring bottom', 'background bottom-left', 'write'),
    ('background bottom-left', 'background top-left', 'write'),
    ('ring left', 'background bottom-left', 'write'),
    ('eyes', 'background top-left', 'write'),
    ('nose', 'background top-left', 'write'),
    ('background bottom-left', 'mouth', 'write'),
]

relay = np.load(args.relayPath)
names = [str(x) for x in relay['readoutNames']]
primary = names.index('selectivity')
edges, window = relay['edges'], int(relay['edgeWindow'])
ring = np.array(boundary.boundaryRingCells)
weight, labels, regionNames = coarse.namedRegionLabels(ring, boundary.featureParts)
idx = {nm: i for i, nm in enumerate(regionNames)}
N = 36
PHASES = dict(flood=range(0, 6), clear=range(6, 12), write=range(12, 36))


def blockNetAt(w):
    total = edges[primary, :, w]
    net = total[0] - total[0].T                                # field channel only, as the artifact draws
    return weight.T @ net @ weight                             # cell-level net -> block-level net (namedRegionLabels' weight)


allNets = [blockNetAt(w) for w in range(N)]

tracked = []
for sender, receiver, phase in TRACKED:
    value = sum(allNets[w][idx[receiver], idx[sender]] for w in PHASES[phase])
    if value < 0:
        sender, receiver, value = receiver, sender, -value
    tracked.append(dict(a=sender, b=receiver, phase=phase, value=round(float(value), 6)))

json.dump(dict(variantKey=args.variantKey, tracked=tracked), open(outputPath, 'w'), indent=1)
print(f'wrote {outputPath}:')
for t in tracked:
    print(f"  {t['phase']:6s} {t['a']} -> {t['b']}  {t['value']:.4f}")
