"""The custom layout's dominant transfers per phase (flood/clear/write), for one ring-code variant, canonicalised
one side per mirror pair -- the same reduction used by hand for the trained code (the Relay Loop diagram), done
automatically so it scales to every steered or knocked-out variant. Reads the FULL, untruncated block net matrix
(weight.T @ net @ weight over the custom layout's namedRegionLabels weight matrix), not the movie's pruned display
edges, exactly as computeBoundaryHarmonicRelay11x11.py's own edges array already holds it.

Writes data/boundaryHarmonicRelayVariantPhaseSummary_<key>1888Hold301FaceMinus60Minus5.json (never overwriting).

    python3 computeBoundaryHarmonicRelayVariantPhaseSummary11x11.py --relayPath <variantRaw.npz> --variantKey steerOrder0
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
parser.add_argument('--topPerPhase', type=int, default=3)
args = parser.parse_args()

outputPath = f'data/boundaryHarmonicRelayVariantPhaseSummary_{args.variantKey}1888Hold301FaceMinus60Minus5.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

relay = np.load(args.relayPath)
names = [str(x) for x in relay['readoutNames']]
primary = names.index('selectivity')
edges, window = relay['edges'], int(relay['edgeWindow'])
ring = np.array(boundary.boundaryRingCells)
weight, labels, regionNames = coarse.namedRegionLabels(ring, boundary.featureParts)
idx = {nm: i for i, nm in enumerate(regionNames)}
N = 36                                                               # the movie's 36 committed windows, as elsewhere


def blockNetsAt(w):
    total = edges[primary, :, w]
    nets = [total[ch] - total[ch].T for ch in (0, 1)]
    return [weight.T @ net @ weight for net in nets]


allNets = [blockNetsAt(w) for w in range(N)]
PHASES = dict(flood=range(0, 6), clear=range(6, 12), write=range(12, 36))    # windows 1-6 / 7-12 / 13-36, 0-indexed

mirrorOf = {'ring left': 'ring right', 'ring right': 'ring left',
            'background top-left': 'background top-right', 'background top-right': 'background top-left',
            'background bottom-left': 'background bottom-right', 'background bottom-right': 'background bottom-left'}


def canon(nm):
    return nm if (nm not in mirrorOf or 'left' in nm) else mirrorOf[nm]


undirectedPairs, seen = [], set()
for a in regionNames:
    for b in regionNames:
        if a == b or canon(a) != a or canon(b) != b:
            continue
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        undirectedPairs.append((a, b))

result = {}
for phase, idxs in PHASES.items():
    totals = {}
    for a, b in undirectedPairs:
        totals[(a, b)] = sum(allNets[w][0][idx[b], idx[a]] for w in idxs)   # field channel; positive = net a -> b
    ranked = sorted(totals.items(), key=lambda kv: -abs(kv[1]))[:args.topPerPhase]
    result[phase] = [dict(a=(a if v >= 0 else b), b=(b if v >= 0 else a), value=round(abs(float(v)), 5),
                          mirrored=bool(a in mirrorOf or b in mirrorOf))
                     for (a, b), v in ranked]

json.dump(dict(variantKey=args.variantKey, regionNames=regionNames, phases=result), open(outputPath, 'w'), indent=1)
print(f'wrote {outputPath}:')
for phase, edgesList in result.items():
    print(f'  {phase}: ' + ', '.join(f"{e['a']} -> {e['b']} {e['value']:.3f}" for e in edgesList))
