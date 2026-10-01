"""Merges the sweep's per-code records (data/relayLoopSweep/<key>.json) into one file, and checks that every code ran and that
no conductance left [0, 2] G_ref.

Writes data/relayLoopSweepNets1888Hold301FaceMinus60Minus5.json (never overwriting), laid out like relayLoopFullNets...json so
the same analyses read either: `pairs`, `phases`, and `codes` keyed by code with multipliers, gap, faceOverlap, field and contact.

    python3 mergeRelayLoopSweep11x11.py
"""
import glob
import json
import os

import relayLoopNets

SUFFIX = '1888Hold301FaceMinus60Minus5'
outputPath = f'data/relayLoopSweepNets{SUFFIX}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

expected = [v['key'] for v in json.load(open(f'data/boundaryHarmonicRingCodeSweep{SUFFIX}.json'))['variants']]
records = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob('data/relayLoopSweep/*.json'))}
missing = [k for k in expected if k not in records]
if missing:
    raise SystemExit(f'{len(missing)} of {len(expected)} codes have no record yet, e.g. {missing[:5]}')
codes = {k: {f: v for f, v in records[k].items() if f not in ('key', 'ringValues')} for k in expected}
lowest, highest = min(c['gpolMin'] for c in codes.values()), max(c['gpolMax'] for c in codes.values())
json.dump(dict(pairs=[list(p) for p in relayLoopNets.PAIRS], phases=list(relayLoopNets.PHASES), nodeOrder=relayLoopNets.NODE_ORDER, codes=codes),
          open(outputPath, 'w'), separators=(',', ':'))
print(f'wrote {outputPath}: {len(codes)} codes; conductance range over every cell, time and code {lowest:.3f} .. {highest:.3f} G_ref '
      f'({"inside" if lowest >= 0 and highest <= 2 else "OUTSIDE"} [0, 2])')
