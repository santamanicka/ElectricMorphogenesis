"""Boils one sweep code's raw relay file down to the small record the sweep analyses use, so the raw file (about 15 MB) need
not be kept: the code's multipliers and ring values, its whole causal net (relayLoopNets.readNet), its selectivity gap, the
face it ends with, and the smallest and largest conductance it ever reached (the check that G_pol stayed inside [0, 2]).

Writes <outputDirectory>/<key>.json (never overwriting).

    python3 extractRelayLoopSweepRecord11x11.py --rawPath <relay.npz> --key sweepGlobal000
"""
import argparse
import json
import os

import relayLoopNets

SUFFIX = '1888Hold301FaceMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--rawPath', type=str, required=True)
parser.add_argument('--key', type=str, required=True)
parser.add_argument('--variantsPath', type=str, default=f'data/boundaryHarmonicRingCodeSweep{SUFFIX}.json')
parser.add_argument('--outputDirectory', type=str, default='data/relayLoopSweep')
args = parser.parse_args()

outputPath = f'{args.outputDirectory}/{args.key}.json'
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
variant = {v['key']: v for v in json.load(open(args.variantsPath))['variants']}[args.key]
record = dict(key=args.key, kind=variant['kind'], multipliers=variant['multipliers'], ringValues=variant['ringValues'],
              **({'radius': variant['radius']} if 'radius' in variant else {}), **relayLoopNets.readNet(args.rawPath))
os.makedirs(args.outputDirectory, exist_ok=True)
json.dump(record, open(outputPath, 'w'), separators=(',', ':'))
print(f"wrote {outputPath}: gap {record['gap']:+.3f}, face overlap {record['faceOverlap']:.3f}, "
      f"G_pol {record['gpolMin']:.3f} .. {record['gpolMax']:.3f} G_ref")
