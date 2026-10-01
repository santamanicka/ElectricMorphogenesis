"""Every ring code's whole causal net, for searching across codes: for each of the trained code, the eight curated codes and
the slider / grid codes of the Relay Loop page (126 in all), the net transfer between every pair of canonical blocks in each
phase, field and contact channel, with the code's coefficient multipliers, its selectivity gap at the write peak and the face it ends with (structural overlap of its
dark cells with the face at recorded iteration 2173, boundaryCodeUtilities.structuralIntersectionOverUnion).

Net transfer is the custom layout's block net (weight.T @ (edges - edges.T) @ weight over the relay's selectivity readout),
summed over the windows of each phase (flood 0-5, clear 6-11, write 12-35, as the page does) and signed along the pair's
canonical direction (the earlier block in the page's node order to the later one). Only the left / centre blocks are kept:
the right-hand blocks are mirror images to ~1e-8 over these windows.

Writes data/relayLoopFullNets1888Hold301FaceMinus60Minus5.json (never overwriting).

    python3 extractRelayLoopFullNets11x11.py
"""
import argparse
import json
import os

import numpy as np

import relayLoopNets

SUFFIX = '1888Hold301FaceMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--outputPath', type=str, default=f'data/relayLoopFullNets{SUFFIX}.json')
outputPath = parser.parse_args().outputPath
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')


placement = json.load(open(f'data/boundaryHarmonicRingCodeFiveLevel{SUFFIX}.json'))
variants = {v['key']: v for v in json.load(open(f'data/boundaryHarmonicRingCodeVariants{SUFFIX}.json'))['variants']}
trained = np.asarray(placement['trainedCoefficients'], float)

# ---- every code's key, raw file and coefficient multipliers
multipliers = {'trained': [1.0] * 4}
for order, points in placement['slider'].items():
    for p in points:
        m = [1.0] * 4
        m[int(order)] = p['multiplier']
        multipliers.setdefault(p['key'], m)
for pairKey, rows in placement['grid'].items():
    i, j = (int(x) for x in pairKey.split('_'))
    for r, mi in enumerate(placement['gridMultipliers']):
        for c, mj in enumerate(placement['gridMultipliers']):
            m = [1.0] * 4
            m[i], m[j] = mi, mj
            key = rows[r][c]
            assert multipliers.setdefault(key, m) == m, (key, multipliers[key], m)
for key, v in variants.items():
    if v['kind'] == 'steering':
        multipliers[key] = [float(c / t) for c, t in zip(v['coefficients'], trained)]
    else:
        multipliers.setdefault(key, [1.0 if o not in v['orders'] else 0.0 for o in range(4)])

paths = {'trained': f'data/boundaryHarmonicRingOnlyRelay{SUFFIX}Raw.npz'}
for key in variants:
    paths[key] = f'data/boundaryHarmonicRelayVariant_{key}{SUFFIX}Raw.npz'
for v in json.load(open(f'data/boundaryHarmonicRingCodeSliderGrid{SUFFIX}.json'))['variants']:
    paths[v['key']] = f"data/boundaryHarmonicRelaySliderGrid_{v['key']}{SUFFIX}Raw.npz"
for v in placement['variants']:
    paths[v['key']] = f"data/boundaryHarmonicRelayFiveLevel_{v['key']}{SUFFIX}Raw.npz"

gaps = {k: v for k, v in json.load(open(f'data/boundaryHarmonicRelayConductanceCurves{SUFFIX}.json'))['curves'].items()}
gaps.update(json.load(open(f'data/boundaryHarmonicRelayConductanceCurvesFiveLevel{SUFFIX}.json'))['curves'])

codes = {}
for n, key in enumerate(sorted(multipliers)):
    net = relayLoopNets.readNet(paths[key])
    c = gaps[key]
    codes[key] = dict(multipliers=multipliers[key], gap=round(float(c['featureMean'][-1] - c['backgroundMean'][-1]), 4),
                      faceOverlap=net['faceOverlap'], field=net['field'], contact=net['contact'])
    print(f'{n + 1}/{len(multipliers)} {key}', flush=True)

json.dump(dict(pairs=[list(p) for p in relayLoopNets.PAIRS], phases=list(relayLoopNets.PHASES), nodeOrder=relayLoopNets.NODE_ORDER, codes=codes),
          open(outputPath, 'w'), separators=(',', ':'))
print(f'wrote {outputPath}: {len(codes)} codes x {len(PHASES)} phases x {len(PAIRS)} pairs x 2 channels')
