"""Mean conductance over time, feature cells against the rest of the interior, for the trained code and every steered or
knocked-out variant -- the curves of the Latching Switch report's figure 2 (G_pol / G_ref of the 14 feature cells and the
other 67 interior cells), read out of the raw relay files computeBoundaryHarmonicRelay11x11.py already wrote (their
trainedState is the whole (Vmem, G_pol) trajectory of the ring code that run decomposed), so nothing is re-simulated.

Each curve also carries its min and max across the cells of its group, so a viewer can tell a group that moves together
from one where only a few cells do. Sampled every `--stride` states from the start to the write peak (state 1766,
recorded iteration 1765), with the phase-end states 302 / 586 / 1766 always included.

Writes data/boundaryHarmonicRelayConductanceCurves1888Hold301FaceMinus60Minus5.json (never overwriting).

    python3 extractBoundaryHarmonicRelayConductanceCurves11x11.py
"""
import argparse
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary
import boundaryHarmonicCoarseGrain as coarse
from boundaryHarmonicStep import Step

parser = argparse.ArgumentParser()
parser.add_argument('--stride', type=int, default=10)
parser.add_argument('--fiveLevel', action='store_true',
                    help='only the two-sided slider / 5x5 grid codes (buildBoundaryHarmonicRingCodeFiveLevelGrid11x11.py)')
args = parser.parse_args()

SUFFIX = '1888Hold301FaceMinus60Minus5'
outputPath = f"data/boundaryHarmonicRelayConductanceCurves{'FiveLevel' if args.fiveLevel else ''}{SUFFIX}.json"
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')

NUM_CELLS = coarse.NUM_CELLS
Gref = float(Step(ringCode=np.zeros(len(boundary.boundaryRingCells))).Gref)
features = sorted(set(boundary.featureCellIndices.tolist()))
background = [int(c) for c in boundary.interiorCellIndices if c not in set(features)]
assert (len(features), len(background)) == (14, 67)

states = sorted(set(range(0, coarse.PEAK + 1, args.stride)) | {coarse.RELEASE, coarse.TROUGH, coarse.PEAK})

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


def summarise(conductance, cells):
    group = conductance[:, cells]
    return [[round(float(v), 3) for v in curve] for curve in (group.mean(1), group.min(1), group.max(1))]


curves = {}
for key, path in paths.items():
    trained = np.load(path)['trainedState']
    assert trained.shape[1] == 2 * NUM_CELLS, (key, trained.shape)
    conductance = trained[states, NUM_CELLS:] / Gref
    featureMean, featureMin, featureMax = summarise(conductance, features)
    backgroundMean, backgroundMin, backgroundMax = summarise(conductance, background)
    curves[key] = dict(featureMean=featureMean, featureMin=featureMin, featureMax=featureMax,
                       backgroundMean=backgroundMean, backgroundMin=backgroundMin, backgroundMax=backgroundMax)
    print(f'{key:28s} feature-background gap at the write peak {featureMean[-1] - backgroundMean[-1]:+.3f}')

json.dump(dict(states=states, features=features, background=background, curves=curves),
          open(outputPath, 'w'), separators=(',', ':'))
print(f'wrote {outputPath}: {len(curves)} ring codes x {len(states)} samples ({os.path.getsize(outputPath) // 1024} KB)')
