"""Reads every saved double-stripe training restart once and writes a compact table for the report page.

    python3 tabulateBoundaryHarmonicDoubleStripesRestarts11x11.py [--overwrite]

One row per restart (every stage of the registered ladder: arms A, B and C, then rungs 1a, 1b, 2, 3a and 3b), taken from the file's
stored best moment: the voltage map, the ring values, the score, the double-stripe overlap and the dark-cell counts. Nothing is
simulated or rescored: the overlap is boundaryCodeUtilities.structuralIntersectionOverUnion on the stored best voltages and the
score is the stored one. The registration's scoring script (D1-D8, G2, C1) is a separate script and is not run here.
Like every script here it refuses to overwrite its output unless --overwrite is given.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301DoubleStripesInteriorMinus60Minus5'
FOLDER = f'data/boundaryHarmonicTraining{SUFFIX}'
# stage key: (folder suffix, title, route, what the stage added to the ladder)
STAGES = (
    ('A', '', 'Arm A', 'contiguous and even-only, small', 'pilot at ceiling 1.3'),
    ('B', 'Ceiling2', 'Arm B', 'contiguous and even-only, small', 'pilot at ceiling 2.0'),
    ('C', 'EvenOrdersPopulation64', 'Arm C', 'even-only', 'even-only chain at ceiling 1.3'),
    ('1a', 'Population64', 'Rung 1a', 'contiguous, orders 3-6', 'population 64 at ceiling 1.3'),
    ('1b', 'Ceiling2Population64', 'Rung 1b', 'contiguous, orders 3-6', 'population 64 at ceiling 2.0'),
    ('2', 'Ceiling2EvenOrdersPopulation64', 'Rung 2', 'even-only {0,2,4} and {0,2,4,6}', 'ceiling 2.0'),
    ('3a', 'Ceiling2HigherOrdersPopulation16', 'Rung 3a', 'contiguous, orders 8-20', 'population 16 at ceiling 2.0'),
    ('3b', 'Ceiling2HigherOrdersPopulation64', 'Rung 3b', 'contiguous, orders 8-20', 'population 64 at ceiling 2.0'),
)

parser = argparse.ArgumentParser()
parser.add_argument('--outputPath', type=str, default=f'data/boundaryHarmonicDoubleStripesRestartTable{SUFFIX}.json')
parser.add_argument('--overwrite', action='store_true')
args = parser.parse_args()
if os.path.exists(args.outputPath) and not args.overwrite:
    raise SystemExit(f'{args.outputPath} exists; pass --overwrite to rebuild it')

interior = set(int(cell) for cell in boundary.interiorCellIndices)
flanks = set(int(cell) for cell in boundary.flankCellIndices)
rows = []
for key, folderSuffix, title, route, note in STAGES:
    for path in sorted(glob.glob(f'{FOLDER}{folderSuffix}/order*_restart*.npz')):
        name = os.path.basename(path)
        match = re.match(r'(orders?)([0-9-]+)_restart(\d+)\.npz', name)
        orderLabel = match.group(1) + match.group(2)
        data = np.load(path)
        vmem = data['bestVmem'].astype(float)
        features = np.flatnonzero(data['scoreGroupMasks'][0])
        assert set(int(cell) for cell in features) == flanks, f'{path}: the feature cells are not the flank cells'
        dark = set(int(cell) for cell in np.flatnonzero(vmem < boundary.hyperpolarizedThresholdMilliVolts))
        rows.append({
            'stage': key, 'file': name, 'orders': [int(order) for order in data['orders']] if 'orders' in data.files else None,
            'orderLabel': orderLabel, 'restart': int(data['restart']), 'startType': str(data['startType']),
            'ceiling': float(data['ceiling']), 'populationSize': int(data['populationSize']),
            'numEvaluations': int(data['numEvaluations']), 'score': float(data['bestScore']),
            'bestIteration': int(data['bestIteration']),
            'overlap': boundary.structuralIntersectionOverUnion(vmem, features),
            'flankDark': len(dark & flanks), 'strayDark': len((dark & interior) - flanks),
            'ringValues': [round(float(value), 4) for value in data['bestRingValues']],
            'vmem': [round(float(value), 1) for value in vmem],
        })
    print(f'{key}: {sum(row["stage"] == key for row in rows)} restarts')
stages = [{'key': key, 'title': title, 'route': route, 'note': note, 'folder': f'{FOLDER}{folderSuffix}'}
          for key, folderSuffix, title, route, note in STAGES]
json.dump({'note': 'Descriptive table of the saved restarts; nothing here is simulated or rescored.', 'threshold': boundary.hyperpolarizedThresholdMilliVolts,
           'stages': stages, 'restarts': rows}, open(args.outputPath, 'w'))
print(f'wrote {args.outputPath}: {len(rows)} restarts')
