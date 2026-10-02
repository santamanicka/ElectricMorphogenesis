"""The scalar record of every restart of every stripe training run, so the stripes report's ladder can be rebuilt without the raw checkpoints.

Each training restart writes an .npz of about 5 MB (every candidate evaluated, with its Vmem at the best moment, for the library the later arms start
from); the report's ladder needs only a handful of numbers from each: the arm (which orders), the population, the ceiling, the restart and how it started,
the best score and where, the best code and the interior's Vmem at its best moment. This reads every restart of every folder
`data/boundaryHarmonicTraining<suffix>*` and writes those, at full precision, to one JSON of well under 1 MB, laid out so that
plotBoundaryHarmonicStripes11x11.py reads it in place of the folders (it does so on its own when the folders are absent, and with --useRestartRecords).

Writes data/boundaryHarmonicTrainingRestartRecords<suffix>.json (never overwriting).

    python3 extractBoundaryHarmonicTrainingRestartRecords11x11.py
"""
import argparse
import glob
import json
import os
import re

import numpy as np

SUFFIX = '1888Hold301StripesInteriorMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--outputPath', type=str, default=f'data/boundaryHarmonicTrainingRestartRecords{SUFFIX}.json')
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')

stem = f'data/boundaryHarmonicTraining{SUFFIX}'
folders = {}
for folder in sorted(glob.glob(f'{stem}*')):
    if not os.path.isdir(folder):
        continue
    records = []
    for path in sorted(glob.glob(f'{folder}/*_restart*.npz')):
        run = np.load(path)
        records.append(dict(
            arm=re.sub(r'_restart.*', '', os.path.basename(path)),
            orders=[int(o) for o in run['orders']] if 'orders' in run else list(range(int(run['maxOrder']) + 1)),
            populationSize=int(run['populationSize']), ceiling=float(run['ceiling']), restart=int(run['restart']), startType=str(run['startType']),
            bestScore=float(run['bestScore']), bestIteration=int(run['bestIteration']), bestCoefficients=[float(c) for c in run['bestCoefficients']],
            bestVmem=[float(v) for v in run['bestVmem']]))
    if records:
        folders[os.path.basename(folder)[len(os.path.basename(stem)):]] = records
json.dump(dict(note='scalar record of every stripe training restart, read from the raw checkpoints by extractBoundaryHarmonicTrainingRestartRecords11x11.py',
               folders=folders), open(args.outputPath, 'w'), separators=(',', ':'))
print(f'wrote {args.outputPath}: {sum(len(r) for r in folders.values())} restarts in {len(folders)} folders ({os.path.getsize(args.outputPath) / 1e6:.2f} MB)')
