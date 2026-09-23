"""Is the cosine basis the ensemble's own basis? (PolyPatterning_Sim.md, Section 12).

A10 tests modes one at a time in the 2D cosine basis, which is fixed by the grid's geometry and is the same for every
code. That makes it a fair coordinate system but not necessarily the ensemble's natural one. This script compares it
with the basis the patterns themselves pick out: the principal components of the 1,024 patterns at the scored moment.
If the two lined up, a per-mode test would lose nothing; where they do not, influence spread across several cosine
modes can hide from A10 and still be found by A13's readout.

Reads the panel data, which already holds every pattern, and writes
data/boundaryHarmonicEnsembleBasis<checkpoint>Hold<hold><target>.json (never overwriting).
"""
import argparse
import json
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--panelsPath', type=str, default='data/boundaryHarmonicReadoutPanels1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--outputPath', type=str, default='data/boundaryHarmonicEnsembleBasis1888Hold301FaceMinus60Minus5.json')
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')

panels = json.load(open(args.panelsPath))
patterns = np.array([entry['vmem'] for entry in panels['codes']])
centred = patterns - patterns.mean(0)
components, singular, directions = np.linalg.svd(centred, full_matrices=False)
share = singular ** 2 / (singular ** 2).sum()
cumulative = np.cumsum(share)

size = 11
cosine = np.column_stack([np.outer(np.cos(np.pi * p * (np.arange(size) + 0.5) / size),
                                   np.cos(np.pi * q * (np.arange(size) + 0.5) / size)).ravel()
                          for p in range(size) for q in range(size)])
cosine = cosine / np.linalg.norm(cosine, axis=0)

readout = np.array(panels['patternDirection'])
readout = readout / np.linalg.norm(readout)
result = dict(numCodes=len(patterns), trainedMoment=panels['trainedMoment'],
              varianceShare=np.round(share[:12], 5).tolist(),
              componentsFor=dict(half=int(np.argmax(cumulative >= 0.5) + 1), ninety=int(np.argmax(cumulative >= 0.9) + 1),
                                 ninetyNine=int(np.argmax(cumulative >= 0.99) + 1)),
              leadingComponents=[], readoutOverlap=[round(float(abs(directions[k] @ readout)), 4) for k in range(5)])
for k in range(4):
    overlaps = np.abs(cosine.T @ directions[k])
    best = int(np.argmax(overlaps))
    result['leadingComponents'].append(dict(
        component=k + 1, varianceShare=round(float(share[k]), 5), bestCosineMode=[best // size, best % size],
        bestOverlap=round(float(overlaps[best]), 4),
        topFiveCoverage=round(float((np.sort(overlaps)[::-1][:5] ** 2).sum()), 4)))
    entry = result['leadingComponents'][-1]
    print(f"component {k + 1} ({entry['varianceShare'] * 100:.1f}% of variance): closest cosine mode "
          f"{tuple(entry['bestCosineMode'])} at {entry['bestOverlap']:.2f}, its best five cover {entry['topFiveCoverage']:.2f}")
print(f"components for half, 90% and 99% of the variance: {result['componentsFor']['half']}, "
      f"{result['componentsFor']['ninety']}, {result['componentsFor']['ninetyNine']}")
print('the readout direction overlaps component 1 by', result['readoutOverlap'][0])
json.dump(result, open(args.outputPath, 'w'))
print('wrote', args.outputPath)
