"""Which interior cells go dark for the best profiles of the two-bump screen (simulateBoundaryHarmonicDoubleStripesBumps11x11.py)? EXPLORATORY.

Replays the best pure two-bump profile, the best profile of the registered grid with a dip, the best plateau profile and the best uniform ring (each chosen by the
overlap at its best moment, as the scoring does) and stores, for each, the 11 x 11 voltage map at its best moment, at the end of the hold and at a few later iterations.
It prints them as maps of dark cells (# dark interior cell inside the flanks, x dark interior cell outside them, . light, o ring).

    python3 inspectBoundaryHarmonicDoubleStripesBumps11x11.py

Writes data/boundaryHarmonicDoubleStripesBumpPatterns<checkpoint>Hold<hold><targetName>.json (never overwriting).
"""
import argparse
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301DoubleStripesInteriorMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--screenPath', type=str, default=f'data/boundaryHarmonicDoubleStripesBumps{SUFFIX}.json')
parser.add_argument('--outputPath', type=str, default=f'data/boundaryHarmonicDoubleStripesBumpPatterns{SUFFIX}.json')
parser.add_argument('--laterIterations', type=str, default='600,1000,1500,2000,2500,2999')
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')
screen = json.load(open(args.screenPath))
one = screen['stage1']
overlap, score = np.array(one['overlapAtBest']), np.array(one['score'])
families = [set(f) for f in one['families']]
T, M, K, S = (np.array(one[key]) for key in 'TMKS')
best = lambda indices: sorted(indices, key=lambda i: (-overlap[i], score[i]))[0]
everything = range(len(families))
chosen = {
    'best pure two-bump profile': best([i for i in everything if 'pure' in families[i]]),
    'best profile with a dip, registered grid': best([i for i in everything if 'wide' in families[i] and M[i] < T[i]]),
    'best plateau profile': best([i for i in everything if 'wide' in families[i] and M[i] == T[i]]),
    'best uniform ring': best([i for i in everything if 'uniform' in families[i]])}
later = [int(v) for v in args.laterIterations.split(',')]
hold = int(screen['hold'])
reference = boundary.loadCheckpoint(int(screen['referenceCheckpoint']))
ringValues = np.array([one['ringValues'][i] for i in chosen.values()])
wanted = {name: sorted({hold - 1, int(one['bestIteration'][i])} | set(later)) for name, i in chosen.items()}
captured = {name: {} for name in chosen}
names = list(chosen)


def onIteration(iteration, vmem):
    for row, name in enumerate(names):
        if iteration in wanted[name]:
            captured[name][iteration] = vmem[row].numpy().copy()


boundary.ringHoldBatchReplay(reference, ringValues, hold, int(screen['numIterations']), onIteration)
flank, ring, interior = set(boundary.flankCellIndices.tolist()), set(boundary.boundaryRingCells.tolist()), set(boundary.interiorCellIndices.tolist())


def asMap(vmem):
    lines = []
    for r in range(boundary.latticeRows):
        cells = []
        for c in range(boundary.latticeCols):
            cell = r * boundary.latticeCols + c
            dark = vmem[cell] < boundary.hyperpolarizedThresholdMilliVolts
            cells.append('o' if cell in ring else (('#' if cell in flank else 'x') if dark else '.'))
        lines.append(' '.join(cells))
    return lines


result = {}
for name, i in chosen.items():
    levels = dict(T=float(T[i]), M=float(M[i]), K=float(K[i]), S=float(S[i]))
    result[name] = dict(levels=levels, overlapAtBest=float(overlap[i]), score=float(score[i]), bestIteration=int(one['bestIteration'][i]),
                        maps={str(it): np.round(v, 1).tolist() for it, v in captured[name].items()})
    print(f"\n{name}: T={levels['T']} M={levels['M']} K={levels['K']} S={levels['S']}  overlap at best {overlap[i]:.3f}, best moment {one['bestIteration'][i]}")
    shown = [hold - 1, int(one['bestIteration'][i])]
    maps = [asMap(captured[name][it]) for it in shown]
    print('   ' + '      '.join(f'iteration {it:<5}' + ' ' * 16 for it in shown))
    for row in zip(*maps):
        print('   ' + '        '.join(row))
    for it in shown:
        v = captured[name][it]
        inside = [v[c] for c in flank]
        print(f'   iteration {it}: dark flank cells {sum(x < boundary.hyperpolarizedThresholdMilliVolts for x in inside)}/54; flank Vmem min {min(inside):.1f} median {np.median(inside):.1f} max {max(inside):.1f} mV')
json.dump(dict(note='EXPLORATORY: voltage maps of the best profiles of the two-bump screen.', screen=args.screenPath, profiles=result), open(args.outputPath, 'w'), separators=(',', ':'))
print('\nwrote', args.outputPath)
