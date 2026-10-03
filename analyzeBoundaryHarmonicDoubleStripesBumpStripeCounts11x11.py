"""How many vertical stripes does a ring code's pattern show over time, whatever their width? EXPLORATORY.

For the best profiles of the two-bump screen (simulateBoundaryHarmonicDoubleStripesBumps11x11.py) and, as the reference, the single stripe's own code
(data/boundaryHarmonicTraining1888Hold301StripesInteriorMinus60Minus5Ceiling2/order2_restart06.npz), the whole voltage history of the interior is
replayed (301-iteration hold, release, 3000 iterations) and, at every iteration after the release, a column of the 9 x 9 interior counts as dark when at least
--columnMinimum of its nine cells are dark (Vmem < -34.6 mV); a stripe is a run of adjacent dark columns, of any width. The single stripe's code is the check on
the counting: it should show two stripes at iteration 1130 and three at 1807, as its report's maps do.

    python3 analyzeBoundaryHarmonicDoubleStripesBumpStripeCounts11x11.py

Writes data/boundaryHarmonicDoubleStripesBumpStripeCounts<checkpoint>Hold<hold><targetName>.json (never overwriting).
"""
import argparse
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

SUFFIX = '1888Hold301DoubleStripesInteriorMinus60Minus5'
parser = argparse.ArgumentParser()
parser.add_argument('--screenPath', type=str, default=f'data/boundaryHarmonicDoubleStripesBumps{SUFFIX}.json')
parser.add_argument('--stripeCodePath', type=str, default='data/boundaryHarmonicTraining1888Hold301StripesInteriorMinus60Minus5Ceiling2/order2_restart06.npz')
parser.add_argument('--outputPath', type=str, default=f'data/boundaryHarmonicDoubleStripesBumpStripeCounts{SUFFIX}.json')
parser.add_argument('--columnMinimum', type=int, default=5, help='dark cells (of 9) that make an interior column dark')
parser.add_argument('--minimumStretch', type=int, default=20, help='shortest run of iterations with one stripe count that is reported as a stretch')
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')
screen = json.load(open(args.screenPath))
one = screen['stage1']
hold, numIterations = int(screen['hold']), int(screen['numIterations'])
overlap, score = np.array(one['overlapAtBest']), np.array(one['score'])
families = [set(f) for f in one['families']]
T, M = np.array(one['T']), np.array(one['M'])
best = lambda indices: sorted(indices, key=lambda i: (-overlap[i], score[i]))[0]
everything = range(len(families))
codes = {
    'best pure two-bump profile': one['ringValues'][best([i for i in everything if 'pure' in families[i]])],
    'best profile with a dip, high sides': one['ringValues'][best([i for i in everything if 'wide' in families[i] and M[i] < T[i]])],
    'best plateau profile': one['ringValues'][best([i for i in everything if 'wide' in families[i] and M[i] == T[i]])],
    'best uniform ring': one['ringValues'][best([i for i in everything if 'uniform' in families[i]])]}
stripeRun = np.load(args.stripeCodePath)
angles = boundary.ringAngles(boundary.boundaryRingCells)
stripeCoefficients = stripeRun['bestCoefficients']
codes = {'REFERENCE: the single stripe\'s own code (ceiling 2.0, order 2)': np.clip(np.cos(np.outer(angles, np.arange(len(stripeCoefficients)))) @ stripeCoefficients, 0, 2).tolist(), **codes}
names = list(codes)

reference = boundary.loadCheckpoint(int(screen['referenceCheckpoint']))
history = np.zeros((len(names), numIterations, boundary.numCells), dtype=np.float32)


def onIteration(iteration, vmem):
    history[:, iteration] = vmem.numpy()


boundary.ringHoldBatchReplay(reference, np.array([codes[n] for n in names]), hold, numIterations, onIteration)

interior = np.array(boundary.interiorCellIndices).reshape(9, 9)         # rows 1-9 x columns 1-9
darkInterior = history[:, :, interior] < boundary.hyperpolarizedThresholdMilliVolts          # (code, iteration, 9 rows, 9 columns)
flank = np.isin(interior, boundary.flankCellIndices)


def runs(columns):
    """Runs of adjacent True entries of a length-9 boolean vector, as (first column, last column) in lattice columns 1-9."""
    found, start = [], None
    for position, flag in enumerate(list(columns) + [False]):
        if flag and start is None:
            start = position
        if not flag and start is not None:
            found.append((start + 1, position))
            start = None
    return found


def stripeRuns(codeIndex, iteration, minimum):
    return runs(darkInterior[codeIndex, iteration].sum(0) >= minimum)


def asMap(codeIndex, iteration):
    lines = []
    for r in range(9):
        lines.append(' '.join(('#' if flank[r, c] else 'x') if darkInterior[codeIndex, iteration, r, c] else '.' for c in range(9)))
    return lines


result = dict(note='EXPLORATORY: vertical stripes over time for the best two-bump profiles; the single stripe\'s code is the reference.', columnMinimum=args.columnMinimum,
              hold=hold, numIterations=numIterations, profiles={})
print(f'a column is dark when >= {args.columnMinimum} of its 9 interior cells are dark; a stripe is a run of adjacent dark columns\n')
for k, name in enumerate(names):
    counts = np.array([len(stripeRuns(k, it, args.columnMinimum)) for it in range(numIterations)])
    after = counts[hold:]
    tally = {str(c): int((after == c).sum()) for c in range(0, int(after.max()) + 1)}
    stretches, begin = [], hold
    for it in range(hold + 1, numIterations + 1):
        if it == numIterations or counts[it] != counts[begin]:
            if it - begin >= args.minimumStretch:
                stretches.append(dict(first=begin, last=it - 1, stripes=int(counts[begin]), columns=stripeRuns(k, (begin + it - 1) // 2, args.columnMinimum)))
            begin = it
    sensitivity = {str(m): {str(it): len(stripeRuns(k, it, m)) for it in (1130, 1807)} for m in (3, 4, 5, 6, 7, 9)}
    result['profiles'][name] = dict(iterationsWithNStripesAfterRelease=tally, stretches=stretches, countsAtIterations1130And1807ByColumnMinimum=sensitivity,
                                    countTrace=counts[hold::5].tolist())
    print(f'{name}\n  iterations (of {numIterations - hold} after the release) showing 0, 1, 2, 3 ... stripes: {tally}')
    print('  stretches of >= %d iterations (first-last: stripes, column runs):' % args.minimumStretch)
    for s in stretches:
        print(f"    {s['first']:>4}-{s['last']:<4}: {s['stripes']} stripe(s) {s['columns']}")
    print(f'  stripe counts at iteration 1130 and 1807 for column minimum 3/4/5/6/7/9: ' + '  '.join(f"{m}:{v['1130']},{v['1807']}" for m, v in sensitivity.items()))
    shown = {}
    for s in stretches:
        if s['stripes'] in (1, 2, 3) and s['stripes'] not in shown:
            shown[s['stripes']] = (s['first'] + s['last']) // 2
    for count, it in sorted(shown.items()):
        print(f'  e.g. {count} stripe(s) at iteration {it}:')
        for line in asMap(k, it):
            print('      ' + line)
    result['profiles'][name]['examples'] = {str(c): dict(iteration=int(it), map=asMap(k, it)) for c, it in shown.items()}
    print()
json.dump(result, open(args.outputPath, 'w'), separators=(',', ':'))
print('wrote', args.outputPath)
