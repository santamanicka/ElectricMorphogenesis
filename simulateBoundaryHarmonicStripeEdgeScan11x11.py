"""Does the stripe depend on the ring cells that the code holds above the bistable edge? EXPLORATORY: nothing here was predicted.

A lone cell has two stable voltages when its polarising conductance G_pol / G_ref is between 0.802 and 1.439 and only the
hyperpolarised one above 1.439 (boundaryCodeUtilities). The stripe code holds exactly ten ring cells above 1.439: the middle five
of the top row and the middle five of the bottom row, the ones facing the centre stripe's ends. This scan moves them against the
rest of the ring. Take the stripe code's ring as it is and add

  delta   to those ten cells (the "end cells"), and
  epsilon to the other thirty ring cells,

on a grid of both, each value clipped to [0, ceiling], and read every code as in training (balanced RMS, best moment, the overlap of the
dark cells with the stripe, the strays), plus the highest overlap at any iteration. Along epsilon = 0 the end cells cross the 1.439 edge at
delta = 1.439 - (the lowest end cell); the map shows whether the stripe's window starts there.

    python3 simulateBoundaryHarmonicStripeEdgeScan11x11.py --trainedRunPath data/boundaryHarmonicTraining...Ceiling2/order2_restart06.npz \\
        --deltaRange=-0.06,0.06,0.0025 --epsilonRange=-0.6,0.6,0.025       (a range that starts with a minus sign needs the = form)

Writes data/boundaryHarmonicStripeEdgeScan<checkpoint>Hold<hold><target>Order<size>Restart<restart>Ceiling<ceiling>.json (never overwriting).
"""
import argparse
import json
import os
import time

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--trainedRunPath', type=str, required=True)
parser.add_argument('--deltaRange', type=str, default='-0.06,0.06,0.0025', help='first,last,step of the shift of the end cells')
parser.add_argument('--epsilonRange', type=str, default='-0.6,0.6,0.025', help='first,last,step of the shift of the other ring cells')
parser.add_argument('--batchSize', type=int, default=64)
parser.add_argument('--outputPath', type=str, default=None)
args = parser.parse_args()
parserRange = lambda text: np.round(np.arange(float(text.split(',')[0]), float(text.split(',')[1]) + float(text.split(',')[2]) / 2, float(text.split(',')[2])), 6)

run = dict(np.load(args.trainedRunPath))
orders = run['orders'] if 'orders' in run else np.arange(len(run['bestCoefficients']))
ceiling, hold, checkpoint = float(run['ceiling']), int(run['holdIterations']), int(run['referenceCheckpoint'])
baseRing = np.clip(np.cos(np.outer(boundary.ringAngles(boundary.boundaryRingCells), orders)) @ run['bestCoefficients'], 0, ceiling)
endCells = np.flatnonzero(baseRing > 1.439)
otherCells = np.flatnonzero(baseRing <= 1.439)
crossing = 1.439 - float(baseRing[endCells].min())
outputPath = args.outputPath or (f"data/boundaryHarmonicStripeEdgeScan{checkpoint}Hold{hold}{run['targetName']}Order{int(orders[-1])}"
                                 f"Restart{int(run['restart']):02d}Ceiling{ceiling:g}".replace('.', 'p') + '.json')
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
started = time.time()
deltas, epsilons = parserRange(args.deltaRange), parserRange(args.epsilonRange)
codes = np.array([[d, e] for d in deltas for e in epsilons])
ringValues = np.repeat(baseRing[None], len(codes), 0)
ringValues[:, endCells] += codes[:, [0]]
ringValues[:, otherCells] += codes[:, [1]]
clippedShare = ((ringValues < 0) | (ringValues > ceiling)).mean(1)
ringValues = np.clip(ringValues, 0, ceiling)
print(f'{len(endCells)} end cells (lowest {baseRing[endCells].min():.4f}, so they reach 1.439 at delta = {crossing:+.4f}); {len(otherCells)} other cells; '
      f'{len(codes)} codes ({len(deltas)} x {len(epsilons)})', flush=True)

target = np.asarray(run['target']).ravel()
results = boundary.scoreRingCodesOverTime(checkpoint, ringValues, hold, int(run['numIterations']), target, boundary.centreStripeCellIndices, batchSize=args.batchSize,
                                          onBatch=lambda done, r: print(f'[{time.time() - started:5.0f}s] {done}/{len(codes)}  best overlap so far {max(r["overlapAtBest"]):.3f}', flush=True))
result = dict(note='EXPLORATORY; no predictions registered.', trainedRunPath=args.trainedRunPath, coefficients=run['bestCoefficients'].tolist(), orders=[int(o) for o in orders],
              baseRing=baseRing.tolist(), ring=[int(c) for c in boundary.boundaryRingCells], endCells=[int(c) for c in endCells], otherCells=[int(c) for c in otherCells],
              endCellValues=baseRing[endCells].tolist(), bistableUpperEdge=1.439, crossingDelta=crossing, delta=deltas.tolist(), epsilon=epsilons.tolist(),
              clippedShare=np.round(clippedShare, 4).tolist(),
              **{key: [round(float(v), 5) for v in value] if key in ('score', 'overlapAtBest', 'maxOverlap') else [int(v) for v in value] for key, value in results.items()})
json.dump(result, open(outputPath, 'w'), separators=(',', ':'))
print('wrote', outputPath, f'({time.time() - started:.0f}s)', flush=True)
