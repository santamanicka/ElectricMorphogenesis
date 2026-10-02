"""The stripe code's program, scored against its registration (P1-P5 of boundaryHarmonicStripeMechanismPredictions...json).

Reads the replay of the best code of every size (recordBoundaryHarmonicRuns11x11.py --mode trained: Vmem and G_pol at every
iteration) and the switch-rule fit of it (analyzeBoundaryHarmonicSwitchRule11x11.py), picks the stripe code's size, and measures:

  trajectory   the interior, stripe-cell and flank-cell mean conductance, the number of dark interior cells and the overlap with the
               stripe, every --traceStride iterations, with the first peak (inside the hold), the trough (lowest interior mean
               between iterations 302 and 1300) and the second peak (highest after the trough) of the whole run
  P1           the switch rule's one-step accuracy for the stripe code and for the best code of every other size
  P2           whether the stripe code's best moment comes before its trough
  P3           the selectivity (stripe minus flank mean conductance, G_ref) at the best moment
  P4           of the interior cells dark at the last iteration of the hold and light at the best moment, the share that are flank cells
  P5           how many of the 27 stripe cells are dark at the last iteration of the hold
and the order in which the stripe cells cross the dark threshold, for the report's figure.

    python3 analyzeBoundaryHarmonicStripeProgram11x11.py --recordPath <record.npz> --switchRulePath <switchRule.json> --summaryPath <summary.json>

Writes data/boundaryHarmonicStripeProgram<rest of the summary's name> (never overwriting).
"""
import argparse
import json
import os

import numpy as np

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--recordPath', type=str, required=True)
parser.add_argument('--switchRulePath', type=str, required=True)
parser.add_argument('--summaryPath', type=str, required=True)
parser.add_argument('--order', type=int, default=2, help='the stripe code\'s size')
parser.add_argument('--traceStride', type=int, default=5)
parser.add_argument('--predictionsPath', type=str, default='data/boundaryHarmonicStripeMechanismPredictions1888Hold301StripesInteriorMinus60Minus5.json')
args = parser.parse_args()

outputPath = args.summaryPath.replace('boundaryHarmonicTrainingSummary', 'boundaryHarmonicStripeProgram')
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
record = np.load(args.recordPath)
orders = [int(o) for o in record['orders']]
index = orders.index(args.order)
hold = int(record['hold'])
best = int(record['bestIterations'][index])
voltage, conductance = record['vmem'][index].astype(float), record['gpol'][index].astype(float)
numIterations = len(voltage)
interior = np.array(boundary.interiorCellIndices)
stripe = np.array(sorted(boundary.centreStripeCellIndices.tolist()))
flank = np.array([c for c in interior if c not in set(stripe.tolist())])
threshold = boundary.hyperpolarizedThresholdMilliVolts

interiorMean = conductance[:, interior].mean(1)
firstPeak = int(interiorMean[:hold + 1].argmax())
trough = hold + 1 + int(interiorMean[hold + 1:1300].argmin())
secondPeak = trough + int(interiorMean[trough:].argmax())
times = list(range(0, numIterations, args.traceStride))
overlap = np.array([boundary.structuralIntersectionOverUnion(voltage[t], stripe) for t in range(numIterations)])
dark = voltage[:, interior] < threshold
darkAtHoldEnd, darkAtBest = dark[hold - 1], dark[best]
stripeDarkAtHoldEnd = int((voltage[hold - 1, stripe] < threshold).sum())
selectivity = float(conductance[best, stripe].mean() - conductance[best, flank].mean())
changed = np.flatnonzero(darkAtHoldEnd & ~darkAtBest)
changedFlank = int(sum(1 for k in changed if interior[k] not in set(stripe.tolist())))
formed = np.flatnonzero(overlap >= 0.9)

# when each stripe cell first goes (and stays) dark through the run, by iteration of its first crossing
firstDark = {int(c): (int(np.argmax(voltage[:, c] < threshold)) if (voltage[:, c] < threshold).any() else None) for c in stripe}

switchRule = json.load(open(args.switchRulePath))
accuracy = {str(o): switchRule['runs'][str(o)]['accuracy'] for o in orders if str(o) in switchRule['runs']}
verdicts = dict(
    P1=dict(accuracyByOrder=accuracy, stripeCode=accuracy.get(str(args.order)), holds=bool(accuracy and min(accuracy.values()) >= 0.99)),
    P2=dict(bestMoment=best, trough=trough, firstPeak=firstPeak, secondPeak=secondPeak, holds=bool(best < trough)),
    P3=dict(selectivity=selectivity, holds=bool(selectivity >= 0.20)),
    P4=dict(darkAtHoldEndLightAtBest=int(len(changed)), ofWhichFlank=changedFlank,
            flankShare=(changedFlank / len(changed)) if len(changed) else None,
            applicable=bool(len(changed) >= 5), holds=bool(len(changed) >= 5 and changedFlank / len(changed) >= 0.8) if len(changed) >= 5 else None),
    P5=dict(stripeCellsDarkAtHoldEnd=stripeDarkAtHoldEnd, of=len(stripe), holds=bool(stripeDarkAtHoldEnd >= 24)))
for key, verdict in verdicts.items():
    print(f"{key}: {'holds' if verdict['holds'] else ('not applicable' if verdict['holds'] is None else 'FAILS')}",
          {k: (round(v, 4) if isinstance(v, float) else v) for k, v in verdict.items() if k != 'holds'}, flush=True)

result = dict(
    note='CONFIRMATORY for P1-P5 (criteria registered before this run); the trajectory and the crossing order are descriptive.',
    order=args.order, hold=hold, bestMoment=best, numIterations=numIterations, predictions=json.load(open(args.predictionsPath)),
    verdicts=verdicts, landmarks=dict(firstPeak=firstPeak, trough=trough, secondPeak=secondPeak, firstPeakValue=float(interiorMean[firstPeak]),
                                      troughValue=float(interiorMean[trough]), secondPeakValue=float(interiorMean[secondPeak])),
    stripeFormedIterations=dict(count=int(len(formed)), first=int(formed.min()) if len(formed) else None, last=int(formed.max()) if len(formed) else None),
    times=times, interiorMean=[round(float(interiorMean[t]), 4) for t in times],
    stripeMean=[round(float(conductance[t, stripe].mean()), 4) for t in times], flankMean=[round(float(conductance[t, flank].mean()), 4) for t in times],
    darkCount=[int(dark[t].sum()) for t in times], stripeDarkCount=[int((voltage[t, stripe] < threshold).sum()) for t in times],
    overlap=[round(float(overlap[t]), 4) for t in times], firstDarkIteration=firstDark,
    snapshots=[dict(iteration=int(t), vmem=[round(float(v), 1) for v in voltage[t]], gpol=[round(float(v), 3) for v in conductance[t]])
               for t in sorted({0, 150, hold - 1, hold + 50, best, trough, secondPeak})])
json.dump(result, open(outputPath, 'w'), separators=(',', ':'))
print('wrote', outputPath, flush=True)
