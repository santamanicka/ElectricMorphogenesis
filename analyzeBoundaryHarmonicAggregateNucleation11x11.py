"""Score the aggregate-nucleation test against its registered predictions.

Predictions and criteria: data/boundaryHarmonicAggregateNucleationPredictions1888Hold301FaceMinus60Minus5.json,
committed before the runner existed. One of them, P3, is void: as registered, its control pool was "interior
cells dark at some point after 1200 but not nucleators", which is exactly the recruit set, so those draws block
the very cells they then score. It is reported as void and an amended pool -- cells that go dark at some point
but are neither nucleators nor scored recruits -- is reported beside it, labelled as the amendment it is.

    python3 analyzeBoundaryHarmonicAggregateNucleation11x11.py --runPath <aggregate.npz>
"""
import argparse
import json

import numpy as np
from scipy import stats

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--runPath', type=str, required=True, help='npz from runBoundaryHarmonicAggregateNucleation11x11.py')
parser.add_argument('--predictionsPath', type=str,
                    default='data/boundaryHarmonicAggregateNucleationPredictions1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--outputPath', type=str,
                    default='data/boundaryHarmonicAggregateNucleation1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--peak', type=int, default=1765, help="the untouched run's second peak")
args = parser.parse_args()

run = np.load(args.runPath)
vmem, gpol, kinds = run['vmem'], run['gpol'], run['kinds']
best = int(run['best'])
nucleators = [int(c) for c in run['nucleators']]
recruits = [int(c) for c in run['recruits']]
dark = vmem[:, best] < boundary.hyperpolarizedThresholdMilliVolts
index = {str(k): i for i, k in enumerate(kinds)}
baseline = dark[index['baseline']]
interior = [int(c) for c in boundary.interiorCellIndices]
featureCells = set(int(c) for c in boundary.featureCellIndices.tolist())

scored = [c for c in recruits if baseline[c]]          # only recruits the untouched run actually leaves dark


def surviving(name):
    return int(sum(1 for c in scored if dark[index[name]][c]))


def driveAt(name):
    return float(gpol[index[name], args.peak, scored].mean())


def quality(name):
    got = {c for c in interior if dark[index[name]][c]}
    return round(len(got & featureCells) / len(got | featureCells), 3)


ladder = [dict(blocked=k, surviving=surviving('baseline' if k == 0 else
                                              ('allNucleators' if k == len(nucleators) else f'ladder{k}')))
          for k in (0, 2, 4, 6, 8, len(nucleators))]
rho, pValue = stats.spearmanr([row['blocked'] for row in ladder], [row['surviving'] for row in ladder])

registered = sorted(surviving(k) for k in index if k.startswith('random'))
amended = sorted(surviving(k) for k in index if k.startswith('amended'))
amendedDrive = float(np.mean([gpol[index[k], args.peak, scored].mean() for k in index if k.startswith('amended')]))
amendedQuality = round(float(np.mean([quality(k) for k in index if k.startswith('amended')])), 3)

allNucleators, base = surviving('allNucleators'), len(scored)
noopChanged = int(sum(1 for c in interior if dark[index['noop']][c] != baseline[c]))

result = dict(
    predictions=json.load(open(args.predictionsPath))['predictions'],
    nucleators=nucleators, recruits=recruits, scoredRecruits=scored, peak=args.peak,
    conditions=dict(
        baseline=dict(surviving=base, drive=round(driveAt('baseline'), 4), quality=quality('baseline')),
        allNucleators=dict(surviving=allNucleators, drive=round(driveAt('allNucleators'), 4),
                           quality=quality('allNucleators')),
        noop=dict(surviving=surviving('noop'), drive=round(driveAt('noop'), 4), quality=quality('noop'),
                  interiorChanged=noopChanged)),
    ladder=ladder, ladderSpearman=dict(rho=round(float(rho), 3), p=round(float(pValue), 3)),
    controls=dict(
        registered=dict(void=True,
                        why='its pool is the recruit set itself, so each draw blocks cells it then scores',
                        draws=registered, median=float(np.median(registered))),
        amended=dict(amendment=True,
                     why='drawn from cells that go dark at some point but are neither nucleators nor scored recruits',
                     draws=amended, median=float(np.median(amended)),
                     percentile10=float(np.percentile(amended, 10)), percentile90=float(np.percentile(amended, 90)),
                     drive=round(amendedDrive, 4), quality=amendedQuality)),
    verdicts=dict(
        P1=dict(surviving=allNucleators, of=base, criterion=f'<= {base // 2}', holds=allNucleators <= base / 2),
        P2=dict(rho=round(float(rho), 3), p=round(float(pValue), 3), holds=bool(rho <= -0.8 and pValue < 0.05)),
        P3=dict(void=True, holds=None,
                amendedComparison=dict(allNucleators=allNucleators, controlMedian=float(np.median(amended)),
                                       aboveEveryDraw=bool(allNucleators > max(amended)))),
        P4=dict(delta=round(driveAt('allNucleators') - driveAt('baseline'), 4),
                noopDelta=round(driveAt('noop') - driveAt('baseline'), 4),
                criterion='baseline minus blocked > 0.02, and |no-op delta| < 0.001',
                holds=bool((driveAt('baseline') - driveAt('allNucleators')) > 0.02
                           and abs(driveAt('noop') - driveAt('baseline')) < 0.001)),
        P5=dict(interiorChanged=noopChanged, holds=noopChanged == 0)))
json.dump(result, open(args.outputPath, 'w'), indent=1)

for name, verdict in result['verdicts'].items():
    print(f"{name}: {'VOID' if verdict.get('void') else ('holds' if verdict['holds'] else 'FAILS')}  {verdict}", flush=True)
print('wrote', args.outputPath, flush=True)
