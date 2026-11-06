"""Scores the rejuvenation search (runRejuvenationSearch.sh) against the predictions registered in data/canalizationTalkRejuvenationPredictions1888Hold301.json
(committed before the trainer's --startState flag existed). Read that file for the protocol, the gates and the criteria.

For each target (stripe, order-3 face, order-4 face): the 16 restarts' best scores and overlaps at their best moments, the baselines (the original code held on the aged state, 64 random
allowed codes of the family on it, the best score of the first training from the uniform start), the gates G1 (some restart reaches overlap 0.9) and G2 (that code scores 5 mV below the best
random code), and V1 (the new replay path reproduces the stored score of the original code from the uniform start). EXPLORATORY additions, labelled as such in the output: the best code
run from the uniform start, and how much of the target's family (stripe family, face family) the best code produces.

    python3 analyzeCanalizationTalkRejuvenation11x11.py

Writes data/canalizationTalkRejuvenationScores1888Hold301.json (never overwriting).
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from scipy.optimize import linprog

import boundaryCodeUtilities as boundary
import canalizationTalkCommon as common
import canalizationTalkFamilies as fam
from boundaryHarmonicStep import ringHoldBatchReplayFromState

parser = argparse.ArgumentParser()
parser.add_argument('--outputPath', type=str, default='data/canalizationTalkRejuvenationScores1888Hold301.json')
parser.add_argument('--seed', type=int, default=20261107)
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')

HOLD, ITERATIONS = 301, 3000
OVERLAP_TARGET, G2_MARGIN, SCORE_GAP, CODE_DIFFERENCE = 0.9, 5.0, 2.0, 0.1
angles = boundary.ringAngles(boundary.boundaryRingCells)
labels = {'Stripe': dict(key='stripe', order=3, orders=3, ceiling=2.0), 'FaceOrder3': dict(key='face', order=3, orders=4, ceiling=1.3), 'FaceOrder4': dict(key='face', order=4, orders=5, ceiling=1.3)}
rng = np.random.default_rng(args.seed)


def randomAllowedCode(basis, ceiling, steps=500):
    """A code drawn uniformly from the polytope 0 <= basis @ a <= ceiling by hit-and-run (as in learnBoundaryHarmonics11x11.py)."""
    matrix, bound = np.vstack([basis, -basis]), np.r_[np.full(len(basis), ceiling), np.zeros(len(basis))]
    n = basis.shape[1]
    norms = np.linalg.norm(matrix, axis=1)
    point = linprog(np.r_[np.zeros(n), -1], A_ub=np.c_[matrix, norms], b_ub=bound, bounds=[(None, None)] * n + [(0, None)]).x[:n]
    for _ in range(steps):
        direction = rng.standard_normal(n)
        direction /= np.linalg.norm(direction)
        rate, slack = matrix @ direction, bound - matrix @ point
        upper = np.min(slack[rate > 1e-12] / rate[rate > 1e-12])
        lower = np.max(slack[rate < -1e-12] / rate[rate < -1e-12])
        point = point + rng.uniform(lower, upper) * direction
    return point


class Context:
    """One target: its cells, its score, and replays of ring codes from the aged state or from the uniform start."""

    def __init__(self, label, spec):
        self.label, self.spec = label, spec
        self.state = dict(np.load(f'data/agedTissueState1888Hold301{label}After300.npz', allow_pickle=True))
        self.cells = np.asarray(boundary.centreStripeCellIndices if spec['key'] == 'stripe' else boundary.featureCellIndices)
        self.original = np.asarray(self.state['coefficients'], float)
        if spec['key'] == 'stripe':
            runPath = common.STRIPE_RUN_PATH
        else:                                                                         # the best face code of this order in the training summary
            summary = json.load(open('data/boundaryHarmonicTrainingSummary1888Hold301FaceMinus60Minus5.json'))
            winner = summary['orders'][str(spec['order'] if spec['key'] == 'face' else 3)]['best']
            runPath = f"{summary['trainingDirs'][winner['round']]}/order{spec['order']}_restart{winner['restart']:02d}.npz"
        run = np.load(runPath, allow_pickle=True)
        assert np.allclose(run['bestCoefficients'], self.original), runPath
        self.firstTrainingScore, self.firstTrainingBest = float(run['bestScore']), int(run['bestIteration'])
        target = np.full(boundary.numCells, -5.0)
        target[self.cells] = -60.0
        self.targetTensor = torch.tensor(target)
        self.featureMask = np.isin(np.arange(boundary.numCells), self.cells)
        self.basis = np.cos(np.outer(angles, np.arange(spec['orders'])))

    def ring(self, coefficients):
        return np.clip(self.basis @ np.asarray(coefficients, float).T, 0, self.spec['ceiling']).T

    def evaluate(self, coefficientRows, aged=True):
        """(score, best iteration, pattern at it, overlap with the target at it) for each code, held for 301 iterations from the aged state (or the uniform start) and released."""
        ringValues = self.ring(coefficientRows)
        count = len(ringValues)
        best = dict(score=torch.full((count,), np.inf, dtype=torch.double), iteration=torch.zeros(count, dtype=torch.long), vmem=torch.zeros(count, boundary.numCells, dtype=torch.double))

        def onIteration(iteration, vmem):
            if iteration >= HOLD:
                squared = (vmem - self.targetTensor) ** 2
                scores = (squared[:, self.featureMask].mean(1).sqrt() + squared[:, ~self.featureMask].mean(1).sqrt()) / 2
                better = scores < best['score']
                best['score'] = torch.where(better, scores, best['score'])
                best['iteration'][better] = iteration
                best['vmem'][better] = vmem[better]
        ringHoldBatchReplayFromState(ringValues, HOLD, ITERATIONS, onIteration, self.state['vmemVolts'] if aged else None, self.state['gpol'] if aged else None)
        patterns = best['vmem'].numpy()
        overlaps = [boundary.structuralIntersectionOverUnion(p, self.cells) for p in patterns]
        return best['score'].numpy(), best['iteration'].numpy(), patterns, np.array(overlaps)

    def family(self, coefficients, aged=True):
        """EXPLORATORY: peak exact IoU and family share of one code's run (the registered measures of the re-canalization runs, dark sets every 5 iterations)."""
        frames = []

        def onIteration(iteration, vmem):
            if iteration % 5 == 0:
                frames.append(vmem[0].numpy()[np.array(common.INTERIOR)] < common.THRESHOLD)
        ringHoldBatchReplayFromState(self.ring(np.array([coefficients])), HOLD, ITERATIONS, onIteration, self.state['vmemVolts'] if aged else None, self.state['gpol'] if aged else None)
        bars, faceBest, exact = fam.frameFlags(np.array(frames))
        tau = np.arange(len(frames)) * 5
        window = (tau >= 301) & (tau < 3000)
        flag = (bars > 0) if self.spec['key'] == 'stripe' else (faceBest >= 0.8)
        return dict(peakExact=float(exact[self.spec['key']].max()), familyShare=float(flag[window].mean()), distinctPatterns=int(len({f.tobytes() for f in frames})))


results = dict(note='EXPLORATORY; scored against data/canalizationTalkRejuvenationPredictions1888Hold301.json', targets={}, predictions={}, validation={})
for label, spec in labels.items():
    context = Context(label, spec)
    files = sorted(glob.glob(f'data/boundaryHarmonicRejuvenationTraining1888Hold301{label}After300/order*_restart*.npz'))
    restarts = []
    for path in files:
        run = np.load(path, allow_pickle=True)
        coefficients = np.asarray(run['bestCoefficients'], float)
        restarts.append(dict(restart=int(run['restart']), start=str(run['startType']), score=float(run['bestScore']), bestIteration=int(run['bestIteration']),
                             overlap=boundary.structuralIntersectionOverUnion(np.asarray(run['bestVmem'], float), context.cells), coefficients=coefficients.tolist(),
                             evaluations=int(run['numEvaluations'])))
    originalScore, originalIteration, _, originalOverlap = context.evaluate(context.original[None])
    randomCodes = np.array([randomAllowedCode(context.basis, spec['ceiling']) for _ in range(64)])
    randomScores, _, _, randomOverlaps = context.evaluate(randomCodes)
    reached = [r for r in restarts if r['overlap'] >= OVERLAP_TARGET]
    bestReached = min(reached, key=lambda r: r['score']) if reached else None
    bestRestart = min(restarts, key=lambda r: r['score']) if restarts else None
    entry = dict(
        restartsFinished=len(restarts), restarts=restarts,
        originalOnAgedState=dict(score=float(originalScore[0]), bestIteration=int(originalIteration[0]), overlap=float(originalOverlap[0])),
        random64=dict(bestScore=float(randomScores.min()), medianScore=float(np.median(randomScores)), bestOverlap=float(randomOverlaps.max())),
        firstTrainingBestScore=context.firstTrainingScore,
        bestAgedScore=bestRestart['score'] if bestRestart else None, bestAgedOverlap=bestRestart['overlap'] if bestRestart else None,
        restartsReachingOverlap0p9=len(reached), G1=bool(reached),
        G2=bool(bestReached and bestReached['score'] <= float(randomScores.min()) - G2_MARGIN),
        bestCodeReachingOverlap0p9=bestReached,
        largestCoefficientDifferenceFromOriginal=float(np.abs(np.array(bestReached['coefficients']) - context.original).max()) if bestReached else None)
    if bestRestart:                                                                                   # EXPLORATORY additions
        entry['exploratory'] = dict(
            bestRestartFromAgedState=context.family(bestRestart['coefficients']),
            bestRestartFromUniformStart=dict(zip(('score', 'bestIteration', 'pattern', 'overlap'), [float(v[0]) if k != 'pattern' else None for k, v in zip(('score', 'bestIteration', 'pattern', 'overlap'), context.evaluate(np.array([bestRestart['coefficients']]), aged=False))])))
    # V1: the new path from the uniform start reproduces the stored score and best moment of the original code
    uniform = context.evaluate(context.original[None], aged=False)
    entry['V1'] = dict(score=float(uniform[0][0]), bestIteration=int(uniform[1][0]), storedScore=context.firstTrainingScore, storedBestIteration=context.firstTrainingBest,
                       holds=abs(float(uniform[0][0]) - context.firstTrainingScore) <= 0.15 and abs(int(uniform[1][0]) - context.firstTrainingBest) <= 5)
    results['targets'][label] = entry
    results['validation'][label] = entry['V1']['holds']

T = results['targets']
P = results['predictions']
P['R1'] = dict(holds=all(t['originalOnAgedState']['overlap'] < OVERLAP_TARGET and t['originalOnAgedState']['score'] > t['random64']['bestScore'] - G2_MARGIN for t in T.values()),
               observed={k: t['originalOnAgedState'] for k, t in T.items()})
P['R2'] = dict(holds=bool(T['Stripe']['G1'] and T['Stripe']['G2']), G1=T['Stripe']['G1'], G2=T['Stripe']['G2'])
P['R3'] = dict(holds=not (T['FaceOrder3']['G1'] or T['FaceOrder4']['G1']), G1={k: T[k]['G1'] for k in ('FaceOrder3', 'FaceOrder4')})
P['R4'] = dict(holds=all(t['bestAgedScore'] is not None and t['bestAgedScore'] - t['firstTrainingBestScore'] >= SCORE_GAP for t in T.values()),
               observed={k: dict(bestAged=t['bestAgedScore'], firstTraining=t['firstTrainingBestScore']) for k, t in T.items()})
passing = {k: t for k, t in T.items() if t['G1']}
P['R5'] = dict(holds=all(t['largestCoefficientDifferenceFromOriginal'] > CODE_DIFFERENCE for t in passing.values()) if passing else None, applicable=bool(passing),
               observed={k: t['largestCoefficientDifferenceFromOriginal'] for k, t in passing.items()})
results['summary'] = {k: v['holds'] for k, v in P.items()}
json.dump(results, open(args.outputPath, 'w'), indent=1)

print('V1 (new path reproduces the first training):', results['validation'])
print(f'\n{"":11}{"restarts":>9}{"best aged":>11}{"overlap":>9}{"first train":>12}{"original":>10}{"(overlap)":>10}{"random best":>12}{"reach 0.9":>10}')
for label, t in T.items():
    print(f'{label:11}{t["restartsFinished"]:9d}{t["bestAgedScore"]:11.2f}{t["bestAgedOverlap"]:9.2f}{t["firstTrainingBestScore"]:12.2f}{t["originalOnAgedState"]["score"]:10.2f}'
          f'{t["originalOnAgedState"]["overlap"]:10.2f}{t["random64"]["bestScore"]:12.2f}{t["restartsReachingOverlap0p9"]:10d}   G1 {t["G1"]}  G2 {t["G2"]}')
print('\npredictions:', results['summary'])
print('wrote', args.outputPath)
