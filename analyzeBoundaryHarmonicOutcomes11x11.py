"""If the code does not paint the pattern, what does it choose? (PolyPatterning_Sim.md, Section 12).

Replays an ensemble of codes and asks how many distinct outcomes the tissue actually produces. Outcomes are compared
as the binarised pattern (cells below the hyperpolarised threshold) at a fixed moment, and as hierarchical clusters of
the Vmem patterns themselves. If a large ensemble of codes yields few distinct outcomes, the code selects among
branches rather than specifying a shape; the selection's predictability from the coefficients is then measured by
nearest-neighbour classification with five-fold cross-validation.

--sampling random draws codes across the feasible region; --sampling slice sweeps a_1 and a_2 on a grid with a_0 and
a_3 at the trained code's values, giving a map of code space.

Writes data/boundaryHarmonicOutcomes<checkpoint>Hold<hold><target><Sampling>.json (never overwriting).
"""
import argparse
import json
import os

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--trainedRunPath', type=str, default='data/boundaryHarmonicTraining1888Hold301FaceMinus60Minus5/order3_restart08.npz')
parser.add_argument('--sampling', type=str, default='random', choices=('random', 'slice'))
parser.add_argument('--numCodes', type=int, default=1024)
parser.add_argument('--gridSize', type=int, default=48)
parser.add_argument('--clusterThresholdMilliVolts', type=float, default=5.0)
parser.add_argument('--seed', type=int, default=17)
args = parser.parse_args()

run = dict(np.load(args.trainedRunPath))
outputPath = (f"data/boundaryHarmonicOutcomes{int(run['referenceCheckpoint'])}Hold{int(run['holdIterations'])}"
              f"{run['targetName']}{args.sampling[0].upper()}{args.sampling[1:]}.json")
if os.path.exists(outputPath):
    raise SystemExit(f'{outputPath} exists; not overwriting')
reference = boundary.loadCheckpoint(int(run['referenceCheckpoint']))
hold, numIterations = int(run['holdIterations']), int(run['numIterations'])
code, trainedMoment = run['bestCoefficients'], int(run['bestIteration'])
target = np.asarray(run['target']).ravel()
angles = boundary.ringAngles(boundary.boundaryRingCells)
basis = np.cos(np.outer(angles, np.arange(len(code))))
featureMask = np.isin(np.arange(boundary.numCells), boundary.featureCellIndices)

generator = np.random.default_rng(args.seed)
if args.sampling == 'random':
    codes = []
    while len(codes) < args.numCodes:
        candidate = np.concatenate([generator.uniform(0.3, 1.3, 1), generator.uniform(-0.6, 0.6, len(code) - 1)])
        if (basis @ candidate).min() >= 0.02 and (basis @ candidate).max() <= 1.98:
            codes.append(candidate)
    codes, sliceAxes = np.array(codes), None
else:
    firstAxis = np.linspace(-0.6, 0.6, args.gridSize)
    secondAxis = np.linspace(-0.6, 0.6, args.gridSize)
    codes = np.array([[code[0], first, second, code[3]] for first in firstAxis for second in secondAxis])
    feasible = np.array([(basis @ row).min() >= 0.0 and (basis @ row).max() <= 2.0 for row in codes])
    codes, sliceAxes = codes[feasible], dict(a1=firstAxis.tolist(), a2=secondAxis.tolist())
    print(f'{feasible.sum()} of {len(feasible)} grid codes feasible', flush=True)

ringValues = np.clip(codes @ basis.T, 0, 2)
print(f'{len(codes)} codes; replaying {numIterations} iterations', flush=True)
scored = dict(best=np.full(len(codes), np.inf), iteration=np.zeros(len(codes), dtype=int),
              vmem=np.zeros((len(codes), boundary.numCells)), atTrained=None)


def onIteration(iteration, vmem):
    if iteration < hold:
        return
    values = vmem.numpy()
    squared = (values - target) ** 2
    scores = 0.5 * np.sqrt(squared[:, featureMask].mean(1)) + 0.5 * np.sqrt(squared[:, ~featureMask].mean(1))
    better = scores < scored['best']
    scored['best'][better], scored['iteration'][better], scored['vmem'][better] = scores[better], iteration, values[better]
    if iteration == trainedMoment:
        scored['atTrained'] = values.copy()


boundary.ringHoldBatchReplay(reference, ringValues, hold, numIterations, onIteration)

# ---------------------------------------------------------------------------------------- the outcomes
result = dict(sampling=args.sampling, numCodes=len(codes), trainedMoment=trainedMoment, codes=np.round(codes, 5).tolist(),
              sliceAxes=sliceAxes, clusterThresholdMilliVolts=args.clusterThresholdMilliVolts,
              bestScore=np.round(scored['best'], 3).tolist(), bestIteration=scored['iteration'].tolist(), moments={})
for name, values in (('trainedMoment', scored['atTrained']), ('ownBest', scored['vmem'])):
    dark = values < boundary.hyperpolarizedThresholdMilliVolts
    shapes, labels, counts = np.unique(dark, axis=0, return_inverse=True, return_counts=True)
    distances = pdist(values)
    clusters = fcluster(linkage(distances, method='average'), t=args.clusterThresholdMilliVolts, criterion='distance')
    clusterCounts = np.bincount(clusters)[1:]
    order = np.argsort(-clusterCounts)
    representatives = {}
    for rank, clusterIndex in enumerate(order[:8]):
        members = np.where(clusters == clusterIndex + 1)[0]
        representatives[str(rank)] = dict(size=int(len(members)), meanScore=round(float(scored['best'][members].mean()), 3),
                                          bestScore=round(float(scored['best'][members].min()), 3),
                                          vmem=np.round(values[members].mean(0), 1).tolist())
    # how well the coefficients predict which cluster a code lands in, five-fold nearest-neighbour
    standardized = (codes - codes.mean(0)) / codes.std(0)
    folds = generator.permutation(len(codes)) % 5
    correct = 0
    for fold in range(5):
        train, test = folds != fold, folds == fold
        for index in np.where(test)[0]:
            distancesToTrain = np.linalg.norm(standardized[train] - standardized[index], axis=1)
            neighbours = clusters[train][np.argsort(distancesToTrain)[:5]]
            correct += int(np.bincount(neighbours).argmax() == clusters[index])
    largest = clusterCounts.max() / len(codes)
    result['moments'][name] = dict(numDistinctShapes=int(len(shapes)), shapeCounts=sorted(counts.tolist(), reverse=True)[:20],
                                   numClusters=int(clusters.max()), clusterSizes=sorted(clusterCounts.tolist(), reverse=True)[:20],
                                   largestClusterShare=round(float(largest), 4), clusterLabels=clusters.tolist(),
                                   shapeLabels=labels.tolist(), representatives=representatives,
                                   nearestNeighbourAccuracy=round(correct / len(codes), 4), chanceAccuracy=round(float(largest), 4))
    summary = result['moments'][name]
    print(f"{name}: {summary['numDistinctShapes']} distinct dark-cell shapes, {summary['numClusters']} clusters at "
          f"{args.clusterThresholdMilliVolts} mV (largest {summary['largestClusterShare']:.2f}); coefficients predict the cluster "
          f"{summary['nearestNeighbourAccuracy']:.3f} vs chance {summary['chanceAccuracy']:.3f}", flush=True)

json.dump(result, open(outputPath, 'w'))
print('wrote', outputPath, flush=True)
