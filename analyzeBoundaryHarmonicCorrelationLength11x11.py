"""How finely does the code decide the pattern? (PolyPatterning_Sim.md, Section 12).

A10 found that no smooth fit on the coefficients predicts the pattern once the face has formed. That could mean the
map is random, or that it is deterministic with a correlation length shorter than the ensemble's spacing. This script
tells the two apart by reading slices through code space sampled at several spacings and measuring how quickly two
codes' outcomes come apart as the codes separate.

For each slice it computes a variogram: the mean squared difference between outcomes against the distance between the
codes that made them. The correlation length is the distance at which that difference reaches half its plateau, which
is the scale below which the map is smooth. Outcome slices give it for the score; ownership slices, whose mode
amplitudes are stored, give it for the pattern at several moments.

Writes data/boundaryHarmonicCorrelationLength<checkpoint>Hold<hold><target>.json (never overwriting).
"""
import argparse
import glob
import json
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--outcomeGlob', type=str, default='data/boundaryHarmonicOutcomes1888Hold301FaceMinus60Minus5Slice*.json')
parser.add_argument('--ownershipGlob', type=str, default='data/boundaryHarmonicModeOwnership1888Hold301FaceMinus60Minus5BaselineSlice*.json')
parser.add_argument('--outputPath', type=str, default='data/boundaryHarmonicCorrelationLength1888Hold301FaceMinus60Minus5.json')
parser.add_argument('--numBins', type=int, default=24)
args = parser.parse_args()
if os.path.exists(args.outputPath):
    raise SystemExit(f'{args.outputPath} exists; not overwriting')


def variogram(codes, values, numBins):
    """Mean squared outcome difference against code distance, and the half-plateau distance."""
    varying = codes.std(0) > 1e-9
    positions = codes[:, varying]
    generator = np.random.default_rng(0)
    pairs = min(400000, len(codes) * (len(codes) - 1) // 2)
    first = generator.integers(0, len(codes), pairs)
    second = generator.integers(0, len(codes), pairs)
    keep = first != second
    first, second = first[keep], second[keep]
    distance = np.linalg.norm(positions[first] - positions[second], axis=1)
    difference = ((values[first] - values[second]) ** 2)
    if difference.ndim > 1:
        difference = difference.sum(1)
    edges = np.geomspace(max(distance[distance > 0].min(), 1e-6), distance.max(), numBins + 1)
    centres, means = [], []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (distance >= low) & (distance < high)
        if inside.sum() >= 30:
            centres.append(float(np.sqrt(low * high)))
            means.append(float(difference[inside].mean()))
    centres, means = np.array(centres), np.array(means)
    plateau = float(np.median(means[-max(3, len(means) // 4):])) if len(means) else float('nan')
    half = np.where(means >= plateau / 2)[0]
    correlationLength = float(centres[half[0]]) if len(half) else float('nan')
    return dict(distance=np.round(centres, 6).tolist(), meanSquaredDifference=np.round(means, 4).tolist(),
                plateau=round(plateau, 4), correlationLength=round(correlationLength, 6))


result = dict(outcomeSlices={}, patternSlices={})
for path in sorted(glob.glob(args.outcomeGlob)):
    data = json.load(open(path))
    codes, scores = np.array(data['codes']), np.array(data['bestScore'])
    spacing = float(np.diff(sorted(set(np.round(codes[:, 1], 8))))[0])
    name = f"halfWidth {data.get('sliceHalfWidth', 0.6)}"
    entry = variogram(codes, scores, args.numBins)
    entry.update(spacing=round(spacing, 6), numCodes=len(codes), scoreSpread=round(float(scores.std()), 4))
    labels = np.array(data['moments']['trainedMoment']['clusterLabels'])
    lookup = {(round(row[1], 8), round(row[2], 8)): index for index, row in enumerate(codes)}
    shared = total = 0
    for (first, second), index in lookup.items():
        for neighbour in ((round(first + spacing, 8), second), (first, round(second + spacing, 8))):
            other = lookup.get(neighbour)
            if other is not None:
                total += 1
                shared += int(labels[other] == labels[index])
    entry['neighboursSharingOutcome'] = round(shared / total, 4) if total else float('nan')
    result['outcomeSlices'][name] = entry
    print(f"score, {name} (spacing {spacing:.5f}): correlation length {entry['correlationLength']:.5f}, "
          f"neighbours sharing an outcome {entry['neighboursSharingOutcome']:.3f}", flush=True)

for path in sorted(glob.glob(args.ownershipGlob)):
    data = json.load(open(path))
    codes = np.array(data['codes'])
    spacing = float(np.diff(sorted(set(np.round(codes[:, 1], 8))))[0])
    name = f"halfWidth {data.get('sliceHalfWidth', 0.6)}"
    result['patternSlices'][name] = dict(spacing=round(spacing, 6), numCodes=len(codes), moments={})
    for moment, amplitudes in data.get('amplitudes', {}).items():
        entry = variogram(codes, np.array(amplitudes), args.numBins)
        result['patternSlices'][name]['moments'][moment] = entry
        print(f"pattern at iteration {moment}, {name} (spacing {spacing:.5f}): correlation length "
              f"{entry['correlationLength']:.5f}", flush=True)

json.dump(result, open(args.outputPath, 'w'))
print('wrote', args.outputPath)
