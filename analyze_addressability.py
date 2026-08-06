"""
Measure ADDRESSABILITY across a field action range sweep.

Participation ratio, Gaussian TSE and spectral entropy are all functions of correlation
structure. They report how much variety a tissue generates, and are blind to whether the
boundary code organises that variety -- which is how PR came to report ~163 effective
dimensions of interior pattern at 30x30 that carried no recoverable relation to the code.

Addressability asks the missing question directly: do similar boundary codes produce similar
interior patterns? It is measured two ways, neither assuming linearity:

  addressability index     For each sample, take its numNeighbours most similar codes, and ask
                           where each of those samples' patterns ranks among all patterns by
                           similarity. Reported as 1 - 2*meanRank/(N-1): 0 at chance, 1 if the
                           code-nearest samples are always the pattern-nearest. Every reported
                           index carries the standard deviation of its own permutation null, so
                           the noise floor is visible next to the estimate -- differences smaller
                           than that floor are not differences.
  Mantel rho               Spearman correlation between code distance and pattern distance over
                           all pairs.

Both are computed under several representations, because cell-wise Euclidean distance lets
fine-scale churn dominate a comparison that should be about tissue-spanning structure:

  raw          all boundary cells / all interior cells
  modes        top-k principal components of each space
  spatialFreq  low-frequency 2D DFT coefficients of the interior pattern (global modes only)

The clamp parameters are identical across screen sizes, but the G_pol code they write is not,
since the field modulates conductance during the clamp window. Both are therefore reported:
clamp->pattern is exactly comparable across screen sizes, G_pol->pattern is the mechanistically
meaningful relation within one.
"""

import argparse, ast
import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import utilities
from embryo import model

parser = argparse.ArgumentParser()
parser.add_argument('--sweepDir',     type=str, default='data/fieldRangeSweep')
parser.add_argument('--screenSizes',  type=str, default='[5,6,8,10,11]')
parser.add_argument('--sourceDat',    type=str, default='data/StigmergicModelParameters_30x30.dat')
parser.add_argument('--modeVariance', type=float, default=0.80,
                    help='fraction of variance the modes representation retains, matched across '
                         'conditions rather than a fixed PC count')
parser.add_argument('--maxWavenumber',type=int, default=3, help='low-frequency DFT band half-width')
parser.add_argument('--numNeighbours',   type=int, default=5,
                    help='code-nearest neighbours averaged per sample; >1 because distances '
                         'concentrate in high-dimensional code space')
parser.add_argument('--numPermutations', type=int, default=400, help='permutations for the chance null')
parser.add_argument('--outputPrefix', type=str, default='data/addressability')
args = parser.parse_args()

screenSizes = ast.literal_eval(args.screenSizes)
PRIMARY_REPRESENTATION = 'G_pol -> raw'   # the code the clamp writes, against the full pattern
rng = np.random.default_rng(0)
torch.set_grad_enabled(False)

# ── Geometry ─────────────────────────────────────────────────────────────────
params = torch.load(args.sourceDat, weights_only=False)
numRows, numCols = params['latticeDims']
numCells = numRows * numCols
params['ATPParameters'] = None
params['latticePeriodicBoundaryGJ'] = False
iv = params['simParameters']['initialValues']
if 'ligandConc' not in iv:
    iv['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)
domeIndices = utilities.utilities().computeDomeIndices(model(params, 1).electricNetwork, mode='tissue')
boundaryMask = np.zeros(numCells, dtype=bool); boundaryMask[domeIndices] = True
interiorMask = ~boundaryMask

# ── Distance and addressability helpers ──────────────────────────────────────
def distanceMatrix(features):
    D = squareform(pdist(features))
    np.fill_diagonal(D, np.inf)
    return D

def addressabilityIndex(codeDistances, patternDistances, numNeighbours):
    """1 - 2*meanRank/(N-1) over each sample's numNeighbours code-nearest others.

    Mean rather than median: the median is a single order statistic, so it discards the shape of
    the rank distribution and barely moves until the well-addressed fraction crosses one half,
    which makes it both noisier and step-like. Averaging over k neighbours rather than trusting
    the single nearest one matters because code space is high-dimensional, where distances
    concentrate and the nearest code is barely nearer than the tenth -- making a single-neighbour
    statistic rest on a nearly arbitrary choice.
    """
    N = len(codeDistances)
    neighbours = np.argsort(codeDistances, axis=1)[:, :numNeighbours]
    thresholds = np.take_along_axis(patternDistances, neighbours, axis=1)      # (N, k)
    ranks = (patternDistances[:, None, :] < thresholds[:, :, None]).sum(axis=2)
    return 1.0 - 2.0 * ranks.mean() / (N - 1), float(np.median(ranks))

def chanceNull(codeDistances, patternDistances, numPermutations, numNeighbours):
    """Null from shuffling which pattern belongs to which code."""
    N = len(codeDistances)
    values = []
    for _ in range(numPermutations):
        order = rng.permutation(N)
        values.append(addressabilityIndex(codeDistances, patternDistances[np.ix_(order, order)],
                                          numNeighbours)[0])
    return float(np.mean(values)), float(np.std(values))

def principalModes(data, varianceFraction):
    """Retain the leading PCs explaining varianceFraction of this condition's own variance.

    A fixed mode count is not comparable across conditions whose dimensionality differs: ten
    modes hold 83% of the 11x11 interior variance but only 20% of the 30x30 range-5 interior,
    so a fixed count silently discards most of the pattern exactly where pattern is richest.
    Matching the retained variance fraction instead keeps the representations comparable.
    """
    centred = data - data.mean(axis=0)
    fitted = PCA(n_components=min(data.shape[0] - 1, data.shape[1])).fit(centred)
    cumulative = np.cumsum(fitted.explained_variance_ratio_)
    numModes = int(np.searchsorted(cumulative, varianceFraction) + 1)
    return fitted.transform(centred)[:, :numModes], numModes

def lowFrequencyModes(patterns, side, maxWavenumber):
    """Complex DFT coefficients within |k| <= maxWavenumber, keeping phase (hence position)."""
    grids = patterns.reshape(len(patterns), side, side)
    spectrum = np.fft.fft2(grids, axes=(1, 2))
    band = maxWavenumber + 1
    block = np.concatenate([spectrum[:, :band, :band], spectrum[:, -maxWavenumber:, :band]], axis=1)
    return np.concatenate([block.real.reshape(len(patterns), -1),
                           block.imag.reshape(len(patterns), -1)], axis=1)

def participationRatio(data):
    centred = data - data.mean(axis=0)
    eigenvalues = np.linalg.svd(centred, compute_uv=False) ** 2
    return eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()

# ── Per screen size ──────────────────────────────────────────────────────────
results = []
for screenSize in screenSizes:
    prefix = f'{args.sweepDir}/screen{screenSize:02d}'
    gpol = np.load(f'{prefix}_gpol_prepatterns.npy')
    vmem = np.load(f'{prefix}_vmem_final.npy')
    clampFile = np.load(f'{prefix}_clamp_params.npz')
    N = len(vmem)

    interior = vmem[:, interiorMask]
    boundaryCode = gpol[:, boundaryMask]
    clampFeatures = np.concatenate([clampFile['frequencies'], clampFile['amplitudes'],
                                    np.cos(clampFile['phases']), np.sin(clampFile['phases'])], axis=1)
    clampFeatures = (clampFeatures - clampFeatures.mean(0)) / (clampFeatures.std(0) + 1e-12)

    interiorSide = numRows - 2
    interiorSquare = vmem.reshape(N, numRows, numCols)[:, 1:numRows-1, 1:numCols-1].reshape(N, -1)

    interiorModes, numInteriorModes = principalModes(interior, args.modeVariance)
    codeModes, numCodeModes = principalModes(boundaryCode, args.modeVariance)
    representations = {
        'clamp -> raw':        (clampFeatures, interior),
        'G_pol -> raw':        (boundaryCode, interior),
        'G_pol -> modes':      (boundaryCode, interiorModes),
        'modes -> modes':      (codeModes, interiorModes),
        'modes -> spatialFreq':(codeModes, lowFrequencyModes(interiorSquare, interiorSide, args.maxWavenumber)),
    }

    row = {'screen': screenSize, 'N': N,
           'spatialStd': float(interior.std(axis=1).mean() * 1000.0),
           'PR': float(participationRatio(interior))}
    for name, (codeFeatures, patternFeatures) in representations.items():
        Dcode, Dpattern = distanceMatrix(codeFeatures), distanceMatrix(patternFeatures)
        index, medianRank = addressabilityIndex(Dcode, Dpattern, args.numNeighbours)
        nullMean, nullStd = chanceNull(Dcode, Dpattern, args.numPermutations, args.numNeighbours)
        rho = spearmanr(pdist(codeFeatures), pdist(patternFeatures))[0]
        row[name] = {'index': index, 'medianRank': medianRank, 'rho': float(rho),
                     'nullStd': nullStd,
                     'z': (index - nullMean) / nullStd if nullStd > 0 else np.nan}
    results.append(row)
    print(f"screen {screenSize:2d}: N={N}, spatialStd={row['spatialStd']:.2f} mV, PR={row['PR']:.1f}, "
          f"modes retaining {args.modeVariance:.0%} variance: {numCodeModes} code / {numInteriorModes} interior")

# ── Report ───────────────────────────────────────────────────────────────────
representationNames = [k for k in results[0] if isinstance(results[0][k], dict)]
print(f"\nAddressability index (0 = chance, 1 = perfect) +- null SD, z against a permutation null, "
      f"N={results[0]['N']}, k={args.numNeighbours} neighbours")
print(f"{'screen':>7} {'spatialStd':>11} {'PR':>7}  " + "  ".join(f"{n:>28}" for n in representationNames))
for row in results:
    cells = "  ".join(f"{row[n]['index']:>+7.3f}+-{row[n]['nullStd']:.3f} (z={row[n]['z']:>+4.1f})" for n in representationNames)
    print(f"{row['screen']:>7} {row['spatialStd']:>8.2f} mV {row['PR']:>7.1f}  {cells}")
print(f"\nMantel rho")
print(f"{'screen':>7}  " + "  ".join(f"{n:>22}" for n in representationNames))
for row in results:
    print(f"{row['screen']:>7}  " + "  ".join(f"{row[n]['rho']:>+22.3f}" for n in representationNames))
# ── Combined objective ───────────────────────────────────────────────────────
# Complexity alone rewards a tissue for generating variety the boundary does not control;
# addressability alone rewards a tissue for going blank, since a near-uniform tissue follows its
# boundary faithfully while having almost no vocabulary. The product penalises both failures.
#
# The maximum's LOCATION is scale-invariant -- rescaling PR cannot move it -- which is why this
# is a product to be maximised rather than two curves to be intersected: an intersection depends
# entirely on the relative scaling of two quantities in different units.
#
# The addressability term is a conservative lower bound, max(0, index - 2*nullSD), not the point
# estimate. Without it, an index statistically indistinguishable from zero multiplied by a large
# PR produces a spurious winner: at 30x30 range 5, +0.067 (z=+1.9) x 97.6 scores highest in the
# table on noise alone.
print(f"\n{'screen':>7} {'PR':>7} {'index':>8} {'nullSD':>8} {'lowerBound':>11} {'PR x lowerBound':>16}")
objective = []
for row in results:
    entry = row[PRIMARY_REPRESENTATION]
    lowerBound = max(0.0, entry['index'] - 2 * entry['nullStd'])
    value = row['PR'] * lowerBound
    objective.append((value, row['screen']))
    print(f"{row['screen']:>7} {row['PR']:>7.1f} {entry['index']:>+8.3f} {entry['nullStd']:>8.3f} "
          f"{lowerBound:>11.3f} {value:>16.2f}")
bestValue, bestScreen = max(objective)
print(f"\n  objective (PR x lower bound, from '{PRIMARY_REPRESENTATION}') peaks at "
      f"action range {bestScreen}, value {bestValue:.2f}")
if bestValue == 0:
    print("  WARNING: every lower bound is zero -- no condition shows addressability above noise.")

print(f"\nPR is capped by N-1 = {results[0]['N']-1}; it is reported for continuity with Section 6, "
      f"not as the headline.")

np.savez(f'{args.outputPrefix}.npz', results=np.array(results, dtype=object))
print(f"Saved {args.outputPrefix}.npz")
