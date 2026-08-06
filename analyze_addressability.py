"""
Measure ADDRESSABILITY across a field action range sweep.

Participation ratio, Gaussian TSE and spectral entropy are all functions of correlation
structure. They report how much variety a tissue generates, and are blind to whether the
boundary code organises that variety -- which is how PR came to report ~163 effective
dimensions of interior pattern at 30x30 that carried no recoverable relation to the code.

Addressability asks the missing question directly: do similar boundary codes produce similar
interior patterns? It is measured two ways, neither assuming linearity:

  nearest-neighbour index  For each sample, find the most similar code among the others, then
                           ask where that sample's pattern ranks among all patterns by
                           similarity. Reported as 1 - 2*medianRank/(N-1): 0 at chance, 1 if the
                           code-nearest sample is always the pattern-nearest.
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
parser.add_argument('--numModes',     type=int, default=10, help='PCs retained for the modes representation')
parser.add_argument('--maxWavenumber',type=int, default=3, help='low-frequency DFT band half-width')
parser.add_argument('--numPermutations', type=int, default=200, help='permutations for the chance null')
parser.add_argument('--outputPrefix', type=str, default='data/addressability')
args = parser.parse_args()

screenSizes = ast.literal_eval(args.screenSizes)
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

def nearestNeighbourIndex(codeDistances, patternDistances):
    """1 - 2*medianRank/(N-1): 0 at chance, 1 when the code-nearest sample is pattern-nearest."""
    N = len(codeDistances)
    codeNearest = codeDistances.argmin(axis=1)
    ranks = np.array([(patternDistances[k] < patternDistances[k, codeNearest[k]]).sum()
                      for k in range(N)])
    return 1.0 - 2.0 * np.median(ranks) / (N - 1), float(np.median(ranks))

def chanceNull(codeDistances, patternDistances, numPermutations):
    """Null from shuffling which pattern belongs to which code."""
    N = len(codeDistances)
    values = []
    for _ in range(numPermutations):
        order = rng.permutation(N)
        values.append(nearestNeighbourIndex(codeDistances, patternDistances[np.ix_(order, order)])[0])
    return float(np.mean(values)), float(np.std(values))

def principalModes(data, numModes):
    numModes = min(numModes, data.shape[0] - 1, data.shape[1])
    return PCA(n_components=numModes).fit_transform(data - data.mean(axis=0))

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

    representations = {
        'clamp -> raw':        (clampFeatures, interior),
        'G_pol -> raw':        (boundaryCode, interior),
        'G_pol -> modes':      (boundaryCode, principalModes(interior, args.numModes)),
        'modes -> modes':      (principalModes(boundaryCode, args.numModes), principalModes(interior, args.numModes)),
        'modes -> spatialFreq':(principalModes(boundaryCode, args.numModes),
                                lowFrequencyModes(interiorSquare, interiorSide, args.maxWavenumber)),
    }

    row = {'screen': screenSize, 'N': N,
           'spatialStd': float(interior.std(axis=1).mean() * 1000.0),
           'PR': float(participationRatio(interior))}
    for name, (codeFeatures, patternFeatures) in representations.items():
        Dcode, Dpattern = distanceMatrix(codeFeatures), distanceMatrix(patternFeatures)
        index, medianRank = nearestNeighbourIndex(Dcode, Dpattern)
        nullMean, nullStd = chanceNull(Dcode, Dpattern, args.numPermutations)
        rho = spearmanr(pdist(codeFeatures), pdist(patternFeatures))[0]
        row[name] = {'index': index, 'medianRank': medianRank, 'rho': float(rho),
                     'z': (index - nullMean) / nullStd if nullStd > 0 else np.nan}
    results.append(row)
    print(f"screen {screenSize:2d}: N={N}, spatialStd={row['spatialStd']:.2f} mV, PR={row['PR']:.1f}")

# ── Report ───────────────────────────────────────────────────────────────────
representationNames = [k for k in results[0] if isinstance(results[0][k], dict)]
print(f"\nAddressability index (0 = chance, 1 = perfect), z against a permutation null, N={results[0]['N']}")
print(f"{'screen':>7} {'spatialStd':>11} {'PR':>7}  " + "  ".join(f"{n:>22}" for n in representationNames))
for row in results:
    cells = "  ".join(f"{row[n]['index']:>+8.3f} (z={row[n]['z']:>+5.1f})" for n in representationNames)
    print(f"{row['screen']:>7} {row['spatialStd']:>8.2f} mV {row['PR']:>7.1f}  {cells}")
print(f"\nMantel rho")
print(f"{'screen':>7}  " + "  ".join(f"{n:>22}" for n in representationNames))
for row in results:
    print(f"{row['screen']:>7}  " + "  ".join(f"{row[n]['rho']:>+22.3f}" for n in representationNames))
print(f"\nPR is capped by N-1 = {results[0]['N']-1}; it is reported for continuity with Section 6, "
      f"not as the headline.")

np.savez(f'{args.outputPrefix}.npz', results=np.array(results, dtype=object))
print(f"Saved {args.outputPrefix}.npz")
