"""How do the trained band-hold boundary codes shape the interior pattern? (PolyPatterning_Sim.md, Section 12)

Uses the late-window mean patterns and face scores from scoreBandHoldFaceMetrics11x11.py. Codes are taken as
full-lattice G_pol fields (zero off the band). Groups are mechanism x depth, 96 seeds each.
  1. mode coupling: which pattern principal component does each code principal component drive?
  2. the dial's rearrangement (Gpol-only D1): the two lobes of pattern PC2 and how the dial moves them
  3. distances: Mantel correlation of code distance with pattern distance, under spatial metrics that respect
     the non-periodic lattice (field-screen smoothing, gap-junction heat kernel, entropic optimal transport)
     and under code geometries restricted to the band itself or, for depth 1, to the ring as a circle
  4. circular harmonics (Gpol-only D1): which boundary orders reach which interior orders, shell by shell
"""
import argparse

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import boundaryCodeUtilities as boundary

parser = argparse.ArgumentParser()
parser.add_argument('--metricsPath', type=str, default='data/bandHoldFaceMetrics11x11_all384.csv')
parser.add_argument('--patternsPath', type=str, default='data/bandHoldPatterns11x11_all384.npz')
parser.add_argument('--numMantelPermutations', type=int, default=2000)
parser.add_argument('--randomSeed', type=int, default=909)
parser.add_argument('--skipDistances', action='store_true', help='skip section 3, the slowest')
args = parser.parse_args()

metrics = pd.read_csv(args.metricsPath).sort_values('fileNumber').reset_index(drop=True)
patternFile = np.load(args.patternsPath)
patternRow = {int(number): index for index, number in enumerate(patternFile['fileNumbers'])}
codes = boundary.loadBandHoldCodes(metrics.fileNumber.tolist())
randomGenerator = np.random.default_rng(args.randomSeed)
groups = [('Gpol-only', 1), ('Gpol-only', 2), ('Gpol+Vmem', 1), ('Gpol+Vmem', 2)]


def groupData(mechanism, depth, readout='windowMean'):
    rows = metrics[(metrics.mechanism == mechanism) & (metrics.depth == depth)].reset_index(drop=True)
    fields = np.stack([codes[number]['field'] for number in rows.fileNumber])
    patterns = np.stack([patternFile[readout][patternRow[number]] for number in rows.fileNumber]).astype(float)
    return rows, fields, patterns


def configurationLabels(rows):
    return rows.holdIterations.astype(str) + '/' + rows.lossMethod


def controlledSpearman(first, second, rows):
    labels = configurationLabels(rows)
    return spearmanr(boundary.rankWithinGroups(first, labels), boundary.rankWithinGroups(second, labels))


def modeScores(fields, patterns, numModes):
    support = fields.std(0) > 1e-9
    codeMatrix = StandardScaler().fit_transform((fields - fields.mean(1, keepdims=True))[:, support])
    patternMatrix = StandardScaler().fit_transform(patterns - patterns.mean(1, keepdims=True))
    codePca, patternPca = PCA(numModes).fit(codeMatrix), PCA(numModes).fit(patternMatrix)
    return (codePca, codePca.transform(codeMatrix), patternPca, patternPca.transform(patternMatrix), support)


# ------------------------------------------------------------------------------------------ 1. mode coupling
print("1. MODE COUPLING (code PCA on the band's values; pattern PCA on self-centred, standardised patterns)")
rows, fields, patterns = groupData('Gpol-only', 1)
codePca, codeScores, patternPca, patternScores, support = modeScores(fields, patterns, 5)
dial = fields[:, support].mean(1)
print(f"   Gpol-only D1: code PC1 carries {codePca.explained_variance_ratio_[0] * 100:.1f}% and tracks the dial at "
      f"rho={spearmanr(codeScores[:, 0], dial).statistic:+.3f}")
print(f"   code PC1 vs pattern PC1: rho={spearmanr(codeScores[:, 0], patternScores[:, 0]).statistic:+.3f}; "
      f"vs pattern PC2: rho={spearmanr(codeScores[:, 0], patternScores[:, 1]).statistic:+.3f}; "
      f"dial vs pattern PC2: rho={spearmanr(dial, patternScores[:, 1]).statistic:+.3f}")
for column, name in (('partCoverage', 'coverage'), ('partSeparation', 'separation'), ('balancedRMS_mV', 'balanced RMS'), ('partScore', 'part score')):
    print(f"   pattern PC2 vs {name}: rho={spearmanr(patternScores[:, 1], rows[column]).statistic:+.3f}")
result = controlledSpearman(codeScores[:, 0], patternScores[:, 1], rows)
print(f"   code PC1 vs pattern PC2, configuration-controlled: rho={result.statistic:+.3f} p={result.pvalue:.1g}")


def residualOnRanks(values, covariate):
    design = np.column_stack([np.ones(len(covariate)), rankdata(covariate)])
    ranked = rankdata(values)
    return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]


result = spearmanr(residualOnRanks(codeScores[:, 0], rows.partCoverage), residualOnRanks(patternScores[:, 1], rows.partCoverage))
print(f"   code PC1 vs pattern PC2 with coverage partialled out: rho={result.statistic:+.3f} p={result.pvalue:.1g}")
for mechanism, depth in groups[1:]:
    rows, fields, patterns = groupData(mechanism, depth)
    _, codeScores, _, patternScores, _ = modeScores(fields, patterns, 3)
    pairs = [(abs(spearmanr(codeScores[:, codeMode], patternScores[:, patternMode]).statistic), codeMode, patternMode)
             for codeMode in range(3) for patternMode in range(3)]
    _, codeMode, patternMode = max(pairs)
    raw = spearmanr(codeScores[:, codeMode], patternScores[:, patternMode]).statistic
    controlled = controlledSpearman(codeScores[:, codeMode], patternScores[:, patternMode], rows)
    print(f"   {mechanism} D{depth}: strongest pair code PC{codeMode + 1} vs pattern PC{patternMode + 1}: rho={raw:+.3f}, "
          f"configuration-controlled {controlled.statistic:+.3f} (p={controlled.pvalue:.4f})")

# ------------------------------------------------------------------------------------ 2. the dial's lobes
print("\n2. THE DIAL'S REARRANGEMENT (Gpol-only D1; dial = mean over the 40 ring cells)")
rows, fields, patterns = groupData('Gpol-only', 1)
_, _, patternPca, patternScores, support = modeScores(fields, patterns, 5)
dial = fields[:, support].mean(1)
loading = patternPca.components_[1] * np.sign(spearmanr(dial, patternScores[:, 1]).statistic)
hyperpolarizingLobe, depolarizingLobe = np.argsort(loading)[:15], np.argsort(-loading)[:15]
centred = patterns - patterns.mean(1, keepdims=True)
contrast = centred[:, hyperpolarizingLobe].mean(1) - centred[:, depolarizingLobe].mean(1)
order = np.argsort(dial)
lowest, middle, highest = order[:24], order[36:60], order[-24:]
changeMap = centred[highest].mean(0) - centred[lowest].mean(0)
print(f"   whole-pattern mean shift, highest 24 minus lowest 24: {patterns[highest].mean() - patterns[lowest].mean():+.2f} mV; "
      f"cells range {changeMap.min():+.1f} to {changeMap.max():+.1f} mV; change map vs PC2 loading r={np.corrcoef(changeMap, loading)[0, 1]:+.3f}")
print(f"   dial vs (hyperpolarising lobe - depolarising lobe): rho={spearmanr(dial, contrast).statistic:+.3f}; "
      f"configuration-controlled {controlledSpearman(dial, contrast, rows).statistic:+.3f}")
for name, indices in (('lowest 24', lowest), ('highest 24', highest)):
    print(f"   {name}: hyperpolarising lobe {centred[indices][:, hyperpolarizingLobe].mean():+.2f} mV, "
          f"depolarising lobe {centred[indices][:, depolarizingLobe].mean():+.2f} mV")
groupMeans = [centred[indices].mean(0) for indices in (lowest, middle, highest)]
print(f"   middle 24 is {np.sqrt(((groupMeans[1] - groupMeans[0]) ** 2).mean()):.2f} mV from the lowest and "
      f"{np.sqrt(((groupMeans[1] - groupMeans[2]) ** 2).mean()):.2f} mV from the highest")

# ------------------------------------------------------------------------------------------- 3. distances
if not args.skipDistances:
    coordinates = np.array([[cell // boundary.latticeCols, cell % boundary.latticeCols] for cell in range(boundary.numCells)], float)
    latticeDistance = squareform(pdist(coordinates))
    adjacency = (np.abs(coordinates[:, None, :] - coordinates[None, :, :]).sum(-1) == 1).astype(float)
    heatKernel = expm(-2.0 * (np.diag(adjacency.sum(1)) - adjacency))

    def neighbourhoodAverage(distance, radius):
        kernel = (distance <= radius).astype(float)
        return kernel / kernel.sum(1, keepdims=True)
    fieldScreen = neighbourhoodAverage(latticeDistance, 4.0)

    def sinkhornDistances(values, groundDistance, regularisation=2.0, iterations=200):
        masses = values - values.min(1, keepdims=True) + 1e-9
        masses /= masses.sum(1, keepdims=True)
        kernel = np.exp(-groundDistance / regularisation)
        distances = np.zeros((len(values), len(values)))
        for first in range(len(values)):
            for second in range(first + 1, len(values)):
                scaleRows = np.ones(len(groundDistance))
                for _ in range(iterations):
                    scaleColumns = masses[second] / (kernel.T @ scaleRows + 1e-300)
                    scaleRows = masses[first] / (kernel @ scaleColumns + 1e-300)
                plan = scaleRows[:, None] * kernel * scaleColumns[None, :]
                distances[first, second] = distances[second, first] = (plan * groundDistance).sum()
        return distances

    def mantel(firstDistances, secondDistances):
        upper = np.triu_indices(firstDistances.shape[0], 1)
        observed = spearmanr(firstDistances[upper], secondDistances[upper]).statistic
        null = []
        for _ in range(args.numMantelPermutations):
            permutation = randomGenerator.permutation(firstDistances.shape[0])
            null.append(spearmanr(firstDistances[np.ix_(permutation, permutation)][upper], secondDistances[upper]).statistic)
        return observed, (1 + (np.abs(null) >= abs(observed)).sum()) / (1 + len(null))

    print("\n3. DISTANCES: Mantel rho (p) between code distance and pattern distance")
    for mechanism, depth in groups:
        rows, fields, patterns = groupData(mechanism, depth)
        _, _, finals = groupData(mechanism, depth, 'final')
        centredCodes = fields - fields.mean(1, keepdims=True)
        centredPatterns, centredFinals = patterns - patterns.mean(1, keepdims=True), finals - finals.mean(1, keepdims=True)
        patternDistances = dict(euclidean=squareform(pdist(centredPatterns)), sinkhorn=sinkhornDistances(centredPatterns, latticeDistance))
        print(f"\n   {mechanism} D{depth}: same metric on both sides")
        for name, codeDistance, patternDistance in (
                ('euclidean', squareform(pdist(centredCodes)), patternDistances['euclidean']),
                ('field screen r=4', squareform(pdist(centredCodes @ fieldScreen.T)), squareform(pdist(centredPatterns @ fieldScreen.T))),
                ('heat kernel t=2', squareform(pdist(centredCodes @ heatKernel.T)), squareform(pdist(centredPatterns @ heatKernel.T))),
                ('sinkhorn', sinkhornDistances(centredCodes, latticeDistance), patternDistances['sinkhorn']),
                ('euclidean, final readout', squareform(pdist(centredCodes)), squareform(pdist(centredFinals))),
                ('screen r=4, final readout', squareform(pdist(centredCodes @ fieldScreen.T)), squareform(pdist(centredFinals @ fieldScreen.T)))):
            observed, pValue = mantel(codeDistance, patternDistance)
            print(f"     {name:28s} {observed:+.3f} (p={pValue:.3f})")
        supportCells = np.where(fields.std(0) > 1e-9)[0]
        bandValues = fields[:, supportCells]
        centredBand = bandValues - bandValues.mean(1, keepdims=True)
        bandDistance = latticeDistance[np.ix_(supportCells, supportCells)]
        codeGeometries = {'whole lattice, raw': squareform(pdist(centredCodes)),
                          'whole lattice, smoothed r=4': squareform(pdist(centredCodes @ fieldScreen.T)),
                          'band only, raw': squareform(pdist(centredBand)),
                          'band only, smoothed r=2': squareform(pdist(centredBand @ neighbourhoodAverage(bandDistance, 2.0).T)),
                          'band only, sinkhorn': sinkhornDistances(bandValues, bandDistance)}
        if depth == 1:
            ringPosition = {int(cell): index for index, cell in enumerate(boundary.boundaryRingCells)}
            cyclicOrder = np.argsort([ringPosition[int(cell)] for cell in supportCells])
            ringValues = bandValues[:, cyclicOrder]
            positions = np.arange(ringValues.shape[1])
            cyclicDistance = np.abs(positions[:, None] - positions[None, :]).astype(float)
            cyclicDistance = np.minimum(cyclicDistance, ringValues.shape[1] - cyclicDistance)
            centredRing = ringValues - ringValues.mean(1, keepdims=True)
            codeGeometries.update({'ring as circle, raw': squareform(pdist(centredRing)),
                                   'ring as circle, smoothed w=2': squareform(pdist(centredRing @ neighbourhoodAverage(cyclicDistance, 2.0).T)),
                                   'ring as circle, sinkhorn': sinkhornDistances(ringValues, cyclicDistance)})
        print(f"   {mechanism} D{depth}: code geometry vs pattern (euclidean | sinkhorn)")
        for name, codeDistance in codeGeometries.items():
            cells = [f"{observed:+.3f} (p={pValue:.3f})" for observed, pValue in
                     (mantel(codeDistance, patternDistances['euclidean']), mantel(codeDistance, patternDistances['sinkhorn']))]
            print(f"     {name:28s} {cells[0]} | {cells[1]}")

# ------------------------------------------------------------------------------------ 4. circular harmonics
print("\n4. CIRCULAR HARMONICS (Gpol-only D1; shells are concentric square rings, shell 0 = the boundary)")
maxOrder = 6
rows, fields, patterns = groupData('Gpol-only', 1)
codeHarmonics = np.stack([boundary.circularHarmonicCoefficients(field[boundary.shellCells(0)], maxOrder) for field in fields])
patternHarmonics = {shell: np.stack([boundary.circularHarmonicCoefficients(pattern[boundary.shellCells(shell)], maxOrder)
                                     for pattern in patterns]) for shell in range(5)}
power = np.abs(codeHarmonics) ** 2
totalPower = power.sum(1).mean()
print("   share of code power by order: " + ", ".join(f"k={order} {power[:, order].mean() / totalPower * 100:.1f}%" for order in range(maxOrder + 1)))


def starred(result):
    return f"{result.statistic:+.3f}{'*' if result.pvalue < 0.01 else (':' if result.pvalue < 0.05 else ' ')}"


print("   same-order transfer |code_k| vs |pattern_k|, shells 0-4:")
for order in range(maxOrder + 1):
    print(f"     k={order}: " + " ".join(starred(spearmanr(np.abs(codeHarmonics[:, order]), np.abs(patternHarmonics[shell][:, order])))
                                      for shell in range(5)))
print("   cross-order at shell 2, |code_k| vs |pattern_m| for m = 0-6:")
for order in range(maxOrder + 1):
    print(f"     k={order}: " + " ".join(starred(spearmanr(np.abs(codeHarmonics[:, order]), np.abs(patternHarmonics[2][:, outputOrder])))
                                      for outputOrder in range(maxOrder + 1)))
print("   |code_k| vs face scores (coverage, separation, part score, balanced RMS):")
for order in range(maxOrder + 1):
    print(f"     k={order}: " + " ".join(starred(spearmanr(np.abs(codeHarmonics[:, order]), rows[column]))
                                      for column in ('partCoverage', 'partSeparation', 'partScore', 'balancedRMS_mV')))
print("   (* p < 0.01, : p < 0.05)")
