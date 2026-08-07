"""
A battery of complexity and addressability metrics, and the geometry to evaluate them fairly.

Every metric here is paired with the same question: how much readable structure does the interior
carry (CX), and how much of it does the boundary code determine (ADDR)? The battery exists because
the obvious choices all failed in ways this project has already paid for:

  - participation ratio and TSE are functions of correlation structure alone, so they score a
    tissue on variety it generates rather than variety the boundary can write, and they cannot
    tell a rich pattern from a rich one nobody is steering.
  - rank-based addressability is scale-free, so it certified control of an interior uniform to
    a nanovolt.
  - anything variance-weighted over the whole interior is dominated by the fringe, because at
    short action range the response is concentrated within a few cells of the boundary and falls
    to nothing beyond. A tissue that addresses only its own edge should not score as one that
    addresses its bulk.

The last point is why depth matters. Interior cells are stratified by Chebyshev distance from the
boundary, and any metric can be evaluated per shell and aggregated with equal weight per shell
rather than per cell -- so the fringe, which holds the most cells and the most variance, cannot
carry the score by itself.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-3, 7, 21)
FLOOR_MV = 0.1          # the smallest voltage difference a downstream reader is assumed to resolve


def crossValidatedR2(code, targets, numFolds=5, seed=0):
    """Per-column cross-validated R^2 of predicting targets from code. Multi-output in one pass."""
    if targets.ndim == 1:
        targets = targets[:, None]
    keep = targets.std(axis=0) > 0
    r2 = np.zeros(targets.shape[1])
    if not keep.any():
        return r2
    X = StandardScaler().fit_transform(code)
    y = targets[:, keep]
    predicted = cross_val_predict(RidgeCV(alphas=ALPHAS, alpha_per_target=True), X, y,
                                  cv=KFold(numFolds, shuffle=True, random_state=seed))
    residual = ((y - predicted) ** 2).sum(axis=0)
    total = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    r2[keep] = 1 - residual / np.maximum(total, 1e-300)
    return r2


def modeDecomposition(patterns, numModes=None):
    """Principal modes of the pattern ensemble, with each mode's amplitude in mV."""
    limit = min(patterns.shape[0] - 1, patterns.shape[1])
    numModes = limit if numModes is None else min(numModes, limit)
    centred = patterns - patterns.mean(axis=0)
    pca = PCA(n_components=numModes).fit(centred)
    return pca.transform(centred), pca


def crossValidatedAmplitude(patterns, numFolds=5, seed=0):
    """Mode amplitudes measured on held-out samples, so sampling noise does not count as structure.

    With 200 samples and 784 cells, PCA yields 199 modes whatever the data, and the trailing ones
    carry sampling noise with non-zero amplitude. Counting modes above a floor would count those.
    Fitting the basis on training samples and measuring amplitude on held-out ones charges a mode
    for failing to generalise: a noise mode has no consistent direction and its held-out amplitude
    collapses, while a real one keeps its size.
    """
    limit = min(patterns.shape[0] - patterns.shape[0] // numFolds - 1, patterns.shape[1])
    if limit < 1:
        return np.zeros(0)
    amplitudes = []
    for train, test in KFold(numFolds, shuffle=True, random_state=seed).split(patterns):
        centre = patterns[train].mean(axis=0)
        pca = PCA(n_components=limit).fit(patterns[train] - centre)
        amplitudes.append(pca.transform(patterns[test] - centre).std(axis=0))
    return np.mean(amplitudes, axis=0)


def metricsForRegion(code, patterns, floor=FLOOR_MV, seed=0):
    """CX and ADDR for one region, both as variance in mV^2 over cross-validated modes.

    CX is the readable variance: the variance carried by modes whose held-out amplitude clears the
    floor. A count of such modes was tried first and censors badly -- at 30x30 the reachable set
    saturates the cross-validation ceiling for several action ranges while range 2 shows a cliff
    at 54 modes -- so variance, which has no ceiling, is used instead.

    ADDR is the part of that variance a cross-validated regression from the boundary code predicts.
    Sharing modes and floor with CX makes the two directly comparable, and ADDR / CX is the
    fraction of readable structure under boundary control. ADDR_dims is kept alongside as the
    equivalent count, since a dimension count is easier to reason about even where it censors.
    """
    blank = dict(CX_variance=0.0, CX_perCell=0.0, CX_modes=0.0, ADDR_variance=0.0,
                 ADDR_perCell=0.0, ADDR_dims=0.0, ADDR_fraction=0.0, ADDR_bits=0.0)
    if patterns.shape[1] == 0 or patterns.std() == 0:
        return blank
    amplitude = crossValidatedAmplitude(patterns, seed=seed)
    if len(amplitude) == 0:
        return blank
    scores, _ = modeDecomposition(patterns, numModes=len(amplitude))
    readable = amplitude >= floor
    if not readable.any():
        return blank

    variance = amplitude[readable] ** 2
    r2 = np.clip(crossValidatedR2(code, scores[:, readable], seed=seed), 0, None)
    numCells = patterns.shape[1]
    # Capacity in bits. For a Gaussian channel the information a predictor carries about one
    # variable is -0.5*log2(1 - R^2), and principal modes are orthogonal, so summing over readable
    # modes gives the total the boundary writes into the interior. Unlike a variance, this is
    # comparable across lattices and conditions of different amplitude; unlike a correlation it is
    # not scale-free, because the floor decides which modes are counted at all. R^2 is capped
    # short of 1 so a mode the regression happens to fit almost exactly cannot contribute
    # unbounded information.
    bits = float((-0.5 * np.log2(1.0 - np.clip(r2, 0.0, 0.999))).sum())
    return dict(ADDR_bits=bits, CX_variance=float(variance.sum()),
                CX_perCell=float(variance.sum() / numCells),
                CX_modes=float(readable.sum()),
                ADDR_variance=float((r2 * variance).sum()),
                ADDR_perCell=float((r2 * variance).sum() / numCells),
                ADDR_dims=float(r2.sum()),
                ADDR_fraction=float((r2 * variance).sum() / variance.sum()))


def depthFairMetrics(code, patterns, shells, floor=FLOOR_MV, seed=0):
    """Aggregate over depth shells with equal weight per shell, and per cell within each shell.

    A variance-weighted measurement over the whole interior is a measurement of the fringe: the
    outermost shell holds the most cells and, at short action range, nearly all of the amplitude.
    Averaging per-cell quantities across shells gives the tissue's core the same say as its edge,
    so a tissue that patterns only its own boundary cannot score as one that patterns its bulk.
    The gap between the whole-interior and depth-fair numbers is itself the measure of how
    fringe-concentrated a given action range is.
    """
    perShell = [metricsForRegion(code, patterns[:, cells], floor=floor, seed=seed)
                for _, _, cells in shells]
    fair = {f'fair_{k}': float(np.mean([s[k] for s in perShell]))
            for k in ('CX_perCell', 'ADDR_perCell', 'ADDR_fraction', 'ADDR_dims', 'CX_modes',
                      'ADDR_bits')}
    fair['profile_CX_perCell'] = [s['CX_perCell'] for s in perShell]
    fair['profile_ADDR_perCell'] = [s['ADDR_perCell'] for s in perShell]
    fair['profile_ADDR_fraction'] = [s['ADDR_fraction'] for s in perShell]
    fair['profile_ADDR_bits'] = [s['ADDR_bits'] for s in perShell]
    return fair


def depthShells(depth, interiorMask, minCells=12):
    """Depth shells over the interior, merging the innermost ones until each has enough cells.

    Shells shrink linearly toward the centre -- at 30x30 the outermost holds 108 cells and the
    innermost 4 -- and a shell with fewer cells than modes cannot support the same measurement.
    Merging inward keeps every shell wide enough to be measured on equal terms.
    """
    shells, current = [], []
    for d in sorted(set(depth[interiorMask])):
        current.append(d)
        cells = interiorMask & np.isin(depth, current)
        if cells.sum() >= minCells:
            shells.append((current[0], current[-1], cells))
            current = []
    if current and shells:                       # fold any remainder into the innermost shell
        first, _, cells = shells[-1]
        shells[-1] = (first, current[-1], cells | (interiorMask & np.isin(depth, current)))
    return shells
