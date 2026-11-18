"""Pattern metrics for the G_pol field of a tissue: how differentiated it is, and how many features it has.   (the page shows both; the same arithmetic is in visualizer/core.js, PotentiaCore.patternMetrics)

State = G_pol / G_ref of a pixel, which lies in [0, 2]; the tissue is the mask (None = the whole grid).

positional_information(g, mask, m)   (bits)   How well the position of a pixel predicts its state: the mutual information between the state (8 bins of 0.25) and the position (squares of 20 um = 2 cells),
    estimated by leave-one-out prediction.  Each position block contributes a 4 x 4 lattice of sample pixels (16 samples at any resolution m >= 2), and every sample is scored by
        log2 [ p(state | the other samples of its block) / p(state | the other samples of the tissue) ],   p = (count + alpha) / (total + 8 alpha),   alpha = 0.1,
    and the scores are averaged.  This is a lower bound of the mutual information that does not grow with the number of pixels: a uniform or a random pattern scores below zero (reported as 0), a pattern that
    is a function of position at the scale of the blocks scores up to its entropy (at most 3 bits).  The blocks are 20 um because that resolves the structures of the model: the half-width of the autocorrelation of
    G_pol in the simulated patterns has a median of 20 um (quartiles 15 to 20 um), a block of 20 um resolves stripes with a period of 40 um and above, and it is the smallest block with 16 samples at the default m = 2.
features(g, mask, m)   (count)   The number of persistent peaks (active regions) and basins (quiet regions) of the cell-level G_pol: the 0-dimensional persistent homology of the superlevel and the sublevel sets of
    the field of cell means (the mean of the tissue pixels of each m x m cell, 4-connected cells, a cell counts if at least half of it is tissue), keeping the features whose persistence is at least 0.5 (a quarter of the
    range 0 to 2; a smaller contrast is noise).  A uniform tissue has 0 features, a tissue with one active and one quiet region 2, a band pattern two per period.  Per 100 cells it can be compared between tissues.
"""
import numpy as np

NBINS = 8                    # states: G_pol / G_ref in [0, 2] in 8 bins of 0.25
ALPHA = 0.1                  # additive smoothing of the leave-one-out probabilities (chosen on known-answer patterns, see test_complexity.py)
NSUB = 4                     # samples per block side
BLOCK_UM = 20.0              # side of the position blocks (um)
PITCH_UM = 10.0              # the cell pitch of the model (um)
DELTA = 0.5                  # smallest persistence of a feature (G_pol / G_ref)


def block_pixels(m, block_um=BLOCK_UM, nsub=NSUB):
    """Side of the position blocks in pixels: block_um micrometres (a cell of 10 um is m pixels), at least nsub pixels so that an nsub x nsub lattice of samples fits in every block."""
    return max(int(round(block_um * m / PITCH_UM)), int(nsub))


def sample_offsets(B, n=NSUB):
    """The n sample positions inside a block of B pixels: the centres of n equal parts."""
    return np.array([int(np.floor((k + 0.5) * B / n)) for k in range(n)])


def positional_information(g, mask, m=2, block_um=BLOCK_UM, nsub=NSUB, nbins=NBINS, alpha=ALPHA, min_samples=4, raw=False):
    g = np.asarray(g, float); mk = np.ones(g.shape, bool) if mask is None else np.asarray(mask, bool)
    R, C = g.shape; B = block_pixels(m, block_um, nsub); offs = sample_offsets(B, nsub)
    s = np.clip(np.floor(g / 2.0 * nbins).astype(np.int64), 0, nbins - 1)
    nbr, nbc = (R + B - 1) // B, (C + B - 1) // B
    I, J = np.indices(g.shape); blk = (I // B) * nbc + (J // B)
    pick = mk & np.isin(I % B, offs) & np.isin(J % B, offs)                                         # the same nsub x nsub lattice of samples in every block, whatever m
    Nbs = np.bincount((blk[pick] * nbins + s[pick]), minlength=nbr * nbc * nbins).reshape(nbr * nbc, nbins).astype(float)
    Nb = Nbs.sum(axis=1); keep = Nb >= min_samples
    Nbs = Nbs[keep]; Nb = Nb[keep]; Ns = Nbs.sum(axis=0); N = Nb.sum()
    if N < 2 * min_samples:
        return 0.0
    la = np.log2
    term = Nbs * (la(np.maximum(Nbs - 1.0 + alpha, 1e-300)) - la(Nb[:, None] - 1.0 + nbins * alpha) - la(np.maximum(Ns[None, :] - 1.0 + alpha, 1e-300)) + la(N - 1.0 + nbins * alpha))
    v = float(np.where(Nbs > 0, term, 0.0).sum() / N)
    return v if raw else max(v, 0.0)


def cell_means(g, mask, m):
    """Mean G_pol of the tissue pixels of each m x m cell (summed in row-major order) and whether the cell is tissue (at least half of its pixels)."""
    g = np.asarray(g, float); mk = np.ones(g.shape, bool) if mask is None else np.asarray(mask, bool)
    nr, nc = g.shape[0] // m, g.shape[1] // m
    tot = np.zeros((nr, nc)); cnt = np.zeros((nr, nc), np.int64)
    for di in range(m):
        for dj in range(m):
            sel = mk[di:nr * m:m, dj:nc * m:m]
            tot = tot + np.where(sel, g[di:nr * m:m, dj:nc * m:m], 0.0); cnt = cnt + sel
    valid = cnt * 2 >= m * m
    return np.where(valid, tot / np.maximum(cnt, 1), 0.0), valid


def _persistence(v, valid, order):
    """0-dimensional persistences of the filtration that adds the cells in the given order (valid cells of the grid, 4-connected): when two components meet, the one whose first cell came later dies,
    with persistence v[first cell] - v[current cell]; the components that survive have persistence v[first cell] - the smallest v in the component.  Returns the list of persistences."""
    nr, nc = valid.shape; n = nr * nc
    parent = np.full(n, -1, np.int64); first = np.zeros(n, np.int64); rank = np.zeros(n, np.int64); rank[order] = np.arange(len(order))
    pers = []

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for c in order:
        parent[c] = c; first[c] = c; i, j = divmod(int(c), nc)
        for nb_ in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= nb_[0] < nr and 0 <= nb_[1] < nc:
                q = nb_[0] * nc + nb_[1]
                if parent[q] >= 0:
                    a, b = find(c), find(q)
                    if a != b:
                        if rank[first[a]] > rank[first[b]]:
                            a, b = b, a                                                           # a is the elder: its first cell came earlier
                        pers.append(v[first[b]] - v[c]); parent[b] = a
    minv = {}
    for c in order:
        r = find(c); minv[r] = min(minv.get(r, np.inf), v[c])
    for r, mn in minv.items():
        pers.append(v[first[r]] - mn)
    return pers


def features(g, mask, m=2, delta=DELTA):
    """(features, peaks, basins, cells): the number of persistent peaks and basins of the cell-level G_pol (see the module docstring) and the number of tissue cells."""
    gc, valid = cell_means(g, mask, m); idx = np.flatnonzero(valid.ravel()); n_cells = int(len(idx))
    if n_cells == 0:
        return 0, 0, 0, 0
    val = gc.ravel()
    sup = idx[np.lexsort((idx, -val[idx]))]                                                          # descending value, ties by index
    peaks = sum(1 for p in _persistence(val, valid, sup) if p >= delta)
    sub = sup[::-1]                                                                                  # the sublevel filtration: the same order reversed (ascending value)
    basins = sum(1 for p in _persistence(-val, valid, sub) if p >= delta)
    return peaks + basins, peaks, basins, n_cells


def pattern_metrics(g, mask, m=2):
    """{'bits': positional information, 'features': count, 'peaks', 'basins', 'cells', 'per100': features per 100 cells}"""
    f, pk, bs, nc = features(g, mask, m)
    return dict(bits=positional_information(g, mask, m), features=f, peaks=pk, basins=bs, cells=nc, per100=(100.0 * f / nc if nc else 0.0))
