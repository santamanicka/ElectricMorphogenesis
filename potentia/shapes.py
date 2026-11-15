"""Tissue shapes for Potentia: boolean masks on the pixel grid (True = tissue).

A mask carves the tissue out of the rectangular grid (Np0 x Np1 pixels).  Pixels outside the mask are not tissue: they carry no charge (so they add
nothing to the field), have no gap junctions to their neighbours (a no-flux edge, as at the border of the rectangular tissue) and are held at V = 0, g = 0.
Shapes are decided at the pixel centres, so the boundary can cut through a cell (cells survive only as length scales).  The same rules are in
visualizer/core.js (PotentiaCore.shapeMask); tests compare the two on many sizes.

Geometry: x to the right, y downwards, in pixel units from the centre of the grid; b = half the shorter side, a = half the longer side.
Elongated shapes run along the longer side (horizontal if the grid is at least as wide as it is tall).

    rectangle   everything (no mask)
    disc        circle of radius b: blastoderm, animal-cap explant, micropatterned colony, organoid
    ellipse     inscribed in the grid: egg or embryo outline
    annulus     disc with a central hole (p = inner radius / outer radius, default 0.5): ring, gut or vessel cross-section, wound-healing assay
    stadium     capsule (rounded rectangle with semicircular ends): elongated body such as a planarian or zebrafish trunk
    hexagon     regular hexagon (flat top) with corner-to-corner width b*2: epithelial cluster
    triangle    equilateral triangle, side = shorter side
    cross       plus sign with arm width p * shorter side (default 0.33): explant with arms
    dumbbell    two discs joined by a neck of width p * disc diameter (default 0.3); p = 0 gives two separate discs with no gap junctions between them; they interact only through the field, and only if the gap is shorter than the field range R
    tadpole     Xenopus tadpole in dorsal view, snout at the negative end of the long axis, the outline mirror-symmetric about the long axis: a large oval body (a fuller front, a longer rear that
                narrows in an S-curve) and a long tail that tapers to a thin whip, joined by a polynomial smooth maximum of their half-depths.  p = depth of the tail at its root as a fraction of the
                box height (default 0.41); p2 = head size / tail size = body height / tail depth (default 2.354; at most 0.97 / p so that the head fits in the box).  The outline is scaled to the box
                (a box about 4.5 times longer than high gives the target's proportions; a shorter one a squatter tadpole).  A margin of max(1 pixel, 4% of the shorter side) is kept clear all round.
                Fitted to the silhouette of a schematic target (overlap 0.975)
    planarian   flatworm outline (dorsal view, head at the negative end of the long axis), the outline of a schematic target traced as a table of half-depths along the length: a pointed head with two
                small auricle spikes (the tallest points), a notch behind them, a spindle-shaped body widest at 40-58% of the length, and a tail tapering to a point; no parameter; use a box about 4.4 times
                longer than high.  A margin of max(1 pixel, 4% of the shorter side) is kept clear all round
"""
import numpy as np

SHAPES = {                      # name: (label, parameter label or None, default parameter, (min, max))
    'rectangle': ('rectangle', None, None, None),
    'disc': ('disc (circle)', None, None, None),
    'ellipse': ('ellipse', None, None, None),
    'annulus': ('annulus (ring)', 'inner radius / outer radius', 0.5, (0.05, 0.95)),
    'stadium': ('stadium (capsule)', None, None, None),
    'hexagon': ('hexagon', None, None, None),
    'triangle': ('triangle', None, None, None),
    'cross': ('cross', 'arm width / shorter side', 0.33, (0.05, 0.9)),
    'dumbbell': ('dumbbell (two discs)', 'neck width / disc diameter', 0.3, (0.0, 0.9)),
    'tadpole': ('tadpole (dorsal view)', 'tail depth (at its root) / box height', 0.41, (0.05, 0.9)),
    'planarian': ('planarian', None, None, None),
}
SHAPES2 = {                     # a second shape parameter, for the shapes that have one: name: (label, default, (min, max))
    'tadpole': ('head size / tail size (depth ratio)', 2.354, (1.5, 5.0)),
}
S3 = float(np.sqrt(3.0))
# planarian: half-depth h along the length s (0 at the head tip, 1 at the tail tip), in units of the half-height of the tallest point (the auricle tips); piecewise linear between these nodes
PLAN_S = (0.0, 0.0147, 0.0245, 0.0664, 0.1028, 0.1182, 0.1336, 0.1406, 0.1448, 0.1503, 0.1713, 0.1769, 0.2063, 0.2692, 0.3196, 0.4063, 0.5769, 0.679, 0.7993, 0.8455, 0.9084, 0.9783, 1.0)
PLAN_H = (0.0, 0.2, 0.287, 0.5135, 0.689, 0.797, 0.97, 1.0, 0.993, 0.964, 0.786, 0.76, 0.757, 0.8232, 0.8915, 0.9592, 0.9592, 0.8712, 0.6845, 0.5931, 0.4354, 0.1856, 0.0)


def shape_mask(name, Np, p=None, p2=None):
    """Boolean mask (Np0, Np1) of the named shape; p is the shape parameter (see SHAPES) and p2 the second one for the shapes that have it (SHAPES2); the default if None."""
    if name not in SHAPES:
        raise ValueError(f'unknown shape {name!r}; choose from {sorted(SHAPES)}')
    R, C = int(Np[0]), int(Np[1])
    if p is None:
        p = SHAPES[name][2]
    if p2 is None and name in SHAPES2:
        p2 = SHAPES2[name][1]
    i = np.arange(R, dtype=float)[:, None]
    j = np.arange(C, dtype=float)[None, :]
    x = (j + 0.5) - C / 2.0
    y = (i + 0.5) - R / 2.0
    x = x + 0.0 * y                                              # broadcast to (R, C)
    y = y + 0.0 * x
    ax, ay = np.abs(x), np.abs(y)
    L = float(min(R, C)); b = L / 2.0; a = float(max(R, C)) / 2.0
    horizontal = C >= R
    axis, perp = (x, y) if horizontal else (y, x)
    if name == 'rectangle':
        m = np.ones((R, C), bool)
    elif name == 'disc':
        m = x * x + y * y <= b * b
    elif name == 'ellipse':
        a1, a2 = C / 2.0, R / 2.0
        m = (x * x) / (a1 * a1) + (y * y) / (a2 * a2) <= 1.0
    elif name == 'annulus':
        r2 = x * x + y * y
        m = (r2 <= b * b) & (r2 >= (p * b) * (p * b))
    elif name == 'stadium':
        c = a - b
        d = np.maximum(np.abs(axis) - c, 0.0)
        m = d * d + perp * perp <= b * b
    elif name == 'hexagon':
        m = (ay <= (S3 * b) / 2.0) & (S3 * ax + ay <= S3 * b)
    elif name == 'triangle':
        H = (S3 * L) / 2.0
        m = (y >= -H / 2.0) & (y <= H / 2.0) & (ax <= (y + H / 2.0) / S3)
    elif name == 'cross':
        w = p * L
        m = (ax <= w / 2.0) | (ay <= w / 2.0)
    elif name == 'dumbbell':
        c = a - b
        d = np.abs(axis) - c
        n = p * b
        m = (d * d + perp * perp <= b * b) | ((np.abs(axis) <= c) & (np.abs(perp) <= n))
    elif name == 'planarian':                                      # head at the negative end of the long axis; half-depth profile h(s) * (b - margin) from a table
        mg = max(1.0, 0.04 * L)                                    # clear margin (pixels) kept all round the outline
        sp = (axis + a - mg) / (2.0 * a - 2.0 * mg)
        xs, ys = np.array(PLAN_S), np.array(PLAN_H)
        i = np.clip(np.searchsorted(xs, sp, side='right') - 1, 0, len(xs) - 2)
        h = ys[i] + (ys[i + 1] - ys[i]) * ((sp - xs[i]) / (xs[i + 1] - xs[i]))   # linear interpolation (the same arithmetic in core.js)
        m = (sp >= 0.0) & (sp <= 1.0) & (np.abs(perp) <= (b - mg) * h)
    else:                                                          # tadpole: snout at the negative end of the long axis; the outline is mirror-symmetric about the long axis
        mg = max(1.0, 0.04 * L)                                    # clear margin (pixels) kept all round the outline
        sp = (axis + a - mg) / (2.0 * a - 2.0 * mg)                # 0 at the snout .. 1 at the tail tip
        w = perp / (2.0 * b - 2.0 * mg)                            # distance from the long axis in units of the height (exactly antisymmetric, so the mask is exactly symmetric)
        xb = sp * 4.82                                             # distance from the snout in units of the height of the target (4.82 times longer than high): the outline is scaled to the length of the box
        hh = np.minimum(p2 * p, 0.97)                              # height of the body as a fraction of the box height: the ratio p2 (head size / tail size) times the tail depth p, at most the box
        K = hh / 0.965                                             # the body scaled about the snout (K = 1: as in the target)
        pr = p / 0.41                                              # the tail's depth relative to the target's
        xn = xb / K
        u = np.where(xn < 0.605, (0.605 - xn) / 0.548, (xn - 0.605) / 1.394)
        r = np.maximum(1.0 - u * u, 0.0)
        hb = K * (0.620 * np.sqrt(r)) * (1.0 + np.where(xn < 0.605, -0.222, -0.199) * r)        # half-depth of the body (0 beyond its ends): an ellipse with a fuller front and a longer rear
        e = 1.0 - sp
        poly = 0.041 + 0.024 * e + 0.034 * (e * e) + 0.857 * (e * e * e)
        ht = pr * (np.maximum(poly, 0.0) * np.minimum(e / 0.025, 1.0) * np.minimum(np.maximum(sp / 0.243, 0.0), 1.0))   # half-depth of the tail: a cubic in the distance from the tip, pointed at the tip, absent at the snout
        ht = np.minimum(ht, np.minimum(0.33 * pr, 0.49))           # the tail's root is bounded (half-depth 0.33 of the height at the default; always inside the box): the cubic rises steeply towards the snout, where a small head would otherwise vanish under it
        k = 0.132 * np.minimum(K, pr)                              # where body and tail meet their depths are joined by a polynomial smooth maximum (C1) of width k
        g = np.maximum(k - np.abs(hb - ht), 0.0)
        bump = np.where((hb > 0.0) & (ht > 0.0), (g * g) / (4.0 * k), 0.0)
        m = (sp >= 0.0) & (sp <= 1.0) & (np.abs(w) <= np.minimum(np.maximum(hb, ht) + bump, 0.4975))   # never beyond the box, whatever the parameters
    return np.asarray(m, bool)
