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
    tadpole     Xenopus tadpole in side view, head (snout) at the negative end of the long axis: an oval body and a lens-shaped finned tail tapering to a point at the far end, the whole outline
                mirror-symmetric about the long axis; p = depth of the tail and its fins as a fraction of the box height (default 0.9; the body keeps its proportions whatever the length of the
                box, so a shorter box gives a shorter tail; use a box about 3.7 times longer than high). A margin of max(1 pixel, 4% of the shorter side) is kept clear all round (as for the planarian)
    planarian   flatworm outline (dorsal view, head at the negative end of the long axis): a pointed head as wide as the body (auricle corners, no neck), a parallel-sided body
                over the first half and a tail tapering to a point over the second (profile fitted to a schematic drawing; a margin of max(1 pixel, 4% of the shorter side) is kept clear all round; use a box about 4.6 times longer than wide)
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
    'tadpole': ('tadpole (body and finned tail)', 'tail (fin) depth / box height', 0.9, (0.1, 0.9)),
    'planarian': ('planarian', None, None, None),
}
SHAPES2 = {                     # a second shape parameter, for the shapes that have one: name: (label, default, (min, max))
    'tadpole': ('head size / tail size (depth ratio)', 0.924, (0.3, 3.0)),
}
S3 = float(np.sqrt(3.0))


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
    elif name == 'planarian':                                      # head at the negative end of the long axis; half-width profile h(s) * (b - margin) along s = 0 (head tip) .. 1 (tail tip)
        mg = max(1.0, 0.04 * L)                                    # clear margin (pixels) kept all round the outline, so that no part of it sits on the edge of the box and looks cut off
        sp = (axis + a - mg) / (2.0 * a - 2.0 * mg)
        sa, st = 0.134, 0.5                                        # auricles (where the head reaches the full body width), start of the tail taper
        x1 = sp / sa
        h1 = 0.6 * np.sqrt(np.maximum(x1, 0.0)) - 0.2 * x1 + 0.6 * x1 * x1   # head: pointed, slightly rounded tip, flaring out to the auricle corners; no neck behind them
        t = (sp - st) / (1.0 - st)
        h4 = 1.0 - t * t * np.sqrt(np.maximum(t, 0.0))             # tail: tapers to a point as 1 - t^2.5 (equal to 1 on the parallel-sided middle where t <= 0)
        h = np.where(sp < sa, h1, np.where(sp < st, 1.0, h4))
        m = (sp >= 0.0) & (sp <= 1.0) & (np.abs(perp) <= (b - mg) * h)
    else:                                                          # tadpole: head at the negative end of the long axis; the outline is mirror-symmetric about the long axis
        mg = max(1.0, 0.04 * L)                                    # clear margin (pixels) kept all round the outline
        sp = (axis + a - mg) / (2.0 * a - 2.0 * mg)                # 0 at the snout .. 1 at the tail tip
        w = perp / (2.0 * b - 2.0 * mg)                            # distance from the long axis in units of the height (exactly antisymmetric, so the mask is exactly symmetric)
        xb = sp * ((2.0 * a - 2.0 * mg) / (2.0 * b - 2.0 * mg))    # distance from the snout in units of the height: the body keeps its proportions in a box of any length
        hh = np.minimum(p2 * p, 0.998)                             # height of the body as a fraction of the box height: the ratio p2 (head size / tail size) times the tail depth p, at most the box
        K = hh / 0.832                                             # the body ellipse scaled about the snout (K = 1: as drawn)
        e1 = (xb - 0.874 * K) / (0.877 * K)
        hb = (0.416 * K) * np.sqrt(np.maximum(1.0 - e1 * e1, 0.0))   # half-depth of the oval body at this position (0 beyond its ends)
        u = (0.556 - sp) / (0.556 - 0.039)                         # tail and fins: a lens about the same axis, deepest at 55.6% of the length;
        t = (sp - 0.556) / (1.0 - 0.556)                           #   half-depth 1 - u^2 towards the head, 1 - t^2.5 towards the tip
        q = np.where(sp < 0.556, 1.0 - u * u, 1.0 - t * t * np.sqrt(np.maximum(t, 0.0)))
        hl = (0.5 * p) * np.maximum(q, 0.0)                        # half-depth of the tail
        k = 0.5 * np.minimum(0.416 * K, 0.5 * p)                   # where the two meet their depths are joined by a polynomial smooth maximum (C1) of width k, which fills the waist
        g = np.maximum(k - np.abs(hb - hl), 0.0)
        bump = np.where((hb > 0.0) & (hl > 0.0), (g * g) / (4.0 * k), 0.0)
        m = (sp >= 0.0) & (sp <= 1.0) & (np.abs(w) <= np.maximum(hb, hl) + bump)
    return np.asarray(m, bool)
