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
    tadpole     round head at one end with a tail of width p * head diameter (default 0.25): Xenopus tadpole, a bud or a sperm-like shape
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
    'tadpole': ('tadpole (head and tail)', 'tail width / head diameter', 0.25, (0.05, 0.9)),
}
S3 = float(np.sqrt(3.0))


def shape_mask(name, Np, p=None):
    """Boolean mask (Np0, Np1) of the named shape; p is the shape parameter (see SHAPES), default if None."""
    if name not in SHAPES:
        raise ValueError(f'unknown shape {name!r}; choose from {sorted(SHAPES)}')
    R, C = int(Np[0]), int(Np[1])
    if p is None:
        p = SHAPES[name][2]
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
    else:                                                          # tadpole: head at the negative end of the long axis
        c = a - b
        t = p * b
        head = (axis + c) * (axis + c) + perp * perp <= b * b
        d = np.maximum(axis - (a - t), 0.0)
        m = head | ((axis >= -c) & (d * d + perp * perp <= t * t))
    return np.asarray(m, bool)
