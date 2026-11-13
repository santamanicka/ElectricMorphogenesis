"""Schematic anatomical targets for the anatomy search, and the dorsal tadpole outline.

Targets live in body coordinates (s, u): s = 0 at the head tip .. 1 at the tail tip along the long axis, u = -1 .. 1 across the body
(lateral position divided by the local half-width).  They are then drawn on whatever pixel grid and outline a run used, so a target
does not depend on tissue size or proportions.

planarian   schematised from the supplied drawing (anatomy/ref/planarian_drawing.png) by colour: orange = nervous system (cerebral ganglion,
            lateral nerve cords, transverse commissures), yellow = intestine (anterior trunk, two posterior branches round the pharynx),
            white/black = eye cups.  Three layers: nerve, gut, eyes.
tadpole     drawn by hand from the supplied dorsal photograph (anatomy/ref/tadpole_photo.png): brain (midline mass in the head), spinal cord
            (midline), myomeres (paired chevron blocks either side of the midline, period 0.026 of the body length, measured on the photo
            by autocorrelation: 8 pixels of 303).  Three layers: brain, cord, myomeres.  The photo shows no eyes, so none are drawn.
            The outline (tadpole_dorsal_mask) is the photo's width profile, with the tail uncurled.
"""
import os
import numpy as np
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'anatomy', 'ref')
NS, NU = 600, 121                       # resolution of the (s, u) template

# ---------------------------------------------------------------- tadpole outline (dorsal view), measured on the photo
# x (pixels from the snout, of 303) and full width (pixels, of 68) of the silhouette; tail tip uncurled
_TX = np.array([0, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 216, 232, 248, 264, 280, 296, 303]) / 303.0
_TW = np.array([0, 12, 44, 55, 62, 65, 68, 67, 66, 68, 67, 65, 60, 57, 51, 45, 35, 29, 27, 26, 24, 20, 18, 16, 14, 13, 10, 9, 8, 8, 7, 5, 4, 0]) / 68.0
TAIL_BASE = 0.40                        # width of the tail where it leaves the body / width of the head, in the photo
S_TAIL = 0.42                           # s where the tail starts


def tadpole_profile(s, tail=TAIL_BASE):
    """Half-width (fraction of the maximum) at s; `tail` rescales the width of the tail (0.40 = the photo)."""
    w = np.interp(s, _TX, _TW)
    k = np.clip((s - 0.30) / (S_TAIL - 0.30), 0, 1)                # blend from the head (unchanged) to the tail (scaled)
    return w * (1 + k * (tail / TAIL_BASE - 1))


def planarian_profile(s, sa=0.134, st=0.5):
    x1 = s / sa
    t = (s - st) / (1.0 - st)
    return np.where(s < sa, 0.6 * np.sqrt(np.maximum(x1, 0)) - 0.2 * x1 + 0.6 * x1 * x1, np.where(s < st, 1.0, 1.0 - t * t * np.sqrt(np.maximum(t, 0))))


def body_coords(Np, shape, tail=TAIL_BASE):
    """(s, u, mask) on the pixel grid Np = (rows, cols); head at the left.  Same margin rule as shapes.py."""
    R, C = Np
    y = (np.arange(R) + 0.5 - R / 2.0)[:, None] + np.zeros((R, C))
    x = (np.arange(C) + 0.5 - C / 2.0)[None, :] + np.zeros((R, C))
    b, a = R / 2.0, C / 2.0
    mg = max(1.0, 0.04 * R)
    s = (x + a - mg) / (2 * a - 2 * mg)
    prof = planarian_profile(s) if shape == 'planarian' else tadpole_profile(s, tail)
    hw = (b - mg) * prof
    with np.errstate(divide='ignore', invalid='ignore'):
        u = np.where(hw > 0, y / hw, 9.0)
    mask = (s >= 0) & (s <= 1) & (np.abs(y) <= hw)
    return s, u, mask


def tadpole_dorsal_mask(Np, tail=TAIL_BASE):
    return body_coords(Np, 'tadpole', tail)[2]


# ---------------------------------------------------------------- planarian layers from the drawing
def planarian_layers():
    from PIL import Image
    im = np.asarray(Image.open(os.path.join(REF, 'planarian_drawing.png')).convert('RGBA')).astype(float)
    r, g, b, al = im[..., 0], im[..., 1], im[..., 2], im[..., 3]
    yellow = (r > 200) & (g > 200) & (b < 140) & (al > 128)
    orange = (r > 190) & (g > 80) & (g < 185) & (b < 100) & (al > 128)
    olive = (np.abs(r - g) < 45) & (b < 0.8 * g) & (g > 50) & (g < 190) & (al > 128)      # body colour (dark to light olive)
    body = ndi.binary_closing(olive | yellow | orange, iterations=4)
    lab, n = ndi.label(body)
    body = lab == (1 + int(np.argmax(ndi.sum(body, lab, range(1, n + 1)))))
    body = ndi.binary_fill_holes(body)
    white = (r > 225) & (g > 225) & (b > 225) & (al > 128) & body
    black = (r < 60) & (g < 60) & (b < 60) & (al > 128) & body
    rows = np.where(body.any(1))[0]
    y0, y1 = rows.min(), rows.max()
    eyes_px = np.zeros_like(body)
    lab, n = ndi.label(white)                                         # eye cups: the white blobs in the head (the pupils are the black beside them)
    for k in range(1, n + 1):
        if ndi.sum(white, lab, k) > 150:
            eyes_px |= ndi.binary_dilation(lab == k, iterations=9) & (white | black)
    eyes_px &= (np.arange(body.shape[0])[:, None] < y0 + 0.3 * (y1 - y0))
    L = {k: np.zeros((NS, NU)) for k in ('nerve', 'gut', 'eyes')}
    src = {'nerve': orange & ~eyes_px, 'gut': yellow, 'eyes': eyes_px}
    ss = (np.arange(NS) + 0.5) / NS
    uu = np.linspace(-1, 1, NU)
    for i, s in enumerate(ss):
        yy = y0 + s * (y1 - y0)
        row = int(round(yy))
        cols = np.where(body[row])[0]
        if cols.size < 3:
            continue
        cx, hw = 0.5 * (cols.min() + cols.max()), 0.5 * (cols.max() - cols.min())
        xs = np.clip(np.rint(cx + uu * hw).astype(int), 0, body.shape[1] - 1)
        for k in L:
            L[k][i] = src[k][row, xs]
    for k in L:                                                       # the drawing is nearly mirror-symmetric; make the target exactly so
        L[k] = 0.5 * (L[k] + L[k][:, ::-1])
    return L


def tadpole_layers():
    ss = ((np.arange(NS) + 0.5) / NS)[:, None] + np.zeros((NS, NU))
    uu = np.linspace(-1, 1, NU)[None, :] + np.zeros((NS, NU))
    w = tadpole_profile(ss)                                           # half-width (of the maximum) at s
    yy = uu * w                                                       # lateral position in units of the maximum half-width
    brain = (((ss - 0.165) / 0.065) ** 2 + (yy / 0.30) ** 2 <= 1.0).astype(float)
    cord = ((ss > 0.23) & (ss < 0.93) & (np.abs(yy) < np.maximum(0.045, 0.16 * w))).astype(float)
    band = (ss > 0.22) & (ss < 0.78) & (np.abs(yy) >= np.maximum(0.045, 0.16 * w)) & (np.abs(yy) < np.minimum(0.42, 0.9 * w))
    phase = (ss + 0.035 * np.abs(yy) / 0.42) / 0.026                  # chevrons: the blocks sweep back away from the midline
    myo = (band & ((phase % 1.0) < 0.6)).astype(float)
    return {'brain': brain, 'cord': cord, 'myomeres': myo}


_CACHE = {}


def layers(shape):
    if shape not in _CACHE:
        _CACHE[shape] = planarian_layers() if shape == 'planarian' else tadpole_layers()
    return _CACHE[shape]


def targets_on_grid(shape, Np, tail=TAIL_BASE):
    """The layers drawn on the pixel grid: dict name -> (rows, cols) array in [0, 1] (area-averaged), and the mask."""
    s, u, mask = body_coords(Np, shape, tail)
    out = {}
    # area-average: supersample each pixel 3 x 3 in (s, u)
    R, C = Np
    ds = 1.0 / C
    for name, T in layers(shape).items():
        acc = np.zeros(Np)
        for fs in (-1 / 3, 0, 1 / 3):
            for fu in (-1 / 3, 0, 1 / 3):
                si = np.clip(((s + fs * ds) * NS).astype(int), 0, NS - 1)
                du = fu * 2.0 / np.maximum(1.0, (np.abs(u) < 5).sum(0))[None, :]
                ui = np.clip(np.rint((np.clip(u + du, -1, 1) + 1) / 2 * (NU - 1)).astype(int), 0, NU - 1)
                acc += T[si, ui]
        out[name] = np.where(mask, acc / 9.0, 0.0)
    return out, mask


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(4, 2, figsize=(15, 9))
    for j, (shape, asp) in enumerate((('planarian', 4.6), ('tadpole', 4.5))):
        for i, rows in enumerate((48, 16)):
            Np = (rows * 2, int(round(rows * asp)) * 2)
            T, mask = targets_on_grid(shape, Np)
            names = list(T)
            rgb = np.zeros(Np + (3,))
            cols = [(1.0, 0.55, 0.1), (1.0, 0.95, 0.2), (1.0, 1.0, 1.0)] if shape == 'planarian' else [(0.3, 0.8, 1.0), (1.0, 1.0, 1.0), (1.0, 0.45, 0.3)]
            rgb[mask] = (0.18, 0.2, 0.12)
            for nm, c in zip(names, cols):
                rgb = rgb * (1 - T[nm][..., None]) + T[nm][..., None] * np.array(c)
            ax[i, j].imshow(rgb, interpolation='nearest'); ax[i, j].set_title(f'{shape}: {" / ".join(names)}  ({rows} cells wide, m=2)', fontsize=9); ax[i, j].axis('off')
        from PIL import Image
        ref = Image.open(os.path.join(REF, 'planarian_drawing.png' if shape == 'planarian' else 'tadpole_photo.png')).convert('RGB')
        if shape == 'planarian':
            ref = ref.rotate(90, expand=True)
        ax[2, j].imshow(ref); ax[2, j].axis('off'); ax[2, j].set_title('supplied reference', fontsize=9)
        ax[3, j].axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'anatomy', 'fig_targets.png'), dpi=100)
    print('ok')
