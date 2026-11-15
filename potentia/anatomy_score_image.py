"""Score a screenshot of the Potentia simulator's G_pol view (magma colour scale, 0 to 2, cut-out background) with the feature match of anatomy_features.py.

The tissue mask is the image's alpha channel (or, without one, the pixels that are not white); G_pol is recovered by taking the nearest colour of matplotlib's magma
(the simulator's G_pol palette); the body is straightened from the mask itself (position along the long axis between its two ends, position across as a fraction of the local
half-width), so no outline model is needed.   python anatomy_score_image.py planarian image.png   |   tadpole image.png
Limits: one frame; the scale is read from colours (screenshots are resampled, so values near edges blur); the target layers are defined for the reference outlines, so a very different outline
matches them only approximately.
"""
import sys
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import matplotlib
import anatomy_features as AF

_LUT = (matplotlib.colormaps['magma'](np.linspace(0, 1, 256))[:, :3] * 255)
_TREE = cKDTree(_LUT)


def load(path):
    im = Image.open(path).convert('RGBA'); a = np.asarray(im)
    mask = a[..., 3] > 128
    if mask.all():
        mask = ~((a[..., :3] > 245).all(2))
    d, idx = _TREE.query(a[..., :3].astype(float))
    g = 2.0 * idx / 255.0
    return np.where(mask, g, 0.0), mask


def rectify_mask(g, mask):
    cols = np.where(mask.any(0))[0]; x0, x1 = cols.min(), cols.max() + 1
    xs = np.clip(np.floor(x0 + AF._S * (x1 - x0)).astype(int), 0, g.shape[1] - 1)
    I = np.zeros((AF.NSR, AF.NUR)); valid = np.zeros((AF.NSR, AF.NUR), bool)
    for i, x in enumerate(xs):
        r = np.where(mask[:, x])[0]
        if r.size < 3:
            continue
        c, hw = 0.5 * (r.min() + r.max()), 0.5 * (r.max() - r.min() + 1)
        if hw < 1.0:
            continue
        ri = np.clip(np.floor(c + AF._U * hw).astype(int), 0, g.shape[0] - 1)
        I[i] = g[ri, x]; valid[i] = mask[ri, x]
    m, sd = I[valid].mean(), I[valid].std()
    I = (I - m) / (sd if sd > 1e-9 else 1.0); I[~valid] = 0.0
    return I, valid


def score(shape, g, mask):
    I, valid = rectify_mask(g, mask)
    F = AF.features(I)
    other = 'tadpole' if shape == 'planarian' else 'planarian'
    return AF.match(F, AF.target_features(shape)), AF.match(F, AF.target_features(other)), I


if __name__ == '__main__':
    shape, path = sys.argv[1], sys.argv[2]
    g, mask = load(path)
    own, oth, _ = score(shape, g, mask)
    print(shape, path, 'tissue px', int(mask.sum()), 'mean G_pol %.2f std %.2f' % (g[mask].mean(), g[mask].std()))
    print(' match to the %s target %.3f' % (shape, own['score']), {k: round(v, 2) for k, v in own.items() if k != 'score'})
    print(' match to the other animal\'s target %.3f' % oth['score'])
