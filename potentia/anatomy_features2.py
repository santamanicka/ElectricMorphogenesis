"""Version 4 of the feature match: the layout channel is replaced by an organ-by-organ enrichment channel.

Why.  In versions 1-3 the body-scale layout channel compared a pattern with the UNION of the schematic's layers, so the largest organ (the planarian's mid-body gut) dominated, and a pattern
with organised structure along the whole body (flank cords, rungs, head and tail structure) scored low on it, though each of those is in the schematic.

organs.  For each layer of the schematic (planarian: nerve, gut, eyes; tadpole: brain, cord, myomeres) the question is whether the pattern has structure where that organ is.  Structure is the local
deviation of the straightened pattern from its own surroundings, E = G_sigma_small(|I - G_sigma_large(I)|), sign-free, so an organ may be marked by high or by low G_pol; a pixel is 'on the organ' where the layer,
blurred by the same tolerance (about 2% of the body length along, 2 samples across), exceeds 0.15, and 'off' where it is below 0.02.  The channel is the area under the ROC curve of E for on against off (0.5 = no
relation to the organ), rescaled as 2 AUC - 1 and floored at 0, averaged over the layers.  A uniformly busy pattern scores 0 here, as does a flat one.
The other three channels (segments, cords, spots) and the overall mean are as in anatomy_features.py.
"""
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import rankdata
import anatomy_features as AF
from anatomy_targets import layers, NS, NU

NSR, NUR = AF.NSR, AF.NUR
_LAY = {}


def layer_arrays(shape):
    if shape not in _LAY:
        si = np.clip((AF._S * NS).astype(int), 0, NS - 1)
        ui = np.clip(np.rint((AF._U + 1) / 2 * (NU - 1)).astype(int), 0, NU - 1)
        k = NS // NSR
        out = {}
        for name, T in layers(shape).items():
            A = T[:NSR * k].reshape(NSR, k, NU).mean(1)[:, ui] if k > 1 else T[si][:, ui]
            B = ndi.gaussian_filter(A, (0.02 * NSR, 2.0), mode='nearest')
            out[name] = (B > 0.15, B < 0.02)
        _LAY[shape] = out
    return _LAY[shape]


def structure(I):
    return ndi.gaussian_filter(np.abs(I - ndi.gaussian_filter(I, (0.06 * NSR, 5.0), mode='nearest')), (0.02 * NSR, 2.0), mode='nearest')


def auc(x_pos, x_neg):
    if x_pos.size < 20 or x_neg.size < 20:
        return 0.5
    r = rankdata(np.concatenate([x_pos, x_neg]))
    return (r[:x_pos.size].sum() - x_pos.size * (x_pos.size + 1) / 2) / (x_pos.size * x_neg.size)


def organs(I, valid, shape):
    E = structure(I)
    vals = {}
    for name, (on, off) in layer_arrays(shape).items():
        vals[name] = max(0.0, 2.0 * auc(E[on & valid], E[off & valid]) - 1.0)
    return float(np.mean(list(vals.values()))), vals


def score_v4(shape, I, valid):
    """(score, channels, per-organ) of a straightened, standardised pattern against the shape's schematic."""
    F = AF.features(I)
    m = AF.match(F, AF.target_features(shape))
    o, per = organs(I, valid, shape)
    ch = {'organs': o, 'segments': m['segments'], 'cords': m['cords'], 'spots': m['spots']}
    return float(np.mean(list(ch.values()))), ch, per
