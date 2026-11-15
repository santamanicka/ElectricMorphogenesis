"""Position-dependent field range for Potentia: cells in narrow regions of the tissue sense over a shorter range than cells in broad regions.

The local width of the tissue at a pixel is its *thickness* T(x): the diameter of the largest circle that fits inside the tissue and contains the pixel (in pixels).  A cell at the edge of a broad
region has the thickness of that region (it lies inside the big circle), not a small one because it is close to the boundary; the tail of a tadpole is thin, its body thick.  The range is the same
fraction of the local width everywhere:  R(x) = max(pct/100 * T(x) * h, 3 h)   (h = pixel pitch; never below 3 pixels, where the kernel is no longer resolved).

T is computed with exact integer arithmetic (squared Euclidean distance transforms), so the JavaScript version (core.js, localThickness) gives the same numbers:
    d2(y)  = squared distance from tissue pixel y to the nearest pixel outside the tissue (the grid border counts as outside)
    S_k    = { y : 4 d2(y) >= k^2 }                       pixels that are the centre of a circle of diameter k inside the tissue
    A_k    = { x : 4 dist2(x, S_k) <= (k+1)^2 }           pixels whose centre lies within half a pixel of one of those circles (the half pixel removes the staircase
                                                          artefacts of the rim of a disc or a capsule, which would otherwise look thinner than their interior)
    T(x)   = max { k : x in A_k }                         (an integer number of pixels; 2 * d for the centre pixels of an even-sized rectangle, so exact there)
Sharp corners and thin tips, where only small circles fit, get short ranges too (3% of a 22 x 60 rectangle is such corner).
How the field uses the range map: potentia.Potentia.set_range_map (a field is computed at a few ranges, then every pixel takes the one for its own range, interpolating linearly).
"""
import numpy as np
from scipy import ndimage


def _d2_to_zero(a):
    """Squared distance of every element to the nearest False element of the boolean array a (exact integers, as floats)."""
    d = ndimage.distance_transform_edt(a)
    return np.rint(d * d)


def local_thickness(mask):
    """Local thickness in pixels (integer array, 0 outside the tissue) of a boolean tissue mask (None = the whole grid is tissue: pass the shape instead via an all-True array)."""
    m = np.asarray(mask, bool)
    d2 = _d2_to_zero(np.pad(m, 1, constant_values=False))[1:-1, 1:-1]          # the border of the grid counts as outside
    T = np.zeros(m.shape, np.int32)
    kmax = int(np.floor(2.0 * np.sqrt(d2.max())))
    for k in range(1, kmax + 1):
        S = 4.0 * d2 >= float(k * k)
        if not S.any():
            break
        ds2 = _d2_to_zero(~S)                                                   # squared distance to the nearest pixel of S
        T[(4.0 * ds2 <= float((k + 1) * (k + 1))) & m] = k
    return T


def range_map(mask, pct, h):
    """Field range in metres at every pixel: pct % of the local thickness (pixels * h metres), at least 3 pixels.  mask: boolean array (all True for a rectangle)."""
    T = local_thickness(mask)
    return np.maximum(pct / 100.0 * (T * h), 3.0 * h)
