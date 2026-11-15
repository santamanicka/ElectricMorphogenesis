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
Smooth mode (the default of the page).  The thickness alone is a stepped map and does not make cells near a boundary sense less far, only cells in narrow regions.  The smooth width is the length of the chord,
through the cell and parallel to the nearest boundary, of the largest inscribed circle: with d(x) the distance from the cell to the nearest pixel outside the tissue and T~(x) a smoothed thickness
    W(x) = 2 sqrt( d (T^ - d) ),   T^ = max(T~, 2 d),   T~ = three passes of a masked box blur (radius 2 pixels) of T
so that W falls to zero towards every boundary (as the square root of the distance: in a 22-pixel strip 0.42 of the middle value at the edge pixel, 0.58 at the next, 0.77 four pixels in), is the thickness on the middle line of a broad strip
(d = T/2; within a pixel, as 2 d is not an integer), scales with the local thickness, has no steps and no kink on the medial axis of a strip (dW/dd = 0 there).  The blur uses prefix sums in a fixed order and the rest is + - * / and sqrt,
so JavaScript gives the same numbers to round-off.  R(x) = max(pct/100 * W(x) * h, 3 h) as before.
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


def _window_sum(A, r, axis):
    """Sum over the window [i - r, i + r] (clipped at the ends) along an axis, by sequential prefix sums."""
    A = np.moveaxis(A, axis, 0)
    n = A.shape[0]
    P = np.zeros((n + 1,) + A.shape[1:])
    P[1:] = np.cumsum(A, axis=0)
    i = np.arange(n)
    return np.moveaxis(P[np.minimum(i + r + 1, n)] - P[np.maximum(i - r, 0)], 0, axis)


def _masked_blur(A, mask, r, passes):
    """Box blur of radius r over the tissue pixels only (normalised by the number of tissue pixels in the window), repeated `passes` times."""
    m = mask.astype(float)
    den = _window_sum(_window_sum(m, r, 0), r, 1)
    out = np.where(mask, A, 0.0)
    for _ in range(passes):
        num = _window_sum(_window_sum(out * m, r, 0), r, 1)
        out = np.where(mask, num / np.maximum(den, 1.0), 0.0)
    return out


def local_width(mask, r=2, passes=3):
    """Smooth local width in pixels (float array, 0 outside the tissue): the chord of the largest inscribed circle through the cell, parallel to the nearest boundary (see the module docstring)."""
    m = np.asarray(mask, bool)
    T = local_thickness(m).astype(float)
    Ts = _masked_blur(T, m, r, passes)
    d = np.sqrt(_d2_to_zero(np.pad(m, 1, constant_values=False))[1:-1, 1:-1])
    That = np.maximum(Ts, 2.0 * d)
    return np.where(m, 2.0 * np.sqrt(d * (That - d)), 0.0)


def range_map(mask, pct, h, mode='thickness'):
    """Field range in metres at every pixel: pct % of the local width (pixels * h metres), at least 3 pixels.  mask: boolean array (all True for a rectangle).
    mode 'thickness': the local thickness (a stepped map); 'smooth': the smooth chord width (local_width), which also falls towards every boundary."""
    if mode not in ('thickness', 'smooth'):
        raise ValueError("mode must be 'thickness' or 'smooth'")
    W = local_width(mask) if mode == 'smooth' else local_thickness(mask)
    return np.maximum(pct / 100.0 * (W * h), 3.0 * h)
