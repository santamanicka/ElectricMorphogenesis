"""A schematic frog developmental sequence for Potentia: the tissue shape morphs over time from a small disc (egg) to a tadpole.

The shape is *prescribed*: it follows a schedule of keyframes and the bioelectric dynamics run on top of it; nothing here decides where tissue grows.
The sequence is a 2D dorsal outline, head on the left, with approximate Nieuwkoop-Faber stage labels; sizes are schematic (see POTENTIA.md section 9).
Each keyframe shape is nested in the next (the tissue only grows): the width of the tail never shrinks and every stage lies inside the following one.

  keyframes   (tau, shape, box) with tau in [0, 1] the fraction of the development duration; the box is where the shape sits on the pixel grid, in units of the
              grid height (x to the right, y downwards; x1 = None means the grid width), so the same table serves any grid whose width is at least about 2.7 heights
  morph       between two keyframes the signed-distance fields of the two shapes are blended (smoothstep easing) and thresholded at 0
  new tissue  pixels that appear inherit the average state (V and G_pol) of their already-assigned 4-neighbours, layer by layer (order independent);
              a new island with no tissue neighbour gets a given background state; pixels that disappear are set to V = g = 0
  range       the field range can be constant, proportional to the shorter extent of the current tissue, or interpolated between per-stage values

Everything is exact integer geometry or plain IEEE arithmetic in the same order as visualizer/core.js (PotentiaCore.Development); tests compare the two.
"""
import numpy as np
from scipy import ndimage

from shapes import shape_mask

STAGES = [
    dict(name='Egg', nf='NF 1', tau=0.00, shape='disc', p=None, box=(0.333, 0.667, 0.333, 0.667)),
    dict(name='Blastula', nf='NF 8', tau=0.18, shape='disc', p=None, box=(0.275, 0.725, 0.275, 0.725)),
    dict(name='Gastrula', nf='NF 11', tau=0.36, shape='disc', p=None, box=(0.20, 0.80, 0.20, 0.80)),
    dict(name='Neurula', nf='NF 17', tau=0.52, shape='ellipse', p=None, box=(0.08, 0.92, 0.20, 0.80)),
    dict(name='Tailbud', nf='NF 25', tau=0.74, shape='tadpole', p=0.28, box=(0.08, 2.0, 0.08, 0.92)),
    dict(name='Tadpole', nf='NF 45', tau=1.00, shape='tadpole', p=0.25, box=(0.0, None, 0.0, 1.0)),
]
A_UM = 5.0                                   # cell radius, um (the cell pitch is 2a = 10 um)


def _fr(x):
    """Round half up (the same in JavaScript: Math.floor(x + 0.5))."""
    return int(np.floor(x + 0.5))


def stage_mask(stage, R, C):
    """Boolean mask (R, C) of one keyframe: its shape generated in its box and placed on the grid."""
    A = C / R
    x0, x1, y0, y1 = stage['box']
    x1 = A if x1 is None else min(x1, A)
    x0 = min(x0, x1)
    c0, c1 = _fr(x0 * R), _fr(x1 * R)
    r0, r1 = _fr(y0 * R), _fr(y1 * R)
    c1 = min(max(c1, c0 + 2), C); c0 = max(0, min(c0, c1 - 2))
    r1 = min(max(r1, r0 + 2), R); r0 = max(0, min(r0, r1 - 2))
    out = np.zeros((R, C), bool)
    out[r0:r1, c0:c1] = shape_mask(stage['shape'], (r1 - r0, c1 - c0), stage['p'])
    return out


def signed_distance(mask):
    """Signed distance (negative inside) from exact Euclidean distance transforms; the grid is padded by one outside pixel so that a full row or column still has a boundary."""
    pad = np.pad(np.asarray(mask, bool), 1, constant_values=False)
    d_in = ndimage.distance_transform_edt(pad)           # inside pixel -> nearest outside pixel
    d_out = ndimage.distance_transform_edt(~pad)         # outside pixel -> nearest inside pixel
    root = lambda d: np.sqrt(np.rint(d * d))              # squared distances are integers: recover them exactly, then one correctly rounded sqrt
    return (root(d_out) - root(d_in))[1:-1, 1:-1]


def smoothstep(u):
    return u * u * (3.0 - 2.0 * u)


class Development:
    """The schedule on a pixel grid Np = (R, C): masks, stage labels and the field-range schedule as functions of the development fraction tau."""

    def __init__(self, Np, duration=300.0, update=0.5, stages=STAGES):
        self.R, self.C = int(Np[0]), int(Np[1])
        self.D = float(duration)
        self.update = float(update)
        self.stages = stages
        self.taus = [s['tau'] for s in stages]
        self.masks = [stage_mask(s, self.R, self.C) for s in stages]
        self.sdf = [signed_distance(m) for m in self.masks]

    def tau(self, t):
        return min(1.0, max(0.0, t / self.D))

    def _segment(self, tau):
        n = len(self.taus)
        if tau >= self.taus[-1]:
            return n - 1, 0.0
        i = 0
        while i < n - 2 and tau >= self.taus[i + 1]:
            i += 1
        u = (tau - self.taus[i]) / (self.taus[i + 1] - self.taus[i])
        return i, smoothstep(u)

    def mask_at(self, tau):
        i, s = self._segment(tau)
        if s == 0.0:
            return self.masks[i].copy()
        return ((1.0 - s) * self.sdf[i] + s * self.sdf[i + 1]) <= 0.0

    def stage_at(self, tau):
        i, _ = self._segment(tau)
        return self.stages[i]

    # ---- field range schedule ----
    def range_um(self, mode, tau, mask, h_um, const_um=28.3, pct=25.0, stage_um=None):
        """Field range in um at tau: 'const', 'prop' (pct % of the shorter extent of the current tissue) or 'stage' (smoothstep between per-stage values).
        Never below three pixels (below that the kernel is not resolved)."""
        if mode == 'prop':
            rows = np.where(mask.any(axis=1))[0]; cols = np.where(mask.any(axis=0))[0]
            ext = min(int(rows[-1] - rows[0] + 1), int(cols[-1] - cols[0] + 1)) * h_um
            r = pct / 100.0 * ext
        elif mode == 'stage':
            i, s = self._segment(tau)
            r = stage_um[i] if s == 0.0 else (1.0 - s) * stage_um[i] + s * stage_um[i + 1]
        else:
            r = const_um
        return max(r, 3.0 * h_um)


def grow_state(V, g, old, new, fill=(0.0, 0.0)):
    """State after the tissue changes from mask `old` to mask `new`: persisting pixels keep their state, removed pixels are zero, new pixels take the mean of their
    already-assigned 4-neighbours, layer by layer (each layer uses only the previous layers, so the result does not depend on any pixel order); a new pixel that
    no chain of new pixels connects to old tissue gets `fill` = (V, g)."""
    old = np.asarray(old, bool); new = np.asarray(new, bool)
    assigned = old & new
    Vo = np.where(assigned, V, 0.0); go = np.where(assigned, g, 0.0)
    remaining = new & ~old
    while remaining.any():
        a = assigned.astype(float)
        cnt = np.zeros_like(a); sv = np.zeros_like(a); sg = np.zeros_like(a)
        for (dr, dc) in ((1, 0), (-1, 0), (0, 1), (0, -1)):           # up, down, left, right: summed in this order
            sl_to = (slice(max(dr, 0), a.shape[0] + min(dr, 0)), slice(max(dc, 0), a.shape[1] + min(dc, 0)))
            sl_from = (slice(max(-dr, 0), a.shape[0] + min(-dr, 0)), slice(max(-dc, 0), a.shape[1] + min(-dc, 0)))
            cnt[sl_to] = cnt[sl_to] + a[sl_from]
            sv[sl_to] = sv[sl_to] + Vo[sl_from]
            sg[sl_to] = sg[sl_to] + go[sl_from]
        cand = remaining & (cnt > 0)
        if not cand.any():
            break
        Vo = np.where(cand, sv / np.where(cnt > 0, cnt, 1.0), Vo)
        go = np.where(cand, sg / np.where(cnt > 0, cnt, 1.0), go)
        assigned = assigned | cand
        remaining = remaining & ~cand
    if remaining.any():
        Vo = np.where(remaining, fill[0], Vo); go = np.where(remaining, fill[1], go)
    return Vo, go


def dev_update(P, dev, n, V, g, rng, fill=(0.0, 0.0)):
    """One development update after step number n (time n * dt): morph the tissue to the shape at this time and move the field range along its schedule.
    rng = dict(mode='const'|'prop'|'stage', const_um, pct, stage_um).  Returns (V, g, status); the model P is changed in place (mask and range)."""
    tau = dev.tau(n * P.dt)
    new = dev.mask_at(tau)
    cur = P.mask if P.mask is not None else np.ones(P.Np, bool)
    changed = not np.array_equal(new, cur)
    if changed:
        V, g = P.set_mask(new, V, g, fill)
    r_um = dev.range_um(rng.get('mode', 'const'), tau, new, P.h * 1e6, rng.get('const_um', 28.3), rng.get('pct', 25.0), rng.get('stage_um'))
    if abs(r_um * 1e-6 - P.R) >= 0.25 * P.h:
        P.set_range(r_um / (A_UM * np.sqrt(2)) - 0.001)
    st = dev.stage_at(tau)
    return V, g, dict(tau=tau, stage=st['name'], nf=st['nf'], tissue=int(new.sum()), range_um=P.R * 1e6, changed=changed)
