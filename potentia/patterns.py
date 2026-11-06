"""Spatio-temporal descriptors and a rule-based regime label for Potentia runs.

Patterns from a homogeneous start in a finite tissue are generally *dynamic* (fronts, then
persistent oscillation), so descriptors are computed over a late time window and include both
spatial and temporal structure.  All voltages are in mV inside this module.

Descriptors (late window = second half of the run):
  fh        mean fraction of tissue hyperpolarised (V < V_BIST, the bistable midpoint)
  Vmean     mean Vmem
  sd_x      spatial standard deviation of Vmem (mean over time)         -> pattern contrast
  act       mean over space of the temporal std of Vmem                 -> 0 for static states
  edge      fraction of neighbouring cell pairs straddling V_BIST       -> spatial fine-scale-ness
  ncomp     connected hyperpolarised components (cell level, 4-conn.)
  f_peak    frequency (Hz) of the peak of the pixel-averaged temporal PSD (nan if act ~ 0)
  peak_rat  peak power / total non-DC power  (1 = pure oscillation, ~0 = broadband)
  act_trend activity in the last third of the run / activity in the middle third (<<1: decaying, i.e. settling)
  novelty   median over the late window of the rms-over-cells distance (mV) from V(t) to the *closest earlier
            state at least tau_min s in the past*.  ~0 for fixed points and for periodic/recurrent orbits,
            several mV for trajectories that keep visiting new states ("never settles, never repeats").
            Caveat: an orbit whose period exceeds the run length is indistinguishable from non-repeating.
  t_nuc     first time any cell >= 3 cells from every edge moves > 5 mV from its initial value
  t_core    first time the central 1/3 x 1/3 of the tissue has half switched from its initial state
"""
import numpy as np
from scipy import ndimage

V_BIST = -30.0          # mV, midpoint of the bistable {-50, -9} states
V_INIT = -9.2


def _cell_blocks(X, m):
    n0, n1 = X.shape[-2] // m, X.shape[-1] // m
    return X.reshape(*X.shape[:-2], n0, m, n1, m).mean(axis=(-3, -1))


def describe(Vt, tt, m, size, late_frac=2 / 3, tau_min=10.0):
    """Vt: (nt, Np0, Np1) pixel-level Vmem in mV; tt: (nt,) seconds."""
    nt = len(tt)
    late = slice(int(nt * (1 - late_frac)), nt)
    Vc = _cell_blocks(Vt, m)                      # (nt, n0, n1)
    Vl = Vc[late]
    d = {}
    d['fh'] = float((Vl < V_BIST).mean())
    d['Vmean'] = float(Vl.mean())
    d['sd_x'] = float(Vl.reshape(len(Vl), -1).std(axis=1).mean())
    d['act'] = float(Vl.std(axis=0).mean())
    hb = Vl < V_BIST
    d['edge'] = float(0.5 * ((hb[:, 1:, :] != hb[:, :-1, :]).mean() + (hb[:, :, 1:] != hb[:, :, :-1]).mean()))
    d['ncomp'] = float(np.mean([ndimage.label(h)[1] for h in hb[::max(1, len(hb) // 20)]]))
    # temporal spectrum, averaged over pixels (cell level), Hann window, DC removed per pixel
    Vx = Vl.reshape(len(Vl), -1)
    Vx = Vx - Vx.mean(axis=0, keepdims=True)
    w = np.hanning(len(Vx))[:, None]
    P = (np.abs(np.fft.rfft(Vx * w, axis=0)) ** 2).mean(axis=1)
    dt_rec = tt[1] - tt[0]
    fr = np.fft.rfftfreq(len(Vx), dt_rec)
    P[0] = 0
    tot = P.sum()
    if d['act'] < 0.05 or tot <= 0:
        d['f_peak'], d['peak_rat'] = np.nan, np.nan
    else:
        k = int(np.argmax(P))
        d['f_peak'] = float(fr[k])
        d['peak_rat'] = float(P[max(0, k - 1):k + 2].sum() / tot)
    # settling and novelty (recurrence against the whole earlier history)
    third = nt // 3
    a_mid = Vc[third:2 * third].std(axis=0).mean()
    a_last = Vc[2 * third:].std(axis=0).mean()
    d['act_last'] = float(a_last)
    d['act_trend'] = float(a_last / a_mid) if a_mid > 1e-9 else np.nan
    X = Vc.reshape(nt, -1).astype(np.float64)
    G = X @ X.T
    sq = np.diag(G)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0) / X.shape[1])   # rms mV between states
    lag = int(round(tau_min / dt_rec))
    i0 = max(lag, int(nt * (1 - late_frac)))
    nov = [D[i, :i - lag + 1].min() for i in range(i0, nt)]
    d['novelty'] = float(np.median(nov)) if nov else np.nan
    d['novelty_p10'] = float(np.percentile(nov, 10)) if nov else np.nan
    # nucleation / core collapse (early, deterministic)
    n0, n1 = size
    mar = 3
    inner = Vc[:, mar:n0 - mar, mar:n1 - mar] if min(n0, n1) > 2 * mar + 1 else Vc
    moved = (np.abs(inner - V_INIT).reshape(nt, -1).max(axis=1) > 5.0)
    d['t_nuc'] = float(tt[np.argmax(moved)]) if moved.any() else np.nan
    c0, c1 = n0 // 3, n1 // 3
    core = Vc[:, c0:n0 - c0, c1:n1 - c1].reshape(nt, -1)
    sw = (np.abs(core - V_INIT) > 20.0).mean(axis=1)   # fraction of core cells that have switched by > 20 mV
    d['t_core'] = float(tt[np.argmax(sw > 0.5)]) if (sw > 0.5).any() else np.nan
    d['core_final_sw'] = float(sw[-1])
    return d


def classify(d, sd_thresh=2.0, act_thresh=0.5, nov_thresh=3.0):
    """Regime label; the main axis is the user's criterion: does it settle, does it repeat?
    Thresholds are descriptive (calibrated on the sweep distributions, see POTENTIA.md), not theoretical."""
    if d['act_last'] < act_thresh:
        if d['sd_x'] < sd_thresh:
            if d['Vmean'] < -40:
                return 'settled: uniform hyperpolarised'
            if d['Vmean'] > -20:
                return 'settled: uniform depolarised'
            return 'settled: uniform intermediate'
        return 'settled: static pattern'
    if not np.isfinite(d['novelty']) or d['novelty'] < nov_thresh:
        return 'recurrent: periodic oscillation' if d['sd_x'] < sd_thresh else 'recurrent: periodic pattern'
    return 'perpetual novelty: oscillation' if d['sd_x'] < sd_thresh else 'perpetual novelty: pattern'


REGIMES = ['settled: uniform depolarised', 'settled: uniform intermediate', 'settled: uniform hyperpolarised',
           'settled: static pattern', 'recurrent: periodic oscillation', 'recurrent: periodic pattern',
           'perpetual novelty: oscillation', 'perpetual novelty: pattern']
