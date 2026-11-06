# Potentia: a continuous-space generalisation of the bioelectric field model

Potentia does for the cellular field network what Lenia does for the Game of Life: the states were
already continuous (Vmem, G_pol), so the generalisation makes **space** continuous.

Scope of this first version (all deliberate):

* pure bioelectric: **no GRN is loaded** (`geneRegulatoryNetwork` is never imported); the original's GRN
  provision is kept as an inert hook (`grn=None`); no ligand either (disabled in the Stigmergic file)
* **finite, non-periodic tissue** (no-flux gap-junction edges; the field comes only from charge inside the tissue)
* **endogenous** patterning from a **homogeneous start** (Vmem = -9.2 mV, G_pol = 1.0 G_ref, eV = 0); **no clamping**

Everything here is numpy/scipy; torch is not needed (`torchfree_io.py` reads the repo's `.dat` files).

## 1. Model

State: `V(x,t)` (membrane potential) and `g(x,t) = G_pol/G_ref`, on a pixel grid of pitch `h = p/m`.
Lengths: `a` = cell radius (5 um), `p = 2a` = cell pitch, `R = a*sqrt(2)*(R_s + .001)` = field range.

```
C dV/dt = I_ion(V,g) + div( D * gate(p |dV|/V0) grad V ),        D = G0 p^2 / C,   gate(x) = 2/(1+cosh x)
dg/dt   = (10/tau) * ( -leak*g + w * tanh((gamma*ebar + beta)/2) ),     g in [0, 2]
E(x)    = fc * (C/p^2) * INT V(y) K(x-y) d2y,     K(r) = r/|r|^2 for |r| <= R       (screened 2D Coulomb)
ebar(x) = mean over s in {+-1}^2 of |E(x + a s)|                                       (perception at cell corners)
```

| lattice model | Potentia |
|---|---|
| cells on an N x N grid | fields on a pixel grid, `m` pixels per cell pitch (`m=1, source='point'` is the lattice exactly) |
| gap junction G_ij(Vi-Vj) between 4 neighbours | gated nonlinear diffusion; the gate acts on the cell-scale difference `p*grad V` |
| Coulomb sum over cell charges Q = C V | convolution of the charge density `C V / p^2` with the truncated kernel (FFT); source pixel area-integrated |
| cell averages the 4 corner probes | same 4 probes, at the physical offsets `(+-a, +-a)` |
| `latticeDims = (11,11)` | tissue extent: `size_cells=(11,11)` or, in physical units, `size_um=(110,110)` |

**What "grid" means here.** Two different things:

1. *Tissue extent* (110 x 110 um). A property of the model: the boundary matters. `size_um` states it in length units.
2. *Pixel grid* (`m`). A numerical device (like Lenia's pixels). The answer should not depend on it, and the lattice is its coarsest setting.

The cell survives only as length scales: `p` in the GJ coupling and gate, `C/p^2` charge density, `a` in the perception offsets, `R` in the field range.
Potentia keeps the lattice's square symmetry (axis-aligned GJ gate, square-corner sensing); it is not isotropic.

## 2. Facts about the original model found along the way (evidence in this directory)

1. **G_pol is a clipped, essentially leak-free integrator.** In the repo `dp = 10*(-G_pol + tanh*w)/tau` with `G_pol` in siemens (~1e-9)
   and `tanh*w` dimensionless, so the effective leak coefficient is 1e-9. Replaying the recorded Stigmergic run with a dimensionally consistent
   leak (`leak=1`) gives loss 4.009 instead of the recorded 2.084 and a 46 mV final-state mismatch (`validate_against_recorded.py`, `LEAK=1`);
   from a homogeneous start the lattice port (11x11) then collapses to a uniform Vmem = -5.7 mV with no pattern. The trained behaviour depends on the integrator.
   Potentia defaults to the as-implemented equation; `leak=1.0` gives the consistent one.
2. **Four of the field parameters are one knob.** At the Stigmergic point the tanh argument never exceeds 5e-4, so the sigmoid is effectively linear.
   `weight=2000` vs `fieldStrength=2, bias=2*bias*` differ by 2.5e-4 mV; vs `gain=-2, bias=2*bias*` by 1e-5 mV; vs `tau=5` by 9e-7 mV
   (while `weight=2000` vs the reference differs by 14.6 mV). The identifiable parameters are range `R_s`, threshold `bias/(|gain| fs)`,
   rate `weight|gain| fs/tau`, and GJ strength. The degeneracy breaks only as `|x|` approaches 1 (17.6 mV at `fs=1000`).
3. **The Stigmergic field parameters were fixed, not learned** (per the authors' clarification): they were chosen to maximise *pattern complexity*, and only the clamp
   protocol was then learned. The repo's complexity measure is TSE complexity (`analyzeSensitivityTSEComplexity.py`, over a 625-point grid of GJ strength, screen size,
   transduction weight and bias). They equal the fixed values in `runLearnCellularFieldNetworkLocal.sh` (`R_s=4, fs=1, weight=1000, gain=-1, GJ=0.05`) and in
   `StigmergicModelParameters.dat`. An earlier version of this note read the bias as learned; that was wrong.
4. *(by reading the code; not executed, torch is unavailable here)* the current `model.py:57` / `cellularFieldNetwork.py:85-93,175-179` appear unable to load
   `StigmergicModelParameters.dat` as-is: its `GRNParameters` has no `GRNEnabled` key (KeyError), and with `GRNEnabled` False the attributes `numGenes`/`GRNTarget` are never set.

## 3. Validation

| check | result |
|---|---|
| Potentia(`m=1`, point source) vs the independent numpy lattice port; finite + periodic, vector + scalar field, homogeneous + random states, non-square domain | max abs difference 4.8e-14 V / 3.6e-13 g (`validate_potentia.py`) |
| the port vs the **recorded** Stigmergic training run (clamped, 1000 it.) | loss 2.084212 vs recorded 2.084209 (rel. 2e-6); final Vmem within 0.0003 mV over 121 cells (`validate_against_recorded.py`). The clamp is replayed only here, as a test. |
| the same check with a consistent leak | fails (loss 4.009, 46 mV): the test discriminates |
| GRN provision | inert by default (no GRN/torch import); bit-identical to no hook when weights and leak are zero; live when given weights (`validate_provision.py`) |
| continuum limit | statistical, **not** pointwise (below) |

"The original" in every comparison is this numpy port, not the torch code itself.

**Continuum limit.** Early (deterministic) dynamics converge; long-time trajectories do not, because the dynamics are chaotic.
On parameter sets that have left the initial state by t = 3 s (63 of 88 in S1): Potentia `m=2` vs `m=4` differ by a median 0.01 mV (0.3% of the departure),
whereas original vs Potentia differ by 0.31 mV (8%, range 2-20% by `R_s`) **independent of `m`** (0.31 at `m=2`, 0.38 at `m=4`):
a systematic lattice-vs-continuum difference, not noise.

**Chaos.** Largest Lyapunov exponent at the Stigmergic point, 11x11, renormalised twin runs (1500 s; the quoted errors are probably underestimates):

| kick | original `m=1` | Potentia `m=2` | Potentia `m=4` |
|---|---|---|---|
| asymmetric (all pixels) /s | +0.054 +- 0.008 | +0.033 +- 0.007 | +0.057 +- 0.009 |
| symmetric (inside the D4 subspace) /s | +0.046 +- 0.010 | +0.021 +- 0.010 | +0.049 +- 0.011 |

Both are chaotic. `m=2` is lower in both rows while `m=1` and `m=4` agree; I could not explain this.

## 4. What patterns appear (Potentia, homogeneous start, finite tissue)

A homogeneous interior has zero field, so `g` integrates upward and the interior hyperpolarises; only the rim senses a field and depolarises.
The rim-core interface (strong field) then sweeps inward as a contracting square front, leaving nested rings; fronts from edges and corners interfere and the
pattern keeps changing. Solutions stay D4-symmetric (square tissue + homogeneous start; checked to within 1e-6 mV over 160 s at `m=2`, 31x31), although the symmetric state is unstable to asymmetric kicks (Lyapunov > 0).

`results/fig_atlas_25_potentia2.png` (25x25 tissue, 88 parameter sets, t = 300 s) shows the families: uniform hyperpolarised (small `R_s`, high threshold);
depolarised sheets with faint corner-seeded diagonal lines (large `R_s`, low threshold); concentric rings and rosettes (`R_s` 4-6, threshold 1-2x);
diamonds and fine kaleidoscopic textures (`R_s` 2-3). A longer field range needs a higher threshold to leave the depolarised state.
`results/fig_regimes_S{1,2,3}_11.png` give the regime maps for range x threshold, rate x threshold and GJ x range.

Regimes are labelled by the criterion you described: does it **settle**, does it **repeat**? The *novelty score* is the median, over the late window, of the rms distance (mV)
between V(t) and the closest state at least 10 s earlier (about 0 for fixed points and periodic orbits, several mV for trajectories that keep visiting new states).
**Caveats:** the 3 mV label threshold is arbitrary (the score distribution has no gap); absolute novelty depends on run length (original Stigmergic point: 7.6 at T=300 s, 3.3 at T=600 s),
so it is used as a relative score at fixed T=300 s; a period longer than the run is indistinguishable from non-repetition (the Lyapunov test above is the independent check);
patterns at `R_s <= 2` are only 1-3 pixels of field range at `m=2` and are not resolved continuum structure.

## 5. Original vs Potentia on the same parameter sets

| sweep (tissue) | parameter sets | regime agreement exact / coarse | rank corr. of late statistics (fh, Vmean, act, novelty) |
|---|---|---|---|
| S1 range x threshold (11x11) | 88 | 0.74 / 0.76 | 0.96, 0.98, 0.84, 0.79 |
| S2 rate x threshold (11x11) | 99 | 0.86 / 0.88 | 0.99, 1.00, 0.97, 0.89 |
| S3 GJ x range (11x11) | 40 | 0.57 / 0.70 | 0.97, 0.98, 0.74, 0.86 |
| S1 sub-grid (25x25) | 20 | 0.90 / 0.90 | 0.99, 1.00, 0.97, 0.99 |

* **Regime labels are too noisy to attribute disagreement to the lattice**: Potentia `m=2` vs `m=4` (the continuum-vs-continuum control) agree 0.76 / 0.80 on S1, the same as original vs Potentia.
  The "disagreement" example in `fig_side_by_side_B.png` is novelty 2.6 vs 4.3/4.0 mV straddling the 3 mV cutoff.
* **Early snapshots show a real, systematic difference** (8%, above). **Late snapshots differ pointwise everywhere** (chaos); mean quantities agree (the hyperpolarised-fraction curves of the
  three models in `fig_fine_F1.png` lie almost on top of each other).
* **A lattice artifact Potentia removes:** in the original, `R_s=1` and `R_s=2` are identical (the field range jumps in shells; the next shell is at `R_s=2.24`).
  The original's novelty maximum at `R_s=1-2` (14.9 mV) does not survive: Potentia `m=2` and `m=4` both settle to uniform hyperpolarised there (`fig_side_by_side_A.png`).
* Potentia's novelty is lower than the original's in all 10 cells of the 25x25 sub-grid where either model has novelty >= 3 mV (by 14-51%); the weak cells go both ways. Cause not tested.
* Same family, different detail at the Stigmergic point (`fig_side_by_side_A.png`, `fig_side_by_side_25.png`): near-identical early fronts, then 4-fold rosettes of rings and spots in both, not the same pattern.
  Potentia's structures are smooth curves; the original's are cell-pixelated.

## 6. Is the Stigmergic parameter set special in Potentia?

"Optimal" here means the field parameters chosen to maximise pattern complexity (section 2, item 3). **I did not compute TSE complexity.** As a proxy I used the property described for
the optimal model (patterns that are continuously generated, never settle, never repeat) via the novelty score at T=300 s, 11x11 tissue (`analysis_optimum.py`, `results/analysis_optimum.txt`).
Novelty is not TSE complexity; whether the two peak at the same parameters is untested.

* **The Stigmergic point has the property in all three models**: perpetual novelty in the original and in Potentia (`m=2`, `m=4`), positive Lyapunov exponents (section 3), and the 25x25 sub-grid
  gives 15.5 mV (original) / 13.4 mV (Potentia).
* **The chosen bias is at the novelty peak.** Along bias at `R_s=4` (F1, 20 values x 3 jitter replicates) the peak is at 1.12x (original), 0.89x (`m=2`), 1.26x (`m=4`) of the Stigmergic bias; the
  Stigmergic value is within the half-maximum window in all three and reaches 92% / 91% / 62% of the peak. Across `R_s = 2.5-4.5` the best bias in the 2-D grid (F3) is 1.0x (0.84-1.19x) in both models.
* **It sits near the edge of a cliff.** Novelty collapses between about 1.4x and 1.8x the Stigmergic bias in every model (sharply near 1.5x in the original, over a wider range in Potentia; `fig_fine_F1.png`), so the chosen bias has roughly 40% headroom on the high side.
* **The hand-set field range is not at the optimum in any model.** Along `R_s` at the Stigmergic bias (F2) novelty peaks at `R_s` = 2.5 (original, `m=2`) or 3 (`m=4`); `R_s=4` reaches 45% / 66% / 65% of the peak.
  In the 2-D grid the Stigmergic cell is 51% (original, rank 16/72) and 50% (Potentia, rank 21/72) of the grid maximum (14.9 mV at `R_s=2`, bias 0.71x; 10.3 mV at `R_s=2.5`, bias 1.0x).
* So under this proxy the chosen threshold carries over from the lattice to Potentia; the field range does not sit at the optimum in either. The landscape is qualitatively the same, with lower novelty magnitudes in Potentia.

**What this does not show:** it is not the criterion the parameters were chosen by (TSE complexity); the novelty score is my proxy; the jitter "replicates" perturb parameters by 0.3% and measure
sensitivity to that, not run-to-run noise; all are single deterministic trajectories. The direct test is TSE complexity (or the repo's other complexity measures) evaluated on Potentia over the same four-parameter grid.

## 7. Limitations and open items

* The 11x11 tissue is small: for `R_s >= 8` the field range spans most of the tissue, so results there are boundary-dominated.
* Patterns are D4-symmetric by construction; the effect of noise on pattern selection was not studied (the symmetric state is unstable to asymmetric kicks).
* The `m=2` Lyapunov dip and the lower Potentia novelty are unexplained. The 3 mV label threshold is arbitrary.
* No GRN, ligand, or clamp; Potentia is not isotropic (square GJ gate and perception footprint).
* TSE complexity of Potentia over the repo's (GJ, range, weight, bias) grid has not been computed; it is the direct test of whether the chosen set maximises complexity in Potentia.
* An isotropic variant, a convergence study at `R_s <= 2` with `m >= 8`, and finite-tissue noise runs are the obvious next steps.

## 8. Reproduce

```bash
cd potentia
python validate_potentia.py && python validate_against_recorded.py && python validate_provision.py     # checks
python atlas.py S1 11 lattice; python atlas.py S1 11 potentia:2                                      # sweeps (S1/S2/S3, models lattice | potentia:m)
python compare_models.py S1 11                                                                       # original vs Potentia, with m=2 vs m=4 control
python lyapunov.py 2 1500 asym; python fine_scans.py F1 potentia:2; python analysis_optimum.py       # chaos and optimum location
python make_figures.py                                                                               # figures -> results/
```

## 9. Interactive visualizer

```bash
python potentia/visualizer.py --open          # then http://127.0.0.1:8765  (numpy + scipy only; binds to localhost)
```

![visualizer](results/fig_visualizer.png)

Inputs: tissue size (cells, rows x columns), pixel resolution `m`, field parameters (`R_s`, threshold bias, field strength, gain, weight, tau, vector/scalar field) and GJ strength,
all defaulting to the Stigmergic set read from `data/StigmergicModelParameters.dat` ("Stigmergic defaults" restores it); run time `T`; and the initial condition:
**homogeneous** (V0, g0), **random** (Gaussian around V0/g0 with sigma_V, sigma_g and a seed, or V uniform in [-55, -5] mV) or **manual painting** (drag on the field to set V or g, brush radius, left-right /
top-bottom mirroring). Views: Vmem, G_pol, and sensed field / bias (above 1 a cell's G_pol falls). A strip shows mean Vmem, hyperpolarised fraction and mean g against time. Painting works while paused,
so it can also perturb a running tissue. "Save PNG" and "Export state (.npz)" are provided. Changing a setting needs "Initialise" (the status chip says so); the simulation runs in the Python server,
so the UI shows exactly the validated `Potentia` model (no separate JavaScript implementation).

Tested here without a display: `test_visualizer_api.py` (17 checks, including 500 server steps matching a direct `Potentia` run to float32 transport precision, painted-state round trip, seeded random starts,
size limits, run-time clamping, export) and `visualizer/ui_smoke_test.js` (15 checks in headless Chromium: run/pause/stop at `T`, painting with mirroring, painted values reaching the server, refusal to paint
while running, view switching, resize, error display, no console errors). I have not tested it in Firefox/Safari or on a touch device. It runs locally because the cloud session is not reachable from a browser.

Notes: one simulation per server (a second tab shares it); grids are limited to 512 pixels per side and 200,000 pixels in total; at m=2 on 11x11 to 25x25 cells it ran at roughly 30-50 simulated seconds per
wall second in the headless test (cost grows roughly with m^2). The sensed-field view needs one extra field evaluation per frame.
