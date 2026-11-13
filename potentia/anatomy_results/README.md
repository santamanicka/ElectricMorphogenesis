# Anatomy search results

Potentia run from a uniform start inside a planarian outline and a dorsal tadpole outline, varying only tissue size, proportions and
field range, and matched to schematic targets made from two reference images. Scripts: `../anatomy_*.py`.

| file | what |
|---|---|
| `feature_search_results.jsonl` | one line per run of the feature-filter search (`anatomy_feature_search.py`, `anatomy_feature_controls.py`, `anatomy_feature_m4.py`): 231 runs in all: size x range sweeps, proportions, the drawing-traced planarian outline on 24-64 cell tissues, outline and resolution controls |
| `pixel_search_results.jsonl` | the earlier pixel-by-pixel search (`anatomy_search.py`), kept for comparison; it preferred plain patterns |
| `planarian_r40_a4.4_f20_pinch_sim.npz`, `tadpole_r24_a4.5_f25_w0.4_sim_p0.4_q2.42.npz` | the two runs shown in the talk, on the online simulator's own outlines (`../shapes_simulator.py`): G_pol and Vmem at the best moment, the score trace, snapshots |
| `planarian_r40_a4.4_f20_pinch.npz`, `tadpole_r24_a4.5_f25_w0.4.npz` | the same settings on the outlines traced for the search (`../anatomy_targets.py`) |
| `fig_feat_map2.png` | feature match at the best moment of each run, size x field range (planarian: outline traced from the drawing) |
| `fig_feat_shapes_head2.png` | the two runs' settings in five outlines |

Headline numbers (feature match; white noise scores 0.63, one target against the other 0.73):

| | planarian (drawing outline, 40 cells wide, range 20%) | tadpole (24 cells wide, range 25%) |
|---|---|---|
| own outline | 0.86 | 0.80 |
| ellipse / rectangle / capsule | 0.87 / 0.86 / 0.83 | 0.80 / 0.80 / 0.79 |
| the other animal's outline | 0.68 | 0.80 |
| the same frames against the other target | 0.75 | 0.74 |
| twice the pixel resolution (m = 4) | 0.85 | 0.73 |

The planarian outline is traced from the supplied drawing (auricles and the pinch behind the head); Potentia's no-neck preset scores the
same (best 0.87 against 0.86 at 40-64 cells). Larger tissues help: mean best match 0.80 at 24 cells, 0.83 at 40-64.

The best match does not depend on the animal's own outline. What the outline sets is the pattern's symmetry: head-tail asymmetry is 0.00 in
the capsule, ellipse and rectangle and 0.53-0.75 in the two animal outlines. The typical frame (median over the run) is better in the
planarian outline (0.83) than in the capsule or rectangle (0.71) or the ellipse (0.79); for the tadpole settings there is no such difference.

Reproducing a frame in the online simulator: the sweeps used outlines traced from the reference images (`anatomy_targets.py`); the simulator's
`tadpole (dorsal view)` and `planarian` shapes (`shapes_simulator.py`, copied from branch `claude/clever-babbage-lizngo`) agree with them on 96.5%
(tadpole, 24 x 108 cells) and 98.9% (planarian, 40 x 176 cells) of pixels. That is enough to change the frame (the two tadpole patterns correlate 0.64
at 67 s) but not the score (0.80 and 0.86 on both). The model is not chaotic enough over 67 s to explain it: a 1e-9 perturbation of G_pol or a start at
-9.24 instead of -9.2 mV leaves the frame unchanged (correlation 1.0). The frames shown in the talk are the simulator-outline runs:
tadpole: shape tadpole (dorsal view), 24 rows x 108 columns, m = 2, tail depth 0.4, head size / tail size 2.42, field range 25%, G_pol at t = 100 s;
planarian: shape planarian, 40 rows x 176 columns, m = 2, field range 20%, G_pol at t = 232 s; homogeneous start and Stigmergic defaults otherwise.

Not tracked: the reference images (third-party; put them in `../anatomy/ref/` as `planarian_drawing.png` and `tadpole_photo.png` to
rebuild the targets), the per-run output folder `../anatomy/`, and figures that embed the reference images.
