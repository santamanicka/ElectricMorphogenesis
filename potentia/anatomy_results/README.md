# Anatomy search results

Potentia run from a uniform start inside a planarian outline and a dorsal tadpole outline, varying only tissue size, proportions and
field range, and matched to schematic targets made from two reference images. Scripts: `../anatomy_*.py`.

| file | what |
|---|---|
| `feature_search_results.jsonl` | one line per run of the feature-filter search (`anatomy_feature_search.py`, `anatomy_feature_controls.py`, `anatomy_feature_m4.py`): 231 runs in all: size x range sweeps, proportions, the drawing-traced planarian outline on 24-64 cell tissues, outline and resolution controls |
| `pixel_search_results.jsonl` | the earlier pixel-by-pixel search (`anatomy_search.py`), kept for comparison; it preferred plain patterns |
| `planarian_r40_a4.4_f20_pinch.npz`, `tadpole_r24_a4.5_f25_w0.4.npz` | the two runs shown in the talk: G_pol and Vmem at the best moment, the score trace, snapshots |
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

Not tracked: the reference images (third-party; put them in `../anatomy/ref/` as `planarian_drawing.png` and `tadpole_photo.png` to
rebuild the targets), the per-run output folder `../anatomy/`, and figures that embed the reference images.
