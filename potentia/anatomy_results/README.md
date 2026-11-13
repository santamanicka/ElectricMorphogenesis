# Anatomy search results

Potentia run from a uniform start inside a planarian outline and a dorsal tadpole outline, varying only tissue size, proportions and
field range, and matched to schematic targets made from two reference images. Scripts: `../anatomy_*.py`.

| file | what |
|---|---|
| `feature_search_results.jsonl` | one line per run of the feature-filter search (`anatomy_feature_search.py`, `anatomy_feature_controls.py`): 108 size x range runs, 48 proportion runs, 20 outline and resolution controls |
| `pixel_search_results.jsonl` | the earlier pixel-by-pixel search (`anatomy_search.py`), kept for comparison; it preferred plain patterns |
| `planarian_r32_a4.6_f22.5.npz`, `tadpole_r24_a4.5_f25_w0.4.npz` | the two runs shown in the talk: G_pol and Vmem at the best moment, the score trace, snapshots |
| `fig_feat_map.png` | feature match at the best moment of each run, size x field range |
| `fig_feat_shapes_head.png` | the two runs' settings in five outlines |

Headline numbers (feature match; white noise scores 0.63, one target against the other 0.73):

| | planarian (32 cells wide, range 22.5%) | tadpole (24 cells wide, range 25%) |
|---|---|---|
| own outline | 0.83 | 0.80 |
| ellipse / rectangle / capsule | 0.85 / 0.82 / 0.83 | 0.80 / 0.80 / 0.79 |
| the other animal's outline | 0.65 | 0.80 |
| the same frames against the other target | 0.77 | 0.74 |
| twice the pixel resolution (m = 4) | 0.84 | 0.73 |

The match does not depend on the animal's own outline. What the outline sets is the pattern's symmetry: head-tail asymmetry is 0.00 in
the capsule, ellipse and rectangle and 0.45-0.66 in the two animal outlines.

Not tracked: the reference images (third-party; put them in `../anatomy/ref/` as `planarian_drawing.png` and `tadpole_photo.png` to
rebuild the targets), the per-run output folder `../anatomy/`, and figures that embed the reference images.
