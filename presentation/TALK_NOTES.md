# Can bulk pattern formation be canalized from the boundary?

Notes for a 10-minute talk at GSO-2026, the conference on guided self-organization (Binghamton, 14–15 October 2026). The system is an 11 × 11 bioelectric tissue model: its 40 outer cells are the *boundary*, its 81 inner cells the *bulk*.
Two targets are used, a **stripe** (the simple pattern, 27 cells) and a **face** (the complex pattern, 14 feature cells). The slides carry pictures and ideas, not numbers; numbers are in section 8. Quantities marked **new** are exploratory (nothing registered beforehand).

## 1. Motivation

The aim is to learn the *grammar of guidance*: how little information, in what form and applied where, is needed to steer a self-organizing tissue to a target pattern, and what limits it.
Development already works this way. Organizers and prepatterns, a few localized signals, guide a tissue that does most of the organizing itself. Guided self-organization asks the same of any self-organizing system: steer it toward an outcome with non-specific guidance.
A bioelectric tissue makes the question concrete, because the guide can be a boundary condition and the bulk's own rules stay untouched. If a few boundary parameters suffice, tissue patterning becomes controllable with few inputs.

The approach: take two targets of different complexity on one frozen tissue, find the smallest boundary code that reaches each, trace how the guidance travels inward, and map where it fails.
The answer to the title: **yes, as compression and as canals (many codes, one pattern); no, as robustness.** The simpler the target, the more readable and the more canalized the code.

## 2. The code is a spatial organizer

An organizer is a localized source that guides a field of cells without specifying the outcome cell by cell:

| Organizer | What it does | Lesson |
|---|---|---|
| Spemann–Mangold (amphibian dorsal lip, 1924) | grafted ventrally, induces a second axis built mostly from host tissue | the guide is small and portable; the response belongs to the tissue (its *competence*). Replacing the boundary code is the analogue of the graft |
| Hydra head organizer (Browne, 1909; Gierer–Meinhardt model) | induces and maintains a head | organizer and self-organizing response belong together |
| Limb bud ZPA (Shh) and AER (FGF) | a grafted ZPA mirror-duplicates the digits | organizers sit at the *margin* of the field they pattern |
| Fly embryo: Bicoid at the anterior pole, the terminal (Torso) system | a few parameters of a polar source specify the axis | a low-dimensional code at the ends: level, gradient between poles, both poles at once are the dial, tilt and oval |
| Neural tube floor plate (Shh) | an edge source sets stripes of cell types (Wolpert's French flag in a real tissue) | the stripe target is the central band of that picture |

Three properties carry over. **Guidance is instructive but small**: different codes select different patterns, and 2–4 numbers steer 81 cells. **It sits at the boundary**: organizers do, edge cells can find themselves from their neighbour count, and a diffusively coupled
tissue is steered most cheaply from its edge (the data show a boundary code is sufficient; inner placement is untested). **It is transient**: the code is held, then released.

## 3. Two phases: guidance, then self-organization

| Phase | What happens | Evidence |
|---|---|---|
| guidance (the hold, iterations 0–300) | the boundary is overwritten every iteration and the code is copied inward | each order of the code owns its own modes of the bulk pattern (share owned by one order 1.00 early in the hold; face). The stripe is already written: 27/27 cells dark at 300 |
| self-organization (the release) | the tissue's own machinery, electric-field feedback and bistable latching cells, turns the imprint into the pattern, and the one-to-one link to the code dissolves | share owned 0.61 at release, 0.06 when the face appears; the information is delocalised, not destroyed (held-out readout 0.89 near the trained code). With the field feedback off the imprint stays and no cell goes dark. The stripe forms even if the boundary is never released; the face needs the release (held throughout it reaches 21–57%) |

A simple organizer writes its pattern while it is held; a complex one is finished by the tissue long after it lets go (stripe read at iteration 504, face at 2173). Instructive then permissive: while held the code instructs, once released the tissue's competence completes the pattern.
The ownership result exists for the face only, and the organizer is an organising picture, not a mechanism claim.

## 4. Canalization as compression

* **The code is lower-dimensional than the pattern.** 81 interior cells (45 independent under the left–right mirror) are steered by 4 numbers for the face (orders 0–3; more add nothing) and 2 effective numbers for the stripe (a dial and an oval; the tilt is 0.001), a compression of 20–40×.
  Behind the code sit 4 region levels and 2 switches: the stripe's two ends, the face's two modules.
* **Many codes, one pattern** (Waddington's sense; figure 5). Turn the dial and the pattern holds, then jumps. Along 201 dial settings the stripe visits 18 distinct patterns (11 settings each), the face 57 (3.5 each). Over random space-filling codes the stripe lands on an effective
  22 patterns from 160 codes (7 codes each, spread over about 7 of the 81 dimensions), the face on an effective 167 from 280 (1.7 each, about 16 dimensions) (**new**).
* **The qualification.** The compression holds for the code that writes the pattern directly. The stripe's two-wave code needs boundary cells driven past the bistable edge (ceiling 2.0; the face stayed at 1.3). Under the face's ceiling the stripe needs all 11 independent boundary values:
  no compression. And canalized is not robust: the exact target is a thin slice (dial window 0.008 for the stripe, under 0.004 for the face; 1% noise on the 40 values keeps the stripe at overlap ≥ 0.9 in 23 of 100 draws, the face in 0).

## 5. What can be claimed

1. **Canalized** means compression and canals, not robustness. Gloss it early: a few numbers steer a high-dimensional pattern into a few discrete outcomes.
2. **Readable, not linear.** The boundary profile is a sum of waves, but the response is switch-like (whole rows of three cells, a window 0.01 wide).
3. **Simple pattern, readable code; complex pattern, distributed code.** The stripe's two bumps write its two ends independently; the face has no single writing part, its orders are not separable brushes (0 of 16 role predictions held), and sliding any order gives a largely shared family of fragments.
4. **Passing states, not attractors.** Both targets are moments the tissue passes through (stripe 34 iterations, face 55–119 of 20,000). Say "guided to a specific state", and point to the canals as the nearest thing to attractors.

## 6. The visuals (dark theme, 1920 × 1080) and a 10-minute order

Boundary cells are violet (brighter = higher held conductance); interior cells glow where hyperpolarised, cyan for the stripe and amber for the face. The boundary's colour scale is stretched per target so the stripe's two bumps show; the glow is decoration.
In movie A each panel stops at its target's moment (stripe 504, face 2173) because the tissue keeps moving afterwards.

| File | What it shows | Use |
|---|---|---|
| `movies/A_guideThenLetGo.mp4` (9 s) | the organizer appears, is held (guidance), lets go (self-organization); stripe forms, then face | opener, or close |
| `1_theSpatialOrganizer.png` | a few waves → the boundary profile → the pattern | the idea |
| `2_twoPhases.png` | seven moments of each target, bracketed by guidance and self-organization | simple is written while held, complex long after |
| `3_stripeTwoSwitches.png` | top end only, bottom end only, both | the simple organizer is readable |
| `movies/B_turningTheKnobs.mp4` (9 s) | each knob turned: three for the stripe, four for the face | few knobs, many cells; tidy versus fragmenting |
| `5_canalsOfTheDial.png` | long flat steps = canals, with thumbnails | many codes, one pattern |
| `4_faceNoSinglePart.png` | upper wall alone, lower wall alone, whole boundary | the complex organizer is a distributed key |
| `6_thePush.png` | strongest transfers over the pattern: a direct push into the stripe, a loop through the face | how the boundary reaches the bulk |

Order: A (1:00) → 1 (1:15) → 2 (1:15) → 3 (1:30) → B and 5 (2:00) → 4 and 6 (1:30) → close on the last frame of A or on 1 (0:45). Number-heavy companions are in `backup_quantitative/` (they call the boundary "ring").

## 7. Where it sits: background, and what sets it apart

**Background** (the links come from a quick, non-systematic search; Kauffman is cited from memory):
* *Guided self-organization:* steering a self-organizing system with non-specific guidance (Prokopenko, Ay, Polani), mostly information-theoretic and agent-based ([overview](https://arxiv.org/pdf/1304.1842)).
* *Organizers and self-organization:* Gierer–Meinhardt's Hydra model ([paper](https://bio.mpg.de/255219/gierer-and-meinhardt-1972)); Wolpert's French Flag was posed as a problem, and his first solution was self-organizing ([Sharpe 2019](https://cob.silverchair.com/dev/article-pdf/1969505/dev185967.pdf));
  gradients orient Turing-like patterns ([Hiscock and Megason](https://pmc.ncbi.nlm.nih.gov/articles/PMC4707970)).
* *Bioelectric:* the [electric-face prepattern](https://now.tufts.edu/2011/07/18/face-frog-time-lapse-video-reveals-never-seen-bioelectric-pattern); a brief bioelectric perturbation rewrites a planarian body plan ([Durant et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5443973));
  [BETSE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4933718/); Manicka and Levin ([2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6901451/), [2022](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8774453/)).
* *Control and inverse design:* boundary control of Turing patterns ([example](https://math.ucsd.edu/seminar/optimal-control-weakly-nonlinear-pattern-formation)); [network controllability](https://barabasi.com/publications/23/controlling-networks);
  [neural cellular automata](https://mlanthology.org/distill/2020/mordvintsev2020distill-growing) learn the rules that grow a target; [indirect encodings](https://stars.library.ucf.edu/scopus2015/4494) compress a phenotype into a short description;
  [positional information](https://pmc.ncbi.nlm.nih.gov/articles/PMC4286692) measures how precisely a pattern is read out.
* *Self-organization and evolution:* Kauffman's "order for free" (*The Origins of Order*, 1993) holds that generic self-organizing dynamics supply much of the order that selection then acts on, with canalizing functions and neutral networks making outcomes robust and evolvable.
  This work is developmental, with no heredity or selection among tissues, so it neither tests nor refutes that claim. It is consistent with the developmental half (generic dynamics supply most of the organization, a compressed and canalized guide steers it, the plateaus resemble neutral networks),
  while the narrow peaks, the search needed for specific targets and the unreachable double stripe show that self-organization alone did not deliver them. A safe line: *a self-organizing tissue offers a compressed, canalized control surface, the kind of substrate selection could act on.*

**What sets it apart** (beyond the bioelectric modality):
* *A different inverse problem.* Most work fixes the guide and asks what emerges, or fixes the target and learns the rules. Here the responder is fixed and the unknown is the minimal guide: its dimension, readability and where it must act.
* *The guide is characterised, not just found.* How its size and readability change with target complexity, with the causal relay showing which boundary parts write which parts of the pattern.
* *Guidance to a passing state:* the guide's grip is gone before the pattern appears, and the pattern is a moment, not an attractor.
* *Limits are results.* The null (0 of 632 double-stripe restarts), the thin windows, and pre-registered, confirmatory testing in a simulation study.

**Two cautions.** The responder is not naive: the tissue (checkpoint 1888) was trained earlier to make the face under a per-cell boundary clamp (100-iteration hold), so it is competent for boundary guidance by construction and the stripe is a transfer to a second target.
Here the tissue stays frozen and new low-dimensional codes (hold 301) are searched. And the search above was quick, so claims of novelty are provisional until boundary control of reaction–diffusion and bioelectric inverse design are checked directly.

## 8. Questions and numbers

* *Is this canalization?* Compression and canals, not robustness.
* *Is the organizer a mechanism?* No: it organises the two phases and the loss of correspondence.
* *Why is the simple target not the easier one to find?* The stripe's two-wave code was found by 1 restart in 20 (ceiling 2.0), the face's by 14 of 30; under the face's ceiling the stripe needs 11 harmonics.
* *Does it work for any pattern?* No: the inverted stripe was reached by none of 632 restarts (best overlap 0.70).
* *What was pre-registered?* The stripe gate, mechanism and ends tests, the face's module confirmation (8 of 9) and the order-role knockouts; everything marked **new** is exploratory.

| Quantity | Value | Source |
|---|---|---|
| Codes | stripe a₀ 1.350, a₁ 0.0012, a₂ 0.137 (ten boundary cells above 1.439); face a₀ 0.817, a₁ 0.250, a₂ 0.434, a₃ −0.280 | trained runs |
| Reach | stripe 8.92 mV vs best random 18.13; face 8.93 vs 20.6–22.4 | Stripes, Training the Ring |
| Independent values | boundary 40 (21 / 11 under one / two mirrors); interior 81 (45 / 25) | lattice |
| Dial window | stripe overlap ≥ 0.9 at 4 of 81 settings (0.008), face 1 of 81; overlap ≥ 0.5 over 59% of the ±0.08 dial (stripe) vs 5% (face) | `data/canalizationTalkNumbers.json` (**new**) |
| Ensemble of random codes | stripe 160 codes: 58 distinct patterns, effective 22, participation ratio 7.3; face 280 codes: 219 distinct, effective 167, participation ratio 16.3 | `data/canalizationTalkPatternEnsemble1888Hold301.json` (**new**) |
| Noise on the boundary | 1%, 100 draws: overlap ≥ 0.5 in 98 (stripe) vs 50 (face); ≥ 0.9 in 23 vs 0; 3%: 56 vs 13 | `data/boundaryHarmonicCodeJitter….json` (**new**) |
| Stripe ends | each half forms 69–75% inside a window 1.485–1.495, 0 of 500 outside; whole stripe 79% with both ends in, 0.9% otherwise | pre-registered, scored |
| Face modules | lower channel (bottom and left of the boundary into the bottom-left background) and flood push (top into the top-left background); face-like 61% with both, 18% otherwise; 1 of 320 a full face | pre-registered, scored |
| Net vs pattern reach | top-3 transfers kept ≥ 0.5 out to a code distance of 0.05–0.2, while the pattern loses most of its excess over baseline by 0.02 | backup figure 07 (**new**) |

Caveats: the two sweeps were sampled differently (compare each with its own baseline); offsets are absolute (±0.08 in G_pol / G_ref); the face slider replays clip to [0, 2.0]; nothing has a confidence interval.

## 9. Files and how to regenerate

```
presentation/   1_theSpatialOrganizer.png … 6_thePush.png, movies/A_guideThenLetGo.mp4, movies/B_turningTheKnobs.mp4
                backup_quantitative/ (figures 01–08, two movies, NOTES_quantitativeSet.md)    TALK_NOTES.md (this file)    TALK_NOTES_previous.md (the spatial-gene framing, see the PS)
```
Scripts (repo root): `canalizationTalkCommon.py`, `buildCanalizationTalkEssence11x11.py` (the set above), `buildCanalizationTalkFigures11x11.py` and `buildCanalizationTalkMovies11x11.py` (the quantitative set), `analyzeBoundaryHarmonicCodeJitter11x11.py`, `analyzeCanalizationTalkPatternEnsemble11x11.py`.
Data: `data/canalizationTalk*.json|npz`, `data/boundaryHarmonicCodeJitter*.json`.
```
python3 buildCanalizationTalkEssence11x11.py --ffmpeg <ffmpeg with libx264> --overwrite        # all; or --parts spatialOrganizer,twoPhases,stripeSwitches,faceNoSinglePart,canals,push,guideThenLetGo,turningTheKnobs
```
The cluster's ffmpeg module has no H.264 encoder; the movies used the static build inside the `imageio-ffmpeg` wheel (unpacked in a scratch folder, not installed). Replays are cached in `data/canalizationTalkReplays1888Hold301.npz` (made by the quantitative-set figure script) and `data/canalizationTalkPartialHolds1888Hold301.npz`.

---

**PS — a gene-style reading is also possible.** The code can be cast as a *spatial gene*: the hold is transcription (the boundary is copied inward, each order owning its own modes of the bulk) and the release is translation (the tissue's own machinery turns the copy into a pattern;
with the field feedback off there is a transcript and no product). The structure fits (two phases, loss of correspondence, dependence on machinery), and it brings the genome-as-compressed-description angle with it. The molecular words fit less well, because an organizer induces a response rather than being
decoded and there is no codebook, which is why these notes use organizer, guidance and self-organization. The fuller treatment of the gene framing, with the evidence for each step, is in `TALK_NOTES_previous.md`.
