> **Superseded write-up, kept for the gene framing.** This is the 2026-10-03 version of the talk notes. It casts the code as a *spatial gene* read by transcription (the hold) and translation (the release),
> keeps the title "Can spatial bulk pattern development be canalized from the boundary?" and calls the 40 outer cells the "rim". The current notes are `TALK_NOTES.md` (organizer, guidance and self-organization, "boundary").
> File names used below map to the current ones: `1_theSpatialGene.png` → `1_theSpatialOrganizer.png`, `2_twoStepsOfReading.png` → `2_twoPhases.png`, `movies/A_theGeneIsRead.mp4` → `movies/A_guideThenLetGo.mp4`; the others are unchanged.

# "Can spatial bulk pattern development be canalized from the boundary?" — the talk, in pictures

Prepared 2026-10-03. The slides carry the picture and the idea; numbers live only in `backup_quantitative/` and in section 5 below, for questions.
Everything marked **new** was computed for this talk, is exploratory (nothing registered beforehand) and is not committed.

## 1. The idea in one breath

*A pattern in the bulk is written, like a gene, as a short spatial code on the boundary. The tissue reads it in two steps, and the whole pattern follows from a handful of waves.*
The two undercurrents you asked for fit the data, with one qualification each.

### Undercurrent 1 — canalization is compression: the code is lower-dimensional than the pattern

What the data say, from coarse to fine:

* **Description length.** The pattern lives in 81 interior cells (45 independent under the left–right mirror, 25 under both mirrors). The rim has 40 cells (21 and 11 independent). The code that
  steers the tissue to the target is 4 numbers for the face (orders 0–3; orders above 3 add nothing) and 2 effective numbers for the stripe (a dial and an oval; the tilt is 0.001). Behind the code
  sit 4 ring-region levels and, in the causal net, 2 switches: the stripe's two ends, the face's two modules. So 81 cells are steered by 2–4 numbers, a compression of 20–40×.
* **Many codes, one pattern** (the sense Waddington gave the word, and what Fig 5 shows). Turn the dial and the pattern does not change continuously: it holds, then jumps. Along a sweep of 201 dial settings the
  stripe visits 18 distinct patterns (11 settings per pattern), the face 57 (3.5 per pattern). Over 160 random space-filling codes the stripe lands on 58 distinct patterns, an effective 22: about 7 codes per pattern, spread over about 7 dimensions
  of the 81. Over 280 random codes the face lands on 219 distinct patterns, an effective 167: 1.7 codes per pattern, spread over about 16 dimensions (**new**).
  So the stripe's map is strongly compressive and the face's is barely: **the simpler the target, the more canalized the code**. That gives your two claims a single axis.
* **The qualification.** The compression holds for the code that *writes* the pattern directly. The stripe's two-wave code needs ring cells driven past the bistable edge (ceiling 2.0; the face's training for its features alone stayed at 1.3). Under the face's ceiling of 1.3 the
  stripe needs all 11 independent rim values (even orders 0–20): no compression at all (backup figure 08). And "canalized" in the sense of robust is not what the data show: the *exact* target is a thin slice (see section 5).
  Say "compressed and canalized into a few states", not "robust".

### Undercurrent 2 — the code is a spatial gene, transcribed during the hold and translated afterwards

The mapping, with what backs each step:

| Gene | Here | Evidence |
|---|---|---|
| the gene | a profile along the rim, written as a few waves (a level, a tilt, an oval, a trefoil) | the trained codes: 3 and 4 waves |
| transcription | the hold (iterations 0–300): the rim is overwritten every iteration and the code is copied inward | during the hold each order owns its own modes of the bulk pattern (share owned by a single order 1.00 early in the hold; face) |
| the transcript's fidelity is lost | after the release the correspondence between parts of the code and parts of the pattern dissolves | 0.61 at the release, 0.06 when the face appears (face); the information is delocalised, not destroyed (held-out readout 0.55 across codes, 0.89 near the trained code) |
| translation | the release: the tissue's own machinery (the electric-field feedback and the bistable cells that latch) turns the copy into a pattern | with the field feedback off the imprint stays for ever (0.68 owned) and not one cell goes dark: a transcript with no translation |
| co-transcriptional or post-transcriptional | the simple gene is read as it is written, the complex one matures long after | stripe: 27/27 cells dark at iteration 300 and it forms even if the rim is never released; face: features darken at 1,853 and the face appears at 2,173, and holding the rim for ever gives only 21–57% of it |
| the interpreter is fixed | the same frozen tissue reads both genes | checkpoint 1888 throughout |

**Where the analogy ends** (say it before someone else does): there is no codebook and no molecular sequence; the code is analogue, the "machinery" is the tissue's fixed dynamics, and the quantitative ownership result exists for the face only
(the stripe's evidence is the direct one above). It is an organising picture of the two-step structure and of the loss of one-to-one correspondence, not a mechanism claim.
**Hooks to dev-bio** for the audience (names only; check wording before you use them): Waddington's canalization and landscape (1942); Wolpert's positional information, of which the stripe is the central band of the French flag, pattern 6 of the design document (1969);
Turing's prepattern (1952); the bioelectric prepattern of the face; the many-to-one genotype–phenotype map and the idea that a small genome specifies a large body plan.

## 2. What the visuals are (all dark, 1920 × 1080)

Rim cells are violet (brighter = higher held conductance); interior cells glow where they are hyperpolarised, cyan for the stripe and amber for the face. Two fidelity notes: the rim's colour scale is stretched per target so the
stripe's two bumps are visible (the stripe code spans 1.2–1.5, the face's 0.15–1.3); and the glow is decoration that follows the cells' hyperpolarisation. In movie A each panel stops at its own target's moment (stripe 504, face 2173), because the tissue
keeps moving afterwards and these patterns are moments, not rest states.

| File | What it shows | Use |
|---|---|---|
| `movies/A_theGeneIsRead.mp4` (9 s) | the gene appears on the rim, is held, released; the stripe forms, then the face; a bar marks transcription and translation | opener, or the take-home at the end |
| `1_theSpatialGene.png` | a few waves sum to the rim profile, painted on the rim, read into the pattern; stripe above, face below | the idea, undercurrent 2 and 1 together |
| `2_twoStepsOfReading.png` | seven moments of each target, bracketed by transcription and translation | the two steps; simple is written in the hold, complex long after |
| `3_stripeTwoSwitches.png` | only the top end held, only the bottom end, both | the simple gene is readable |
| `movies/B_turningTheKnobs.mp4` (9 s) | each knob turned in turn, three for the stripe and four for the face | few knobs, many cells; the stripe tidy, the face fragmenting |
| `5_canalsOfTheDial.png` | height = how much of the target; long flat steps are canals; thumbnails of the patterns on them | canalization as many codes to one pattern |
| `4_faceNoSinglePart.png` | upper wall alone, lower wall alone, whole rim | the complex gene is a distributed key |
| `6_thePush.png` | the strongest transfers drawn over the pattern: a direct push into the stripe, a loop through the face | how the rim reaches the bulk |

Suggested order for 10 minutes: A (1:00) → 1 (1:15) → 2 (1:15) → 3 (1:30) → B and 5 (2:00) → 4 and 6 (1:30) → close on the last frame of A or on 1 (0:45).
One-line captions: *A few waves on the rim are a gene. The tissue reads it in two steps. A simple gene is read as it is written, a complex one is finished later by the tissue. Turn a knob and the stripe changes in tidy steps while the face
breaks into fragments. Many codes give one pattern: that is the canal.*

## 3. The story, with the four wording changes the data ask for

1. **"Canalized"** — use it for compression and for the canals (many codes, one pattern), not for robustness: the exact stripe is a thin slice (dial window 0.008), the face thinner (under 0.004).
2. **"Linear"** — drop it. The ring profile is a sum of waves, but the tissue's response is switch-like (whole rows of three cells, a window 0.01 wide). Say "readable".
3. **"Simple pattern → readable code"** — true for the code that writes the stripe directly; state the freedom (the ring may be driven past the bistable edge). At the face's limits the stripe's code is 11 spiky harmonics.
4. **"Complex pattern: unique patterns and nets"** — the face's sliding gives a largely shared family of fragments whichever order is turned (62% of its patterns also appear under another order, against 42% for the stripe). Its causal net is as orderly as the stripe's; its pattern is not.

## 4. Questions to expect

* *Is this canalization?* It is compression and it is canals; it is not robustness. The exact pattern needs a thin slice of code space.
* *Is the gene analogy a mechanism?* No: it organises the two-step structure and the loss of correspondence; there is no codebook.
* *Why is the simple target not the easier one to find?* The stripe's two-wave code was found by 1 restart in 20 (ceiling 2.0); the face's by 14 of 30; under the face's ceiling the stripe needs 11 harmonics.
* *Does it work for any pattern?* No: the inverted stripe (flanks dark, centre light) was reached by none of 632 restarts (best overlap 0.70).
* *Is the pattern stable?* No: both are moments the tissue passes through (stripe 34 iterations, face 55–119 of 20,000).
* *What was pre-registered?* The stripe gate, mechanism and ends tests, the face's module confirmation (8 of 9) and the order-role knockouts. Everything marked **new** and all the sliding and compression comparisons are exploratory.

## 5. Numbers (for questions only)

| Quantity | Value | Source |
|---|---|---|
| Codes | stripe a₀ 1.350, a₁ 0.0012, a₂ 0.137 (ceiling 2.0, ten rim cells above 1.439); face a₀ 0.817, a₁ 0.250, a₂ 0.434, a₃ −0.280 (ceiling 1.3) | trained runs |
| Reach | stripe 8.92 mV vs best random 18.13; face 8.93 vs best random 20.6–22.4 | Stripes, Training the Ring |
| Independent values | rim 40 (21 / 11 under one / two mirrors); interior 81 (45 / 25) | counted from the lattice |
| Dial window | stripe overlap ≥ 0.9 at 4 of 81 settings of step 0.002 (0.008); face 1 of 81. Overlap ≥ 0.5 over 59% of the ±0.08 dial (stripe), 21% tilt, 27% oval; face 5%, 6%, 6%, 9% | `data/canalizationTalkNumbers.json` (**new**) |
| Many to one, along a slider | 201 settings: stripe 18 / 25 / 28 distinct patterns (orders 0 / 1 / 2), face 57 / 73 / 57 / 59 | `…PatternSmoothness….json` (exploratory) |
| Many to one, across random codes | stripe: 160 space-filling codes, 58 distinct patterns, effective 22, 7.2 codes per pattern, participation ratio 7.3, 9 components for 90%; face: 280 codes, 219 distinct, effective 167, 1.7 per pattern, participation ratio 16.3, 21 components. All 400 codes: stripe 179 distinct / 4.8 per pattern, face 268 / 2.2 | `data/canalizationTalkPatternEnsemble1888Hold301.json` (**new**) |
| Noise on the rim | 1% independent noise on the 40 values, 100 draws: overlap ≥ 0.5 in 98 (stripe) vs 50 (face); ≥ 0.9 in 23 vs 0; 3%: 56 vs 13; 10%: 9 vs 1 | `data/boundaryHarmonicCodeJitter….json` (**new**) |
| Ends of the stripe | each half forms 69–75% inside a window 1.485–1.495, 0 of 500 outside; whole stripe 79% with both ends in the window, 0.9% otherwise | pre-registered, scored |
| Face modules | face-like 61% with both modules, 18% otherwise; only 1 of 320 a full face; 0 of 16 order-role predictions held | pre-registered, scored |
| Ownership (face) | single order owns the pattern: 1.00 early in the hold, 0.61 at release, 0.06 when the face appears; field off: 0.68 for ever and no dark cell | Training the Ring |
| Net vs pattern reach | top-3 transfers kept: ≥ 0.5 out to a ring-code distance of 0.05–0.2. The pattern's median overlap has lost most of its excess over the no-ring baseline by 0.02 (stripe 0.82 → 0.46 against a baseline of 0.33; face 0.42 → 0.28 against 0.19) | backup figure 07 (**new**) |

Caveats of the new comparisons: the stripe and face sweeps were sampled differently (compare each with its own baseline; the space-filling subsets are the like-for-like ones, but their boxes differ); the slides' offsets are absolute (±0.08 in G_pol / G_ref);
the face slider replays clip ring values to [0, 2.0], not the training ceiling of 1.3; nothing has a confidence interval.

## 6. Files and how to regenerate (code last)

```
presentation/
  1_… 6_*.png, movies/A_*.mp4, movies/B_*.mp4      the essence set (this file's section 2)
  backup_quantitative/                              the first, number-heavy set: 01–08 figures, 2 movies, NOTES_quantitativeSet.md
```
Scripts (repo root, nothing committed): `canalizationTalkCommon.py` (trained codes, replay, palette), `buildCanalizationTalkEssence11x11.py` (the essence set), `buildCanalizationTalkFigures11x11.py` and
`buildCanalizationTalkMovies11x11.py` (the backup set), `analyzeBoundaryHarmonicCodeJitter11x11.py`, `analyzeCanalizationTalkPatternEnsemble11x11.py` (both refuse to overwrite). Data: `data/canalizationTalk*.json|npz`, `data/boundaryHarmonicCodeJitter*.json`.

```
python3 buildCanalizationTalkEssence11x11.py --ffmpeg <ffmpeg with libx264> --overwrite      # all essence figures and movies
python3 buildCanalizationTalkEssence11x11.py --parts spatialGene,canals --overwrite         # some
```
The cluster's ffmpeg module has no H.264 encoder; the movies were encoded with the static build inside the `imageio-ffmpeg` wheel (unpacked in the session scratch folder, not installed). Replays are cached in `data/canalizationTalkReplays1888Hold301.npz`
(made by the backup figure script) and `data/canalizationTalkPartialHolds1888Hold301.npz`.
