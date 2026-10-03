# "Can spatial bulk pattern development be canalized from the boundary?" — materials for the 10-minute talk

Prepared 2026-10-03 from the five reports (Training the Ring, The Latching Switch, The Relay's Loop, The Stripes, The Double Stripes) and a few new replays.
Everything marked **new** below was computed for this talk, is **exploratory** (nothing registered beforehand) and is not committed.

## 1. Verdict on the storyline

The skeleton is sound: *the boundary does not paint the bulk pattern, it picks which of the tissue's own trajectories is taken*, and a simple target leaves a readable
code while a complex one does not. Four parts of the wording need changing, because the data disagree with them or an audience will probe them.

| Your wording | What the data say | Suggested wording |
|---|---|---|
| "canalized" | Canalization means robustness. The *outcome* is a thin slice of code space for **both** targets: the stripe stays at overlap ≥ 0.9 over 0.008 of the dial, the face over less than 0.004 (one stop of 0.002); 1% independent noise on the 40 ring values leaves the stripe at overlap ≥ 0.9 in 23 of 100 draws and the face in 0 of 100 (**new**). What *is* wider is the **mechanism**: the causal net keeps half of its top transfers out to a ring-code distance of 0.05–0.2, the pattern needs under 0.02 (Fig 7, **new**). | "steered", or "channelled": the boundary selects a channel; the pattern is a thin slice of it, the mechanism a wider one. Keep "canalized" only if you define it this way. |
| "ring codes are interpretable / **linear**" | Interpretable, yes; linear, no. The ring *profile* is linear in the orders by construction, but the tissue's response is switch-like: a window 0.01 wide just above the 1.439 bistable edge, and a staircase of whole three-cell rows (27 → 21 → 15) as the dial is lowered. The only linear statement the data back is that which transfers appear in the causal net can be read from four ring-region levels (held-out AUC 0.81–0.93). | "modular and readable": top bump writes the upper end, bottom bump the lower end, independently. |
| "sliding the orders somewhat systematically alters the patterns and the relay networks" (stripe) | True for the stripe, with a number to attach: along the dial the stripe stays at overlap ≥ 0.5 over **59%** of the ±0.08 range (21% along the tilt, 27% along the oval); the face over 5–9% along each of its four orders (**new**, Fig 5). The pattern does *not* change more smoothly at the finest scale: at a step of 0.001 the stripe changes in 18–26% of steps, the face in 21–28% (earlier exploratory comparison). So claim graceful degradation, not smoothness. | "degrades gracefully, in whole rows". |
| "face: sliding yields unique patterns and relay networks, far less systematic, less interpretable, nonlinear" | Half right. The patterns are *not unique per order*: the four face orders give a largely shared family of partial faces (corner dots and a short nose, then a T of eyes-row plus nose; Fig 5b) — 62% of the face's slid patterns also appear when a different order is moved, against 42% for the stripe's (20 distinct dark sets in 32 face tiles, 17 in 24 stripe tiles; **new**). The code acts through a few ring-region levels, not through four separate brushes: a pre-registered test of order roles failed (0 of 16 predictions held). The causal net is *not* less systematic than the stripe's: it decays with distance from the trained code about as fast (Fig 7, top right) and its edges are predicted from region levels about as well on unseen regions (AUC 0.80 vs 0.81). What *is* worse for the face: the pattern is less predictable from the ring (overlap R² 0.09 vs 0.39, gap R² 0.34 vs 0.55), less robust to noise (50 vs 98 of 100 draws keep overlap ≥ 0.5 at 1%), and the roles are not separable. | "the net still changes in an orderly way, but the pattern it produces is rough and the code no longer reads as parts". |

Two further things to decide before the talk:

* **The stripe's readable code depends on a freedom the face did not have.** The two-bump code holds ten ring cells above 1.439 (the cell's bistable upper edge), which needs a
  ceiling of 2.0; the face code never exceeds 1.3. At the face's ceiling no code of up to six orders forms the stripe; an even-order code of 11 harmonics does (10 of 16 restarts),
  and it is a spiky, unreadable profile (backup Fig 8; its mechanism was not analysed). So "simple target → readable code" holds for the code that *writes* the stripe during the hold.
  The cleanest axis may therefore be **written versus triggered** (Fig 2) rather than simple versus complex. Whether a face trained with the same freedom would also become readable was not tested.
* **Both targets are transient moments, not states the tissue settles into.** The stripe is at overlap ≥ 0.9 for 34 iterations (479–512), the face for 55–119 iterations once in 20,000.
  "Development" here means being steered *through* the target; movie 1 shows both dissolving afterwards.

## 2. The story I would tell (answer first)

**Yes, in the sense that matters: a handful of numbers on the 40 boundary cells selects the pattern that forms in the 81 bulk cells. No, if canalized means robust: the boundary is a key, not a basin.**

1. A code of 3 numbers (stripe) or 4 (face), held on the ring for 301 iterations and then released, steers the tissue to its target. Random codes do not (best random 18.1 mV vs 8.9 mV for the stripe, 20.6–22.4 vs 8.9 mV for the face). The code does not draw the pattern; the tissue's own field feedback does (without it not one interior cell goes dark).
2. **Simple target — the stripe: the code is readable.** Two bumps on the ring, at the top and bottom, write the two ends; they are separate switches (each forms its half 69–75% of the time inside a window 0.01 wide, never outside it); the relay is a direct push, ring top → upper stripe and ring bottom → lower stripe. Slide the dial and the stripe shortens by whole rows and stays a clean shorter stripe over a long stretch.
3. **Complex target — the face: the code is a distributed key.** No part of the ring writes it (the two wall halves each give half of the lead and neither makes a face); the orders are not brushes (0 of 16 role predictions held); two ring-level modules go with face-like patterns (61% against 18%) but only 1 of 320 codes reaches a full face. Slide the dial and the face breaks into fragments, largely the same fragments whichever order you move.
4. **How wide is the channel?** The outcome needs a thin slice; the causal net behind it persists 5–10 times further. That is the nearest thing to canalization the data support, and it is the same for both targets.
5. Take-home: *steerable from the boundary, as a trigger that selects among the tissue's own trajectories; the more of the pattern the code writes directly, the more it reads as parts and the more gracefully it degrades.*

## 3. Slide plan (about 9.5 minutes)

| # | Time | Slide | Material | Say |
|---|---|---|---|---|
| 1 | 1:00 | The question and the set-up | `movies/movie_01_codeToPattern.mp4`, `01_twoTargetsOneHandle.png` | 121 cells, 40-cell ring = boundary, 81 = bulk. Code held 301 iterations, released. Stripe = simple, face = complex. 3 and 4 numbers reach them (8.9 mV vs ≥ 18 mV for random codes). |
| 2 | 1:00 | Written or triggered | `02_whenThePatternIsWritten.png` | Stripe is dark in the ring-held tissue (27/27 at iteration 300) and is left standing when the flanks relight (504). The face appears 1,872 iterations after release, after the code's grip is gone. Both need the field feedback. |
| 3 | 1:30 | Simple: the stripe's code reads as parts | `03_stripe_ringBumpsWriteTheEnds.png` | Bumps at top and bottom; top+bottom rows carry 99% of the effect, sides 19%; two separate switches; 79% vs 0.9%. |
| 4 | 1:30 | Slide the dial | `movies/movie_02_slidingTheDial.mp4` or `05_slidingTheDial.png` | Stripe shrinks by rows 27→21→15 and stays; face falls to fragments. Overlap ≥ 0.5 over 59% of the range vs 5%. |
| 5 | 1:00 | The relay as the dial slides | `06_slidingRelayNets.png` | Stripe: the same two pushes, ring top→upper and ring bottom→lower, 0.27 each at the trained code, ~20× weaker 0.04 away. Face: a loop, edges reorganise. Both nets can be read from four ring-region levels. |
| 6 | 1:30 | Complex: the face's code is a distributed key | `04_face_noPartAlone_twoModules.png` | Walls add but neither makes a face; roles fail 0/16; modules 61% vs 18% but 1/320 full faces. |
| 7 | 1:30 | How wide is the channel? | `07_howWideIsTheChannel.png` | Pattern: thin slice. Net: 5–10× further. Noise: 98 vs 50 of 100 keep half the pattern at 1%; exact pattern only 23 vs 0. Ring tells you more about the stripe (R² 0.39 vs 0.09). |
| 8 | 0:30 | Take-home | — | Steerable, as a key; readable in proportion to how directly the code writes. Next: the face-window probe. |

If you must cut to six slides: drop 2 (move its two numbers into 1) and merge 5 into 4.

## 4. Numbers and where they come from

C = pre-registered and scored; E = exploratory in the reports; N = new for this talk (exploratory).

| Quantity | Value | Source | Status |
|---|---|---|---|
| Tissue, hold | 11 × 11, 121 cells, ring 40, interior 81; hold 301 iterations; checkpoint 1888 | all reports | — |
| Stripe code | a₀ 1.350, a₁ 0.0012, a₂ 0.137; ceiling 2.0; ten end cells above 1.439; best moment 504; 8.92 mV; 27/27 dark, 0 strays | Stripes report; `data/boundaryHarmonicTraining1888Hold301StripesInteriorMinus60Minus5Ceiling2/order2_restart06.npz` | C (gate G1, G2) |
| Stripe random codes | best 18.13 mV of 64 | Stripes report appendix; training summary | C |
| Stripe restarts | order 2, ceiling 2.0: 1 of 20 forms; ceiling 1.3, orders ≤ 6: 0; even orders 0–20, ceiling 1.3: 10 of 16 | restart records JSON | E |
| Face code | a₀ 0.817, a₁ 0.250, a₂ 0.434, a₃ −0.280; ceiling 1.3; best moment 2173; 8.93 mV; 14/14 dark, 1 stray (IoU 0.933) | Training the Ring; `…FaceMinus60Minus5/order3_restart08.npz` | E |
| Face random codes, restarts | random allowed codes: median ≈ 25.5, best 20.6–22.4 mV; at order 3, 14 of 30 restarts end below 10 mV | Training the Ring | E |
| Stripe written in the hold | 27/27 stripe cells dark at iteration 300; 34 iterations at overlap ≥ 0.9 (479–512) | Stripes report | C (P5) |
| Face timing | features darken at 1,853; face at 2,173; one visit of 55–119 iterations in 20,000 | Training the Ring | E |
| Field and release | field off: no interior cell ever dark (both). Ring held throughout: stripe still forms (overlap 1.0 at 472); face only 21–57% | Stripes appendix A1, A3; Training the Ring | C for the stripe (S8), E for the face |
| Ring segments (stripe) | top + bottom rows 98.6%, six end cells 53.0%, left + right 18.9% of the effect; random 6-cell sets: 95th percentile 0.06 vs 0.34 | `…StripeRingSegments…Order2Restart06Ceiling2.json` | C (W1–W5) |
| Two ends (stripe) | upper half 375/500 = 75% inside the window 1.485–1.495, 0/500 outside; lower 347/500 = 69%, 0/500; top alone never writes the lower half (0/250), nor bottom alone the upper (0/250); whole stripe 198/250 = 79% with both ends in the window vs 7/750 = 0.9% otherwise | `…StripeEndsConfirmationScoring….json` | C (E1–E4 all hold) |
| Relay push (stripe) | ring top → stripe upper and ring bottom → stripe lower: 0.269 and 0.267 at the trained code; 0.0125 at a₀ −0.04 | `relayLoopStripesPageData…json` | E |
| Wall halves (face) | upper wall alone 0.183, lower wall alone 0.187, whole ring 0.356 of selectivity lead; replayed Vmem at 2173: upper alone 5/14 dark + 8 strays, lower alone 3/14 + 0, none 3/14 + 2, whole 14/14 + 1 | `…WallCounterfactual….json`; replays **N** | E; replay N |
| Orders as brushes (face) | 0 of 16 pre-registered knockout predictions supported; share of the pattern owned by one order: 1.00 early, 0.61 at release, 0.06 when the face appears | Training the Ring | C (knockout), E |
| Modules (face) | face-like (overlap ≥ 0.3): both 49/80 = 61%, flood push only 23/80, lower channel only 14/80, neither 7/80; others pooled 18%; Fisher p 1.8e-12; only 1 of 320 reached overlap ≥ 0.5; 8 of 9 criteria passed (C7 failed) | `relayLoopModulesConfirmation….json` | C |
| Dial slide (new) | stripe overlap ≥ 0.9 at 4 of 81 stops (width 0.008), ≥ 0.5 at 48 (59%); face ≥ 0.9 at 1 of 81, ≥ 0.5 at 4 (5%). Tilt: stripe 21%, face 6%; oval: 27% vs 6%; face a₃ 9% | `data/canalizationTalkNumbers.json` (`sliding`) | N |
| Stripe staircase | dark stripe cells 27 → 21 → 15 as a₀ falls; 15/27 with 0 strays from −0.016 to −0.08 | same | N |
| Pattern smoothness | at step 0.001: stripe changes in 18–26% of steps, face 21–28%; at each code's own best moment the stripe is rougher | `…PatternSmoothness1888Hold301StripesAndFace.json` | E |
| Channel width (new) | distance = rms difference of the four ring-region levels from the trained code. Median overlap, stripe: 0.82 (< 0.01), 0.46 (0.01–0.02), 0.41, 0.41, 0.36, 0.33, 0.33; face: 0.42, 0.28, 0.29, 0.25, 0.21, 0.19, 0.15. No-ring baselines 0.33 and 0.19. | `data/canalizationTalkNumbers.json` (`channel`) | N |
| Net retention (new) | mean share of the trained code's top-3 field transfers kept (flood, clear), same bins: stripe 0.74, 0.71, 0.61, 0.56, 0.50, 0.26, 0.15; face 0.94, 0.83, 0.69, 0.51, 0.37, 0.28, 0.18 | same | N |
| Noise on the code (new) | independent multiplicative noise on each of the 40 held values, 100 draws, same generator and draw order as the face's earlier test (which it reproduces: 50 and 13 of 100). Overlap ≥ 0.5 / ≥ 0.9 — stripe: 1%: 98 / 23, 3%: 56 / 0, 10%: 9 / 0; face: 1%: 50 / 0, 3%: 13 / 0, 10%: 1 / 0 | `data/boundaryHarmonicCodeJitter….json` | N |
| Predictability (exploratory, earlier) | edges from four ring-region levels, median held-out AUC: unseen region 0.81 stripe / 0.80 face; the page's own codes 0.93 / 0.81. Learnable edges 20 of 22 / 25 of 35. R² for the selectivity gap 0.55 / 0.34, for the overlap 0.39 / 0.09 | `relayLoop[Stripes]OrderEdgeMap….json` | E |
| Double stripes | 632 restarts, 2.3 million codes, both ceilings, orders up to 20: none reaches overlap 0.9; best 0.70 | Double Stripes report | C (negative) |

## 5. Questions to expect

* **"Is this canalization?"** Define it: the code selects a channel; the pattern is a thin slice of it (the stripe's window is 0.008 on the dial, the face's under 0.004; 1% noise keeps the exact pattern in 23 of 100 draws for the stripe, 0 for the face); the causal net is 5–10× more forgiving. Do not claim robust canalization.
* **"Why is the simple target not the easier one to find?"** Under its own freedom the stripe was found by 1 restart in 20 (order 2, ceiling 2.0); the face by 14 of 30 at order 3. Under the face's ceiling the stripe needs 11 even harmonics. Search difficulty and readability of the final code are different things.
* **"Does it work for any pattern?"** No: the inverted stripe (two flank stripes dark, centre light) was not reached by any of 632 restarts (best overlap 0.70).
* **"Is the pattern stable?"** No: both are moments the tissue passes through (stripe 34 iterations, face 55–119 of 20,000).
* **"Is the code the cause or a trigger?"** A trigger: field feedback is required for both, and in the face the code's grip on the pattern is gone (0.06 owned) before the features darken.
* **"Which results were pre-registered?"** The stripe gate and its mechanism and ends tests, the face's module confirmation (8 of 9) and the order-role knockouts. Everything marked N, and the sliders, smoothness and edge-map comparisons, are exploratory.

## 6. Caveats of the new comparisons

* The stripe and face **sweeps were sampled differently** (stripe: 160 space-filling + 120 near the code + 120 inside the window the stripe forms in; face: 280 space-filling + 120 near). Bin-wise medians and means are fair within a bin, but the stripe's first bin is enriched near the window by design, and the stripe's no-ring overlap (0.33) is higher than the face's (0.19), so compare each curve with its own dotted baseline.
* **Distance** is in ring-region space (four levels, G_pol / G_ref). The face sweep codes are multiples of the trained coefficients, the stripe's are coefficient values; both are converted to region levels before measuring.
* **Slide offsets are absolute** (±0.08 in G_pol / G_ref, the physical unit). The face's page stops in Fig 6 are multiples (×0.5 … ×2 = a₀ −0.41 … +0.82), the stripe's are offsets (−0.15 … +0.15), because those are the stops the Relay Loop pages have; the nets do not exist at matched distances.
* The face slider replays use ring values clipped to [0, 2.0], as the report's sliders did, not to the training ceiling of 1.3.
* The noise test reads the pattern at one fixed moment (504 and 2173). A noisy code may form the pattern slightly earlier or later; within ±100 iterations of the readout the best overlap reaches ≥ 0.5 in 100 / 76 / 22 of 100 draws for the stripe and 88 / 48 / 3 for the face (σ = 1%, 3%, 10%), and ≥ 0.9 in 36 / 1 / 0 and 0 / 0 / 0.
* Fig 7 bottom right and Fig 5b are exploratory summaries of existing files and fresh replays; none of the new quantities has a confidence interval.

## 7. Files

Figures are 200 dpi PNG at slide proportions (about 13.3 × 7.5 in) and use the reports' palette (stripe teal, face ochre; dark = hyperpolarised).

| File | Use |
|---|---|
| `01_twoTargetsOneHandle.png` | targets, ring codes, patterns, coefficients |
| `02_whenThePatternIsWritten.png` | snapshots at 100, 300, 504, 1000, 1850, 2173 |
| `03_stripe_ringBumpsWriteTheEnds.png` | ring profile, which part of the ring writes it, the two ends |
| `04_face_noPartAlone_twoModules.png` | face ring profile, wall halves, modules |
| `05_slidingTheDial.png`, `05b_slidingEveryOrder.png` | patterns as one order slides (main: dial; appendix: every order) |
| `06_slidingRelayNets.png` | causal net, clear phase, as the dial slides |
| `07_howWideIsTheChannel.png` | pattern vs net vs noise vs predictability |
| `08_stripeTwoRoutes_ringCodes.png` | backup: the stripe's two routes against the face code |
| `movies/movie_01_codeToPattern.mp4` (14 s), `movies/movie_02_slidingTheDial.mp4` (9 s) | H.264, 1280 × 720 |

Regenerate (everything reads existing data; the replays are cached in `data/canalizationTalkReplays1888Hold301.npz`):

```
python3 buildCanalizationTalkFigures11x11.py --overwrite
python3 buildCanalizationTalkMovies11x11.py --ffmpeg <ffmpeg with libx264> --overwrite
python3 analyzeBoundaryHarmonicCodeJitter11x11.py --target stripesInterior     # and --target face; refuses to overwrite
```

The cluster's ffmpeg module has no H.264 encoder; the movies were encoded with the static build shipped in the `imageio-ffmpeg` wheel (unpacked in the session scratch folder, not installed).
Nothing is committed. New files: the four scripts (`canalizationTalkCommon.py`, `buildCanalizationTalkFigures11x11.py`, `buildCanalizationTalkMovies11x11.py`, `analyzeBoundaryHarmonicCodeJitter11x11.py`), `data/boundaryHarmonicCodeJitter*.json` (2), `data/canalizationTalkNumbers.json`, the replay cache, and this folder.
