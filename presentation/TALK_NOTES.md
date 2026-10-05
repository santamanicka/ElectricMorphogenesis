# Can bulk pattern formation be steered from the boundary? Canalization and constraint in a bioelectric tissue.

Notes for a 25-minute talk at GSO-2026, the conference on guided self-organization (Binghamton, 14–15 October 2026). The system is an 11 × 11 bioelectric tissue model: its 40 outer cells are the *boundary*, its 81 inner cells the *bulk*.
Two targets are used, a **stripe** (the simple pattern, 27 cells) and a **face** (the complex pattern, 14 feature cells). The slides carry pictures and ideas, not numbers; numbers are in section 11. Quantities marked **new** are exploratory (nothing registered beforehand).

## 1. Motivation

**The ultimate goal is a grammar of steering**: a compact account of how a self-organizing tissue is guided to a pattern. How little information, in what form and applied where? Which guide reaches which pattern, through what mechanism, and why can some patterns not be reached at all?
A grammar of that kind would let one read, predict and design guides, which matters for developmental biology (what do organizers and prepatterns specify, and what does the tissue supply?) and for synthetic biology (what is the minimal input that programs a tissue?). This talk is a first step toward it.

Development already works this way. Organizers and prepatterns, a few localized signals, guide a tissue that does most of the organizing itself. Guided self-organization asks the same of any self-organizing system: steer it toward an outcome with non-specific guidance.
A bioelectric tissue makes the question concrete, because the guide can be a boundary condition and the bulk's own rules stay untouched. If a few boundary parameters suffice, tissue patterning becomes controllable with few inputs.

The approach: take two targets of different complexity on one frozen tissue, find the smallest boundary code that reaches each, trace how the guidance travels inward, and map where it fails.

**The answer to the title.** Can bulk pattern formation be steered from the boundary? Yes: 2–4 numbers on the 40 boundary cells steer 81 interior cells to a stripe or a face, where random codes do not. The deeper questions are how the guide narrows the tissue's options, and for how long.
Canalization is used **with a stated scope, not as a bare word**. It is the strong, buffered, endpoint-seeking end of developmental bias (a bias in which outcomes a system produces), so the question is how much, and against what:
* **Compression:** many codes to one pattern. A few numbers steer a high-dimensional pattern into a few discrete outcomes; the simpler the target, the stronger it is.
* **Constraint:** the family bias across time. The code keeps the tissue near its family of patterns: graded, partly buffered against code noise (the stripe more than the face), and ending at about 3,000 iterations, about ten times the length of the hold.

Against what it is canalized: noise on the code, partly (stripe); perturbation of the tissue's own state, untested; time, no. In the sense of robustness of the exact target the answer is no. Compression and the lingering imprint (constraint) are the two results that go beyond "yes, it can be steered".

## 2. The talk in 25 minutes

One through-line, **guide, then let go**, and the title's question answered in three labelled parts: *steered* (does it work), *how* (what the guide is and how it reaches the bulk), *how far* (canalization as compression and as constraint, and the limits). The ending looks forward to the grammar of steering.
Assumes 25 minutes of talking with questions separate; the plan sums to 24.5 minutes, leaving half a minute of slack. If questions come out of the 25, drop the slides marked optional.

| # | Min | Slide | Material |
|---|---|---|---|
| | | **Motivation (3:00)** | |
| 1 | 0:45 | Hook: a few waves on the boundary, then it lets go | movie A |
| 2 | 1:00 | The question: how little information, in what form and applied where, steers a self-organizing system? Quote the GSO definition. The ultimate goal: a grammar of steering | new text slide |
| 3 | 1:15 | Development already works this way (organizers). A tissue lets us ask it with a boundary guide and untouched bulk rules. Preview "canalized" in two senses, each in one sentence: compression, and constraint across time | new |
| | | **Background (2:30)** | |
| 4 | 1:00 | Organizers and prepatterns: Spemann–Mangold, ZPA/AER, Bicoid, the electric face | new, simple icons |
| 5 | 0:45 | Guided self-organization, controllability, inverse design | new |
| 6 | 0:45 | The gap: the responder is fixed and the unknown is the minimal guide | new |
| | | **Model (3:15)** | |
| 7 | 1:15 | The tissue: 11 × 11 cells with ion channels, gap junctions and field feedback. State that it was trained earlier for the face | `0_theModel.png` |
| 8 | 1:00 | The guide: waves on the 40 boundary cells, held then released | figure 1 |
| 9 | 1:00 | Two targets, the stripe (simple) and the face (complex); the search for the smallest code, with random controls | new, or a variant of figure 1 |
| | | **Results (11:00)** | |
| 10 | 1:45 | *Steered.* A few numbers reach each target (8.9 mV against at least 18 for random codes). Two phases: the stripe is written in the hold, the face long after | figure 2 |
| 11 | 2:30 | *How.* The simple organizer is readable (two bumps, two switches); the complex one is distributed (no part alone, two modules). Link: a readable code is a compressed one | figures 3 and 4 |
| 12 | 2:00 | *How.* First the phases, read off the polarising conductance as it sweeps its switching window twice (flood, clear, write, latch: the build-up); then the net of each phase, drawn in the phase's colour, as a knob turns. Link: "so which parts push which, in each phase?" | figure 8 or movie E (0:30), then movie D `_thresh` |
| 13 | 1:30 | *How far: compression.* Many codes, one pattern (about 7 codes per stripe pattern against 1.7 for the face) | figure 5 |
| 14 | 2:30 | *How far: constraint.* The imprint lingers about ten times the hold, then the tissue wanders off | figure 7 and movie C |
| 15 | 0:45 | Limits: thin windows, passing states, the double stripe reached by 0 of 632 restarts | **new figure** |
| | | **Conclusion (1:15)** | |
| 16 | 1:15 | Answer the title in its own words: yes, steered with a few numbers; canalized with a stated scope (compression; a graded, partly buffered constraint ending at about 3,000 iterations; not robustness of the exact pattern) | **new summary** |
| | | **Implications and the way forward (3:30)** | |
| 17 | 1:00 | *Toward a grammar of steering*, the ultimate goal (below) | new |
| 18 | 0:50 | For developmental biology | new |
| 19 | 0:50 | For synthetic biology | new |
| 20 | 0:50 | Evolution (the Kauffman line) and next steps; close on "guide, then let go" | new |

**The forward-looking point (slides 17–20).** The grammar of steering is the ultimate goal, and the data already give its first words:
* *Already in hand.* The ring's four region levels (top, upper sides, lower sides, bottom) are the language in which the guide acts on the net (25 of 35 face edges and 20 of 22 stripe edges are predictable from them on held-out blocks of code space); the net is organised in a few switches (the stripe's two ends, the face's two modules);
  thresholds sit at the single cell's bistable edge (1.439); a pattern is either written during the hold (stripe) or triggered and finished by the tissue (face); and readability goes with compression.
* *Still missing.* A predictive map from target to minimal guide; an account of why some patterns cannot be reached (the double stripe); the mechanism of the lingering imprint and a test of its robustness to perturbation of the tissue's state; whether the grammar survives larger tissues, learned codes and interior placement of the guide.
* *For developmental biology.* Organizers and bioelectric prepatterns read as low-dimensional guides. A grammar would say which boundary or prepattern changes should alter which structures, and for how long a transient guide still biases the tissue (labile specification before determination, competence windows).
  It would also give testable predictions for regeneration, where a brief bioelectric perturbation can rewrite a body plan ([Durant et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5443973)).
* *For synthetic biology.* The design question is the minimal input that programs a tissue: engineered organizer sources, or optogenetic control of morphogen production at a tissue's edge (an example from the search: [optogenetic control of morphogen production](https://www.biorxiv.org/content/10.1101/2024.06.11.598403.full.pdf); only its summary was read). The grammar's design rules would say what to engineer
  (written-in-the-hold targets give readable guides), what to expect (thin windows, passing states), and what to avoid (patterns it cannot reach).
* *Evolution and next steps (slide 20).* The Kauffman line from section 9, then the next experiments: perturb the tissue's state during the lingering stretch, place the guide inside the tissue, scale up, learn the codes, and test a boundary guide in a living tissue.

**Linking sentences between the parts.** Motivation to background: "so who has steered self-organization, and how?" Background to model: "here is the smallest tissue that asks the question." Model to results: "does a few-number boundary guide work at all?" Results *steered* to *how*: "what does the guide look like, and how does it reach the bulk?"
*How* to *how far*: "a readable code is a compressed one, so how much does the guide compress, and for how long does it hold?" Results to conclusion: "so, can it be steered, and in what sense is it canalized?" Conclusion to implications: "and what would a grammar of steering give us?"

**Still to build:** the organizer icons (slide 4), the limits figure (slide 15, from the Double Stripes report in the same style), the summary slide (16) and the implications slides (17–20). Slides 10–14 use existing figures and movies.

## 3. The code is a spatial organizer

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

## 4. Three stretches of time: guidance, a lingering imprint, free self-organization

| Stretch | What happens | Evidence |
|---|---|---|
| guidance (the hold, iterations 0–300) | the boundary is overwritten every iteration and the code is copied inward | each order of the code owns its own modes of the bulk pattern (share owned by one order 1.00 early in the hold; face). The stripe is already written: 27/27 cells dark at 300 |
| the imprint lingers (about 300 to 3,000) | the boundary has let go, the tissue's own machinery (electric-field feedback, bistable latching cells) is at work, and the code still keeps the tissue near its family of patterns | the trained stripe code visits stripe-family patterns in 30% of its frames, in about five separate visits; the trained face code visits a deformed face once. Unrelated codes of the same symmetry almost never do (section 6) |
| free self-organization (beyond about 3,000) | the family bias is gone: the tissue wanders through a limited repertoire of patterns the code no longer picks | no family visits to 20,000 iterations; a 1%-different code is as far from the trained one as any unrelated code; the field-off control shows the machinery is required throughout (no cell goes dark without it) |

Instructive then permissive: while held the code instructs, once released the tissue's competence completes the pattern, and for a while longer the code still biases which patterns the tissue tends to visit. A simple organizer writes its pattern while it is held (stripe read at iteration 504);
a complex one is finished by the tissue long after it lets go (face read at 2173). The one-to-one link between parts of the code and parts of the pattern is lost sooner than the family bias: the share of the face pattern owned by a single order is 0.61 at the release and 0.06 when the face
appears, although the information is delocalised, not destroyed (held-out readout 0.89 near the trained code). The ownership result exists for the face only, and the organizer is an organising picture, not a mechanism claim.

**The phases come from the polarising conductance (flood, clear, write, latch).** The Latching Switch report (its figure 2) gives the origin: a cell's voltage settles far faster than its polarising conductance G_pol / G_ref, so a cell sits at a stable root of its own equation. Above a critical conductance G_up only the hyperpolarised state exists (the cell is forced to polarize: it floods, a lit cell in the figures); below G_down only the depolarised one (forced to depolarize: dry); between them both exist and the cell keeps the one it is on, which is the memory (an isolated cell: about 0.80 to 1.44; each dark neighbour lowers the thresholds by about 0.03, each light one raises them by about 0.04). The field the tissue generates drives the conductance up across the whole interior, lets it collapse through the bottom of the window, then drives it up a second time: the conductance sweeps the window twice. The phases are the stretches of that sweep, and the relay pages and the relay-net visuals use the same names (the nets' edges are coloured by phase in movie D).

| Phase | Iterations | The conductance (mean over cells) | The tissue | The relay net |
|---|---|---|---|---|
| flood | 0–301 (the hold; the stripe's relay windows: 0–299) | the first rise: the interior's mean climbs from 1.00 to a first peak near iteration 245 (1.42 face, 1.47 stripe) and is 1.32 (face) or 1.36 (stripe) at the release; the target cells run slightly above the rest | cells above G_up flood: the stripe's channel (27/27) and 14 flank cells; for the face a block of the interior. The first rise writes nothing about the face: its flooded cells are no more likely to be face cells than chance | face: the boundary's top pushes straight into the top corners, the one clearly one-way transfer of the run; stripe: each end of the boundary pushes into the stripe |
| clear | 301–585 (stripe: 300–504) | the collapse through the bottom of the window. Face: the mean falls to a trough of 0.81 at 585 (the clamp brings it forward from 1,055 without a clamp). Stripe: by 504 the rest of the interior is at 0.70, below G_down, while the stripe cells are still at 1.16, inside the window | the first rise's flooding drains, unevenly. Stripe: the flank cells fall out of the window first and for 34 iterations (479–512) the stripe stands alone, latched; it is read at 504. Face: a small remnant at the trough | the direct push shuts off; face: the top corners drain into the eyes and nose and feed back into the ring's sides |
| write (face only) | 585–1765 | the second rise: the mean climbs to a second peak of 1.15 at 1765, but the feature cells climb to 1.48 against 1.08 for the rest (a lead of 0.40): only now does the tissue select | the background floods in places (up to 13 stray cells near 950) while the future feature cells are lifted clear and cross G_up. The features are not yet visible in the voltage map | the bottom channel takes over and the loop reverses and keeps circulating (eyes and nose discharge into the top corners) |
| latch (face only) | 1765–2173 | the fall back: the feature cells go from 1.48 to 1.02, the rest from 1.08 to 0.80 | as the conductance falls through the window the lifted cells stay on the flooded branch and the rest go dry: the face sets (3 of 14 feature cells at 1750, all 14 by 2000) | outside the relay: its last saved window ends at 1765, so nothing is claimed about the nets of this phase |

The stripe has no write or latch phase in the relay's sense: it is written in the hold and read in the clear phase, where its cells outlast the rest in the window. (Its interior conductance does sweep a second time, trough 1130 and second peak 1807, long after the stripe has been read.) The curves in figures and movies are replays of the trained codes (`data/canalizationTalkConductance1888Hold301.npz`); the face's replay reproduces the report's numbers (first peak 1.421, release 1.317, trough at 585, second peak at 1765).

**How the relay nets are built, and what an arrow means.** The nets of figure 6 and movie D come from the Relay Loop and Stripes reports. There "transfer" is a technical term, not a current.

*The question a net answers.* The boundary code makes the trained run differ from a run without it. At the readout the target's cells end up with a higher polarising conductance than the rest of the interior (the selectivity gap: 0.40 G_ref for the face at the write peak, 0.46 for the stripe at its readout). The relay asks which parts of the tissue carried that difference, and in which phase.

*How a net is built.*
1. Two runs of the same frozen tissue: the trained run (the code held for 301 iterations) and a baseline, the same tissue with the tissue-wide extra voltage update the clamp performs on every hold iteration (a side effect of how the clamp is implemented) but with the ring never held. Keeping that update in the baseline isolates what the forty held values add. D(n) is the difference between the two runs in every cell's voltage and polarising conductance at iteration n.
2. The model's one-step map gives an exact relation, D(n+1) = J(n) D(n) + s(n). J(n) is the step's Jacobian averaged along the straight line from the baseline state to the trained state (Gauss–Legendre quadrature, refined at clip bounds and field turns until the identity closes to 1e-6); s(n) is what the trained map does differently at the baseline state (non-zero only during the hold). Nothing is linearised: it decomposes the actual difference, so the chaos over two thousand iterations does not enter.
3. A readout: the selectivity, the mean G_pol / G_ref of the target's cells (the face's 14 feature cells, the stripe's 27 cells) minus the mean of the other interior cells, at the primary moment (face: the write peak, iteration 1765; stripe: its readout, 504).
4. Walking backward from the readout gives, for every iteration, how much a deviation in each cell (voltage and conductance) matters for the readout later (the adjoint). Multiplied by the actual deviation it is the cell's flux, how much of the final difference that cell carries at that moment; after the hold the fluxes sum to the final difference at every iteration (checked).
5. An edge from cell j to cell i in one step is the adjoint of i times the step's response of i to j times j's deviation: j's deviation, passed on by that step to i, weighted by how much i's deviation matters for the readout. It is split by the channel the start-of-step voltage entered through: the field (the shared extracellular field, at any distance) or contact (gap junctions). A cell's own terms are excluded.
6. Cells are grouped into blocks (face: 11, namely ring top, bottom, left and right, eyes, nose, mouth and four background quadrants; stripe: 10, namely the four ring sides, stripe upper and lower, and each flank upper and lower). The net transfer between blocks A and B is the sum of the cell edges from B into A minus those from A into B, summed over the 50-iteration windows of a phase (face: flood windows 0–5, clear 6–11, write 12–35; stripe: flood to the release, clear to the readout). Positive means A to B in the canonical order (left or centre block first). Left–right twins are equal to about 1e-8, so one is computed and the other is drawn as its mirror.

*What an arrow means.* "A to B in phase X": during phase X, the deviation the boundary caused in region A, passed on to region B through the shared field, contributed this much (in units of G_ref) to the target's final conductance lead, net of any flow the other way; thickness is that size. The trained face's biggest field transfers: flood, ring top to the top-left background 0.11 (and to the eyes 0.04); clear, top-left background to the eyes 0.19 (the report's "the top corners drain into the eyes"); write, ring bottom to the bottom-left background 0.15. The trained stripe's: flood about 0.04 at each end, clear 0.27 from ring top to stripe upper and from ring bottom to stripe lower, so the stripe's push is mostly delivered just after the release.

*What it is not.* It is not a measured current or a signal that could be cut. It is not a gap-junction current: those are the contact channel, drawn nowhere in the talk's figures, and for the face at most about a fifth the size (0.04 against 0.19). It is not an intervention: it is an exact accounting of two runs, and the interventions are separate experiments (the partial holds of figures 3 and 4, the wall counterfactuals, the module confirmation). It is not a partition of the gap: transfers are signed, many pairs contribute, and opposite flows cancel inside a block net, so within-block movement is hidden. It is one trained code against one baseline and one readout, and it stops at the readout (face: 1765), so nothing is said about the latch phase.

*What the figures draw.* Figure 6 carries a compressed key at the top ("run the tissue with and without the boundary's code, then ask which regions carried the difference"; an A-to-B arrow: "how much of that difference one region passed on to another, through the shared field; thicker = more") and a foot line (the three strongest field transfers of each phase, all phases overlaid, net of flow the other way; an accounting of two runs, not a current). It draws the field channel only, the three biggest net transfers per phase of the trained code (face: flood, clear, write; stripe: flood and clear), direction by sign, mirror twins added, width proportional to size, between block centres over the pattern the run ends with. Movie D draws a different object, the pages' steering lab, which does not rerun the relay: for each knob setting it computes the ring's four region levels (top, upper sides, lower sides, bottom) and averages, with a Gaussian kernel in that space, the top-three sets of the 846 (face) or 499 (stripe) simulated codes. An edge is drawn when at least a quarter (half in `_thresh`) of the nearby codes carry it among their top three in that phase, thicker when more do. So in movie D thickness is how many nearby codes agree, not a transfer size, and the tissue underneath is simulated afresh.

*One sentence for the talk.* "An arrow says how much of the final difference between the steered tissue and the unsteered tissue passed from one region to another, through the shared field, during that phase." If pressed: it is an exact accounting of two runs, not a current, and not an intervention.

*Where it lives (code).* `computeBoundaryHarmonicRelay11x11.py` (the raw relay of one code), `relayLoopNets.py` (`readNet`, the block layouts, the phase windows), `extractRelayLoopFullNets11x11.py`, `assembleRelayLoopSteeringData11x11.py` and `assembleRelayLoopStripesData11x11.py` (the pages' data); the Relay Loop and Stripes reports.

## 5. Canalization as compression

* **The code is lower-dimensional than the pattern.** 81 interior cells (45 independent under the left–right mirror) are steered by 4 numbers for the face (orders 0–3; more add nothing) and 2 effective numbers for the stripe (a dial and an oval; the tilt is 0.001), a compression of 20–40×.
  Behind the code sit 4 region levels and 2 switches: the stripe's two ends, the face's two modules.
* **Many codes, one pattern** (Waddington's sense; figure 5). Turn the dial and the pattern holds, then jumps. Along 201 dial settings the stripe visits 18 distinct patterns (11 settings each), the face 57 (3.5 each). In figure 5's terms: the stripe's dial has 20 canals (four of at least 10 settings cover 88% of the dial; the longest is 113 settings, 56%), the face's 60 (five of at least 10 cover 57%; 39 are a single setting). The trained setting is a one-setting canal for both. An earlier version of figure 5 swept only plus or minus 0.08 around the trained code in steps of 0.002 and looked almost equal for the two (22 against 28 canals, 3.7 against 2.9 settings per canal, longest canal 33 against 27 settings, both at the weak end of the dial), because that window shows only the trained neighbourhood, where both are thin; it is available as `--canalSweep zoom` (written as `5_canalsOfTheDial_zoom.png`). Over random space-filling codes the stripe lands on an effective
  22 patterns from 160 codes (7 codes each, spread over about 7 of the 81 dimensions), the face on an effective 167 from 280 (1.7 each, about 16 dimensions) (**new**).
* **The qualification.** The compression holds for the code that writes the pattern directly. The stripe's two-wave code needs boundary cells driven past the bistable edge (ceiling 2.0; the face stayed at 1.3). Under the face's ceiling the stripe needs all 11 independent boundary values:
  no compression. And canalized is not robust: the exact target is a thin slice (dial window 0.008 for the stripe, under 0.004 for the face; 1% noise on the 40 values keeps the stripe at overlap ≥ 0.9 in 23 of 100 draws, the face in 0).

## 6. Canalization as constraint: the imprint lingers (the headline result)

**What kind of canalization this is.** Canalization is a special case of developmental bias or constraint (Maynard Smith et al. 1985: a bias in the production of variant phenotypes caused by the dynamics of the developmental system), distinguished by strength: Waddington's canal also buffers against perturbation and returns the trajectory to its channel.
What is measured here is the genus, with part of the species. The family bias is real and graded; against noise on the code it is partly buffered at the family level, for the stripe (1%-noisy copies still visit stripe-like patterns, 14% of frames, while the exact pattern is not buffered), much less for the face (0.1%); it ends at about 3,000 iterations; and the trajectory has no endpoint, only
intermittent visits. The defining test of a canal, whether the tissue returns to its family after its *state* is displaced during the wander (for example noise on the cell voltages at around iteration 1,500, compared with controls), has not been run. Until it is, say "constraint" or "priming" for the lingering, and "canalization with a stated scope" for the whole.
In the classical vocabulary the lingering stretch is a *labile specification* (a reversible bias left by a transient signal) that is not followed by *determination* (irreversible commitment), which is why it fades.

**The idea.** A code that has steered a tissue to a target may keep biasing the tissue's later wandering toward that target's family: stripe-like patterns for the stripe code, face-like patterns for the face code, visited every now and then, not at every step.
**The test**, fixed before the first run: both trained codes were replayed for 20,000 iterations in 64-bit (in 32-bit the face code's tissue turns lopsided from rounding alone), together with 24 + 24 symmetry-matched random codes (stripe-type controls keep both mirror symmetries, face-type controls the left–right one),
12 + 12 copies of each trained code with 1% noise on every coefficient, and the tissue with no boundary held. The dark set of the interior was stored every 5 iterations. A pattern belongs to a family broadly:
* **stripe family**: at least 6 dark cells and at least 80% of them in solid, tall, thin bars (one, two, three or more bars);
* **face family**: at least 80% overlap with a *deformed face*, any union of at least two of eyes, nose and mouth with each part moved up to one cell.

**What it shows** (figure 7, movie C):
* **It lingers.** Between iterations 301 and 3,000 the trained stripe code is in the stripe family in 30% of its frames, in about 5 separate visits of about 170 iterations (one, two and three stripes all occur: 87, 41 and 31 frames). The trained face code is in the deformed-face family in 9% of its frames,
  in one visit of about 245 iterations. The matched controls have a median of 0 in both cases; only 1 of 24 stripe-type controls reaches the stripe code's broad share (bars come easily to symmetric codes), and none of the 24 face-type controls reaches the face code's. On the exact target (the 27 stripe cells or the 14 feature cells)
  none of the controls matches the trained code. The two codes show a clean double dissociation: the stripe code never visits the face family, the face code never visits the stripe family.
* **Nearby codes share it, up to a point.** The stripe code's 1%-noisy copies visit the stripe family in 14% of their frames. The face code's copies mostly do not visit a deformed face (0.1% of frames; the exact face, 7%): the face's passage is a single fragile event, the stripe's is a robust habit.
* **It does not last.** Beyond about 3,000 iterations neither trained code visits its family again; the trained codes fall at or below the controls. Nearby codes follow the trained code's exact pattern until about 1,000 iterations (the face copies, about 2,000), are equally far from it as unrelated codes by
  about 3,000 (the stripe curves cross and re-cross between 2,300 and 3,800), and stay at the level of unrelated codes to 20,000. The late repertoire has the same density of dark cells as the controls'.
* **So the imprint outlasts the hold by about tenfold but does not reach long horizons.** This refines the earlier statement that the code's grip ends with the hold: three levels can be told apart. Attribution of the pattern to individual orders is lost by the time the face forms; the family bias lasts to about 2,500–3,000;
  memory of the exact state lasts to about 1,000–3,000.

**How to read it honestly.** The codes were trained on the first 3,000 iterations (the score stops at 2,999), so visits inside that window are partly selection. What the test adds is that the visits are repeated and long, occur at times other than the selected best moment, are rare in controls, are shared by nearby stripe codes,
and stop in the held-out window. A single trained code per class is tested (its copies are noisy versions of it), and the controls are random, not optimized.

**Two follow-up tests, documented but not presented** (backup figures 09 and 10):
* **Pattern space** (figure 09, `analyzeCanalizationTalkPatternClusters11x11.py`). A PCA of the interior patterns of the stripe set and the face set (trained, nearby and control codes of each; the first two components explain 17–24% of the variance) shows overlapping, only modestly separable clouds. A classifier scored on codes it never saw
  tells the two sets apart at 0.63–0.65 accuracy early, 0.61 at 4,000–10,000 and 0.60 at 10,000–20,000 (chance reaches about 0.53), so the sets weaken toward each other but do not fully merge; the symmetrised-pattern version gives the same picture, with a trace of the symmetry difference left in it.
  Inside each symmetry class, the neighbourhood of the stripe code stays distinct from matched controls at every horizon (balanced accuracy 0.75, 0.75, 0.73; chance about 0.58–0.61), while the face code's neighbourhood is distinct only early (0.61) and merges with generic face-type codes after about 4,000 (0.51, 0.47).
  The stripe, then, keeps a faint statistical territory even after its family visits and exact-state memory are gone; the face does not.
* **Distance to the families** (figure 10, `analyzeCanalizationTalkFamilyDistance11x11.py`). Each pattern's closeness to the stripe family (22,944 explicit members) and to the face family (9,923 deformed faces) is its best overlap with a member, expressed as a percentile among the controls' patterns (the raw numbers favour the larger stripe family: a random pattern is 0.31 close
  to it and 0.27 to the face family). The expectation that each set stays close to its own family and away from the other is half met. Early, the stripe set's patterns are unusually close to the stripe family (76th percentile, against 53–58 for the face family) and the trained code is nearer than 22 of 24 controls; after 3,000 the
  difference vanishes. The face set is not unusually close to the face family (55th–61st percentile, against 64th–71st for the stripe family), and its later mild closeness (55 against 44) is shared by the generic face-type controls, so it reflects the symmetry class and not the trained code. In the two-number space of the two closenesses a classifier
  separates the sets at only 0.53–0.55 (chance about 0.50). Family distance is a weak lens; the pattern-space result is the more telling one.

## 7. What can be claimed

1. **Canalization with a stated scope.** Compression: many codes to one pattern. Constraint: the family bias across time, graded, partly buffered against code noise, ending at about 3,000 iterations. It is not robustness of the exact target (a thin slice), and its robustness to perturbation of the tissue's state is untested.
2. **Readable, not linear.** The boundary profile is a sum of waves, but the response is switch-like (whole rows of three cells, a window 0.01 wide).
3. **Simple pattern, readable code and robust habit; complex pattern, distributed code and fragile passage.** The stripe's two bumps write its two ends independently and its family bias survives small changes of the code; the face has no single writing part, its orders are not separable brushes (0 of 16 role predictions held), and its one visit to a deformed face does not survive them.
4. **Passing states, not attractors.** Both targets are moments the tissue passes through (stripe 34 iterations, face 55–119 of 20,000), and the family bias ends. Say "guided to a specific state", and point to the canals as the nearest thing to attractors.

## 8. The visuals (dark theme, 1920 × 1080)

Boundary cells are violet (brighter = higher held conductance); interior cells glow where hyperpolarised, cyan for the stripe and amber for the face. The boundary's colour scale is stretched per target so the stripe's two bumps show; the glow is decoration.
In movie A each panel stops at its target's moment (stripe 504, face 2173) because the tissue keeps moving afterwards. Movie C and figure 7 use a broken time axis: 0 to 3,000 iterations at full width, 3,000 to 20,000 squeezed.

| File | What it shows | Use |
|---|---|---|
| `movies/A_guideThenLetGo.mp4` (9 s) | the organizer appears, is held (guidance), lets go (self-organization); stripe forms, then face | opener |
| `0_theModel.png` | the model in three panels: a cell (a slow polarising channel whose arrow points down and a fixed depolarising one whose arrow points up, i.e. the direction each pushes the voltage, and the two stable voltages of an isolated cell computed from the model), a tissue (the 11 × 11 lattice with gap junctions between neighbours, every cell shaded by its voltage in a real snapshot, and the electric field those voltages create drawn as arrows on the model's own 12 × 12 field grid at the cell corners; arrow lengths are square-root compressed so the weak far field shows, the directions are exact), and the loop (voltage writes into the shared field, every cell reads the field and slowly adjusts its polarising conductance, which sets the voltage). The boundary is not singled out and the guide (hold and release) is not shown: it is introduced on the next slide (figure 1). The caveat line says the tissue is fixed and was trained beforehand, with no ligands or gene network; that it was trained for the face with a per-cell boundary clamp is said aloud | the model slide (slide 7) |
| `1_theSpatialOrganizer.png` | a few waves, each with a dial whose pointer shows the size of its wave (up: none, clockwise: positive; faint: the dial turned by 0.3 and the wave it would give) → the boundary profile → the pattern | the idea; introduces the dials before movies B and D. In the movies the dials rest straight up at the trained code and the sweeps show departures from it |
| `2_twoPhases.png` | seven moments of each target, bracketed by guidance and self-organization | simple is written while held, complex long after |
| `3_stripeTwoSwitches.png` | top end only, bottom end only, both | the simple organizer is readable |
| `movies/B_turningTheKnobs.mp4` (9 s) | each knob turned: three for the stripe, four for the face | few knobs, many cells; tidy versus fragmenting |
| `5_canalsOfTheDial.png` | the dial (order 0) turned over its whole allowed range in 201 settings, read at each target's own readout (504, 2173): height is how much of the target is present, each flat stretch is a canal (a run of settings giving the identical pattern, longer and brighter the more codes share it), with thumbnails of the trained setting (outlined; a canal of one setting in both targets) and of the longest canals that look different from those already shown (at least four dark cells apart, boundary included, so two near-identical patterns are not both drawn). The stripe's dial is four long canals, the face's is long canals at the weak end and a fragmented rest | canalization as compression |
| `movies/C_theImprintLingers.mp4` (16 s) | both tissues run on to 20,000 iterations; a frame lights up whenever the tissue shows a pattern from its code's family, labelled one, two, three stripes or a deformed face | the presented result |
| `7_theImprintLingers.png` | for each code: family visits (the code, nearby codes, other codes of the same symmetry), how far nearby codes have drifted, and a gallery of family patterns visited | the presented result, as a still |
| `4_faceNoSinglePart.png` | upper wall alone, lower wall alone, whole boundary | the complex organizer is a distributed key (short) |
| `6_thePush.png` | strongest transfers (what an arrow means: section 4) over the pattern: a direct push into the stripe, a loop through the face | optional; has the compressed key for reading an arrow (section 4) |
| `8_thePhases.png` | the phases on the conductance: for each target the mean G_pol / G_ref of its cells (stripe cells, face feature cells) and of the rest of the interior over time (to scale), over the switching window (forced to polarize above, two stable voltages inside, forced to depolarize below), with a lane above marking the hold (iterations 0–301, the boundary's conductance fixed to the code) and the release (after it; a violet dashed line crosses both panels at the release), the phases as tinted stretches named flood, clear, write, latch and a tissue snapshot at the end of each phase underneath; the stripe's curves stop at its readout (504) with a note that it is complete; a two-line note at the foot says what the curves, the shading and the snapshots are | slide 12, before movie D: the nets' coloured edges are these phases |
| `movies/E_thePhasesOfBuilding.mp4` (16 s) | the same on a clock that gives each phase screen time (flood 3 s, clear 3 s, write 4.5 s, latch 3.5 s): the tissue glyphs run, the two conductance curves grow behind a moving point, the current phase is lit and named with a one-line reading in terms of the conductance, with the same hold / release lane above the curves; the stripe stops at its readout with a frame, the face runs on to 2173 and locks in | slide 12, optional if time is short; figure 8 is the still |
| `movies/D_theNetAsTheKnobsTurn.mp4` (26 s) | figure 6 animated: each knob is swept from the trained setting to its lowest, to its highest and back, while the arrows (strongest transfers per phase: flood, clear, write) and the pattern follow | optional; how the net depends on the dial. Arrows are the Relay Loop and Stripes pages' own steering-lab estimate (a kernel average over 846 face and 499 stripe simulated codes; thicker = more of the nearby codes carry the edge), not a new simulation per frame; the tissue underneath is simulated afresh at each setting |
| `movies/D_theNetAsTheKnobsTurn_thresh.mp4` | the same movie with arrows drawn only when at least *half* of the nearby simulated codes carry them (D uses a quarter) | at the trained setting this is close to figure 6: the runner-up edges (flood leaks into the stripe's flanks; three flood edges of the face) are gone |
| `movies/D_theNetAsTheKnobsTurn_thresh_ghost.mp4` | the half threshold, plus faint arrows for edges carried by a quarter to a half of the nearby codes, and dotted grey arrows for an edge of the trained code (figure 6's) that fewer than a quarter now carry | shows what the simpler net leaves out, and where the trained net's own edges drop away |

The order in the 25-minute talk is in section 2. Movie B and figure 6 are not on the main path (they overlap figures 3, 4 and movie D); `_thresh` is the version of D that matches figure 6, `_thresh_ghost` shows what it leaves out.
One-line captions: *A few waves on the boundary are an organizer. Hold it, then let go. A simple organizer writes its pattern while it is held; a complex one is finished by the tissue long after. Turn a knob and the stripe changes in tidy steps while the face breaks into fragments.
Many codes give one pattern: that is the canal. After the boundary lets go, the imprint lingers, and then the tissue wanders off.*

## 9. Where it sits: background, and what sets it apart

**Background** (the links come from a quick, non-systematic search; Kauffman is cited from memory):
* *Guided self-organization:* steering a self-organizing system with non-specific guidance (Prokopenko, Ay, Polani), mostly information-theoretic and agent-based ([overview](https://arxiv.org/pdf/1304.1842)).
* *Organizers and self-organization:* Gierer–Meinhardt's Hydra model ([paper](https://bio.mpg.de/255219/gierer-and-meinhardt-1972)); Wolpert's French Flag was posed as a problem, and his first solution was self-organizing ([Sharpe 2019](https://cob.silverchair.com/dev/article-pdf/1969505/dev185967.pdf));
  gradients orient Turing-like patterns ([Hiscock and Megason](https://pmc.ncbi.nlm.nih.gov/articles/PMC4707970)).
* *Bioelectric:* the [electric-face prepattern](https://now.tufts.edu/2011/07/18/face-frog-time-lapse-video-reveals-never-seen-bioelectric-pattern); a brief bioelectric perturbation rewrites a planarian body plan ([Durant et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5443973)), a persistent memory where here the bias fades;
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
* *Guidance that fades.* Attribution to the code is lost when the pattern appears, a family bias outlasts the hold by about tenfold, and then the tissue wanders off: the pattern is a moment, not an attractor, and the imprint is finite.
* *Limits are results.* The null (0 of 632 double-stripe restarts), the thin windows, the imprint that ends, and pre-registered, confirmatory testing in a simulation study.

**Two cautions.** The responder is not naive: the tissue (checkpoint 1888) was trained earlier to make the face under a per-cell boundary clamp (100-iteration hold), so it is competent for boundary guidance by construction and the stripe is a transfer to a second target.
Here the tissue stays frozen and new low-dimensional codes (hold 301) are searched. And the search above was quick, so claims of novelty are provisional until boundary control of reaction–diffusion and bioelectric inverse design are checked directly.

## 10. Questions to expect

* *Is steering from the boundary surprising?* Not by itself: boundary conditions steering a bulk is familiar in physics. What the work adds is how little it takes (2–4 numbers for 81 cells), how readable the guide is (simple targets) or not (complex ones), how long the tissue stays biased (about ten times the hold), and where it fails (0 of 632 double-stripe restarts).
* *Is this canalization?* Yes with a stated scope: compression (many codes to one pattern) and a graded, partly buffered constraint across time that ends at about 3,000 iterations. Canalization is a strong form of developmental bias; the strict test, return to the family after the tissue's state is displaced, is untested. It is not robustness of the exact pattern.
* *Does the imprint persist to long horizons?* No: the family bias stops at about 3,000 iterations. A faint statistical territory remains for the stripe code's neighbourhood (backup figure 09), none for the face's.
* *Isn't the lingering just selection?* Partly, because the codes were trained on the first 3,000 iterations. The visits are repeated, long, rare in controls, shared by nearby stripe codes, and absent in the held-out window.
* *Why does the face linger less than the stripe?* Its pattern is a narrow arrangement the training found as one passage; small changes of the code lose it. The stripe is a generic structure that symmetric codes produce easily, and its code is robust to small changes.
* *What exactly is an arrow in the relay net?* How much of the final difference between the steered and the unsteered tissue passed from one region to another through the shared field in that phase; an exact accounting of two runs, not a current and not an intervention (section 4).
* *Is the organizer a mechanism?* No: it organises the stretches of time and the loss of correspondence.
* *Why is the simple target not the easier one to find?* The stripe's two-wave code was found by 1 restart in 20 (ceiling 2.0), the face's by 14 of 30; under the face's ceiling the stripe needs 11 harmonics.
* *Does it work for any pattern?* No: the inverted stripe was reached by none of 632 restarts (best overlap 0.70).
* *What was pre-registered?* The stripe gate, mechanism and ends tests, the face's module confirmation (8 of 9) and the order-role knockouts; everything marked **new**, including every result of section 6, is exploratory.

## 11. Numbers

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
| Family visits, 301–3,000 | stripe code in the stripe family 30% of frames (1.9 episodes per 1,000 iterations, mean dwell 169; bars: 87 one, 41 two, 31 three, 4 four-plus frames); face code in the deformed-face family 9% (0.4 per 1,000, dwell 245; exact face 14%, dwell 385). Controls: median 0; stripe-type max 0.40 (1 of 24 at least as high as the trained code), face-type max 0.02 (0 of 24) | `data/canalizationTalkFamilyVisits1888Hold301.json` (**new**) |
| Family visits, held out | 3,000–10,000 and 10,000–20,000: 0 for both trained codes; stripe-type controls up to 0.07 | same |
| Nearby codes | stripe copies in the stripe family 14% of frames (301–3,000); face copies in the deformed-face family 0.1% (exact face 7%). Mean fraction of cells different from the trained code's pattern, copies vs unrelated codes: stripe 0.08 vs 0.22 (301–1,000), 0.15 vs 0.22 (1,000–3,000), 0.23–0.24 vs 0.21–0.24 after 3,000; face 0.005 vs 0.10, 0.10 vs 0.19, equal after 3,000 | same |
| Territory (identification of the code from one pattern, 25 classes, chance 0.04) | library from 3,000–10,000, test on 10,000–20,000 (the same runs, later in time): trained-code recall stripe 0.22, face 0.07; the stripe code's noisy copies are assigned to their parent in 28% of frames, the face's in 4%. (Within 301–3,000 the recall is 0.76 and 0.97, but library and test are then drawn from the same stretch of the same runs, so it is an upper bound and is not used.) | same |
| Pattern space (PCA) | first two components 17–24% of the variance; stripe set vs face set accuracy 0.63–0.65 / 0.61 / 0.60 (1,000–4,000 / 4,000–10,000 / 10,000–20,000; null 95th percentile about 0.53). Near-trained vs controls inside a class (balanced accuracy): stripe 0.75 / 0.75 / 0.73 (null 95th 0.58–0.61); face 0.61 / 0.51 / 0.47 (p 0.01 / 0.35 / 0.82) | `data/canalizationTalkPatternClusters1888Hold301.json` (**new**) |
| Distance to the families (percentile among controls) | stripe set, own vs other family: 76 vs 53 (trained), 76 vs 58 (copies), 301–3,000; 47 vs 53 and 51 vs 57 for 3,000–10,000; face set: 61 vs 64 (trained), 55 vs 71 (copies) early, 53 vs 47 and 56 vs 43 later (controls 54 vs 47). Two-coordinate classification of the set: 0.53 / 0.55 / 0.53 (null 95th 0.48–0.50) | `data/canalizationTalkFamilyDistance1888Hold301.json` (**new**) |

Movie D's knob ranges are limited to where at least about seven simulated codes lie nearby (stripe: ±0.3 on each coefficient; face: ×0.6–×1.4 for the level, ×0.2–×1.8 for the tilt and oval, ×0–×2 for the trefoil), arrows fade where fewer lie nearby, and the face's tilt and oval change the net little until moved far, because the ring's four region levels barely move.
Caveats: the two sweeps were sampled differently (compare each with its own baseline); offsets are absolute (±0.08 in G_pol / G_ref); the face slider replays clip to [0, 2.0]; nothing has a confidence interval. For the lingering tests: one trained code per class, random controls, symmetry-matched but not optimized;
the family definitions were fixed before the first run (the stripe family extended to several stripes and the face family to deformed faces at the user's request, before any result was computed), while the late-territory test and the 100-iteration smoothing of the drift curves were added after the first results were seen.

## 12. Files and how to regenerate

```
presentation/   0_theModel.png, 1_theSpatialOrganizer.png … 8_thePhases.png, movies/A_guideThenLetGo.mp4, B_turningTheKnobs.mp4, C_theImprintLingers.mp4, D_theNetAsTheKnobsTurn*.mp4, E_thePhasesOfBuilding.mp4
                backup_quantitative/ (figures 01–10, two movies, NOTES_quantitativeSet.md)    TALK_NOTES.md (this file)    TALK_NOTES_previous.md (the spatial-gene framing, see the PS)
```
Scripts (repo root): `canalizationTalkCommon.py`, `buildCanalizationTalkEssence11x11.py` (figures and movies above), `buildCanalizationTalkFigures11x11.py` and `buildCanalizationTalkMovies11x11.py` (the quantitative set), `analyzeBoundaryHarmonicCodeJitter11x11.py`, `analyzeCanalizationTalkPatternEnsemble11x11.py`,
`analyzeCanalizationTalkFamilyVisits11x11.py` (the lingering test; writes the trajectories, the per-frame flags and the visits), `analyzeCanalizationTalkPatternClusters11x11.py` and `analyzeCanalizationTalkFamilyDistance11x11.py` (the two follow-ups, which read the trajectories).
Data: `data/canalizationTalk*.json|npz`, `data/boundaryHarmonicCodeJitter*.json`.
```
python3 buildCanalizationTalkEssence11x11.py --ffmpeg <ffmpeg with libx264> --overwrite        # all; or --parts spatialOrganizer,twoPhases,stripeSwitches,faceNoSinglePart,canals,push,guideThenLetGo,turningTheKnobs,lingering,lingerThenWander,clusters,familySpace,relayKnobs,relayKnobsThresh,relayKnobsThreshGhost,model,phases,phasesMovie
python3 analyzeCanalizationTalkFamilyVisits11x11.py                                            # about 15 minutes for the 75 runs; refuses to overwrite
```
The cluster's ffmpeg module has no H.264 encoder; the movies used the static build inside the `imageio-ffmpeg` wheel (unpacked in a scratch folder, not installed). Replays are cached in `data/canalizationTalkReplays1888Hold301.npz` (made by the quantitative-set figure script),
`data/canalizationTalkPartialHolds1888Hold301.npz` and `data/canalizationTalkLongHorizon64Trained1888Hold301.npz` (the 20,000-iteration 64-bit trajectories of the two trained codes, for movie C) and `data/canalizationTalkNetStopPatterns1888Hold301.npz` (the tissue at the readout for every knob setting of movie D).

---

**PS — a gene-style reading is also possible.** The code can be cast as a *spatial gene*: the hold is transcription (the boundary is copied inward, each order owning its own modes of the bulk) and the release is translation (the tissue's own machinery turns the copy into a pattern;
with the field feedback off there is a transcript and no product). The structure fits (two phases, loss of correspondence, dependence on machinery), and it brings the genome-as-compressed-description angle with it. The molecular words fit less well, because an organizer induces a response rather than being
decoded and there is no codebook, which is why these notes use organizer, guidance and self-organization. The fuller treatment of the gene framing, with the evidence for each step, is in `TALK_NOTES_previous.md`.
