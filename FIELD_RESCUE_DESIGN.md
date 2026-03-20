# Field-Based Rescue via Collective Stress Signaling and ATP Bifurcation

## 1. Design Evolution

### Phase 1: Stress-Triggered Parameter Modulation (Superseded)

The initial design attempted to use field mismatch as a stress signal triggering CaMKII → parameter modulation. Critical analysis revealed gaps:

1. **Mismatch detection requires dedicated circuitry.** Unlike field superposition (physics), computing "how different am I from my neighbor" requires the cell to have a reference and perform subtraction — that's computation, not passive physics.

2. **Missing ATP success characteristics.** The ATP model succeeds because of four properties:
   - Bistable with healthy/unhealthy attractors
   - Low ATP (≈0) → zero GRN weights (null state)
   - High ATP (≈1) → native GRN weights (normal patterning)
   - Asymmetric reaction favoring the healthy attractor

   The stress-triggered design inherited bistability but lacked the clean attractor semantics (why would reduced G_0 specifically push toward the *correct* pattern?) and had no asymmetric reaction mechanism.

### Phase 2: Field-Coupled ATP via Healthy Template Correction (Superseded)

The second design kept the ATP model intact and replaced the diffusion term with a field-driven coupling term. The transducer read E_total (self + superposed neighbor fields) and output positive for healthy patterns, negative for unhealthy. Rescue worked by healthy neighbors' fields shifting E_total toward normal.

This was elegant but failed a critical experimental test: **the CEMA paper shows that adding healthy (untreated) embryos does NOT rescue treated embryos** (Section 2). The healthy-template model predicts the opposite — that healthy neighbors should provide the strongest rescue signal.

### Phase 3: Collective Stress Signaling with Bifurcation Control (Current)

The current design addresses the CEMA constraint by inverting the rescue logic: the stress signal comes from *perturbed* embryos, not healthy ones. Each embryo's field transducer outputs zero for normal patterns and a positive stress signal for abnormal patterns. Neighboring embryos' stress signals sum up and act as a **bifurcation parameter** on the receiver's ATP dynamics, eliminating the unhealthy basin when the collective signal is strong enough.

#### Transducer Design Evolution (within Phase 3)

The transducer — the component that converts an embryo's developmental state into a scalar stress signal — went through its own design evolution:

**Phase 3a: Centralized transducer (Superseded).** The initial transducer design read the embryo's entire field pattern E_self (~100-400 values) through a single learned function (linear weights, CNN, or template correlation) and output a scalar stress signal. This was computationally convenient but biologically implausible: no known biological entity reads an entire embryo's field pattern at once. Every real transduction pathway operates at the single-cell level.

**Phase 3b: Decentralized cell-local transducer with simple threshold (Superseded).** Replaced the centralized readout with identical per-cell transducers, each sensing only local Vmem and Ca²⁺. Each cell's CaMKII pathway (already present in the model) integrates the local developmental trajectory over time, and a sigmoid threshold on CaMKII outputs a per-cell stress signal. The embryo-level stress is the mean of all cell outputs (physically: total eATP released). This maps one-to-one onto known biology (VGCCs, Ca²⁺, CaMKII, pannexin channels). However, two problems emerged:

1. *Heterogeneous final states*: The healthy CaMKII pattern spans the full [0,1] range (some cells locked ON, others OFF due to bistability). A uniform threshold can't distinguish "healthy heterogeneous" from "perturbed heterogeneous" — both produce intermediate mean values.
2. *Pattern-blind*: If a perturbation produces a different spatial pattern but similar Ca²⁺ levels (e.g., rearranged facial features — a "Picasso face"), the simple integrator is blind to it. Detecting pattern rearrangement seems to require either positional reference information or a richer per-cell network.

**Phase 3c: Decentralized reaction-diffusion bistable stress system (Current).** The current design addresses both problems by making the stress computation itself a spatially-coupled bistable dynamical system with decay. Each cell has a stress variable S governed by bistable reaction-diffusion dynamics, with local Ca²⁺ acting as a bifurcation parameter. The key insight: diffusion + decay + bistability together create a **spatial frequency filter** that is sensitive to the *structure* of the Ca²⁺ pattern, not just its level. A decay term ensures that S can only cross the bistable threshold when enough neighboring cells simultaneously provide sufficient drive and their spatial arrangement allows constructive reinforcement via diffusion. The healthy pattern's specific spatial frequency signature is "tuned out" by the RD parameters, while perturbations with different spatial structure trigger nucleation and tissue-wide stress propagation. Rotated or translated versions of the healthy pattern are naturally invariant (the Laplacian is rotation/translation invariant). See Section 7 for full details.

---

## 2. Key Experimental Constraint from the CEMA Paper

Tung et al. (2024) established a critical finding that constrains any model of inter-embryo rescue:

**Adding untreated (healthy) embryos does NOT rescue treated embryos. Only adding more treated (perturbed) embryos does.**

Specific data:
- 150 treated embryos alone: ~37% defect rate
- 150 treated + 150 untreated (mixed): ~37% defect rate (no improvement)
- 300 treated embryos: significantly lower defect rate

The paper explicitly states: **"only perturbed individuals have a role in CEMA."**

Additional mechanistic findings:
- Stressed embryos release extracellular ATP (higher eATP in treated groups: 1.073 μM vs 0.989 μM control)
- eATP binds P2 receptors on neighboring embryos, triggering Ca²⁺ influx
- Blocking P2 receptors with suramin or PPADS eliminates the protective effect
- Physical contact is not required, but adjacency is (diffusible signal)
- The paper's computational model found that "supportive signals were sent in response by exposed (but not unexposed) embryos" and "compromised embryos weighted neighbors' inputs more"

**Any valid model must satisfy:**
1. Healthy embryos contribute zero rescue signal
2. More perturbed embryos → stronger rescue
3. Rescue is persistent (once recovered, embryo stays healthy)
4. The mechanism is collective — individual signals are insufficient alone

---

## 3. Core Insight: Collective Stress Signaling via Bifurcation Control

The rescue mechanism is not about healthy embryos providing a "correct template" — it is about perturbed embryos collectively generating a stress signal strong enough to reshape each other's dynamical landscape.

### The Key Idea

Each perturbed embryo emits a stress signal derived from its abnormal field pattern. These signals sum at each receiver. When the collective signal exceeds a critical threshold, it acts as a **bifurcation parameter** that eliminates the unhealthy ATP basin via saddle-node bifurcation, forcing ATP toward the (now unique) healthy attractor.

### Why This Is Different from the Phase 2 Design

| Aspect | Phase 2 (Template Correction) | Phase 3 (Stress Signaling) |
|--------|-------------------------------|----------------------------|
| Signal source | Healthy embryos (strong) | Perturbed embryos (strong) |
| Healthy embryo output | Positive (healthy template) | Zero (no stress) |
| Mechanism | Shift E_total toward normal | Collective bifurcation of ATP |
| CEMA prediction | Adding healthy helps (WRONG) | Adding healthy doesn't help (CORRECT) |
| Rescue origin | Healthy → perturbed | Perturbed → perturbed (mutual) |

### Biological Precedents for Bifurcation Control by Extracellular Signals

Extracellular signals modifying intracellular bifurcation structure is a well-characterized motif:

1. **EGF/MAPK cascade** [Xiong & Ferrell 2003]: Extracellular EGF acts as a bifurcation parameter — above a critical concentration, the low-ERK state disappears via saddle-node bifurcation, forcing cells into the active state. *This is exactly the architecture proposed here.*

2. **Wnt/beta-catenin** [Ferrell 2012]: Extracellular Wnt eliminates the low-beta-catenin state, committing cells to a specific developmental fate.

3. **TNF-alpha/NF-kappaB**: Sufficient TNF eliminates the inactive state, forcing the inflammatory transition.

4. **Apoptosis via Bcl-2/Bax**: Withdrawal of extracellular survival signals eliminates the survival basin — the reverse direction but the same mathematical principle.

The common principle: **extracellular ligand enters the ODE system as a parameter, not a state variable**, modulating effective rate constants or thresholds without participating in the intracellular feedback loops directly.

---

## 4. The Stress-Signal ATP Model

### 4.1 Original ATP Model (Bistable Reaction)

```
dATP/dt = reaction(ATP)
```

where `reaction(ATP)` is a bistable reaction with:
- Stable fixed point at ATP ≈ 0 (unhealthy)
- Unstable fixed point at ATP ≈ 0.3 (threshold/separatrix)
- Stable fixed point at ATP ≈ 1 (healthy)

The asymmetry: the unhealthy fixed point is closer to the unstable equilibrium than the healthy fixed point is.

**Biological basis for ATP bistability**: Glycolysis exhibits multiple steady-state behavior, segregating glucose metabolism into high-flux and low-flux states. Two regulatory loops involving phosphofructokinase and pyruvate kinase each give rise to bistable behavior [Mulukutla et al. 2014].

### 4.2 Stress-Signal Variant with Bifurcation Control

```
dATP/dt = reaction(ATP) + S_collective
```

where `S_collective` is the **sum of stress signals received from neighboring embryos**:

```
S_collective = g(ATP) * Σ_neighbors  stress_signal_j
```

Each embryo's stress signal is computed by a **field transducer**:

```
stress_signal_i = transducer(E_self_i)
```

The transducer is trained to output:
- **Zero** for healthy/normal field patterns
- **Positive values** for abnormal/perturbed field patterns

The receiver gain `g(ATP)` is **state-dependent**: stressed embryos (low ATP) are more sensitive to incoming signals than healthy embryos (high ATP). This implements the CEMA paper's observation that "compromised embryos weighted neighbors' inputs more."

### 4.3 Bifurcation Mechanism

The collective stress signal `S_collective` acts as a bifurcation parameter on the ATP dynamics. As `S_collective` increases:

```
ATP phase space evolution:

S_collective = 0 (isolated perturbed embryo):
                        reaction + 0
    ─────●───────────●─────────────────●─────
    ATP=0         ATP≈0.3           ATP=1
   (stable)     (unstable)        (stable)
    Two basins: embryo stays trapped at ATP≈0

S_collective < S_crit (few perturbed neighbors):
                    reaction + small S
    ─────●─────────●───────────────────●─────
    ATP=0       ATP≈0.35             ATP=1
   (stable)    (unstable)           (stable)
    Unhealthy basin shrinks but persists

S_collective ≥ S_crit (enough perturbed neighbors):
                reaction + large S
    ─────────────────────────────────●─────
                                  ATP=1
                                 (stable)
    Saddle-node bifurcation: unhealthy basin DISAPPEARS
    ATP is forced toward healthy attractor
```

This is a classic saddle-node bifurcation [Ferrell 2012; Huang et al. 2007]. The collective signal shifts the reaction curve upward until the unhealthy fixed point and the unstable equilibrium collide and annihilate.

### 4.4 Why This Works

1. **Perturbed embryo alone**: Abnormal field → transducer outputs positive stress signal → but no neighbors to receive it → S_collective ≈ 0 → ATP stays in unhealthy basin → developmental arrest

2. **Perturbed embryos in a group**:
   - Each perturbed embryo emits stress signal S > 0
   - Each receives S_collective = g(ATP) × (N-1) × S from neighbors
   - When N is large enough, S_collective exceeds S_crit
   - Saddle-node bifurcation eliminates unhealthy basin
   - ATP forced to healthy state → field normalizes → stress signal drops to zero
   - S_collective decreases → unhealthy basin reappears → but embryo is already in healthy basin
   - **Persistent rescue via bistable hysteresis**

3. **Healthy embryos don't help**: Their transducer outputs zero → they contribute nothing to S_collective → adding them doesn't increase the bifurcation parameter

4. **Healthy embryos are protected**: Even if they receive stress signals from perturbed neighbors, they are deep in the healthy basin AND their receiver gain g(ATP) is low (healthy embryos are less sensitive). The asymmetric reaction pulls them back.

5. **Group-size scaling** (emergent):
   - More perturbed embryos → larger S_collective
   - Threshold crossing at critical group size N_crit
   - Sigmoidal rescue probability vs group size

---

## 5. The Self-Cure Paradox and Its Resolution

### The Paradox

If a perturbed embryo generates a positive stress signal from its own abnormal field, and this signal helps other perturbed embryos escape the unhealthy basin — why doesn't the embryo use that signal on itself?

### The Resolution: Collective Threshold

Each individual embryo's stress signal `S` is too weak to trigger the bifurcation alone. The signal must satisfy:

```
1 × S  <  S_crit    (one embryo's signal is insufficient)
S alone  <  S_crit   (self-signal cannot rescue)
N_crit × S  ≥  S_crit   (collective signal crosses threshold)
```

This is not arbitrary — it reflects the physical reality that a single embryo's stress response (eATP release, Ca²⁺ signaling) is diluted in the extracellular medium. Only when multiple embryos contribute to the same shared medium does the concentration reach effective levels. The paper confirms this: physical adjacency is required but physical contact is not, consistent with a concentration-dependent diffusible signal.

There is no symmetry-breaking problem: all embryos start in the same perturbed state, all emit the same signal, all receive the same collective input, and all rescue simultaneously. This is consistent with the population-level dose-response observed in the CEMA data.

---

## 6. State-Dependent Receiver Sensitivity

The CEMA paper found that "compromised embryos weighted neighbors' inputs more." This is implemented via the state-dependent gain function:

```
effective_input = g(ATP) × Σ(neighbor_stress_signals)
```

where `g(ATP)` is high when ATP is low (unhealthy basin) and low when ATP is high (healthy basin).

### Why This Matters

1. **Amplifies rescue in the right context**: Stressed embryos are MORE sensitive to incoming signals, so the collective input is amplified by the receiver's own compromised state
2. **Protects healthy embryos**: Even if they receive some stress signals, their low receiver gain attenuates the effect
3. **Creates cooperative dynamics**: The combination of many incoming signals AND high receiver sensitivity enables collective escape

### Biological Basis

State-dependent sensitivity is well-established:
- Stressed cells upregulate P2 receptors [Tung et al. 2024]
- AMPK activity (which mediates ATP recovery) is enhanced when ATP is low [Hardie 2011]
- Calcium-dependent signaling sensitivity varies with metabolic state

---

## 7. The Decentralized Stress Transducer

### 7.1 Design Principles

The transducer converts an embryo's developmental state into a scalar stress signal for inter-embryo communication. Three principles guide the design:

1. **Decentralized**: Each cell has its own transducer with identical parameters. No cell reads the global pattern. The embryo-level stress signal emerges from local cell-level computations.
2. **Trajectory-sensing**: The transducer integrates over the developmental trajectory (t=100 to 1000), not a single snapshot. This captures timing, rate, and developmental milestone information.
3. **Pattern-sensitive via physics**: The spatial structure of the Ca²⁺ pattern matters, not just its average level. Pattern sensitivity arises from the physics of reaction-diffusion dynamics (diffusion + decay + bistability), not from explicit pattern matching.

### 7.2 Architecture: Reaction-Diffusion Bistable Stress System

Each cell (i,j) has a stress variable S(i,j,t) governed by bistable reaction-diffusion dynamics with decay:

```
dS/dt = (1/τ_S) * [k_on_S * OR_gate(S, Ca) * (1 - S) - k_off_S * S] - γ*S/(K_decay + S) + D_S * ∇²S
        \_________________________________________________________/   \_______________/   \_______/
                          bistable reaction                           Michaelis-Menten     diffusion
                                                                          decay
```

where:
```
self_activation(S) = (S² - K_S²) / (S² + K_S²)                          → [-1, +1]
OR_gate(S, Ca) = sigmoid( gain_S × sigmoid((Ca - Ca_threshold) / σ_ca)
                          + self_activation(S) - θ_or )                    → [0, 1]
                          └────── ca_drive ──────┘
∇²S_i = Σ_j A(i,j) * (S_j - S_i)                [discrete Laplacian from adjacency matrix]
```

The **embryo-level stress signal** is the mean of all cell outputs:
```
stress_signal = mean_i(S_i)
```

This physically corresponds to the total eATP released by all cells into the shared extracellular medium.

**Concurrent Ca²⁺ and S dynamics**: Ca²⁺ and S evolve simultaneously at each timestep during the Vmem drive phase, rather than pre-computing Ca²⁺ trajectories and then running S on the final pattern. This is biologically more plausible — stress pathways respond to Ca²⁺ in real-time, not after the bioelectric pattern has fully equilibrated. The simulation proceeds in two phases:

1. **Concurrent phase** (during Vmem drive, t=0 to ~1000): At each Vmem timestep, Ca²⁺ is updated from the current Vmem (via VGCCs), then S is advanced using the current Ca²⁺. Both variables co-evolve as the bioelectric pattern develops.
2. **Equilibration phase** (after Vmem drive ends): Ca²⁺ is held at its final value while S continues to equilibrate via the bistable RD dynamics. This allows S to reach its steady-state given the terminal Ca²⁺ pattern.

The Ca²⁺ parameters are **fixed** from a previously learned CaMKII model (`bestLearnedCaMKIIParams_0.dat`), ensuring the Ca²⁺ dynamics are biologically calibrated. Only the stress-specific parameters (τ_S, k_on_S, K_S, γ, K_decay, D_S, etc.) are learned.

### 7.3 How Ca²⁺ Acts as the Bifurcation Parameter

Empirical observation: when the GRN is perturbed (damping factor 0.7–0.9, representing teratogen exposure), average Ca²⁺ levels at t=1000 become **higher** than in the healthy case. More perturbation → higher Ca²⁺. This is biologically expected — essentially every form of cellular stress causes sustained Ca²⁺ elevation [Walter & Ron 2011].

The Ca²⁺ level controls the stress bistable switch via saddle-node bifurcation:

```
S (stress)
   |
   |         ●────────────── high stress (stable)
   |        /
   |       /  saddle-node bifurcation
   |      ●
   |     /
●──●──●────────────────── low stress (stable)
   |
   └───────────────────────── Ca²⁺ (level)
        ^                ^
     healthy          perturbed
    (damping=1.0)    (damping=0.7)
```

- **Healthy Ca²⁺ (below threshold)**: ca_drive is near 0. OR gate depends only on self_activation. If S < K_S, self_activation is negative → S decays to 0. Stable low state.
- **Perturbed Ca²⁺ (above threshold)**: ca_drive is near 1. OR gate receives strong positive input → S rises. Once S > K_S, self_activation locks it in. Stable high state.
- **Bistable memory**: If S has crossed K_S, self_activation maintains S high even if Ca²⁺ later fluctuates. Persistent stress signal.

### 7.4 Pattern Sensitivity via Spatial Frequency Filtering

The Michaelis-Menten decay term `γ*S/(K_decay + S)` is the key ingredient that makes the stress system sensitive to the *spatial structure* of the Ca²⁺ pattern, not just its average level. The saturable decay provides two critical properties: (1) at low S, strong relative decay (unsaturated phosphatase rapidly clears small stress signals), and (2) at high S, constant decay rate (saturated phosphatase cannot keep up with large stress signals). This creates an effective threshold: weak, spatially isolated stress signals are efficiently cleared, while strong, spatially coherent signals overwhelm the decay capacity and propagate.

**The mechanism**: Each cell's S is a competition between Ca²⁺ drive (pushes up), Michaelis-Menten decay (pulls down with saturation), and diffusion (redistributes spatially). The diffusion operator ∇² is a **low-pass spatial filter** that attenuates features at a rate proportional to their spatial frequency squared. For low S values (S << K_decay), the effective decay rate is approximately γ/K_decay, and the effective diffusion length scale is `l ≈ √(D_S * K_decay / γ)`, which determines which spatial features survive after smoothing.

**Nucleation dynamics as pattern classifier**: In a bistable RD system with decay, a local region of high S can only trigger tissue-wide propagation if it exceeds a **critical nucleation radius** r_c. Below that radius, diffusion + decay dissipate the signal before self-activation can lock it in. The healthy Ca²⁺ pattern's specific spatial structure — features of specific sizes at specific spacings — is "tuned out" by the RD parameters, so that no region exceeds the nucleation radius.

```
Healthy Ca²⁺ pattern:              Perturbed Ca²⁺ pattern:
  Ca²⁺ drive          S (steady)     Ca²⁺ drive          S (steady)
  ┌──────────┐        ┌──────────┐   ┌──────────┐        ┌──────────┐
  │ ·  ■■ ·  │        │ · ·· ·   │   │ ■■■■■ ·  │        │ ████ ·   │
  │ · ■■■■ · │  D,γ   │ · ·· ·   │   │ · · · ·  │  D,γ   │ ████ ·   │
  │ ·  ■■ ·  │ ────▶  │ · ·· ·   │   │ · ■■■■■ │ ────▶  │ · ████   │
  │ · ■■■■ · │        │ · ·· ·   │   │ · · · ·  │        │ · ████   │
  │ ·  ·· ·  │        │ · ·· ·   │   │ · · · ·  │        │ · · · ·  │
  └──────────┘        └──────────┘   └──────────┘        └──────────┘
  Features spread out → diffusion     Features clumped → diffusion
  smooths S below threshold           can't dissipate → nucleation!
  → no stress                         → tissue-wide stress
```

**Natural invariances**: The Laplacian ∇² is rotation and translation invariant. The RD dynamics depend only on the spatial frequency content of the pattern, not its position or orientation. Therefore:

| Perturbation type | Spatial frequency change | Detected? |
|---|---|---|
| Elevated Ca²⁺ (level shift) | DC component increases | Yes |
| Failed patterning (too uniform) | Loss of spatial features | Yes |
| Noisy/patchy pattern | Gain of high frequencies | Yes |
| Features wrong size or spacing | Shifted spatial spectrum | Yes |
| Rotated face | Same spectrum, phase shift | No (intended) |
| Translated face | Same spectrum, phase shift | No (intended) |
| Mirror-image face | Same spectrum | No (acceptable) |

This is exactly the invariance class desired: the stress system is pattern-sensitive but tolerant of rigid transformations that preserve the overall spatial structure. Biologically, a rotated or reflected face with correct feature proportions should not trigger stress — only genuinely malformed patterns should.

### 7.5 Diffusion-Based Synchronization Within the Embryo

The diffusion term D_S∇²S serves a dual role: spatial filtering (Section 7.4) and **synchronization of the tissue-level decision**.

In a perturbed embryo, cells with above-threshold Ca²⁺ develop high S. Diffusion propagates this to neighboring cells, pushing their S past the bistable threshold even if their local Ca²⁺ is borderline. This is **front propagation in a bistable medium** — once a critical mass of cells commits to the stressed state, the front sweeps across the tissue.

Three mechanisms contribute to synchronization:
1. **Diffusion of S**: Explicit spatial coupling via the ∇²S term
2. **Shared perturbation**: All cells in a perturbed embryo experience the same teratogen, producing correlated Ca²⁺ elevations. No active synchronization needed — the perturbation provides it for free.
3. **Gap junction coupling** (upstream): Cells are already electrically coupled through the bioelectric network, so each cell's Vmem and Ca²⁺ trajectory is influenced by its neighbors.

**Timescale constraint**: D_S should be moderate — fast enough to synchronize the 11×11 tissue within ~200–500 time units, slow enough that local bistable switching happens first. For a grid of width L~10, equilibration time is ~L²/D_S. The constraint τ_S << L²/D_S << T_communication gives D_S in the range 0.05–0.2.

### 7.6 Biological Interpretation

The decentralized stress transducer maps onto a concrete molecular pathway:

```
Per cell (i,j), continuously during development:

Vmem(i,j,t) → VGCCs → Ca²⁺ influx           [Catterall 2011, Pall 2013]
    │
    ▼
Ca²⁺(i,j,t) → bifurcation parameter         [local sensing, no global readout]
    │
    ▼
Stress RD system: S(i,j,t)                   [Ca²⁺-dependent bistability]
    │  - Bistable reaction (self-activation)  [ROS-Ca²⁺ feedback, or NFAT switch]
    │  - Decay (Michaelis-Menten)              [phosphatase with saturation kinetics]
    │  - Diffusion (D_S)                      [gap junction IP3/ROS, or paracrine]
    │
    ▼
S(i,j) → Pannexin-1 open probability → eATP  [Tung et al. 2024]
    │
    ▼
Embryo stress = mean(S) = total eATP release  [quorum-sensing analog]
```

**Molecular candidates for the stress variable S**:
- **ROS (H₂O₂)**: Membrane-permeable, diffuses between cells, produced in proportion to Ca²⁺-dependent mitochondrial stress, exhibits positive feedback via Ca²⁺-ROS loop [Walter & Ron 2011]
- **IP3**: Diffuses through gap junctions, triggers Ca²⁺ release in neighbors, well-characterized in developmental Ca²⁺ wave propagation [Goldberg et al. 2010]
- **NF-κB / NFAT activity**: Ca²⁺-activated transcription factors with switch-like nuclear translocation dynamics and positive feedback loops

**Key biological mechanisms** (with literature support):

1. **Ca²⁺-dependent bistability**: Multiple molecular systems exhibit Ca²⁺-parameterized bistability. The Ca²⁺ → mitochondrial ROS → Ca²⁺ release feedback loop creates a bistable switch between low-ROS/low-Ca²⁺ and high-ROS/high-Ca²⁺ states. The calcineurin-NFAT axis shows cooperative, switch-like nuclear translocation [Walter & Ron 2011].

2. **Diffusible stress signals**: H₂O₂ is membrane-permeable and diffuses freely between cells (D ~ 1000 μm²/s). IP3 diffuses through gap junctions and propagates Ca²⁺ waves across tissues [Goldberg et al. 2010]. eATP itself diffuses in extracellular space (D ~ 300 μm²/s) and amplifies stress via P2R-Ca²⁺ positive feedback [Tung et al. 2024].

3. **Voltage-gated Ca²⁺ channels as local sensors**: VGCCs respond to the membrane potential of the cell in which they reside — inherently a cell-local measurement. The S4 voltage sensor segments physically move in response to the transmembrane field [Catterall 2011]. At least 23 studies show VGCC blockers abolish EMF effects [Pall 2013].

4. **Homogeneous transducer parameters**: All cells in an early embryo express the same genetic program. Using identical transducer parameters across cells is not a simplification — it is the biological reality. The spatial variation in stress output arises from spatial variation in the *input* (Vmem/Ca²⁺ pattern), not from cell-to-cell differences in the transducer.

5. **Quorum sensing analogy**: The embryo-level stress signal (mean of all cell S values = total eATP) is a eukaryotic analog of bacterial quorum sensing. Individual cells produce noisy, imprecise signals; the collective output averages out noise and reflects the population-level state.

### 7.7 Why Simpler Alternatives Were Insufficient

Several simpler transducer designs were considered and rejected:

**Simple CaMKII threshold** (`stress = sigmoid(CaMKII - θ)`): Fails because the healthy CaMKII pattern is heterogeneous, spanning [0,1] due to bistability. Some healthy cells have high CaMKII (depolarized region) and some have low. A uniform threshold can't distinguish "healthy heterogeneous" from "perturbed heterogeneous."

**Trajectory-based measures** (cumulative Ca²⁺, CaMKII indecision time, rate of change): CaMKII is empirically just a temporally stretched version of Ca²⁺ — same spatial information, no qualitative gain. Trajectory-based measures help detect level perturbations but not pattern rearrangements.

**Spatial coherence measures** (local Laplacian variance, gap junction current magnitude): Detects disorganized patterns but fails for rearranged-but-organized patterns (Picasso faces). A face with eyes where the nose should be has perfectly smooth gradients, just in wrong positions.

**Multi-node GRN transducer** (6-10 genes per cell with intercellular signaling, as in [Manicka et al. 2023]): More expressive and could detect multiple perturbation types, but unnecessarily complex if the RD bistable system achieves pattern sensitivity through physics alone. The RD approach is preferred because:
- Fewer parameters (3 key: D_S, γ, K_S) vs ~50-100 for a GRN
- Pattern sensitivity emerges from diffusion physics, not learned weights
- Natural invariance to rotations/translations (from Laplacian symmetry)
- Direct biological mapping (Ca²⁺-ROS feedback, gap junction diffusion)

The multi-node GRN remains a fallback if simulation reveals perturbation types that the RD system cannot discriminate (e.g., perturbations requiring comparison of morphogen coordinates against bioelectric state).

### 7.8 Training the Stress System Parameters

The RD bistable stress system has a small number of learnable parameters:

**Core parameters** (4 critical):
| Parameter | Role | Suggested range |
|-----------|------|-----------------|
| D_S | Diffusion coefficient; sets spatial smoothing scale | 0.01–0.3 |
| γ | Michaelis-Menten V_max; max decay rate (saturated phosphatase) | 0.01–0.5 |
| K_decay | Michaelis-Menten K_m; half-saturation for decay (phosphatase affinity) | 0.01–0.5 |
| K_S | Bistable threshold; nucleation decision boundary | 0.1–0.8 |

**Supporting parameters** (from CaMKII-like architecture):
| Parameter | Role | Suggested range |
|-----------|------|-----------------|
| τ_S | Stress system time constant | 30–80 |
| k_on_S | Activation rate | 0.5–10.0 |
| k_off_S | Inactivation rate | 0.01–0.05 |
| Ca_stress_threshold | Ca²⁺ level for bifurcation | Set empirically between healthy and perturbed Ca²⁺ distributions |
| σ_ca | Ca²⁺ sensing sharpness | 0.005–2.0 |
| gain_S | Ca²⁺ drive weight vs self-activation | 1.0–6.0 |
| θ_or | OR gate threshold | 0.1–2.0 |

**Training objective**: Find D_S, γ, K_decay, K_S (and supporting parameters) such that:
1. Healthy embryo (GRN damping 1.0) → stress ≈ 0 (OFF state)
2. Intermediate perturbation (damping 0.95) → stress ≈ 0.5
3. Strong perturbation (damping 0.9) → stress ≈ 1.0 (ON state)
4. Self-cure constraint: single embryo's mean(S) < S_crit for ATP bifurcation

**Training pipeline** (`learnStressBistableSwitch.py`):
- Ca²⁺ parameters are **fixed** from learned CaMKII model (5 params from `bestLearnedCaMKIIParams_0.dat`)
- Only stress-specific parameters are learned (11 params via sigmoid parameterization for bounded optimization)
- Ca²⁺ and S evolve **concurrently** during the Vmem drive phase, followed by S equilibration
- **Weighted loss**: healthy target (stress=0) receives 3x weight to ensure the OFF state reaches ~0, addressing the asymmetry where Michaelis-Menten decay must overcome bistable self-activation
- Variance penalty discourages all outputs collapsing to the same value

**Training data**:
- Run stigmergic model with GRN damping levels {1.0, 0.95, 0.9} → different Vmem patterns → different Ca²⁺ trajectories
- Concurrent Ca²⁺ + S dynamics during Vmem drive, then S equilibration
- Optimize stress parameters to match target stress outputs [0.0, 0.5, 1.0]

### 7.9 Empirical Analysis of Learned Parameters

Analysis of learned parameter file `bestLearnedStressParams_6.dat` reveals that the optimizer converged to a **Ca-level threshold switch** rather than the intended bistable regime. The single-cell phase portrait analysis (`visualize_stress_bistability.py`) exposes the mechanism and its implications for CEMA.

#### Decoded parameters and their consequences

| Parameter | Learned value | Design default | Implication |
|-----------|--------------|----------------|-------------|
| Ca_stress_threshold | **8.06** | 0.8 | Ca drive sigmoid activates only when Ca >> 1 |
| sigma_ca | **0.005** | 0.2 | Extremely sharp sigmoid — effectively a step function |
| or_threshold_S | **1.73** | 0.6 | Self-activation alone (max +1) cannot open the OR gate |
| gain_S | **1.16** | 2.0 | Even full Ca drive contributes only ~1.16 to OR input |
| gamma | **0.25** | 0.08 | Strong Michaelis-Menten decay |
| K_decay | **0.01** | 0.3 | Tiny half-saturation → decay ≈ constant γ for all S > 0.01 |
| D_S | **0.01** | 0.1 | Negligible spatial diffusion |
| k_on_S | **1.73** | 3.0 | Moderate activation rate |
| k_off_S | **0.001** | 0.02 | Very low inactivation |

#### Why self-activation fails

The competitive self-activation term `(S² - K_S²)/(S² + K_S²)` ranges over [-1, +1]. For self-activation to sustain S without Ca drive, the OR gate must open sufficiently:

```
or_input = gain_S × ca_drive + self_activation - or_threshold_S

At Ca = 0: ca_drive ≈ 0 (since Ca << Ca_stress_threshold = 8.06)
Best case: self_activation = +1 (when S >> K_S)

or_input_max = 1.16 × 0 + 1.0 - 1.73 = -0.73
or_gate_max = sigmoid(-0.73) ≈ 0.33
```

An OR gate of 0.33 produces a reaction rate of `k_on × 0.33 × (1-S) / tau_S`, which peaks at ~0.12 near S = 0. But the Michaelis-Menten decay is `γ × S / (K_decay + S) ≈ 0.25` for any S > 0.01 (because K_decay = 0.01 saturates immediately). **Decay dominates everywhere** — the dS/dt nullcline has only one zero-crossing near S ≈ 0.

**Requirement for bistability**: `or_threshold_S < 1.0` (so self-activation alone can push or_input > 0), or equivalently, the decay must be weak enough relative to the reaction at intermediate S. A parameter sweep confirms that bistable configurations exist nearby in parameter space (e.g., `k_on_S = 3.0, or_threshold_S = 1.45, k_off_S = 0.01, gamma = 0.21`) producing three fixed points at S ≈ {0.03, 0.08, 0.70} at Ca = 0. The OFF basin is narrow (S < 0.08), the ON state is at S ≈ 0.70. The learned parameters are close but not in this regime.

#### Decomposed phase portrait: step-function Ca sensitivity

At Ca = 0 (no external drive), the dS/dt curve is entirely negative -- all initial conditions decay to S ~ 0. As Ca approaches the threshold (~8.06), the entire transition from "S decays to zero" to "S rises to a high steady state" occurs within a window of ~6 sigma ~ 0.03 Ca units:

```
Ca = 8.0441  (ca_drive = 0.047):  dS/dt negative everywhere -- S -> 0
Ca = 8.0541  (ca_drive = 0.269):  reaction begins to lift, still sub-threshold
Ca = 8.0591  (ca_drive = 0.500):  threshold -- total dS/dt crosses zero at mid-range S
Ca = 8.0641  (ca_drive = 0.731):  strong positive region -- S -> high steady state
Ca = 8.0741  (ca_drive = 0.953):  nearly saturated -- maximum reaction lift
```

The reaction term (blue curve in `data/stress_decomposed_phase_portrait.png`) is visibly lifted panel-by-panel while the Michaelis-Menten decay (red curve) remains unchanged -- it depends only on S, not Ca. The total dS/dt (black curve) transitions from entirely negative to having a large positive region within this narrow Ca window. This is a sharp threshold switch, not a graded bifurcation.

#### Consequences for CEMA

**What works with these parameters:**

1. **Healthy → stress 0**: If healthy embryos produce Ca < 8, S → 0 everywhere. ✓
2. **Perturbed → stress > 0**: If perturbed embryos produce Ca > 8, S → 0.62. ✓
3. **Healthy don't help**: Zero stress signal from healthy embryos. ✓
4. **Group rescue via `runGroupRescue.py`**: The diffusive field F on the embryo lattice sums neighbor stress emissions, modulates effective damping, and can rescue perturbed embryos. The threshold-switch behavior of S is sufficient for this.

**What is lost:**

1. **No bistable memory**: S tracks Ca instantaneously. If Ca drops (e.g., during a transient fluctuation), S drops immediately. The design intended S to "lock in" via self-activation (Section 7.3), providing persistent stress signaling even through Ca fluctuations.

2. **No spatial frequency filtering**: With D_S = 0.01 (vs design spec 0.05–0.2) and no active bistability, the nucleation mechanism (Section 7.4) does not operate. A "Picasso face" — rearranged features with similar mean Ca — would not trigger stress. The system is a Ca-level detector, not a pattern detector.

3. **No graded response**: The sharp threshold (sigma_ca = 0.005) means stress is binary (0 or 0.62) rather than proportional to perturbation severity. The training target of intermediate stress = 0.5 for damping = 0.95 likely cannot be achieved — the system either fires or doesn't.

4. **Self-cure paradox is trivially resolved**: Since S is entirely Ca-controlled, a single embryo's stress output is determined by its Ca pattern alone. The collective threshold in the CEMA mechanism (Section 5) operates through the inter-embryo diffusive field, not through within-embryo S dynamics.

#### Why the optimizer found this solution

The learning objective (Section 7.8) minimizes `Σ weight_i × (stress_i - target_i)²` across damping levels. The optimizer likely found that a sharp Ca threshold cleanly separates the training damping levels (1.0, 0.95, 0.9), which produce distinct Ca distributions. This solution achieves low loss without needing the more complex bistable regime. The bistable regime is harder to find because:

- The OFF basin is inherently narrow (S_low ≈ 0.03, S_unstable ≈ 0.08 — a gap of only 0.05)
- Random initialization rarely lands in the bistable parameter region
- Gradient-based optimization (Rprop) can easily push through the bistable window into the monostable-ON regime (lower or_threshold) or monostable-OFF regime (higher or_threshold)

#### Recommendations for recovering bistability

To steer learning toward the intended bistable regime:

1. **Constrain or_threshold_S < 1.0**: This ensures self-activation (max +1) can push or_input > 0, a necessary condition for bistability at Ca = 0.
2. **Constrain D_S ∈ [0.05, 0.2]**: Forces spatial coupling sufficient for nucleation dynamics.
3. **Add a bistability penalty**: Require that S trajectories from S₀ = 0.01 and S₀ = 0.99 at Ca = 0 converge to different steady states. Loss term: `-|S_final(S₀=0.99) - S_final(S₀=0.01)|`.
4. **Constrain K_decay ∈ [0.1, 0.5]**: Prevents the Michaelis-Menten decay from degenerating into a constant (when K_decay << S).
5. **Multi-phase training**: First learn parameters that produce bistability at Ca = 0, then fine-tune to match the healthy/perturbed stress targets.

#### Visualization

The analysis scripts and diagnostic figures:
- `visualize_stress_bistability.py`: Single-cell phase portraits, bifurcation diagrams, decomposed dS/dt, internal signals, and bistable-vs-learned comparison
- `data/stress_decomposed_phase_portrait.png`: Reaction vs decay vs total dS/dt at multiple Ca levels
- `data/stress_internal_signals.png`: self_activation, or_gate, and ca_drive as functions of S
- `data/stress_bistability_comparison.png`: Learned (monostable) vs true-bistable parameter regime, with timeseries and phase portraits
- `data/stress_bifurcation_diagram.png`: Steady-state S vs Ca showing the sharp threshold at Ca ≈ 8.06

---

## 8. Signal Flow Diagram

### 8.1 Intra-Embryo: Decentralized Stress Computation

```
┌─────────────────────────────────────────────────────────────────────┐
│                  WITHIN A SINGLE EMBRYO (11×11 grid)                │
│                                                                     │
│  Per cell (i,j) — same parameters for all cells:                    │
│                                                                     │
│  Vmem(i,j,t) ──▶ VGCC ──▶ Ca²⁺(i,j,t)                             │
│                              │                                      │
│                              │ bifurcation parameter                │
│                              ▼                                      │
│                    ┌──────────────────────┐                         │
│                    │  Stress RD system    │                         │
│                    │                      │                         │
│                    │  dS/dt =             │                         │
│                    │   reaction(S, Ca²⁺)  │ ← bistable (OR gate    │
│                    │   - γS/(K_decay+S)   │   + self-activation)   │
│                    │   + D_S*∇²S   ◄──────────── from neighbors    │
│                    │                      │                         │
│                    └──────────┬───────────┘                         │
│                               │                                     │
│                               ▼                                     │
│                    S(i,j) → Pannexin-1 → eATP(i,j)                  │
│                                                                     │
│  Embryo-level stress = mean_ij(S) = total eATP released             │
│                                                                     │
│  Key dynamics:                                                      │
│  - Healthy Ca²⁺ pattern: S stays below threshold (spatial           │
│    frequency filter tuned to pass healthy pattern without stress)    │
│  - Perturbed Ca²⁺ pattern: S nucleates above threshold,             │
│    diffusion propagates → tissue-wide high S → eATP release         │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Inter-Embryo: Collective Stress Signaling

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                         MULTI-EMBRYO SYSTEM                                   │
│                                                                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐            │
│  │    Embryo 1       │  │    Embryo 2       │  │    Embryo N       │            │
│  │   (perturbed)     │  │   (perturbed)     │  │   (perturbed)     │            │
│  │                   │  │                   │  │                   │            │
│  │ [11×11 cells]     │  │ [11×11 cells]     │  │ [11×11 cells]     │            │
│  │ each: Vmem→Ca²⁺   │  │ each: Vmem→Ca²⁺   │  │ each: Vmem→Ca²⁺   │            │
│  │   →S(RD bistable) ��  │   →S(RD bistable) │  │   →S(RD bistable) │            │
│  │   →mean(S)=S₁     │  │   →mean(S)=S₂     │  │   →mean(S)=Sₙ     │            │
│  │    │   broadcast   │  │    │   broadcast   │  │    │   broadcast   │            │
│  └────┼──────────────┘  └────┼──────────────┘  └────┼──────────────┘            │
│       │                      │                      │                           │
│       └──────────┬───────────┴──────────┬───────────┘                           │
│                  │     sum at each       │                                       │
│                  │      receiver         │                                       │
│                  ▼                       ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐                   │
│  │  At Embryo i:                                             │                   │
│  │                                                           │                   │
│  │  S_collective = g(ATP_i) × Σⱼ≠ᵢ Sⱼ                       │                   │
│  │                    ↑                                      │                   │
│  │         state-dependent gain                              │                   │
│  │         (high when ATP low,                               │                   │
│  │          low when ATP high)                               │                   │
│  │                    │                                      │                   │
│  │                    ▼                                      │                   │
│  │         ┌───────────────────┐                             │                   │
│  │         │  ATP dynamics     │                             │                   │
│  │         │  dATP/dt =        │                             │                   │
│  │         │  reaction(ATP)    │  ← bistable [Mulukutla 2014]│                   │
│  │         │  + S_collective   │  ← bifurcation parameter   │                   │
│  │         └────────┬──────────┘                             │                   │
│  │                  │                                        │                   │
│  │                  ▼                                        │                   │
│  │            ATP level                                      │                   │
│  │                  │                                        │                   │
│  │                  ▼                                        │                   │
│  │          GRN weight scaling                               │                   │
│  │                  │                                        │                   │
│  │                  ▼                                        │                   │
│  │        Developmental patterning                           │                   │
│  └──────────────────────────────────────────────────────────┘                   │
│                                                                                 │
│  ┌──────────────────┐                                                           │
│  │  Healthy Embryo   │  All cells: Ca²⁺ below bifurcation → S ≈ 0 everywhere    │
│  │                   │  mean(S) ≈ 0 → contributes nothing to S_collective        │
│  └──────────────────┘                                                           │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Rescue Dynamics: Temporal Narrative

### Phase 1: Pre-Rescue (Isolated or Small Group)

```
t=0: All perturbed embryos at ATP ≈ 0 (unhealthy basin)
     Each emits stress signal S > 0
     S_collective = g(ATP_low) × (N-1) × S
     If N < N_crit: S_collective < S_crit → unhealthy basin persists
     ATP stays at ≈ 0 → developmental arrest
```

### Phase 2: Rescue Triggered (Sufficient Group Size)

```
t=0: N ≥ N_crit perturbed embryos
     S_collective = g(ATP_low) × (N-1) × S ≥ S_crit
     Saddle-node bifurcation → unhealthy basin disappears
     ATP begins rising toward healthy attractor
```

### Phase 3: Recovery and Signal Decay

```
t=T_rescue: ATP approaches 1 (healthy basin)
     Field pattern normalizes → transducer output drops toward 0
     S_collective decreases as embryos recover
     Unhealthy basin may reappear → but embryo is already in healthy basin
     Bistable hysteresis provides persistence
```

### Potential Race Condition

As early-recovering embryos stop emitting stress signals, S_collective drops. Late-recovering embryos receive less collective support. This predicts:
- Partial rescue in marginal group sizes (near N_crit)
- Possible bimodal outcomes: most embryos rescued, some left behind
- This is worth simulating and comparing to experimental dose-response data

### Edge Effects

If the stress signal is diffusible (decays with distance), embryos at the center of a group receive more signal than edge embryos. This predicts spatially heterogeneous rescue — center embryos rescue first. If not observed experimentally, consider well-mixed conditions (shared medium).

---

## 10. Molecular Identity of the Stress Signal

The stress signal in our model maps onto a concrete molecular pathway from the CEMA paper:

| Level | Identity | Role |
|-------|----------|------|
| Intracellular origin | Abnormal Vmem → aberrant VGCC activity → elevated Ca²⁺ | Stress detection |
| Between embryos | Extracellular ATP (eATP) released via pannexin hemichannels | Diffusible stress signal |
| At receiver membrane | eATP binds P2X/P2Y receptors | Signal reception |
| Receiver cytoplasm | Ca²⁺ influx (P2X) and IP3-mediated Ca²⁺ release (P2Y) | Second messenger |
| Effector | CaMKK2 → AMPK activation | Bifurcation parameter modifier |

**Critical distinction**: Extracellular ATP (signaling molecule) and intracellular ATP (bistable state variable) are in **different compartments** and regulated by **different processes**. There is no circular logic — a stressed embryo releases eATP (depleting intracellular stores slightly) as a paracrine signal, which when received by neighbors through P2 receptors triggers Ca²⁺-mediated pathways that shift the receiver's intracellular ATP bifurcation parameter.

**Ca²⁺ as the bifurcation parameter**: Ca²⁺ modulates kinase/phosphatase balance in several ways:
- Ca²⁺/calmodulin activates CaMKII and CaMKK2 [Hardie 2011]
- CaMKK2 directly activates AMPK (master metabolic switch promoting ATP production)
- AMPK promotes mitochondrial biogenesis, fatty acid oxidation, autophagy
- This effectively shifts the production rate in the ATP bistable system, moving the saddle-node bifurcation point

---

## 11. ATP Success Characteristics Mapping

| ATP Characteristic | How Stress-Signal Model Inherits It |
|--------------------|-------------------------------------|
| **Bistable** | Same reaction kinetics, unchanged [Mulukutla et al. 2014] |
| **Low ATP → zero GRN weights** | Same, unchanged — unhealthy state = developmental arrest |
| **High ATP → native GRN weights** | Same, unchanged — healthy state = normal patterning |
| **Group-size scaling** | Collective stress signal sums from N neighbors; threshold crossing |
| **Asymmetric reaction favoring healthy** | Same reaction kinetics — healthy basin larger |
| **Healthy don't help (NEW)** | Healthy transducer output = 0 → no contribution to S_collective |

**All original success characteristics preserved, plus the new CEMA constraint is satisfied.**

---

## 12. Comparison Across All Three Design Phases

| Aspect | Phase 1: Stress-Triggered | Phase 2: Template Correction | Phase 3: Stress Signaling |
|--------|---------------------------|------------------------------|---------------------------|
| **New components** | FieldStressDetector, StressCaMKIISwitch, FieldRescueModulator | FieldTransducer only | FieldTransducer + receiver gain |
| **Mismatch detection** | Required (problematic) | Not required | Not required |
| **Attractor semantics** | Unclear | Clear | Clear |
| **Asymmetry mechanism** | Missing | From ATP reaction | From ATP reaction + receiver gain |
| **CEMA: healthy don't help** | Not addressed | WRONG prediction | CORRECT prediction |
| **CEMA: more treated helps** | Not addressed | Partial (via E_total) | CORRECT (collective threshold) |
| **Rescue persistence** | Unclear | Via ATP hysteresis | Via ATP hysteresis (saddle-node) |
| **Biological plausibility** | Requires mismatch computation | Passive but wrong mechanism | eATP/P2R/Ca²⁺ pathway [Tung 2024] |

---

## 13. Implementation Plan

### 13.1 New File: `stressBistableSwitch.py`

```python
class StressBistableSwitch:
    """
    Decentralized reaction-diffusion bistable stress system.

    Each cell has a stress variable S governed by bistable RD dynamics
    with decay. Local Ca²⁺ acts as bifurcation parameter. Diffusion
    couples neighboring cells. Decay + diffusion + bistability create
    a spatial frequency filter for pattern-sensitive stress detection.

    Biological interpretation:
    - S represents a Ca²⁺-dependent bistable pathway (ROS-Ca²⁺ feedback,
      NFAT switch, or p38/JNK stress kinase cascade)
    - Diffusion via gap junction-permeable molecules (IP3, H₂O₂)
    - Decay via protein degradation / phosphatase activity
    - Output: S drives pannexin-1 → eATP release [Tung et al. 2024]
    """

    def __init__(self, num_cells, adjacency_matrix, params=None):
        """
        Args:
            num_cells: number of cells in the tissue grid
            adjacency_matrix: (num_cells, num_cells) lattice connectivity
            params: dict of learnable parameters (or use defaults)
        """
        self.num_cells = num_cells
        self.A = adjacency_matrix  # for discrete Laplacian
        self.S = torch.zeros(num_cells)  # stress variable per cell

        # Default parameters (learnable)
        defaults = {
            'tau_S': 50.0,           # stress time constant
            'k_on_S': 3.0,          # activation rate
            'k_off_S': 0.02,        # inactivation rate
            'K_S': 0.4,             # bistable threshold
            'Ca_stress_threshold': 0.5,  # SET EMPIRICALLY
            'sigma_ca': 1.0,        # Ca²⁺ sensing sharpness
            'gain_S': 2.5,          # Ca²⁺ drive weight
            'or_threshold_S': 0.5,  # OR gate threshold
            'D_S': 0.1,             # diffusion coefficient
            'gamma': 0.08,          # Michaelis-Menten V_max (max decay rate)
            'K_decay': 0.3,         # Michaelis-Menten K_m (half-saturation)
        }
        self.params = params if params else defaults

    def step(self, dt, Ca):
        """
        Args:
            dt: timestep
            Ca: (num_cells,) local Ca²⁺ at each cell
        """
        p = self.params

        # Ca²⁺ drive (per cell)
        ca_drive = torch.sigmoid(
            (Ca - p['Ca_stress_threshold']) / p['sigma_ca']
        )

        # Competitive self-activation (per cell)
        self_act = (self.S**2 - p['K_S']**2) / (self.S**2 + p['K_S']**2)

        # OR gate
        or_input = p['gain_S'] * ca_drive + self_act - p['or_threshold_S']
        or_gate = torch.sigmoid(or_input)

        # Bistable reaction
        reaction = (or_gate * p['k_on_S'] * (1 - self.S)
                     - p['k_off_S'] * self.S) / p['tau_S']

        # Michaelis-Menten decay (phosphatase kinetics)
        # Low S: unsaturated phosphatase → strong relative decay
        # High S: saturated phosphatase → constant decay rate
        decay = -p['gamma'] * self.S / (p['K_decay'] + self.S)

        # Diffusion: ∇²S_i = Σ_j A(i,j) * (S_j - S_i)
        laplacian = torch.matmul(self.A, self.S) - self.A.sum(dim=1) * self.S
        diffusion = p['D_S'] * laplacian

        # Update
        self.S = self.S + dt * (reaction + decay + diffusion)
        self.S = torch.clamp(self.S, 0, 1)

    def get_embryo_stress(self):
        """Returns embryo-level stress = mean(S) = total eATP proxy."""
        return self.S.mean()
```

### 13.2 Modifications to ATP Model (Unchanged from Phase 3)

Add collective stress signal as bifurcation parameter:

```python
def step(self, dt, neighbor_stress_signals=None):
    # Bistable reaction (unchanged) [Mulukutla et al. 2014]
    reaction = self.compute_reaction(self.ATP)

    # Collective stress signal as bifurcation parameter
    if neighbor_stress_signals is not None and len(neighbor_stress_signals) > 0:
        # State-dependent receiver gain [Tung et al. 2024]
        receiver_gain = self.compute_receiver_gain(self.ATP)
        S_collective = receiver_gain * sum(neighbor_stress_signals)
    else:
        S_collective = 0.0

    # Update: S_collective acts as bifurcation parameter
    self.ATP = self.ATP + dt * (reaction + self.coupling_strength * S_collective)
    self.ATP = torch.clamp(self.ATP, 0, 1)

def compute_receiver_gain(self, ATP):
    """
    State-dependent sensitivity: stressed embryos weight neighbors' inputs more.
    """
    return 1.0 - ATP  # gain ∈ [0, 1], inversely proportional to health
```

### 13.3 New File: `run_field_coupled_rescue.py`

Multi-embryo simulation with decentralized stress signaling:

1. Initialize N embryos (mix of perturbed and healthy)
2. For each embryo: run bioelectric dynamics → Vmem → Ca²⁺ at each cell
3. For each embryo: step the StressBistableSwitch with local Ca²⁺ values
4. For each embryo: compute embryo stress = mean(S) via `get_embryo_stress()`
5. For each embryo: receive S_collective = g(ATP) × Σ neighbor stresses
6. Update ATP with reaction + S_collective (bifurcation control)
7. ATP modulates GRN weights → developmental patterning
8. Repeat; visualize rescue trajectories and group-size dependence

### 13.4 New File: `learnStressParameters.py`

Train the RD bistable stress system parameters:

1. Generate training data: run stigmergic model with healthy and perturbed parameters, record Ca²⁺(i,j,t) trajectories for multiple GRN damping levels (0.7–1.0)
2. Optimize D_S, γ, K_S, Ca_stress_threshold (and supporting params) to maximize separation: mean(S) ≈ 0 for healthy, mean(S) > 0 for perturbed
3. Validate pattern sensitivity: test on perturbations that change spatial pattern but not average Ca²⁺
4. Verify self-cure constraint: single embryo's mean(S) < S_crit
5. Verify healthy embryo invariance: rotated/translated healthy patterns → mean(S) ≈ 0

---

## 14. Verification Plan

1. **Transducer accuracy**: Train transducer, verify it correctly classifies healthy (output ≈ 0) vs unhealthy (output > 0) field patterns (>90% accuracy on held-out set)

2. **Single embryo dynamics**:
   - Healthy embryo alone → stress signal ≈ 0 → S_collective = 0 → ATP stays high → normal development
   - Perturbed embryo alone → stress signal > 0 → but S_collective = 0 (no neighbors) → ATP stays low → developmental arrest

3. **Self-cure constraint**: Verify that a single embryo's stress signal S < S_crit (cannot trigger its own bifurcation)

4. **Group-size sweep (perturbed only)**: {1, 2, 5, 10, 25, 50, 100} perturbed embryos → verify sigmoidal rescue probability with threshold at N_crit

5. **CEMA test (critical)**: Compare:
   - 10 perturbed alone → rescue rate R₁
   - 10 perturbed + 10 healthy → rescue rate R₂
   - 20 perturbed → rescue rate R₃
   - **Must show**: R₂ ≈ R₁ (healthy don't help) and R₃ > R₁ (more perturbed helps)

6. **Healthy embryo stability**: Healthy embryo surrounded by many perturbed embryos → verify it remains healthy (asymmetric basin + low receiver gain protect it)

7. **Rescue persistence**: After rescue (ATP → 1), remove neighbors → verify embryo stays in healthy basin (hysteresis)

8. **Race condition test**: Monitor temporal dynamics in marginal group sizes → check for partial rescue

9. **Comparison with ATP diffusion model**: Same perturbation conditions → compare rescue rates and dynamics

---

## 15. Open Questions

### Resolved (from earlier phases)

4. ~~**Temporal dynamics of stress signal**: Should the transducer read instantaneous field or time-averaged field?~~ **Resolved**: The decentralized RD bistable stress system integrates over the full developmental trajectory (t=100–1000) via the bistable S dynamics. The time constant τ_S (~50) provides natural temporal filtering. No explicit time-averaging needed.

### Open

1. **Receiver gain function**: What form should g(ATP) take? Options:
   - Linear: `g = 1 - ATP` (simplest)
   - Sigmoidal: `g = 1 - sigmoid((ATP - threshold)/sensitivity)` (sharper transition)
   - The paper suggests a qualitative difference (stressed vs healthy), favoring sigmoidal

2. **Coupling strength and N_crit**: The coupling strength determines the critical group size. Too low → N_crit is very large; too high → even 2 embryos rescue each other. Must calibrate against experimental dose-response data.

3. **Distance dependence**: If stress signal is diffusible eATP, concentration decays with distance. Options:
   - Well-mixed: all embryos in shared medium → distance-independent
   - Diffusion-limited: `S_received = S_emitted / d²` → spatial effects
   - The paper shows adjacency matters but contact isn't required → moderate distance dependence

5. **Noise near bifurcation**: Saddle-node bifurcations are sharp (all-or-nothing). The experimental data shows graded rescue with group size. Adding stochastic terms to ATP dynamics would soften the transition and may be needed for quantitative agreement.

6. **Ca_stress_threshold placement**: The RD bistable stress system requires the Ca²⁺ bifurcation threshold to sit between healthy and perturbed Ca²⁺ distributions. Empirical data shows monotonic separation (damping 0.7→1.0 maps to distinct Ca²⁺ levels), but the exact threshold must be calibrated. **Partial resolution**: The learned File 6 parameters placed Ca_stress_threshold at 8.06 with sigma_ca = 0.005, making it an ultra-sharp step function. This cleanly separates healthy from perturbed but eliminates graded response and pattern sensitivity. See Section 7.9 for full analysis. The open question is whether to constrain this parameter to force a lower, softer threshold that preserves bistable dynamics, or accept the sharp threshold for a simpler CEMA demonstration.

7. **Spatial frequency sensitivity range**: The RD parameters (D_S, γ) define which spatial patterns trigger stress and which don't. Systematic characterization is needed: for the healthy face pattern's specific spatial frequency content, what range of D_S and γ provides the best discrimination margin? Is there a risk of the filter being too narrow (missing subtle perturbations) or too broad (triggering on normal variants)?

8. **Perturbations that evade the RD filter**: The RD bistable stress system detects level perturbations (elevated Ca²⁺) and structural perturbations (different spatial frequencies). But perturbations that produce the same spatial frequency content as the healthy pattern (e.g., exact spatial inversion) may evade detection. How likely are such perturbations biologically? If they arise, the fallback is a multi-node GRN transducer (Section 7.7).

9. **D_S molecular identity**: What molecule mediates the diffusion of S between cells? Gap junction-permeable candidates (IP3, ROS) have D ~100–1000 μm²/s through junctions. Extracellular candidates (eATP itself, H₂O₂) have different diffusion characteristics. The choice affects the effective D_S and whether boundary conditions matter.

---

## 16. References

### Primary Sources

1. **Tung et al. (2024)**. "Embryos assist morphogenesis of others through calcium and ATP signaling mechanisms in collective teratogen resistance." *Nature Communications* 15:535. — The empirical basis: CEMA effect, eATP signaling, P2 receptor dependence, group-size scaling, and the critical finding that healthy embryos don't contribute.

2. **Pall ML (2013)**. "Electromagnetic fields act via activation of voltage-gated calcium channels to produce beneficial or adverse effects." *Journal of Cellular and Molecular Medicine* 17(8):958-965. — Reviews 23 studies showing VGCC blockers abolish EMF effects; proposes VGCCs as primary EMF transducers.

3. **Catterall WA (2011)**. "Voltage-Gated Calcium Channels." *Cold Spring Harbor Perspectives in Biology* 3(8):a003947. — Describes S4 voltage sensor mechanism: S4 segments move outward and rotate in response to electric field, opening the channel pore.

4. **Jouaville LS, Pinton P, Bastianutto C, Rutter GA, Rizzuto R (1999)**. "Regulation of mitochondrial ATP synthesis by calcium: Evidence for a long-term metabolic priming." *Proceedings of the National Academy of Sciences* 96(24):13807-13812. — Demonstrates Ca²⁺ modulates three Krebs cycle dehydrogenases; shows long-lasting metabolic priming effect.

5. **Glancy B, Balaban RS (2012)**. "Regulation of ATP production by mitochondrial Ca²⁺." *Cell Calcium* 52(1):28-35. — MCU and MICU1 as key components in Ca²⁺-dependent ATP regulation.

6. **Mulukutla BC, Yongky A, Grimm S, Daoutidis P, Hu WS (2014)**. "Bistability in Glycolysis Pathway as a Physiological Switch in Energy Metabolism." *PLOS ONE* 9(6):e98756. — Demonstrates bistable ATP states via phosphofructokinase and pyruvate kinase regulatory loops.

7. **Huang S, Guo YP, May G, Enver T (2007)**. "Bifurcation dynamics in lineage-commitment in bipotent progenitor cells." *Developmental Biology* 305(2):695-713. — Models GATA1-PU.1 bistable switch; shows commitment occurs through destabilization of progenitor state followed by attraction to lineage-specific attractors.

8. **Alagha A, Zaikin A (2013)**. "Asymmetry in erythroid-myeloid differentiation switch and the role of timing in a binary cell-fate decision." *Frontiers in Immunology* 4:426. — Shows asymmetric external signals create unequal basins of attraction through imperfect pitchfork bifurcations.

9. **Wang J, Zhang K, Xu L, Wang E (2011)**. "Quantifying the Waddington landscape and biological paths for development and differentiation." *PNAS* 108(20):8257-8262. — Quantifies cell fate landscapes showing how basin sizes determine fate probabilities.

10. **Ferrell JE Jr (2012)**. "Bistability, bifurcations, and Waddington's epigenetic landscape." *Current Biology* 22(11):R458-R466. — Reviews bistability in developmental systems; shows cell fate induction occurs via saddle-node bifurcation (valley disappearance).

11. **Xiong W, Ferrell JE Jr (2003)**. "A positive-feedback-based bistable 'memory module' that governs a cell fate decision." *Nature* 426:460-465. — Demonstrates saddle-node bifurcation in MAPK cascade controlled by extracellular signal concentration.

12. **Levin M (2019)**. "Bioelectrical controls of morphogenesis: from ancient mechanisms of cell coordination to biomedical opportunities." *Current Opinion in Genetics & Development* 57:51-59. — Electric fields provide positional information; overlapping fields give spatial context for morphogenesis.

13. **Levin M (2021)**. "Bioelectric signaling: Reprogrammable circuits underlying embryogenesis, regeneration, and cancer." *Cell* 184(6):1621-1636. — Cells are regulated by both their own Vmem and neighbors' Vmem; bioelectricity enables collective morphogenetic decision-making.

14. **Manicka S, Levin M (2025)**. "Field-mediated bioelectric basis of morphogenetic prepatterning." *Cell Reports Physical Science*. — Demonstrates field superposition from multiple sources creates morphogenetic prepatterns.

15. **Hardie DG (2011)**. "AMP-activated protein kinase: an energy sensor that regulates all aspects of cell function." *Genes & Development* 25(18):1895-1908. — AMPK as master metabolic switch; CaMKK2-dependent activation independent of AMP:ATP ratio.

### Supporting Sources

16. **Nerbonne JM, Guo W (2002)**. "Differential Distribution of Cardiac Ion Channel Expression as a Basis for Regional Specialization in Electrical Function." *Circulation Research* 90:1225-1232. — Documents spatially varying ion channel density in tissues.

17. **Modolo J, Thomas AW, Stodilka RZ (2018)**. "Physiological effects of low-magnitude electric fields on brain activity." *Current Opinion in Biomedical Engineering*. — Weak fields affect networks through accumulated parameter changes at synapses, not direct firing.

18. **Goldberg M, De Pittà M, Bhalla US, Ben-Jacob E (2010)**. "Nonlinear Gap Junctions Enable Long-Distance Propagation of Pulsating Calcium Waves in Astrocyte Networks." *PLOS Computational Biology* 6(8):e1000909. — Gap junction-mediated Ca²⁺ wave propagation provides spatial integration.

19. **Mycielska ME, Djamgoz MB (2004)**. "Cellular mechanisms of direct-current electric field effects: galvanotaxis and metastatic disease." *Journal of Cell Science* 117(Pt 9):1631-1639. — Voltage-gated Ca²⁺ channels on anodal side create asymmetric Ca²⁺ influx.

20. **Thrivikraman G, Boda SK, Basu B (2018)**. "Unraveling the mechanistic effects of electric field stimulation towards directing stem cell fate and function: A tissue engineering perspective." *Biomaterials* 150:60-86. — EF directs cell fate through Ca²⁺/MAPK/PI3K pathways.

### Decentralized Transducer Sources

21. **Walter P, Ron D (2011)**. "The Unfolded Protein Response: From Stress Pathway to Homeostatic Regulation." *Science* 334(6059):1081-1086. — ER stress sensing via IRE1/PERK/ATF6; Ca²⁺-dependent stress pathways; ROS-Ca²⁺ positive feedback bistability.

22. **Manicka S, Pai VP, Levin M (2023)**. "Information integration during bioelectric regulation of morphogenesis of the embryonic frog brain." *iScience* 26(12):108398. [PMC10687303] — Minimal dynamical model of collective gene expression driven by multicellular voltage patterns; causal integration analysis reveals higher-order mechanism by which voltage pattern information is spatiotemporally integrated into gene activity; demonstrates cell-local computation of tissue-level developmental decisions. Precedent for decentralized pattern classification via intracellular networks.

23. **Lisman JE, Zhabotinsky AM (2001)**. "A model of synaptic memory: a CaMKII/PP1 switch that potentiates transmission by organizing an AMPA receptor anchoring assembly." *Neuron* 31(2):191-201. — CaMKII as molecular temporal integrator; bistable autophosphorylation dynamics.

24. **Olfati-Saber R, Murray RM (2004)**. "Consensus Problems in Networks of Agents with Switching Topology and Time-Delays." *IEEE Transactions on Automatic Control* 49(9):1520-1533. — Mathematical framework for decentralized consensus in coupled agent networks; convergence conditions for lattice-connected systems.

25. **Elliott MR, Chekeni FB, Trampont PC, et al. (2009)**. "Nucleotides released by apoptotic cells act as a find-me signal to promote phagocytic clearance." *Nature* 461:282-286. — eATP as damage-associated molecular pattern (DAMP); pannexin-1 mediated release from stressed cells.

26. **Nelson DE, Ihekwaba AEC, Elliott M, et al. (2004)**. "Oscillations in NF-κB Signaling Control the Dynamics of Gene Expression." *Science* 306(5696):704-708. — NF-κB oscillatory dynamics encode stress duration via frequency and amplitude; cells count nuclear translocations as temporal integration mechanism.

---

## Appendix A: Why "Mismatch Detection" Was Problematic

The original design assumed cells could detect field mismatch:
```
mismatch = ||E_neighbor - E_self||
```

This requires:
1. Knowing what E_self is (the cell's own contribution to the field)
2. Knowing what E_neighbor is (the neighbor's contribution)
3. Computing the difference

But physically, the cell only experiences E_total = E_self + E_neighbor. It cannot decompose this into components without a reference. This is like asking "what is the temperature contribution from the sun vs the room heater?" — you only experience total temperature.

The stress-signal model avoids this entirely: the transducer reads E_self (the embryo's own field, which it generates and can sense directly) and outputs a stress signal based on whether its own pattern is normal or abnormal.

---

## Appendix B: The Asymmetry Question

**Q**: Won't a healthy embryo near many perturbed embryos get destabilized?

**A**: No, for two reasons:

1. **Asymmetric ATP reaction** — A healthy embryo at ATP ≈ 1 is deep in the healthy basin. The reaction strongly pulls it back toward ATP = 1. Only an embryo near the unstable equilibrium (ATP ≈ 0.3) can be tipped.

2. **State-dependent receiver gain** — Healthy embryos have low g(ATP), so even if they receive stress signals from perturbed neighbors, the effective input is attenuated.

```
ATP reaction shape:
                    ↑
                    |      healthy basin (large)
                    |     ╱
            ────────┼────●─────────────────────
                    |   ╱       ATP = 1
                    |  ╱
                    | ╱
            ────────●──────────────────────────
                   ╱        unstable (ATP ≈ 0.3)
                  ╱
            ─────●─────────────────────────────
                          ATP = 0
                    unhealthy basin (small)
```

This double protection (asymmetric basin + low receiver gain) ensures developmental robustness: healthy embryos resist perturbation while unhealthy embryos remain rescuable.

---

## Appendix C: Biological Pathway Summary

The complete biological pathway underlying the stress-signal rescue mechanism, with decentralized per-cell stress computation:

```
SENDER (perturbed embryo) — decentralized, per-cell computation:

Per cell (i,j), continuously during development (t=100 to 1000):

  Abnormal Vmem(i,j,t) pattern develops
      |
      v
  Voltage-gated Ca2+ channel activation (cell-local)
      | [Catterall 2011, Pall 2013]
      v
  Elevated intracellular Ca2+(i,j,t)
      |
      | --- acts as BIFURCATION PARAMETER on per-cell stress system ---
      v
  Stress RD bistable system at cell (i,j):
      |
      |  dS/dt = reaction(S, Ca2+) - gamma*S/(K_decay + S) + D_S * laplacian(S)
      |
      |  Components:
      |  - Ca2+ drive: sigmoid((Ca - Ca_threshold) / sigma)  [local sensing]
      |  - Self-activation: (S^2 - K^2)/(S^2 + K^2)         [bistable memory]
      |  - Michaelis-Menten decay: gamma*S/(K_decay + S)     [phosphatase kinetics]
      |  - Diffusion (D_S * laplacian): neighbor coupling     [synchronization]
      |
      |  Healthy Ca2+ pattern: S stays below nucleation threshold
      |  Perturbed Ca2+ pattern: S nucleates and propagates tissue-wide
      v
  S(i,j) -> Pannexin-1 open probability -> eATP(i,j)
      | [Tung et al. 2024, Elliott et al. 2009]
      v
  Embryo stress = mean_ij(S) = total eATP released
      |
      |   (diffuses to neighboring embryos)
      v

RECEIVER (neighboring perturbed embryo):

  eATP binds P2X/P2Y receptors
      | [Tung et al. 2024]
      | - P2X: direct Ca2+ influx (fast)
      | - P2Y: IP3 -> ER Ca2+ release (slower, amplified)
      v
  Ca2+ elevation (amplified by stressed state)
      |
      v
  CaMKK2 activation
      | [Hardie 2011]
      v
  AMPK activation (independent of AMP:ATP ratio)
      |
      v
  Metabolic reprogramming:
      | - Mitochondrial biogenesis up
      | - Fatty acid oxidation up
      | - Krebs cycle activity up [Jouaville et al. 1999]
      v
  ATP synthesis rate up (shifts bifurcation parameter)
      | [Mulukutla et al. 2014]
      v
  If collective signal sufficient:
      Saddle-node bifurcation -> unhealthy basin disappears
      | [Ferrell 2012]
      v
  ATP -> healthy attractor
      |
      v
  GRN weights restored -> normal patterning
      |
      v
  Vmem normalizes -> per-cell S decays -> eATP drops to 0
      |
      v
  Persistent rescue via bistable hysteresis
```

**Key distinction from earlier centralized design**: The SENDER pathway is entirely decentralized. No cell reads the global pattern. Each cell's stress variable S is governed by identical parameters, sensing only local Ca2+ and communicating with immediate neighbors via diffusion. The embryo-level stress signal emerges from the collective dynamics of 121 coupled bistable switches, not from any centralized readout. Pattern sensitivity arises from the physics of reaction-diffusion with decay (spatial frequency filtering), not from learned pattern-matching weights.

Each step in this pathway is supported by experimental literature, making the stress-signal rescue model biologically grounded rather than a purely computational construct.

---

## Appendix D: Field-Based Alternative to eATP Signaling

During the design process, we explored whether the inter-embryo signaling could use **direct electric field propagation** instead of the chemical messenger eATP. This appendix documents that exploration, the challenges encountered, and why the Ca²⁺/eATP pathway remains the primary design—while acknowledging that a field-based mechanism may be a viable alternative.

### D.1 Motivation: Why Consider Field-Based Signaling?

The eATP pathway requires:
- eATP synthesis and release (metabolic cost)
- Extracellular diffusion (slow, ~300 μm²/s)
- P2R receptor binding (requires receptor expression)
- Multiple enzymatic steps (pannexin, ecto-ATPases)

In contrast, electric fields:
- Propagate automatically (no synthesis cost)
- Travel at ~c (light speed in medium, vastly faster than diffusion)
- Affect all nearby cells without specific receptors
- Are a direct physical consequence of voltage patterns

This suggests a potential evolutionary advantage: **Why synthesize and release eATP when the field itself could carry the stress information?**

### D.2 Initial Hypothesis: Spatial Harmonic Detection

The first field-based design attempted to exploit spatial frequency content:

**Core idea**: Different spatial patterns have different harmonic content. A field-based system with **natural resonant modes** could distinguish healthy from perturbed patterns via energy dissipation when driven off-resonance.

#### The Mechanism (Initial Version)

```
┌─────────────────────────────────────────────────────┐
│  Each embryo emits field E(x,y) based on Vmem       │
│                                                     │
│  Fourier decomposition:                             │
│  E(x,y) = Σₖ Eₖ * ψₖ(x,y)                          │
│                                                     │
│  Healthy pattern:  dominated by k₁, k₂ (organized) │
│  Perturbed pattern: different k-spectrum (aberrant) │
└─────────────────────────────────────────────────────┘
                    ↓
        (field propagates to neighbors)
                    ↓
┌─────────────────────────────────────────────────────┐
│  Receiver embryo has spatial filter                 │
│  (gap junction network with diffusion length λ)     │
│                                                     │
│  Field enters at boundary cells                     │
│  → Diffuses through gap junction network            │
│  → High-k modes attenuated                          │
│  → Energy dissipation = stress signal               │
└─────────────────────────────────────────────────────┘
```

**Spatial filtering via diffusion**:
```
Boundary cells sense E_external (all frequencies)
Gap junction network acts as low-pass filter
Cutoff frequency k_c ≈ 1/λ where λ = √(D*τ)

Modes k > k_c: strongly attenuated → current flow → Joule heating
Modes k < k_c: propagate well → low dissipation

Stress M = Σ_{k>k_c} |E_k|² (power in high-frequency modes)
```

**Predicted behavior**:
- Healthy (organized): low k content → low M → low stress
- Perturbed (disorganized): high k content → high M → high stress
- **No template needed**: The filter's natural cutoff k_c determines what's "compatible"

### D.3 Critical Challenge: Perturbed Patterns Are Actually Uniform

Empirical observation from Model 253 (see `stress_bistable_test_5.png`):

**Reality:**
```
Healthy face:                    Perturbed embryo:
┌─────────────┐                  ┌─────────────┐
│  ·  ██  ·   │  Vmem            │ ████████████ │  Vmem
│  · ████ ·   │  pattern with    │ ████████████ │  uniform
│  ·  ██  ·   │  facial features │ ████████████ │  depolarization
│  · ████ ·   │  (eyes, nose)    │ ████████████ │  (homogeneous)
│  ·  ··  ·   │                  │ ████████████ │
└─────────────┘                  └─────────────┘

Spatial spectrum:                Spatial spectrum:
Power                            Power
  |  ●                              |     ●
  |   ●                             |
  | ●  ●  ●                         |
  |  ●  ●                           |
  └────────── k                     └────────── k
  Intermediate k                    k ≈ 0
  (features at specific scales)    (DC, uniform)
```

**The problem**: The initial hypothesis predicted backwards!
- Perturbed = uniform = **low k**, not high k
- Healthy = patterned = **intermediate k**, not low k

If the filter suppresses high k, it would pass uniform patterns easily → **low stress for perturbed** (WRONG).

### D.4 Corrected Understanding: RD Filter Detects Lack of Structure

The key insight from Section 7.4: The RD stress system doesn't simply filter high frequencies—it detects patterns **incompatible with its intrinsic dynamics**.

**How the Ca²⁺ RD system actually works with uniform patterns**:

```
Uniform high Ca²⁺ everywhere:
┌─────────────┐
│ ████████████ │ Ca²⁺
│ ████████████ │ drive
│ ████████████ │
└─────────────┘
       ↓ (reaction: Ca drives S up everywhere)
┌─────────────┐
│ ████████████ │ S
│ ████████████ │ (no spatial variation
│ ████████████ │  for diffusion to smooth)
└─────────────┘
       ↓
mean(S) = high → STRESS

Patterned Ca²⁺ (healthy):
┌─────────────┐
│  ·  ██  ·   │ Ca²⁺
│  · ████ ·   │ drive
│  ·  ██  ·   │ (features)
└─────────────┘
       ↓ (reaction + diffusion: smooths out)
┌─────────────┐
│  ·  ··  ·   │ S
│  · ··· ·    │ (diffusion suppresses
│  ·  ··  ·   │  isolated signals)
└─────────────┘
       ↓
mean(S) = low → no stress
```

**The mechanism is not about blocking high-k in the input—it's about the interplay of:**
1. Local activation (Ca²⁺ drives S up)
2. Diffusive smoothing (∇²S averages neighboring S values)
3. Saturable decay (Michaelis-Menten clears weak signals efficiently)

**Uniform drive** → S high everywhere → no diffusive smoothing can help → high mean(S)
**Patterned drive** → S varies spatially → diffusion redistributes → suppresses isolated activation → low mean(S)

### D.5 Field-Based RD Analog: Correct Formulation

The field-based version should work identically:

```python
class FieldStressRD:
    """
    Field-based analog of Ca²⁺ stress system.

    Instead of Ca²⁺ driving S, external electric field E_collective
    drives a field stress variable F with RD dynamics.
    """

    def __init__(self, adjacency_matrix):
        self.A = adjacency_matrix
        self.F = torch.zeros(num_cells)  # field stress per cell

    def step(self, dt, E_collective):
        """
        E_collective: collective external field from neighbors
        """
        # Field drive (analog of Ca²⁺ drive)
        # Could use field energy: E²
        field_drive = torch.sigmoid(
            (E_collective**2 - E_threshold) / sigma_E
        )

        # Competitive self-activation (bistable)
        self_act = (self.F**2 - K_F**2) / (self.F**2 + K_F**2)

        # OR gate (combines field drive + self-activation)
        or_gate = torch.sigmoid(gain_F * field_drive + self_act - θ_or)

        # Bistable reaction
        reaction = (or_gate * k_on * (1 - self.F) - k_off * self.F) / τ_F

        # Michaelis-Menten decay
        decay = -γ * self.F / (K_decay + self.F)

        # Diffusion (spatial coupling)
        laplacian = torch.matmul(self.A, self.F) - self.A.sum(1) * self.F
        diffusion = D_F * laplacian

        # Update
        self.F = self.F + dt * (reaction + decay + diffusion)

    def get_embryo_stress(self):
        return self.F.mean()
```

**Signal flow (field-based)**:

```
┌─────────────────────────────────────────────────────┐
│  PERTURBED EMBRYO                                   │
│                                                     │
│  Uniform high Vmem → Strong uniform E field         │
│                      (high field energy everywhere) │
│                                                     │
│  Emission strength ∝ ∫|E|² dV (total field energy)  │
└─────────────────────────────────────────────────────┘
                    ↓
        (field superposition at neighbors)
                    ↓
┌─────────────────────────────────────────────────────┐
│  RECEIVER EMBRYO                                    │
│                                                     │
│  E_collective = Σ_neighbors E_emission              │
│                                                     │
│  Uniform high E_collective:                         │
│  → field_drive high everywhere                      │
│  → F rises uniformly                                │
│  → no diffusive smoothing possible                  │
│  → mean(F) high → STRESS                            │
│                                                     │
│  Patterned E_collective (from healthy):             │
│  → field_drive varies spatially                     │
│  → diffusion in F dynamics smooths                  │
│  → mean(F) low → no stress                          │
└─────────────────────────────────────────────────────┘
```

**Key point**: Just like the Ca²⁺ system, this detects **uniform high drive** (not specific spatial frequencies).

### D.6 Biological Plausibility Comparison

Let's honestly assess biological support for field-based vs Ca²⁺-based mechanisms:

#### Ca²⁺/eATP Pathway (Current Design)

| Component | Mechanism | Support |
|-----------|-----------|---------|
| Stress detection | Vmem → VGCC → Ca²⁺ ↑ | ✓✓✓ [Catterall 2011] |
| Stress variable S | Ca²⁺-dependent bistable (ROS-Ca²⁺ loop) | ✓✓✓ [Walter & Ron 2011] |
| RD dynamics | IP3 or ROS diffusion via gap junctions | ✓✓✓ [Goldberg 2010] |
| Inter-embryo signal | eATP release (pannexin) + diffusion | ✓✓✓ [Tung 2024, Elliott 2009] |
| Signal reception | P2R → Ca²⁺ → AMPK | ✓✓✓ [Tung 2024, Hardie 2011] |
| ATP bifurcation | AMPK → metabolic reprogramming | ✓✓✓ [Jouaville 1999] |

**Literature**: Every step has strong experimental support. This is the **most conservative, biologically grounded** design.

#### Field-Based Pathway (Alternative)

| Component | Mechanism | Support |
|-----------|-----------|---------|
| Field emission | Vmem pattern → E field (Poisson) | ✓✓✓ (physics) |
| Field sensing | Ephaptic coupling → local Vmem change | ✓✓ (neurons) to ✓ (development) |
| Field → F variable | E²-dependent activation? | ? (speculative) |
| F has RD dynamics | Diffusion via gap junctions | ✓✓ (if F is IP3/ROS) |
| Field as inter-embryo signal | Superposition (automatic) | ✓✓✓ (physics) |
| F → ATP rescue | Metabolic stress pathway | ✓ (indirect) |

**Key challenges**:

1. **What is F biologically?**
   - If F = Ca²⁺ (field → VGCCs → Ca²⁺), we've circled back to the Ca²⁺ system
   - If F = ROS (field → mitochondrial stress → ROS), plausible but less direct
   - If F = something else, unclear what

2. **Field sensing in development**
   - Ephaptic coupling well-established in **neurons** [Anastassiou 2011]
   - Less clear in **embryonic tissues** (though Vmem coupling via gap junctions is established)
   - EM field effects on VGCCs exist but are controversial [Pall 2013]

3. **Why would embryos use fields instead of eATP?**
   - Faster propagation (EM vs diffusion)
   - No metabolic cost (field is free byproduct)
   - But: eATP has specificity (P2R), fields affect all cells indiscriminately

#### Most Plausible Field-Based Scenario: ROS as F

If we must propose a field-based mechanism, the strongest candidate is:

```
External E field → Ephaptic current in boundary cells →
Membrane stress → Mitochondrial dysfunction →
ROS production ↑ [Goodman et al. 2009]
    ↓
ROS = F variable (diffuses via membranes + gap junctions)
    ↓
ROS bistability: Ca²⁺ ↔ ROS positive feedback [Walter & Ron 2011]
    ↓
ROS → AMPK activation (oxidative stress) → ATP rescue
```

**Support**:
- EM fields → mitochondrial ROS: ✓✓ (moderate evidence) [Luukkonen et al. 2014]
- ROS diffusion: ✓ (limited range, ~50-100 nm without carrier)
- ROS → AMPK: ✓✓ (established) [Zmijewski et al. 2010]

**Limitations**:
- ROS is highly reactive → limited diffusion range
- EM → ROS pathway less direct than Vmem → VGCCs → Ca²⁺
- Still less established than purinergic signaling

### D.7 Why We Chose Ca²⁺/eATP (But Field-Based Remains Viable)

**Reasons to prefer Ca²⁺/eATP system**:

1. **Biological precedent**: Every step is well-documented
   - eATP as stress signal: canonical [Elliott 2009, Tung 2024]
   - P2R → rescue pathway: directly observed in CEMA paper
   - No speculative steps

2. **Simplicity**: One signal (eATP), one receptor (P2R)
   - Field-based requires two-stage field sensing (external E → F → emission)
   - More complex pathway, more assumptions

3. **Specificity**: P2R provides targeted signal reception
   - Fields affect all cells indiscriminately
   - Chemical signaling allows receptor-based specificity

4. **Direct experimental support**: Tung et al. measured eATP levels
   - Treated embryos: 1.073 μM
   - Control: 0.989 μM
   - P2R blockers abolish rescue
   - **This is smoking-gun evidence for eATP**

**Reasons field-based remains interesting**:

1. **Physical elegance**: Field superposition is automatic
   - No need to synthesize/release messenger
   - Faster propagation (EM vs diffusion)

2. **RD filtering works identically**: The spatial frequency filtering logic (Section 7.4) applies equally to field-driven F as to Ca²⁺-driven S
   - Same mechanism: diffusion + decay + bistability
   - Same invariances (rotation, translation)
   - Same pattern sensitivity

3. **Evolutionary plausibility**: If rescue is beneficial, why not exploit the field (which exists anyway) rather than paying the metabolic cost of eATP synthesis?

4. **Hybrid possibility**: Both pathways could coexist
   - Primary: eATP (well-established)
   - Secondary: field effects (modulatory)
   - Redundancy increases robustness

### D.8 Conclusion: Complementary Rather Than Competing

The field-based mechanism is **not a replacement** for the Ca²⁺/eATP system—it's a **conceptual alternative** that demonstrates:

1. **RD filtering is mechanism-agnostic**: The spatial frequency filtering principle works whether the input is Ca²⁺, electric field, or any other spatially distributed signal

2. **Multiple pathways possible**: Biology often uses redundant signaling (Wnt, Hedgehog, FGF all pattern the same tissues). Field + eATP could both contribute.

3. **Template-free pattern recognition generalizes**: The key insight—that RD dynamics with decay create intrinsic pattern selectivity—applies beyond the specific molecular implementation.

**Implementation priority**:
1. **Primary**: Ca²⁺/eATP system (most biologically grounded)
2. **Alternative**: Field-based ROS system (if exploring field mechanisms)
3. **Comparison study**: Quantitative predictions from both, test against CEMA data

The field-based design remains in this document as a **design alternative** that validates the core RD filtering principle while acknowledging that the eATP pathway has stronger direct experimental support.

---

### D.9 Summary Table: Ca²⁺ vs Field Pathways

| Feature | Ca²⁺/eATP System | Field-Based ROS System |
|---------|------------------|----------------------|
| **Internal stress signal** | Ca²⁺ (via VGCCs) ✓✓✓ | Ca²⁺ or ROS (via ephaptic) ✓ |
| **RD variable** | S (Ca²⁺-driven bistable) | F (field-driven bistable) |
| **Spatial filtering** | D∇²S (same physics) | D∇²F (same physics) |
| **Pattern detection** | RD nucleation dynamics | RD nucleation dynamics |
| **Inter-embryo signal** | eATP (chemical) ✓✓✓ | E field (physical) ✓✓ |
| **Signal propagation** | Diffusion (~300 μm²/s) | EM (~c in medium) |
| **Signal reception** | P2R (specific) ✓✓✓ | Ephaptic (nonspecific) ✓ |
| **Rescue pathway** | P2R → AMPK → ATP ✓✓✓ | ROS → AMPK → ATP ✓✓ |
| **Experimental evidence** | Direct (Tung 2024) ✓✓✓ | Indirect/speculative ? |
| **Biological plausibility** | ✓✓✓ High | ✓ Moderate |
| **Physical elegance** | ✓ (requires synthesis) | ✓✓✓ (automatic) |
| **Implementation priority** | **Primary** | Alternative |

**Bottom line**: The Ca²⁺/eATP system is the **primary design** due to stronger biological support. The field-based system demonstrates that the RD filtering principle is robust across different physical implementations and remains a viable **alternative hypothesis** for future exploration.

---

## 17. Pair-Rescue Implementation (`runStressRescue.py`)

### 17.1 Overview

The pair-rescue mechanism tests whether a donor embryo's stress temporal profile can rescue a stressed recipient embryo by dynamically modulating its GRN damping. This is a simplified two-embryo proxy for the full multi-embryo CEMA system: instead of collective eATP in a shared medium, we directly inject the donor's stress time-course into the recipient's damping dynamics.

### 17.2 Core Rescue Formula

The effective GRN damping at each timestep is computed in logit space:

```
effective_damping(t) = sigmoid( logit(d₀) + α · S_donor(t) )
```

where:
- `d₀` = recipient's base (static) GRN damping level
- `S_donor(t)` = donor embryo's stress signal at time t (mean(S) from the RD bistable stress system)
- `α` = rescue rate parameter (learnable in future; fixed CLI argument for now)

**Why logit space?** The sigmoid-logit parameterization ensures:
1. When `S_donor = 0`: `effective_damping = sigmoid(logit(d₀)) = d₀` (no effect)
2. As `S_donor` increases: damping rises toward 1.0 (full restoration)
3. Damping is bounded in (0, 1) regardless of α or S values
4. The effect is symmetric in logit space — equal stress increments produce equal logit shifts

### 17.3 Two-Phase Simulation Architecture

Following `runStressBistableSwitch.py`, the sweep uses a two-phase approach:

**Phase 1 — Pre-compute donor stress profiles (5 simulations):**
For each damping level d ∈ {1.0, 0.95, 0.9, 0.5, 0.1}:
1. Run Model 253 bioelectric simulation with static GRN damping = d for `numBioSteps` iterations
2. Collect Vmem timeseries
3. Feed Vmem through StressBistableSwitch (concurrent Ca²⁺ + S dynamics) to get `stress_history[t]`

**Phase 2 — Pairwise rescue sweep (25 pairs):**
For each (donor_d, recipient_d) pair:
- **Diagonal (donor == recipient):** Use Phase 1 baseline results directly — no rescue simulation needed
- **Off-diagonal (donor != recipient):**
  1. Load Model 253 with **undamped** weights (grn_damping=1.0)
  2. Store original `tissueGRNWeights` and `GRNtoLigandWeights`
  3. At each timestep t: compute `effective_damping(t)`, scale both weight matrices by it
  4. Collect rescued Vmem timeseries → run through stress system → get rescued stress

### 17.4 Dynamic Weight Modulation

The GRN damping acts on two weight matrices at each timestep:

```python
# Store originals after model construction (undamped)
original_tissueGRNWeights = bio_model.geneNetwork.tissueGRNWeights.clone()
original_GRNtoLigandWeights = bio_model.electricNetwork.GRNtoLigandWeights.clone()

# Each timestep:
eff_damp = compute_effective_damping(base_damping, donor_stress_profile[t], alpha)
bio_model.geneNetwork.tissueGRNWeights = original_tissueGRNWeights * eff_damp
bio_model.electricNetwork.GRNtoLigandWeights = original_GRNtoLigandWeights * eff_damp
```

This mirrors the static damping mechanism in `load_model_parameters()` but applies it dynamically at each simulation step.

### 17.5 Rescue Quantification Metrics

Two complementary metrics quantify rescue effectiveness:

1. **Vmem pattern similarity** (Pearson correlation):
   - Compare rescued Vmem with healthy reference (damping=1.0) at the final timestep
   - Handles degenerate uniform patterns (std < 1e-10) gracefully
   - Range: [-1, 1], where 1 = identical to healthy pattern

2. **Final stress level** (mean(S)):
   - Lower stress = better rescue
   - Range: [0, 1]

### 17.6 Visualization

Six-panel figure (`data/stress_rescue_test.png`):

| Panel | Content |
|-------|---------|
| (1,1) | 5×5 heatmap: Vmem similarity (Pearson r) for each donor-recipient pair |
| (1,2) | 5×5 heatmap: rescued final stress for each pair |
| (1,3) | Bar chart: baseline vs best-donor Vmem similarity per recipient |
| (2,1) | Stress timeseries for most-stressed recipient with various donors |
| (2,2) | Effective damping curves over time for selected pairs |
| (2,3) | Bar chart: baseline vs best-donor stress reduction per recipient |

### 17.7 Usage

```bash
# Quick test with 2 levels
python runStressRescue.py --dampingLevels "1.0,0.9" --alpha 3.0 --numBioSteps 500

# Full sweep with learned stress parameters
python runStressRescue.py --stressParamsFile data/bestLearnedStressParams_5.dat

# Custom alpha
python runStressRescue.py --alpha 10.0 --stressParamsFile data/bestLearnedStressParams_5.dat \
    --outputFile data/stress_rescue_test_StressFile5_alpha10.0.png
```

---

## 18. Pair-Rescue Experimental Results

### 18.1 Temporal Mismatch Problem

The initial pair-rescue experiments revealed a fundamental **temporal mismatch**: the rescue signal arrives too late to prevent pattern collapse.

**The problem:**
1. Donor stress `S(t)` starts near 0 and only rises significantly after ~500-1000 bio steps (the stress system must wait for Ca²⁺ to build up, then S must cross the bistable threshold)
2. The recipient's Vmem pattern is determined in the first ~100-500 bio steps — GRN-damped embryos quickly converge to a degenerate (uniform) voltage pattern
3. By the time the donor's stress signal becomes strong enough to meaningfully shift the recipient's damping, the recipient's pattern has already collapsed

```
Donor stress S(t):         Recipient Vmem pattern:

  1.0 ─┤                    Healthy ─┤  ████
       │          ╱────     pattern   │  ██ ██
       │        ╱                     │  ████
  0.5 ─┤      ╱                      │
       │    ╱            Collapsed ─┤  ████████
       │  ╱              (uniform)   │  ████████
  0.0 ─┤╱                            │  ████████
       └────┬────┬────▶              └──┬──┬──▶
           500  1000                   100 500

  Donor stress rises slowly...    ...but recipient pattern is
                                  already determined here
```

**Consequence:** Rescue effectiveness depends almost entirely on the recipient's damping level (which determines how quickly its pattern collapses), not the donor's identity. All donors produce nearly identical rescue outcomes for a given recipient because their stress profiles only differ in the late phase when the recipient's fate is already sealed.

### 18.2 Observed Results Summary

With `alpha=3.0` and learned stress parameters (`bestLearnedStressParams_5.dat`):
- **Vmem similarity matrix**: Rows (recipients) show near-uniform values across columns (donors)
- **Stress matrix**: Similarly, rescued stress depends primarily on recipient damping
- **Rescue magnitude**: Essentially zero donor-dependent rescue effect

### 18.3 Implications and Future Directions

The temporal mismatch suggests several modifications:

1. **Pre-computed stress signal**: Instead of real-time temporal S(t), use the donor's final/steady-state stress level from t=0. This models the scenario where the donor has *already been stressed* before the recipient begins development (biologically: the donor was exposed to teratogen first, reached steady-state eATP release, and the recipient is placed in this eATP-enriched medium).

2. **Faster stress signal**: Use a faster proxy than the RD bistable S (e.g., Ca²⁺ divergence from healthy baseline, which manifests within ~100 steps).

3. **Longer simulation window**: Extend the bio simulation to allow pattern recovery after the rescue signal finally arrives (e.g., numBioSteps=5000).

4. **Higher alpha**: Larger α makes the rescue formula more responsive to small early stress signals, potentially catching the recipient before collapse.

---

## 19. Biophysical Estimate of Inter-Embryonic Signal Propagation Speed

### 19.1 CEMA Paper Physical Parameters

From Tung et al. (2024), Xenopus laevis embryo measurements:

| Parameter | Value | Source |
|-----------|-------|--------|
| Embryo diameter | ~1.4 mm | Stage 10-12 Xenopus |
| Inter-embryo distance (d_inter) | ~1.4 mm (adjacent, touching) | Experimental setup |
| Ca²⁺ wave speed in tissue | 5.28 μm/s | Measured inter-embryo signaling |
| eATP concentration (treated) | 1.073 μM | Measured |
| eATP concentration (control) | 0.989 μM | Measured |

### 19.2 Derivation of Effective Propagation Speed from α

The rescue formula operates in logit space:

```
effective_damping(t) = sigmoid( logit(d₀) + α · S(t) )
```

Define the **logit shift required** for a meaningful damping increase (e.g., from d₀ to d₀ + Δd):

```
Λ = logit(d₀ + Δd) - logit(d₀)
```

The **time to achieve this shift** depends on when α · S(t) ≥ Λ. Approximating the stress rise as linear with rise time τ_rise and plateau S_∞:

```
S(t) ≈ S_∞ · (t / τ_rise)    for t < τ_rise
```

The rescue time (when the shift reaches Λ) is:

```
t_rescue = (Λ · τ_rise) / (α · S_∞)
```

The **effective propagation speed** — the rate at which rescue influence propagates from donor to recipient across distance d_inter — is:

```
v_eff = d_inter / t_rescue_physical = (d_inter · α · S_∞) / (Λ · τ_rise_physical)
```

where τ_rise_physical = τ_rise_steps × dt × κ, with κ being the model-to-physical time calibration factor (seconds per model time unit).

### 19.3 CEMA-Calibrated α Estimate

Setting v_eff = v_CEMA = 5.28 μm/s and solving for α:

```
5.28 = (d_inter · α · S_∞) / (Λ · τ_rise_steps · dt · κ)
```

With reference values:
- d_inter = 1400 μm
- S_∞ = 0.75 (typical stressed embryo final stress)
- Λ = logit(0.15) - logit(0.1) ≈ 0.46 (example: shifting damping from 0.1 to 0.15)
- τ_rise_steps = 750 (model steps for stress to reach ~S_∞)
- dt = 0.01 (bioelectric timestep)

Solving:

```
α = (v_CEMA · Λ · τ_rise_steps · dt · κ) / (d_inter · S_∞)
  = (5.28 × 0.46 × 750 × 0.01 × κ) / (1400 × 0.75)
  = 0.01735 × κ
```

**The constrained quantity is α/κ = 0.01735.** The physical α depends on the time calibration:

| κ (s/model unit) | α | Physical interpretation |
|-------------------|------|------------------------|
| 1 | 0.017 | Very slow model time |
| 5 | 0.087 | Moderate calibration |
| 10 | 0.174 | Based on Ca²⁺/CaMKII timescales |
| 50 | 0.87 | Fast model time |

### 19.4 Time Calibration Factor κ

The factor κ converts model time units to physical seconds. It is estimated by matching model timescales to known biological timescales:

1. **Ca²⁺ dynamics**: Model τ_ca = 2.6 model units. Biological Ca²⁺ transients: 10-50s. → κ ≈ 4-19 s/unit
2. **CaMKII dynamics**: Model τ_camkii = 61 model units. Biological CaMKII activation: 30s-10min. → κ ≈ 0.5-10 s/unit
3. **Average estimate**: κ ≈ 10 s/unit (geometric mean of the above ranges)

**Important caveat**: The CEMA data actually constrains the product α × κ (or equivalently α/κ = 0.01735), not each independently. The decomposition into separate α and κ requires an independent calibration of model timescale against physical time. κ is the weakest link in this estimate.

### 19.5 Key Relationships

**v_eff is linear in α:**

```
v_eff = (d_inter · α · S_∞) / (Λ · τ_rise_physical) ∝ α
```

This means:
- Doubling α doubles the effective rescue propagation speed
- The CEMA-calibrated α* ≈ 0.17 (with κ ≈ 10) predicts v_eff ≈ 5.28 μm/s
- The experimentally used α = 3.0 predicts v_eff ≈ 91 μm/s (17× faster than CEMA), suggesting the model operates in a regime where rescue should be very fast if the temporal mismatch problem (Section 18.1) didn't intervene

### 19.6 Implications for Model Parameters

The biophysical estimate reveals a tension:
- **CEMA-calibrated α ≈ 0.17**: Biophysically realistic but produces very small damping shifts (logit shift of ~0.13 at peak stress), likely insufficient for rescue in the current model
- **Simulation-effective α ≈ 3-10**: Large enough to produce meaningful damping shifts but corresponds to v_eff ≈ 91-300 μm/s, much faster than measured CEMA signaling

This tension may reflect:
1. **The temporal mismatch problem** (Section 18.1): the model requires unnaturally large α to compensate for the late-arriving stress signal
2. **Missing amplification mechanisms**: In biology, the receiver gain g(ATP) (Section 6) amplifies weak signals in compromised embryos — this is not yet implemented in `runStressRescue.py`
3. **Different rescue timescales**: The CEMA rescue operates over hours (developmental timescale), not seconds (signaling timescale). The pair-rescue model collapses these timescales

---

## 20. Group-Level Rescue Implementation (`runGroupRescue.py`)

### 20.1 Overview

The group rescue extends the pair-rescue mechanism (Section 17) from a single donor–recipient pair to a 2D lattice of 1–300 embryos. All embryos simultaneously run their bioelectric models, compute stress, exchange stress signals with lattice neighbors, and dynamically modulate their GRN damping. This enables investigation of group-size effects on collective rescue — a key prediction of the CEMA model where larger groups of perturbed embryos rescue each other more effectively.

### 20.2 Architecture

Each embryo in the lattice has:
1. A **bioelectric model** (Model 253 instance) with its own Vmem, GRN, ion channels, and ligands
2. A **StressBistableSwitch** (Section 7) computing Vmem → Ca²⁺ → bistable stress S ∈ [0,1]
3. A **base GRN damping** level d₀ (lower = more teratogen-exposed)
4. An **effective damping** dynamically modulated by the inter-embryo rescue signal

Embryos are arranged on a 2D grid with von Neumann (4-connected) or Moore (8-connected) neighborhood connectivity.

### 20.3 Synchronized Simulation Loop

At each bioelectric timestep t:

```
Step 1: All embryos advance bioelectric sim by 1 step
        (ThreadPoolExecutor if num_embryos >= parallel_threshold)

Step 2: All embryos update stress
        - Ca²⁺ from Vmem via voltage-gated channels (dt_ca = 0.01)
        - Bistable stress step (dt_stress = 0.1)

Step 3: Compute inter-embryo rescue signal
        - Diffusive field (default): solve dF/dt on embryo lattice
        - Or: mean_neighbor_stress fallback (D_F = 0)

Step 4: Update each embryo's effective GRN damping
        - effective_damping(t) = sigmoid(logit(d₀) + α · F_i(t))
        - Scale tissueGRNWeights and GRNtoLigandWeights by effective_damping
```

After the bioelectric phase, a stress equilibration phase (default 500 steps) continues the stress dynamics and field diffusion with frozen Ca²⁺ and frozen bio models.

### 20.4 The Diffusive Stress Field

#### 20.4.1 Motivation: Why Mean Neighbor Stress Has No Group-Size Effect

On a von Neumann lattice, every interior embryo has exactly 4 neighbors. Computing `mean_neighbor_stress = (1/4) Σ S_j` produces the same value regardless of whether the lattice is 2×2 or 100×100 — when all embryos have similar damage, their stress levels are similar, so the mean is independent of group size.

#### 20.4.2 Solution: Reaction-Diffusion Field with Absorbing Boundary Conditions

Each embryo emits its stress into a shared scalar field F defined on the lattice. The field diffuses and decays:

```
dF/dt = D_F · ∇²F - γ_F · F + emission_i
```

where:
- `D_F` = diffusion rate (default 1.0)
- `γ_F` = decay rate (default 0.1)
- `emission_i = mean(S_i)` = embryo i's stress (same as `get_embryo_stress()`)
- `∇²F_i` = discrete Laplacian on the embryo lattice (see below)

The field is solved via explicit Euler with `n_substeps` sub-steps per bio step.

**Absorbing boundary conditions** are critical for group-size dependence. The Laplacian uses the maximum degree (`max_degree`) rather than each cell's actual neighbor count:

```
∇²F_i = Σ_j A(i,j) · F_j  -  max_degree · F_i
```

This treats missing neighbors (beyond the lattice edge) as having F = 0, so boundary cells leak signal into the exterior.

**Clarification on "F = 0 at the boundary":** The F = 0 condition applies to the **virtual exterior** just outside the lattice, not to the boundary embryos themselves. Edge and corner embryos are fully functional — they emit stress, receive field, and can be rescued. They simply lose signal faster because some of their diffusive flux goes to virtual exterior cells held at F = 0, representing the surrounding medium where eATP is diluted to negligible concentration:

```
Interior embryo:  ∇²F = (F_left + F_right + F_up + F_down) - 4·F    [4 real neighbors]
Corner embryo:    ∇²F = (F_right + F_up + 0 + 0)           - 4·F    [2 real + 2 virtual at F=0]
                                              ↑   ↑
                                    virtual exterior (absorbing BC)
```

Without absorbing BCs, reflecting BCs (using actual degree) would give every interior cell the same steady-state F = e/γ_F regardless of group size.

**Why reflecting BCs fail:** With reflecting (Neumann) BCs, the Laplacian uses each cell's actual degree: `∇²F_i = Σ_j (F_j - F_i)`. At steady state (∇²F = 0 in the interior), F becomes spatially uniform at F = e/γ_F. This value is identical for a 1×2 lattice and a 100×100 lattice — no group-size effect.

**Why absorbing BCs work:** Boundary cells lose field to the exterior at rate `k · D_F`, where k = max_degree − actual_degree is the number of missing neighbors. This creates a position-dependent effective decay rate:

```
γ_eff(i) = γ_F + k_i · D_F
```

Interior cells (k = 0) have γ_eff = γ_F, while boundary/corner cells have much higher γ_eff. Small groups are boundary-dominated (high average γ_eff), large groups have proportionally more interior cells (low γ_eff). This geometric asymmetry creates the CEMA effect.

#### 20.4.3 How This Creates Group-Size Dependence

The characteristic communication length is:

```
λ = sqrt(D_F / γ_F)
```

This controls how far a single embryo's stress signal reaches (in lattice spacings) before decaying.

**Key physical insight:** Small groups are boundary-dominated; large groups are bulk-decay-limited.

For a small group (e.g., 1×2), each embryo has k ≈ 3 missing neighbors (von Neumann), giving:
```
γ_eff(1×2) = γ_F + 3·D_F
F_ss(1×2) ≈ emission / γ_eff ≈ e / (γ_F + 3·D_F)
```

For a large group interior (k = 0), the steady state follows from the 1D Poisson solution with absorbing BCs at the edges. For a square group of side L:
```
F_center ≈ e·L² / (8·D_F)      (when γ_F → 0)
F_center ≈ e / γ_F              (when γ_F >> D_F/L²)
```

**Concrete example** (e = 0.5, D_F = 1.0, γ_F = 0):

| Group | k (missing neighbors) | γ_eff | F_ss | α·F (α=10) |
|-------|----------------------|-------|------|-------------|
| 1×1 (isolated) | 4 | 4.0 | 0.125 | 1.25 |
| 1×2 | 3 | 3.0 | 0.167 | 1.67 |
| 3×3 center | 0 | ~0.5 (effective) | ~0.56 | 5.6 |
| 5×5 center | 0 | ~0.13 | ~1.6 | 16 |
| 10×10 center | 0 | ~0.04 | ~6.25 | 62.5 |

The ratio F(10×10)/F(1×2) ≈ 37× arises from geometry alone — this is the CEMA effect.

**Special case γ_F = 0:** With no bulk decay, boundary leakage is the *only* decay mechanism. Interior cells of large groups have near-zero effective decay, so F grows as L². This produces the strongest possible group-size discrimination but requires careful α tuning (see 20.4.4).

#### 20.4.4 Parameter Guidance

| Parameter | Role | Default | Typical Range |
|-----------|------|---------|---------------|
| `D_F` | Diffusion rate | 1.0 | 0.1 – 5.0 |
| `γ_F` | Decay rate | 0.1 | 0.0 – 1.0 |
| `λ = sqrt(D_F/γ_F)` | Communication range (lattice spacings) | 3.2 | 1 – ∞ |
| `α` | Rescue sensitivity | 10.0 | 1.0 – 100.0 |
| `n_substeps` | Sub-steps per bio step | 10 | 8 – 20 |

**Critical α condition:** For rescue to occur at a given cell, α must overcome the local effective decay:

```
α > γ_eff(i) / e    where γ_eff(i) = γ_F + k_i · D_F
```

If α is below this threshold, the field value F_i is too small to shift the damping sigmoid, and the cell receives no rescue regardless of group size. The diagnostic output at startup prints the worst-case (boundary) and interior γ_eff values along with the critical α.

**Example:** With D_F = 10.0, γ_F = 0, e ≈ 0.5, a group-of-2 cell has k = 3, so γ_eff = 30 and α_crit = 60. Setting α = 10 is **subcritical** — no rescue occurs.

**Choosing D_F — the key tradeoff:**

D_F controls the balance between self-rescue (bad for CEMA) and boundary leakage (limits rescue). The critical insight: each embryo emits into the field at its own location and reads it back, so F always includes a self-contribution. The D_F tradeoff for α = 10, e ≈ 0.5, γ_F = 0 is:

| D_F | F_self (group-of-2) | α·F_self | Behavior |
|-----|---------------------|----------|----------|
| 0.01 | 16.7 | 167 | Self-rescue saturates sigmoid — no group effect |
| 0.1 | 1.67 | 16.7 | Still strong self-rescue |
| **1.0** | **0.17** | **1.7** | **Marginal self-rescue — sweet spot for CEMA** |
| 5.0 | 0.033 | 0.33 | Subcritical for boundary cells |
| 10.0 | 0.017 | 0.17 | No rescue anywhere |

**Recommended approach:** Set γ_F = 0 (or very small) and tune D_F so that:
- α · F_self ≈ 1–3 for boundary cells (marginal rescue)
- α · F_interior >> 1 for large-group interior cells (strong rescue)

This maximizes the contrast between small and large groups.

**Choosing γ_F for group discrimination (with D_F = 1.0):**

| γ_F | λ | F(1×2) | F(10×10 center) | Behavior |
|-----|---|--------|-----------------|----------|
| 0.0 | ∞ | ~0.17 | ~6.25 | Max discrimination; boundary leakage only |
| 0.05 | 4.5 | ~0.16 | ~10.0 | Very long range; even small groups may saturate |
| 0.10 | 3.2 | ~0.16 | ~5.0 | Good balance for 2 vs 100 discrimination |
| **0.15** | **2.6** | **~0.16** | **~3.3** | **Recommended starting point (with γ_F > 0)** |
| 0.25 | 2.0 | ~0.16 | ~2.0 | Short range; mostly nearest-neighbor effect |
| 1.00 | 1.0 | ~0.13 | ~0.5 | Negligible accumulation; no group effect |

**Stability constraint:** `dt_sub = 1/n_substeps` must satisfy `dt_sub < 1/(2·D_F·max_degree)`. With D_F = 1.0 and max_degree = 4: need `n_substeps ≥ 8`.

#### 20.4.5 Self-Contribution and the Self-Rescue Problem

The emission term includes the embryo's own stress (`emission_i = mean(S_i)`), so each embryo's field value includes a self-contribution. With absorbing BCs, the self-contribution for an embryo with k missing neighbors is:

```
F_self ≈ e / (γ_F + k · D_F)
```

**Derivation:** Each embryo's field variable evolves as `dF_i/dt = e_i − γ_F · F_i + D_F · Σ_neighbors(F_j − F_i)`. For an isolated embryo at steady state (dF/dt = 0), all k missing neighbors have F_j = 0 (absorbing BC), so the diffusion term reduces to `−D_F · k · F`. Setting `0 = e − γ_F · F − D_F · k · F` and solving gives `F_self = e / (γ_F + k · D_F)` — the balance between emission (source) and the two loss channels: intrinsic decay (γ_F) and diffusive leakage to absent neighbors (k · D_F).

**The self-rescue problem:** When D_F is too small, F_self becomes large enough to saturate the damping sigmoid (α · F_self >> 1). In this regime, every embryo rescues itself regardless of neighbors, eliminating any group-size dependence. This was observed empirically: D_F = 0.01 with α = 10 produced perfect rescue for a group of 2.

**Does a group of 1 always self-rescue when γ_F = 0?** No. An isolated embryo has k = 4 (all neighbors missing), giving F_self = e/(4·D_F). Self-rescue requires α · F_self > 1, i.e., D_F < α·e/4. For α = 10, e = 0.5: D_F < 1.25 allows self-rescue; D_F > 1.25 prevents it.

**Key constraint for group-of-2:** A group-of-2 cell has k = 3, so F_self = e/(3·D_F), which is always larger than the group-of-1 value. Therefore, for any D_F where a group of 1 self-rescues, a group of 2 also self-rescues. Clean CEMA discrimination requires D_F large enough that small groups cannot self-rescue but small enough that large groups accumulate sufficient field from collective contributions.

#### 20.4.6 The Optimal Diffusion–Decay Operating Regime

The fundamental tension in the CEMA field model is between two failure modes:

```
D_F too small (or γ_F too small):
  → Signal pools at each emitter's own location
  → F_self is large → every embryo rescues itself
  → No group-size dependence — CEMA disappears

D_F too large (or γ_F too large):
  → Signal leaks out of the group before it can accumulate
  → F is small everywhere, even for large groups
  → No rescue at all — CEMA also disappears
```

The optimal regime sits between these extremes, where boundary cells (small groups) receive marginal rescue signal while interior cells (large groups) accumulate strong signal. This is the regime where the surface-to-volume ratio creates maximum discrimination between group sizes.

**The role of D_F.** Diffusion rate controls how quickly signal spreads from emitter to neighbors, but also how quickly it leaks through the boundary. The self-rescue condition `α·F_self > 1` gives `D_F < α·e/k`, where k is the number of missing neighbors. The no-rescue condition `α·⟨F⟩ < 1` gives `D_F > α·e·L/4` for interior cells of a large group. The optimal D_F lies between these bounds:

```
α·e / k_boundary  <  D_F  <  α·e·L / 4
     ↑                           ↑
  Above this:                Above this:
  self-rescue                large-group interior
  is PREVENTED               rescue FAILS
  (signal diffuses            (too much leakage
   away too fast               even for big groups)
   to pool locally)
```

Below the left threshold, D_F is so small that signal pools at each emitter — every embryo rescues itself regardless of neighbors. Above the right threshold, D_F is so large that even collective accumulation in large groups can't overcome boundary leakage. The CEMA regime lives in between.

For α = 10, e = 0.5, k = 3 (group-of-2 boundary): D_F > 1.67 prevents self-rescue (signal diffuses away before pooling). For a group of side L = 10: D_F < 12.5 allows interior rescue (collective accumulation still overcomes leakage). So D_F ∈ [1.67, 12.5] is the operating window where small groups fail but large groups succeed. With D_F = 1.0 (just below the self-rescue threshold), both small and large groups get some rescue but large groups get dramatically more — this is the sweet spot.

**The role of γ_F.** Bulk decay provides a floor on the effective decay rate that is independent of position: `γ_eff(i) = γ_F + k_i·D_F`. Setting γ_F = 0 means the *only* signal loss is boundary leakage, which maximizes the contrast between boundary and interior cells. However, γ_F = 0 also means that interior cells of very large groups have near-zero effective decay, so F can grow without bound (limited only by the negative feedback where rescued embryos emit less). This can cause numerical issues or unrealistically strong rescue.

Setting γ_F > 0 introduces a cap: no matter how large the group, `⟨F⟩ ≤ e/γ_F`. This bounds the maximum rescue signal but also limits group-size discrimination for groups larger than L ≈ 4D_F/(γ_F·L), i.e., L ≈ 2√(D_F/γ_F) = 2λ. Beyond this size, additional embryos don't help because the field has already saturated at the bulk-limited value.

**Summary of operating regimes:**

| | Small D_F (pooling) | **Optimal D_F (CEMA regime)** | Large D_F (leaking) |
|---|---|---|---|
| **F_self** | Very large | Marginal (~1/α) | Negligible |
| **F_interior** | Very large | Large (>> 1/α) | Small (< 1/α) |
| **Rescue** | All embryos (self-rescue) | **Large groups only** | None |
| **Group effect** | None | **Strong** | None |

**Practical recipe for choosing D_F and γ_F:**

1. Estimate emission e (typical stress level; ~0.5 with `--initialStress 1.0`)
2. Set γ_F = 0 initially (maximum group-size discrimination)
3. Choose D_F so that α·e/(k_max·D_F) ≈ 1–2, where k_max is the number of missing neighbors for the smallest group you want to test. For α = 10, e = 0.5, k = 3: D_F ≈ 0.8–1.7
4. Verify: run group-of-1 (should NOT rescue) and large group (SHOULD rescue)
5. If large groups saturate too strongly, increase γ_F to cap the field (start with γ_F = 0.01–0.1)
6. If even large groups fail to rescue, decrease D_F or increase α

#### 20.4.7 Backward Compatibility

Setting `--D_F 0` disables the diffusive field and falls back to the original `mean_neighbor_stress` method (Section 17 behavior).

### 20.5 Interaction with the Intra-Embryo Feedback Loop

The rescue mechanism creates a negative feedback loop:

```
F_i ↑  →  effective_damping ↑  →  better GRN  →  better Vmem
  →  lower Ca²⁺  →  lower stress S  →  lower emission  →  F ↓
```

The feedback is stabilizing (rescue), but the **strength** of the rescue signal depends on group size through the diffusive field. Larger groups accumulate more F at interior positions, driving stronger rescue.

**Important consequence:** The negative feedback self-limits the rescue. As F rises and embryos become healthier, their stress drops, reducing emission, reducing F. The system settles at an equilibrium where F is much lower than the "all maximally stressed" prediction of `emission/γ_F`. The α parameter controls how responsive the damping formula is to small F values.

### 20.6 The Temporal Mismatch Problem (Revisited)

As described in Section 18.1, the rescue signal arrives too late when embryos start with zero stress. By the time the stress → F → damping chain builds up (~500-750 steps), the Vmem pattern has already committed to an abnormal attractor.

This is visible in group rescue experiments: effective damping reaches ~1.0, but Vmem similarity to the healthy reference remains near 0.

**Mitigation strategies:**
1. **Pre-seeded stress** (`--initialStress 0.5`): Start all embryos with nonzero S, giving the diffusive field a head start
2. **Milder damage** (`--dampingGaussian "0.5,0.1"` instead of `"0.1,0.3"`): The initial Vmem pattern deviates less and remains redirectable once rescue kicks in
3. **Longer simulations** (`--numBioSteps 5000`): Give the restored GRN more time to correct the pattern
4. **Higher α**: Makes the damping formula more responsive to small early F values

### 20.7 Interpreting the Negative Feedback Loop: Stress, Field, and Rescue Quality

#### 20.7.1 The Stress–Similarity Paradox

A counterintuitive observation in group rescue simulations: embryos can have **high Vmem similarity** (successful rescue) while maintaining **high stress levels**. This is not a bug — it reflects the bistable nature of the stress switch and the negative feedback structure of the rescue loop.

The causal chain:

```
t=0:  initialStress=1.0 → high emission → large F → high α·F
      → damping restored to ~1.0 → Vmem improves → similarity ↑

t>500: Vmem is now healthy, but the stress switch is BISTABLE
       → Ca²⁺ has dropped (healthier Vmem), but stress S was already
         locked HIGH by the bistable dynamics
       → S stays high → emission stays high → F stays high
       → damping stays high → Vmem stays healthy
```

**Stress is the fuel, not the outcome.** High stress + high similarity = successful rescue being sustained by its own stress signal. The stress remaining high is actually *sustaining* the rescue. If stress dropped, emission would drop, F would drop, damping would drop, and the embryo could relapse.

#### 20.7.2 Why Higher F Can Mean Worse Rescue

A second paradox: comparing groups of different sizes, the **larger group** (with better rescue rates) can show **lower steady-state F** than the smaller group (with worse rescue rates).

This is the negative feedback loop in action:

```
Large group (better rescue):
  More interior cells → F rises fast early → damping restored quickly
  → Vmem improves → emission slightly lower (healthier embryos)
  → F settles at a LOWER steady state

Small group (worse rescue):
  More boundary cells → F rises slower, weaker early signal
  → some embryos miss the rescue window → Vmem stays abnormal
  → those embryos keep emitting MAXIMUM stress
  → F stays HIGH because unrescued embryos keep pumping signal
```

**F measures the disease burden of the group, not the rescue quality.** A group where everyone is sick but nobody is being rescued has the highest F. Successfully rescued groups have marginally lower emission, which paradoxically lowers F.

#### 20.7.3 Interpreting Simulation Output

| Observation | Interpretation |
|-------------|---------------|
| High similarity + high stress | Successful rescue sustained by bistable stress fuel |
| Low similarity + high stress | Failed rescue — stress is high but F was insufficient to restore damping in time |
| High similarity + low stress | Would indicate stress switch flipped LOW — rescue may be unstable |
| High F in small group vs low F in large group | Small group has more unrescued embryos emitting maximally |
| F drops over time after initial rise | Rescue is working — healthier embryos emit less |

### 20.8 Damping Assignment Modes

| CLI Option | Description |
|------------|-------------|
| `--dampingLevels "1.0,0.5"` | Alternating pattern across the grid |
| `--dampingRange "0.1,0.3"` | Uniform random in [min, max] |
| `--dampingGaussian "0.2,0.05"` | Gaussian N(mean, std), clipped to [0.01, 1.0] |
| `--dampingMap "1.0,0.5,..."` | Explicit per-embryo values (row-major) |
| `--dampingCenter 0.1` | Center embryo gets this value, all others get 1.0 |
| (default) | Alternating 1.0, 0.5 |

### 20.9 Rescue Rate Metric

The **rescue rate** quantifies the fraction of embryos whose Vmem similarity to a healthy reference exceeds a user-defined threshold:

```
rescue_rate(t) = |{i : similarity_i(t) > threshold}| / N
```

CLI argument: `--rescueThreshold` (default 0.5). The rescue rate is plotted as a timeseries (overall in solid black, per-damping-group in colored dashed lines) and printed as a summary at the end of the run.

This metric is more interpretable than mean similarity because it counts how many embryos are functionally "rescued" rather than averaging over a mix of rescued and unrescued.

### 20.10 Output Files

Two files are produced with auto-generated suffixes encoding run parameters:

**Main figure** (`group_rescue_test_g{N}_a{α}_d{min}-{max}_t{steps}_D{D_F}_g{γ_F}.png`):
- Row 1: Base damping heatmap | Final stress heatmap | Vmem similarity heatmap
- Row 2: Stress timeseries | Effective damping timeseries | Diffusive field timeseries
- Row 3 (optional): Vmem similarity timeseries (left) | Rescue rate timeseries (right)

**Vmem grid** (`..._vmem_grid.png`): Per-embryo final Vmem patterns as small heatmaps.

### 20.11 Usage Examples

```bash
# Quick smoke test (4 embryos, 100 steps)
python runGroupRescue.py --groupSize 4 --numBioSteps 100 --alpha 3.0 \
    --dampingLevels "1.0,0.5"

# Compare group sizes (2 vs 100) with diffusive field
python runGroupRescue.py --groupSize 2 --dampingRange "0.1,0.3" --alpha 2.5 \
    --D_F 1.0 --gamma_F 0.15 --stressParamsFile data/bestLearnedStressParams_6.dat

python runGroupRescue.py --groupSize 100 --dampingRange "0.1,0.3" --alpha 2.5 \
    --D_F 1.0 --gamma_F 0.15 --stressParamsFile data/bestLearnedStressParams_6.dat

# Gaussian damping distribution
python runGroupRescue.py --groupSize 100 --dampingGaussian "0.2,0.05" --alpha 2.5 \
    --D_F 1.0 --gamma_F 0.15 --stressParamsFile data/bestLearnedStressParams_6.dat

# Center-stressed configuration (single damaged embryo surrounded by healthy)
python runGroupRescue.py --groupSize 25 --dampingCenter 0.1 --alpha 5.0 \
    --D_F 1.0 --gamma_F 0.1 --numBioSteps 2000

# Disable diffusive field (fall back to mean_neighbor_stress)
python runGroupRescue.py --groupSize 25 --dampingLevels "1.0,0.5" --alpha 3.0 \
    --D_F 0 --numBioSteps 2000

# Pre-seeded stress to address temporal mismatch
python runGroupRescue.py --groupSize 100 --dampingGaussian "0.3,0.1" --alpha 2.5 \
    --D_F 1.0 --gamma_F 0.15 --initialStress 0.5 \
    --stressParamsFile data/bestLearnedStressParams_6.dat

# Absorbing BC with γ_F=0 (boundary leakage only — max CEMA discrimination)
python runGroupRescue.py --groupSize 2 --dampingGaussian "0.5,0.01" --alpha 10.0 \
    --D_F 1.0 --gamma_F 0.0 --initialStress 1.0 \
    --stressParamsFile data/bestLearnedStressParams_6.dat --numBioSteps 2000

python runGroupRescue.py --groupSize 100 --dampingGaussian "0.5,0.01" --alpha 10.0 \
    --D_F 1.0 --gamma_F 0.0 --initialStress 1.0 \
    --stressParamsFile data/bestLearnedStressParams_6.dat --numBioSteps 2000

# Custom rescue threshold
python runGroupRescue.py --groupSize 25 --dampingGaussian "0.3,0.1" --alpha 5.0 \
    --D_F 1.0 --gamma_F 0.0 --rescueThreshold 0.6 --numBioSteps 2000
```

### 20.12 GPU Parallelization Decision

The bioelectric `model` class uses `torch.float64` throughout (incompatible with Apple MPS), and the simulation requires step-level synchronization (all embryos must complete step t before stress exchange for step t+1). This rules out fire-and-forget GPU parallelism. Instead:

- **ThreadPoolExecutor** parallelizes the N `model.simulate(numSimIters=1)` calls within each timestep, with barrier sync before stress exchange
- Activated when `num_embryos >= parallel_threshold` (default 16)
- **Stress computation** and **field diffusion** are lightweight numpy/torch ops, no parallelism needed

---

## 21. Implementation File Reference

### 20.1 Core Files

| File | Role | Status |
|------|------|--------|
| `stressBistableSwitch.py` | StressBistableSwitch class (RD bistable stress system) | Implemented |
| `learnStressBistableSwitch.py` | Learn stress parameters via Rprop optimization | Implemented |
| `runStressBistableSwitch.py` | Run stress system on single embryo at various damping levels | Implemented |
| `runStressRescue.py` | Pairwise donor-recipient rescue sweep | Implemented |
| `learnStressRescue.py` | Learn α parameter (and potentially stress params) for rescue | Planned |

### 20.2 Parameter Files

| File | Contents |
|------|----------|
| `data/bestLearnedStressParams_5.dat` | Best learned stress parameters (primary, used for rescue experiments) |
| `data/bestLearnedStressParams_0.dat` through `_6.dat` | Alternative learned parameter sets from different training runs |
| `data/bestLearnedCaMKIIParams_0.dat` | Fixed Ca²⁺ parameters (shared between CaMKII and stress systems) |

### 20.3 Output Files

| File | Contents |
|------|----------|
| `data/stress_rescue_test.png` | Default rescue sweep visualization (6 panels) |
| `data/stress_rescue_test_StressFile5_alpha*.png` | Rescue results at various α values |
| `data/stress_bistable_test_5.png` | Single-embryo stress profiles at various damping levels |

---

## 22. Initial Hypothesis: Electric Field-Based Rescue Signaling

> **Note**: This section documents an initial hypothesis developed from the α=5000 simulation result. Section 22 provides a systematic mechanistic critique that substantially revises several conclusions here. In particular, the v_eff formula is shown to compute a kinematic ratio rather than a physical propagation speed, and the "electrotonic speed coincidence" is argued to be spurious. Section 22 should be read as the current position; this section is preserved to document the reasoning and the physics exploration that led to it.

### 24.1 The α = 5000 Result

Pair-rescue experiments revealed that α = 5000 successfully rescues even severely stressed embryos (GRN damping = 0.1), while biophysically calibrated values (α ≈ 0.17 for eATP diffusion) produce no measurable rescue. Using the propagation speed formula from Section 19.2:

```
v_eff = (d_inter · α · S_∞) / (Λ · τ_rise_physical)
      = (1400 · 5000 · 0.75) / (0.46 · 750 · 0.01 · 10)
      ≈ 152,000 μm/s
      ≈ 0.15 m/s
```

This speed falls in a **distinctive biophysical regime** that uniquely identifies the signaling mechanism.

### 24.2 Speed Regime Comparison Across Biological Mechanisms

```
Speed (μm/s)       Mechanism                                  Match?
─────────────────────────────────────────────────────────────────────
0.3                Pure ATP diffusion (~1mm distance)          ✗
3                  Pure Ca²⁺ diffusion (~100μm)                ✗
5.28               CEMA measured (inter-embryo Ca²⁺)           ✗
10-50              IP3-relay Ca²⁺ waves (gap junction)         ✗
100                Fastest Ca²⁺ waves (cardiac, hepatocyte)    ✗
─────────── 3 orders of magnitude gap (chemical → electrical) ──────
60,000-170,000     Electrotonic spread (embryonic tissue)      ✓ ←──
500,000+           Action potentials (excitable tissue)        ✗
~3×10¹⁰            EM wave in tissue                           ✗
```

**The 150,000 μm/s speed is separated by ~3 orders of magnitude from the fastest biochemical wave and sits squarely in the electrotonic propagation regime.** No biochemical mechanism — diffusion, active Ca²⁺ relay, or IP3-mediated wave — can produce speeds in this range.

### 24.3 Electrotonic Spread: Physics and Parameter Match

Passive voltage propagation through gap junction-coupled cells follows cable theory:

```
v_electrotonic = 2λ / τ_m
```

where:
- λ = space constant = √(R_m · d / (4 · ρ_i_eff)) — how far voltage decays spatially
- τ_m = R_m × C_m — membrane charging time constant

**Xenopus embryonic tissue parameters:**

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Specific membrane resistance | R_m | 1,000-10,000 Ω·cm² | Embryonic cells |
| Specific membrane capacitance | C_m | ~1 μF/cm² | Universal |
| Membrane time constant | τ_m | 1-10 ms | R_m × C_m |
| Cell diameter | d | 20-50 μm | Post-blastula cells |
| Effective intracellular resistivity | ρ_i_eff | ~2,000 Ω·cm | Including gap junction resistance |
| Gap junction conductance | G_j | 10 nS - 1 μS per pair | Cx38/Cx43 |
| Connexins present | — | Cx38 (maternal), Cx43 | Xenopus embryo |

**Calculated electrotonic speed across embryonic tissue parameter ranges:**

| R_m (Ω·cm²) | τ_m (ms) | λ (μm) | v_electrotonic (m/s) | Tissue type |
|--------------|----------|--------|----------------------|-------------|
| 1,000 | 1 | 190 | 0.19-0.38 | Early blastomere (leaky) |
| 3,000 | 3 | 330 | 0.11-0.22 | Mid-embryonic |
| **5,000** | **5** | **430** | **0.09-0.17** | **Typical embryonic epithelium** |
| 10,000 | 10 | 600 | 0.06-0.12 | Differentiated epithelium |

For typical embryonic epithelium (R_m ≈ 5,000 Ω·cm², C_m ≈ 1 μF/cm²):

```
τ_m = 5,000 × 10⁻⁶ = 5 ms
λ = √(5000 × 30×10⁻⁴ / (4 × 2000)) ≈ 430 μm
v = 2 × 430 / 5 = 172,000 μm/s ≈ 0.17 m/s
```

**This matches the model-derived 0.15 m/s within measurement uncertainty.**

### 24.4 Why Other Mechanisms Are Excluded

**Biochemical diffusion (eATP, Ca²⁺):**
- For pure diffusion: v_eff = D/x, where D ~ 300 μm²/s and x ~ 1000 μm → v ≈ 0.3 μm/s
- Even at shortest relevant distance (100 μm): v ≈ 3 μm/s
- Factor of **50,000-500,000× too slow**

**IP3-relay Ca²⁺ waves:**
- Theoretical maximum: v ≈ √(D_IP3 / τ_release) ≈ √(280/0.1) ≈ 53 μm/s
- Fastest measured (cardiac tissue): ~100 μm/s
- Factor of **1,500-3,000× too slow**

**Ephaptic coupling (extracellular field):**
- Effectively instantaneous at tissue scales (microseconds across ~300 μm)
- Would not produce a finite measurable propagation speed
- But could contribute as the **mechanism that establishes the field pattern** that electrotonic spread then propagates through the tissue

**Action potentials:**
- 0.5-120 m/s depending on tissue type
- Embryonic cells are not classically excitable (no fast Na⁺ channels)
- Would require regenerative voltage-gated currents not present in early embryos

### 24.5 Mechanistic Interpretation

The electrotonic speed match suggests the following rescue signaling pathway:

```
DONOR (perturbed embryo):
  Abnormal Vmem pattern
    → Altered electric field E(x,y)
    → Field propagates to neighboring embryo (~instantaneous)
    → Enters receiver at boundary cells

RECEIVER (neighboring embryo):
  E_external at boundary cells
    → Electrotonic spread through gap junction network
    → Propagation speed: v = 2λ/τ_m ≈ 0.15 m/s
    → Establishes across 11×11 grid (~300 μm) in ~2 ms
    → Modulates local Vmem at each cell
    → Vmem change → VGCC activation → Ca²⁺ → downstream rescue
```

**Key distinction from the eATP pathway (Section 10):**

| Aspect | eATP Pathway (CEMA) | Electric Field Pathway |
|--------|---------------------|------------------------|
| Inter-embryo signal | eATP molecule (diffusion) | Electric field (EM + electrotonic) |
| Propagation speed | ~5 μm/s | ~150,000 μm/s |
| Required α | ~0.17 | ~5,000 |
| Signal type | Chemical concentration | Voltage pattern |
| Reception mechanism | P2R binding → Ca²⁺ | VGCC activation → Ca²⁺ |
| Temporal mismatch? | Yes (too slow) | No (fast enough) |
| Pattern information | Lost (scalar eATP level) | **Preserved** (spatial Vmem) |
| Rescues damping=0.1? | No | Yes |

### 24.6 The Pattern Information Advantage

A critical qualitative difference: the electric field pathway **preserves spatial pattern information**, while the eATP pathway reduces the donor's state to a scalar (mean stress).

In the eATP pathway:
```
Donor Vmem pattern → Ca²⁺ pattern → S pattern → mean(S) = scalar eATP → receiver
                                                 ↑
                                    All spatial information lost here
```

In the electric field pathway:
```
Donor Vmem pattern → E field(x,y) → receiver boundary → electrotonic spread → Vmem modulation
                     ↑                                                         ↑
              Spatial pattern preserved ─────────────────────────────────────────┘
```

This means a field-based rescue mechanism could, in principle, **impose the correct spatial pattern** on the recipient — not just boost its damping uniformly. This is a qualitatively stronger rescue: the donor provides a template, not just a signal.

### 24.7 Reconciliation with CEMA Experimental Data

The CEMA paper (Tung et al. 2024) presents strong evidence for the eATP/P2R pathway:
- Measured eATP elevation in treated groups
- P2R blockers (suramin, PPADS) abolish rescue
- Ca²⁺ waves observed between embryos at ~5.28 μm/s

**How can the electric field hypothesis be consistent with these findings?**

**Hypothesis: Dual-timescale signaling** — both pathways operate simultaneously but on different timescales:

```
                                         Rescue
                                         effect
Fast pathway (field):                      |
  Electrotonic coupling                    |    ╱── Fast rescue onset
  v ≈ 0.15 m/s                            |  ╱     (seconds-minutes)
  Provides spatial template                | ╱
  Necessary but insufficient alone         |╱
                                           ├──────────────
Slow pathway (eATP/Ca²⁺):                 |╲
  Chemical diffusion                       | ╲
  v ≈ 5 μm/s                              |  ╲── Sustained rescue
  Provides bifurcation control             |    ╲   (minutes-hours)
  Amplifies and stabilizes rescue          |     ╲
                                           └──────────────▶ time
                                               10s  1min  1hr
```

**Why suramin blocks rescue even if the field pathway exists:**
1. The field provides the **spatial pattern** (fast, electrotonic)
2. eATP/P2R provides the **metabolic rescue** (slow, bifurcation control of ATP dynamics)
3. Both are necessary: the field alone modulates Vmem but cannot restore metabolic health; eATP alone lacks spatial information but enables metabolic recovery
4. Blocking P2R eliminates the metabolic component → rescue fails even though field coupling persists

**Why only perturbed embryos help (CEMA constraint):**
- Healthy embryos produce **organized** Vmem patterns → their field at the receiver is spatially structured → electrotonic filtering in the receiver produces low stress (Section 7.4)
- Perturbed embryos produce **uniform** Vmem → strong, coherent field → drives receiver stress pathway → triggers eATP release → collective rescue
- The field pathway **inherits the same CEMA-compatible logic** as the Ca²⁺ pathway because it drives the same downstream stress system

### 24.8 Testable Predictions

The dual-pathway hypothesis generates predictions distinct from either pathway alone:

| Prediction | eATP-only | Field-only | Dual pathway |
|------------|-----------|------------|--------------|
| P2R blockers (suramin) | Abolish rescue | No effect | Abolish rescue |
| **Gap junction blockers (Cx38 DN)** | **Partial effect** | **Abolish rescue** | **Abolish rescue** |
| **Faraday cage between embryos** | **No effect** | **Abolish rescue** | **Abolish rescue** |
| Physical separation (>5mm) | Abolish (diffusion limited) | Abolish (1/r² decay) | Abolish both |
| Increased medium viscosity | Slows rescue | No effect | **Slows but doesn't abolish** |
| Ca²⁺ channel blockers | Abolish rescue | Partial effect | Abolish rescue |

**Key discriminating experiments:**
1. **Gap junction knockdown** (dominant-negative Cx38 or Cx43): Should abolish field-mediated rescue but leave eATP pathway intact. If rescue is abolished, field pathway is necessary.
2. **Faraday cage / EM shielding** between embryos: Should block field propagation without affecting chemical diffusion. If rescue is reduced, field component exists.
3. **Medium viscosity increase**: Should slow eATP diffusion (v ∝ D ∝ 1/η) without affecting field propagation. If rescue onset is delayed but not eliminated, both pathways contribute.

### 24.9 Implications for Model Architecture

If the electric field pathway is the primary rescue mechanism, the model architecture should be revised:

**Current model** (scalar stress coupling):
```
Donor mean(S) → α × S → sigmoid(logit(d₀) + α·S) → uniform damping modulation
```

**Proposed field-coupled model** (spatially resolved):
```
Donor Vmem(x,y) → E_field(x,y) → electrotonic spread → ΔVmem(x,y) at receiver
                                                         → spatially patterned Ca²⁺
                                                         → spatially patterned rescue
```

This would replace the uniform damping modulation with **spatially varying damping** — cells at the receiver that experience stronger field effects get more rescue. This is a richer model that could explain how the donor's *pattern* (not just stress level) influences the recipient's development.

### 24.10 Summary: Three Speed Regimes, Three Mechanisms

```
α regime          v_eff          Mechanism            Rescue?
─────────────────────────────────────────────────────────────
α ≈ 0.17          5.28 μm/s      eATP diffusion       No (temporal mismatch)
α ≈ 3-10          91-300 μm/s    Ca²⁺ wave?           Marginal
α ≈ 5000          0.15 m/s       Electrotonic spread   Yes (even damping=0.1)
─────────────────────────────────────────────────────────────
                  ↑               ↑                    ↑
            CEMA measured    Transitional          Model requirement
```

The fact that successful rescue requires α values corresponding to electrotonic speeds — rather than diffusion speeds — appeared to be computational evidence for a **bioelectric (field-based) rescue mechanism**. However, Section 22 argues this inference is flawed: the v_eff formula assigns embryo geometry to the donor's internal bistable rise time rather than to a physical propagation event, making the "speed" an artifact of the temporal mismatch problem rather than a biophysical measurement. The electrotonic speed coincidence is likely numerically spurious.

---

## 23. Critical Analysis of Inter-Embryo Signaling Mechanisms

This section documents a systematic mechanistic analysis of every candidate physical process that could explain inter-embryo Ca²⁺ wave propagation at 5.28 μm/s, and critically reassesses what the CEMA measurement actually captures. The analysis substantially revises Section 21's interpretation.

### 24.1 CEMA Medium: 0.1× MMR, Not Saline or Deionized Water

All CEMA inter-embryo experiments used **0.1× Marc's Modified Ringers solution (MMR), pH 7.8** — a dilute physiological salt buffer. This has major implications for field-based mechanisms.

| Medium | σ (S/m) | ρ (Ω·m) | Field screening |
|--------|---------|---------|-----------------|
| Physiological saline (~154 mM NaCl) | ~1.5 | ~0.67 | Strong |
| **0.1× MMR (~10 mM NaCl equiv.)** | **~0.12** | **~8** | **Moderate** |
| Deionized water | ~5.5×10⁻⁶ | ~180,000 | Negligible |

0.1× MMR is ~12× more resistive than physiological saline, making extracellular electric fields ~12× stronger than Section 21's estimates assumed. However, it is still ~22,000× more conductive than deionized water. The medium is an ionic solution, not deionized water — the earlier assumption of physiological saline slightly underestimated field strengths, but the order-of-magnitude conclusions are unchanged.

**Revised ephaptic coupling (0.1× MMR):**

| Scenario | Previous estimate (saline) | Revised (0.1× MMR) | VGCC threshold |
|----------|---------------------------|---------------------|----------------|
| Single embryo pair | ~10–25 μV | ~120–300 μV | ~1–5 mV |
| 150 coherent embryos | ~1–2 mV | ~12–24 mV | ~1–5 mV |

Single-pair ephaptic coupling remains sub-threshold. A large coherent group could in principle reach suprathreshold levels, but random embryo orientation in the dish means dipoles cancel statistically.

### 24.2 What Does 5.28 μm/s Actually Measure?

The CEMA paper defines inter-embryo wave speed as:

```
v = distance(injury site → closest point on embryo B) / time(injury → first Ca²⁺ response in B)
```

This definition **includes the time for the CICR wave to travel within embryo A** from the injury site to the A–B interface. Given embryo A's intra-tissue CICR speed of 6.7 μm/s, traversing a ~1.4 mm embryo takes ~210 s. The reported 5.28 μm/s is then dominated by intra-embryo A travel time, not the gap-crossing time.

**Observed speeds:**

| Measurement | Value | Interpretation |
|-------------|-------|----------------|
| Intra-A (injured embryo) | 6.7 μm/s | CICR wave speed in injured tissue |
| "Inter-embryo" | 5.28 μm/s | Dominated by CICR travel within A |
| Intra-B (uninjured embryo) | 2.36 μm/s | CICR wave speed in uninjured tissue |

The inter-embryo speed (5.28 μm/s) is intermediate between the two intra-embryo speeds — not faster than either, which would be expected if a distinct fast inter-embryo mechanism existed. **This pattern is consistent with 5.28 μm/s being an intra-embryo speed in disguise**, where the gap crossing (eATP diffusion over a near-zero gap between nearly-touching embryos) is fast and contributes little to the total time.

### 24.3 Why IP3/CICR Waves Cannot Explain the Gap Crossing

IP3/CICR waves require **ER (endoplasmic reticulum)** as the Ca²⁺ source and **IP3 receptors** as the gating mechanism. The extracellular MMR between embryos contains neither. A regenerative IP3 wave cannot propagate through aqueous medium — it can only travel within cells.

The correct picture is a **relay** mechanism:
1. CICR wave sweeps embryo A at ~6.7 μm/s (ER-mediated, intra-cellular)
2. Ca²⁺ or eATP released at A's surface diffuses across the narrow gap
3. Triggers CICR in embryo B at ~2.36 μm/s (ER-mediated, intra-cellular)

The extracellular gap crossing is passive diffusion. The CICR wave does not traverse the medium.

### 24.4 Why Shear Waves Do Not Apply

**Shear waves cannot propagate through liquids.** Liquids have zero shear modulus — a transverse deformation flows rather than propagates. The 0.1× MMR between embryos supports only longitudinal (acoustic) waves, which travel at ~1500 m/s. These arrive in ~1 μs for a 1 mm gap — essentially instantaneous but carrying negligible chemical signal.

Shear transmission would require direct embryo-embryo contact (solid pathway) or coupling through the acrylic substrate. If this were the mechanism, the observed wave speed would be m/s, not μm/s.

### 24.5 Electrophoresis: Correct Framework but Wrong Medium

The **Nernst-Planck equation** provides the correct framework for field-assisted Ca²⁺ transport:

```
J = -D∇C + μCE    where μ = zDF/RT  (Nernst-Einstein)
```

For Ca²⁺ (z = +2, D ≈ 790 μm²/s): μ = 61,500 μm²/(s·V)

The drift velocity needed for 5 μm/s: E_required = 5 / 61,500 ≈ **81 mV/m**

However, the extracellular field generated by an embryo in 0.1× MMR is far weaker:

```
φ_e(a) ≈ -J_m × a / (3σ_e)
       = 0.05 A/m² × 7×10⁻⁴ m / (3 × 0.12 S/m) ≈ 0.1 mV

E_surface ≈ 0.1 mV / 700 μm ≈ 0.14 V/m
v_drift = 6.15×10⁻⁸ × 0.14 ≈ 0.009 μm/s
```

**Electrophoretic drift is ~70× slower than passive diffusion** in 0.1× MMR. The ionic conductivity screens the extracellular field: Vmem = 50 mV drops almost entirely across the 7 nm lipid bilayer, leaving only ~0.1 mV in the extracellular space.

**Conductivity dependence — the key insight:** Electrophoretic transport is inversely proportional to medium conductivity. In deionized water (σ ≈ 5.5×10⁻⁶ S/m):

```
φ_e(a) ≈ 2.1 V (at surface)     →     v_drift ≈ 60 μm/s  at 1 mm distance
```

The same embryo in deionized water would drive Ca²⁺ electrophoretically at ~60 μm/s — 10× faster than the CEMA measurement and comfortably above the required 81 mV/m threshold. **The electrophoresis mechanism is viable in principle but is screened out by the ionic MMR medium in CEMA's experiments.**

This generates a testable prediction: repeat the CEMA inter-embryo wave experiment across a gradient of medium conductivities from ~10⁻⁵ S/m up to 1× MMR. If electrophoresis contributes, wave speed should scale inversely with conductivity.

### 24.6 The Swept-Source Interpretation: How Field-Based Mechanisms Produce Finite Speeds

A field (electric, acoustic, or other) propagates essentially instantaneously at tissue scales. However, the **apparent wave speed observed in experiments** is not the field propagation speed — it is the speed at which the **source pattern changes**.

When embryo A's CICR depolarization front sweeps at ~6.7 μm/s, the electric field configuration at embryo B updates at that same rate — not because the field travels slowly, but because the source driving it moves slowly. Embryo B experiences a time-varying field whose pattern shifts at the CICR wave speed of A.

This "swept source" effect naturally explains:
- Inter-embryo speed ≈ intra-A CICR speed (both ~6.7 μm/s)
- Inter-embryo speed intermediate between intra-A and intra-B speeds
- No need for a novel inter-embryo propagation mechanism

**The 5.28 μm/s does not require a mechanism that propagates at 5.28 μm/s across the gap.** It is the CICR wave speed of embryo A projected onto the measurement geometry.

### 24.7 Suramin: The Critical Discriminator for Field vs. Chemical Mechanisms

The CEMA paper shows that **suramin (P2 ATP receptor blocker) attenuates inter-embryo wave transfer**. This is the sharpest experimental constraint on mechanism:

- Suramin blocks P2X/P2Y purinergic receptors (eATP receptors)
- It does not alter medium conductivity, permittivity, or any physical property relevant to electric fields
- The primary CICR wave within embryo A is IP3R/ER-mediated, not P2X-dependent
- Therefore suramin should **not** affect field coupling between embryos

Yet inter-embryo transfer is attenuated. This directly implicates eATP binding at P2 receptors as a necessary step in the inter-embryo causal chain — not merely an amplifier, but a required component.

**A pure electric field mechanism cannot explain suramin sensitivity** without introducing eATP as an essential intermediate, at which point the mechanism has become chemical.

### 24.8 The Separation Experiment: A Confounded Design

The CEMA Discussion states: *"When embryos were physically distanced and prevented from touching, the average survival rate decreased significantly."* However, the experimental design confounds **physical separation** with **chemical diffusion restriction**.

| Condition | Physical contact | Diffusion path | Survival |
|-----------|-----------------|----------------|---------|
| Unseparated | Allowed | Direct (≈ zero gap) | 96% |
| Solid wells (1.80 mm) | Prevented | Over top of 1.80 mm walls (long, tortuous) | 1.3% |
| Windowed wells | Prevented | Lateral through windows | 63% |

The critical comparison is **solid vs. windowed wells**: same physical separation, same embryo-to-embryo distance, but different diffusion path lengths. This alone accounts for the 1.3% → 63% improvement. The dominant variable is diffusion access, not physical proximity.

There is **no experiment** in the paper that varies physical distance while holding diffusion path length constant at the same value as unseparated embryos. The paper's own figure caption correctly states *"CEMA requires diffusion but not physical contact"* — the Discussion's characterization of "physical distancing" as the key variable is an overreach not directly supported by the data.

**Implications for field-based mechanisms:** Solid acrylic wells are insulators that restrict chemical diffusion. They also partially restrict electric field coupling by forcing current to travel a longer path through the solution above the wells. The near-complete abolishment of CEMA (1.3%) is more consistently explained by diffusion blockade than by field attenuation, since the field can still couple through the connected solution above.

### 24.9 Revised Verdict: 5.28 μm/s and the α=5000 Result

**On CEMA's 5.28 μm/s:**

The speed is most parsimoniously explained as the intra-embryo CICR wave speed of the injured embryo A (6.7 μm/s), slightly reduced by geometric projection. The actual inter-embryo gap crossing is fast (eATP diffuses a negligible distance between nearly-touching embryos) and contributes little to the total measured time. The CEMA paper's own evidence — suramin sensitivity, diffusion-dependent survival, required adjacency — points to eATP as the primary inter-embryo messenger.

**On the α=5000 result:**

The earlier interpretation (Section 21) that α=5000 implies electrotonic signaling at 0.15 m/s should be treated with caution. The v_eff formula:

```
v_eff = d_inter × α × S_∞ / (Λ × τ_rise_physical)
```

assigns the physical distance d_inter to the donor's internal bistable rise time τ_rise, not to a physical propagation event. Moving the embryos 10 mm apart would give v_eff = 1.5 m/s for identical dynamics — an absurd result. **The formula computes a kinematic ratio, not a propagation speed.**

α=5000 is required because of the **temporal mismatch** (Section 18.1): donor stress rises over ~500–1000 steps while recipient pattern collapses in ~100–500 steps. The large α compensates for the late-arriving signal by making small early stress signals produce large damping shifts. The "speed" assigned to this is an artifact of the measurement geometry, not a biophysical propagation rate.

**The coincidence with cable theory electrotonic speed (0.15–0.17 m/s) is likely numerically spurious.**

### 24.10 Summary: Mechanism Scorecard for CEMA Inter-Embryo Signaling

| Mechanism | Predicted speed | CEMA speed (5.28 μm/s) | Suramin? | Solid wall? | Verdict |
|-----------|----------------|------------------------|----------|-------------|---------|
| eATP free diffusion | ~0.36 μm/s | 15× too slow | ✓ | ✓ | Speed mismatch |
| Ca²⁺ free diffusion | ~0.64 μm/s | 8× too slow | ✗ | ✓ | Suramin mismatch |
| IP3/CICR relay (intra-embryo) | ~6.7 μm/s | ✓ (intra-A) | ✓ (secondary) | ✓ | Best fit |
| Electrophoresis (0.1× MMR) | ~0.009 μm/s | 600× too slow | ✗ | ✓ | Field screened |
| Shear waves (liquid) | Not applicable | — | ✗ | ✗ | No shear in liquid |
| Acoustic waves | ~1,500 m/s | 10⁸× too fast | ✗ | Partial | Mechanosensitive trigger only |
| Ephaptic (electric field) | Source-speed limited | ✓ (swept source) | ✗ | Partial | Suramin problem |
| **Intra-A CICR + eATP gap crossing** | **~6.7 / gap→0 μm/s** | **✓** | **✓** | **✓** | **Best overall** |

**The most parsimonious explanation consistent with all CEMA experiments:** the 5.28 μm/s reflects the intra-embryo CICR wave speed in injured embryo A. The gap crossing uses eATP diffusion over a negligible near-zero gap (embryos nearly touching in the acrylic channel). The suramin sensitivity and separation experiments confirm eATP/P2R as the primary inter-embryo signal.

Electric field mechanisms remain theoretically interesting and are the basis of the `runStressRescue.py` pair-rescue model, but they are best understood as a **computational hypothesis** to be tested rather than a mechanism with strong current experimental support from CEMA.

---

## 24. Bulk Flow as an Alternative Inter-Embryo Transport Mechanism

A separate hypothesis, distinct from all mechanisms discussed in Sections 21–22, is that the apparent 5.28 μm/s inter-embryo wave speed is explained by **bulk flow** (advective transport) of eATP through the shared medium, rather than passive diffusion.

### 24.1 Bulk Flow Is Not Related to Shear Waves

These two mechanisms are frequently conflated but are physically unrelated:

| Property | Shear Wave | Bulk Flow (Advection) |
|---|---|---|
| Medium requirement | Solid only (nonzero shear modulus) | Fluid |
| Speed | ~1–10 m/s in soft tissue | 1–50 μm/s (flow velocity) |
| Carries chemical signal | No (mechanical only) | Yes |
| Decays with distance | 1/r amplitude | Depends on geometry |
| Relevant for CEMA | No | Potentially yes |

Shear waves require a solid-like restoring force against transverse deformation. Liquids have zero shear modulus and simply flow in response to shear stress — they cannot sustain transverse mechanical waves. Bulk flow is a net coordinated fluid current carrying dissolved solutes along with it. The Ca²⁺ wave in CEMA is a biochemical reaction wave (a concentration front through a reactive medium), not a mechanical wave of any kind.

### 24.2 Sources of Bulk Flow in Xenopus Embryo Cultures

Several mechanisms could generate sustained fluid currents among 300 Xenopus embryos in a shared dish:

**1. Multiciliated cell (MCC) surface flows (dominant candidate)**

Xenopus epidermal epithelium develops dense arrays of motile cilia that beat in coordinated metachronal waves, generating tangential surface flows of **10–50 μm/s** near the embryo surface. This ciliary pumping is the same mechanism responsible for establishing left-right asymmetry via nodal flow. With 300 embryos each generating surface flows, coupled mesoscale circulation patterns could sustain inter-embryo medium currents in the range of **1–20 μm/s** across embryo-to-embryo gaps.

**2. Osmotic flows**

In hypo-osmotic 0.1× MMR, net water uptake by embryos creates weak inward radial flows (~0.1–1 μm/s). Less directional but present.

**3. Mechanical fluid displacement from cleavage contractions**

Cortical actomyosin contractions during cell division displace fluid locally. In Stokes flow (Re << 1), these disturbances decay as 1/r and are weak at inter-embryo distances, but asynchronous divisions among 300 embryos create a stochastic flow field.

**4. Thermal convection**

Metabolic heat from embryos creates temperature gradients, but estimated convective velocities (~5×10⁻³ μm/s) are negligible compared to cilia-driven flows.

**Critical stage-dependent caveat**: MCC maturation occurs during neurulation (~stage 18–20). If CEMA inter-embryo signaling occurs at early cleavage (stages 4–9, before MCC differentiation), cilia-driven bulk flow is unavailable as a mechanism. The developmental staging of CEMA experiments in Tung et al. must be verified to evaluate whether this mechanism is applicable.

### 24.3 Two Routes by Which Bulk Flow Could Explain 5.28 μm/s

#### Route A: Direct Advection (Péclet-dominated transport)

If inter-embryo medium flows at velocity U along the propagation axis, the signal front advances at approximately U. For eATP with D = 335 μm²/s and gap L = 200 μm (see Section 23.4), the Péclet number is:

```
Pe = U × L / D = 5 μm/s × 200 μm / 335 μm²/s ≈ 3
```

At Pe ≥ 1, advection contributes meaningfully to transport. A sustained inter-embryo flow of ~5 μm/s would directly account for the observed wave speed. This is at the lower end of MCC-driven flows but plausible for bulk medium currents away from the embryo surface.

#### Route B: Taylor Dispersion (advection-enhanced effective diffusivity)

In a shear flow, the effective longitudinal diffusivity is dramatically enhanced (G.I. Taylor, 1953):

```
D_eff = D_molecular + (U × a)² / (48 × D_molecular)
```

where a is the channel half-width (≈ inter-embryo gap half-width, ~100–200 μm).

For U = 20 μm/s, a = 200 μm, D = 335 μm²/s:

```
D_eff = 335 + (20 × 200)² / (48 × 335)
      = 335 + 1,600,000 / 16,080
      ≈ 1735 μm²/s    (5× enhancement)
```

Using the Fisher-KPP reaction-diffusion wave speed formula with CICR rate k ≈ 0.005 s⁻¹:

```
v_wave = 2√(D_eff × k) = 2√(1735 × 0.005) ≈ 5.9 μm/s  ✓
```

A flow velocity of ~20 μm/s — well within the range of Xenopus cilia-driven surface flows — produces sufficient Taylor dispersion to explain the observed speed. The required flow velocity for a given target speed:

| Flow velocity (μm/s) | D_eff (μm²/s) | Wave speed (μm/s) | Mechanism |
|---------------------|---------------|-------------------|-----------|
| 0 (pure diffusion) | 335 | ~0.4 | Baseline |
| 10 | 578 | ~1.7 | Weak enhancement |
| **20** | **1735** | **~5.9** | **Matches CEMA** |
| 50 | 7244 | ~12 | Over-prediction |

### 24.4 The Gap Geometry Argument: Simplest Resolution

The most important correction to the 15× speed discrepancy may be purely geometric. The earlier diffusion estimate assumed L = 1.4 mm (the embryo diameter). But if embryos are nearly touching in the CEMA channel, the actual inter-embryo medium gap is far smaller.

For touching spheres of diameter 1.4 mm, the gap at the point of closest approach is ≈ 0, but the effective diffusion distance through the narrow annular gap is approximately **100–200 μm**. For L = 200 μm:

```
t_diffusion = L² / (2D) = (200)² / (2 × 335) ≈ 60 s
v_apparent ≈ L / t = 200 μm / 60 s ≈ 3.3 μm/s
```

This is within **1.6× of 5.28 μm/s** from pure diffusion alone. Adding modest Taylor dispersion from U = 5 μm/s flow closes the remaining gap entirely. **The apparent 15× discrepancy may be largely an artifact of incorrect gap distance assumptions**, not a fundamental mechanistic puzzle.

### 24.5 Advection vs. Diffusion: Scaling Laws and Discriminating Experiments

The key discriminating feature is that advection and diffusion scale differently with medium viscosity:

```
Pure diffusion wave speed:  v ∝ √D ∝ 1/√η   (Stokes-Einstein: D ~ 1/η)
Pure advection speed:       v ∝ U ∝ 1/η       (cilia force ÷ viscous drag ~ 1/η)
Taylor dispersion:          D_eff ∝ U²/D ∝ η/η² = 1/η  (different dependence)
```

This provides a clean experimental discriminator:

| Experiment | Pure diffusion | Direct advection | Taylor dispersion |
|-----------|---------------|-----------------|-------------------|
| **Viscosity increase (5×)** | Speed × 1/√5 ≈ 0.45× | Speed × 1/5 = 0.2× | Speed × 1/5 = 0.2× |
| **μPIV particle tracking** | No net flow | Directed flow ≥ 5 μm/s | Shear flow visible |
| **Directional flow chamber** | Symmetric | Asymmetric (faster downstream) | Asymmetric |
| **Ciliary inhibition** (chlorpromazine) | No effect | Strong reduction | Strong reduction |
| **Geometric confinement** | Weak effect | Strong effect (Pe changes) | Strong effect |

**Recommended priority experiments:**

1. **μPIV (micro-particle image velocimetry)**: Add 0.5–1 μm fluorescent microspheres to 0.1× MMR, image inter-embryo gaps at high frame rate. Directly measures whether sustained fluid currents ≥ 5 μm/s exist. Most direct test with no ambiguity.

2. **Viscosity titration**: Add 0–2% methylcellulose (inert, non-ionic) to increase η by 2–10× without affecting cell physiology. Measure wave speed vs. viscosity. The exponent of the power law (−0.5 for diffusion, −1 for advection) discriminates mechanisms cleanly.

3. **Directional flow chamber**: Apply a controlled laminar flow (1–20 μm/s, achievable with a gentle peristaltic pump) across the embryo culture. If advection matters, wave propagation should be measurably faster downstream than upstream. If diffusion-only, propagation should be symmetric.

4. **Ciliary inhibition** (only valid if CEMA occurs post-stage 18): Apply chlorpromazine to inhibit dynein ATPase and stop cilia beating. Reduced wave speed implicates MCC-driven flow. Control: verify eATP-induced Ca²⁺ release in isolated embryos is unaffected by chlorpromazine.

5. **ATP scavenger kinetics**: Add apyrase at known concentrations. For pure diffusion, apyrase reduces effective D and wave speed predictably. For bulk flow (Pe >> 1), the signal arrives faster, partially escaping enzymatic degradation. Comparing observed apyrase IC50 to the pure-diffusion prediction quantifies the advective contribution.

### 24.6 Relationship to the eATP/CICR Relay Model

Bulk flow does not replace the eATP/CICR relay mechanism — it modifies the gap-crossing step. The revised relay model with advection is:

```
CICR wave sweeps embryo A at ~6.7 μm/s  (ER-mediated, intra-cellular)
    ↓
eATP released at A's surface into the medium
    ↓
eATP crosses inter-embryo gap via:
    ├── Passive diffusion (D ≈ 335 μm²/s, gap ≈ 100–200 μm)    → ~60 s
    └── + Taylor dispersion from cilia-driven flow (U ≈ 10–20 μm/s)  → ~12–30 s
    ↓
eATP binds P2X/P2Y on embryo B → CICR → sweeps B at ~2.36 μm/s
```

The measured 5.28 μm/s is still dominated by intra-embryo A CICR transit time, but the gap-crossing step is faster than pure diffusion would suggest, consistent with bulk flow assistance.

Suramin sensitivity remains fully explained: P2R is required at the reception step regardless of whether eATP crosses the gap by diffusion or advection.

### 24.7 Updated Mechanism Scorecard

Adding bulk flow to the Section 22.10 scorecard:

| Mechanism | Speed | CEMA 5.28 μm/s | Suramin? | Solid wall? | Verdict |
|-----------|-------|----------------|----------|-------------|---------|
| eATP diffusion (L=1.4 mm) | ~0.36 μm/s | 15× too slow | ✓ | ✓ | Gap overestimated |
| eATP diffusion (L=200 μm) | ~3.3 μm/s | 1.6× too slow | ✓ | ✓ | Close — gap geometry matters |
| **eATP + bulk flow (Taylor)** | **~5–6 μm/s** | **✓** | **✓** | **✓** | **Plausible if cilia active** |
| Intra-A CICR + near-zero gap | ~6.7 μm/s | ✓ | ✓ | ✓ | Best fit (Section 22) |
| Electrophoresis (0.1× MMR) | ~0.009 μm/s | 600× too slow | ✗ | ✓ | Screened by ions |
| Ephaptic (swept source) | Source-limited | ✓ | ✗ | Partial | Suramin problem |

**Two mechanisms are now consistent with all CEMA data:** (1) intra-embryo CICR dominating the apparent inter-embryo speed (Section 22) and (2) eATP diffusion over a short gap (~200 μm) enhanced by Taylor dispersion from cilia-driven bulk flow. These are not mutually exclusive — both could operate simultaneously. The gap geometry correction (Section 23.4) is the single most important unresolved empirical question: how large is the actual medium gap between embryos in the CEMA channel setup?


## 25. Physical and Biological Analogies for Absorbing-BC CEMA

The absorbing-BC field model for CEMA obeys the steady-state diffusion equation on a finite domain with Dirichlet boundary conditions and interior sources:

```
D∇²F − γF + S = 0,    F|_boundary = 0
```

with mean steady-state field `⟨F⟩ ≈ e/(γ_F + 4D_F/L)`, where the `4D_F/L` term encodes the surface-to-volume ratio of the group. This mathematical structure appears across physics, ecology, and biology.

### 25.1 Nuclear Reactor Criticality (Exact Mathematical Isomorphism)

The neutron diffusion equation in a reactor core is identical in form:

```
D∇²Φ − Σ_a·Φ + S = 0,    Φ = 0 at extrapolated boundary
```

| CEMA model | Nuclear reactor |
|------------|----------------|
| Embryo stress emission e | Fission neutron production S |
| Signal diffusion D_F | Neutron diffusion D |
| Bulk decay γ_F | Neutron absorption Σ_a |
| Absorbing BC (F=0 outside) | Extrapolated zero-flux boundary |
| Surface/volume ratio 4/L | Geometric buckling B_g² ~ π²/L² |
| Rescue condition α·⟨F⟩ > 1 | Criticality condition k_eff > 1 |

A small lump of fissile material is **subcritical** — neutrons leak out through the surface faster than fission replaces them. Increasing the mass reduces the surface-to-volume ratio until boundary leakage no longer dominates. At the **critical mass**, interior neutron production sustains the chain reaction.

A small embryo group is a subcritical reactor: stress signal leaks out the edges. A large group reaches "criticality" where the interior field is strong enough to rescue. The nuclear physics literature is the only field where the surface-to-volume ratio mechanism is explicitly named and quantified through the **geometric buckling** formalism.

### 25.2 Penguin Huddling and Bergmann's Rule

Each penguin generates heat volumetrically; heat is lost through the huddle's outer surface. A lone penguin loses heat from all sides. In a large huddle, interior penguins are shielded — their heat accumulates because neighbors absorb the leakage. The benefit scales with the surface-to-volume ratio, exactly as `γ̄_eff = γ_F + 4D_F/L`.

Bergmann's rule (larger body size in colder climates) is the individual-organism analog: larger animals have lower surface-to-volume ratio and retain heat more efficiently.

The negative feedback paradox (Section 25.2) has a penguin analog: warm interior penguins reduce their metabolic rate, producing less heat, so the huddle core is slightly cooler than the "all-maximum-output" prediction.

### 25.3 Bacterial Quorum Sensing and Diffusion Sensing

Bacteria release autoinducer (AI) molecules into the medium. In a small colony, AI diffuses away — the signal leaks. In a large biofilm, interior cells are shielded from the open boundary and AI concentration builds up, triggering collective gene expression (bioluminescence, virulence factors, biofilm matrix production).

Redfield (2002) reframed quorum sensing as **diffusion sensing**: bacteria detect how well their local geometry retains secreted molecules, which is exactly what the absorbing-BC field model computes. Hense et al. extended this to **efficiency sensing**, which explicitly incorporates colony geometry and spatial clustering.

A key result from the quorum sensing literature: "the effect of a single non-reflecting boundary side was equivalent to a 100-fold lower cell concentration" — absorbing vs reflecting BCs matter enormously, consistent with our finding that reflecting BCs eliminate group-size dependence.

Key references:
- Redfield (2002), "Is quorum sensing a side effect of diffusion sensing?" (PMID: 12160634)
- Hense et al. (2007), "Does efficiency sensing unify diffusion and quorum sensing?" Nature Reviews Microbiology

### 25.4 KiSS Critical Patch Size (Ecological Twin)

The Kierstead-Slobodkin-Skellam (1953) model asks: what is the minimum habitat patch size for a population to persist? Organisms reproduce (= emission), diffuse, and die if they cross the patch boundary (= absorbing BC). Below the critical patch size `L_crit = π√(D/r)`, boundary losses exceed interior growth and the population goes extinct. Above it, the population persists.

One large patch always outperforms two smaller patches of equal total area — splitting increases the boundary-to-area ratio. This directly parallels the CEMA prediction that one group of 100 embryos rescues better than ten groups of 10.

### 25.5 Resonance and Frequency Matching

The resonance analogy captures a feature the other analogies miss: **selectivity**.

A tuning fork in isolation vibrates and decays — energy radiates away (absorbing BC). Place it next to another tuning fork of the same frequency, and they reinforce each other through the shared acoustic field. More tuning forks packed together → stronger standing wave in the interior, because interior forks are shielded from radiation loss.

The CEMA paper showed that embryos stressed by **different teratogens don't rescue each other** — only same-challenge embryos help. This is exactly **frequency matching**: tuning forks only resonate when they share the same natural frequency. The stress signal must be specific to the morphogenetic challenge for constructive interference to occur.

The "super-embryo" in this framing is a **resonant cavity** — a group large enough that the coherent stress signal builds up past the threshold for sustained morphogenetic rescue, analogous to a laser cavity reaching the lasing threshold through stimulated emission feedback.

### 25.6 Coupled Oscillator Arrays and Synchronization

The surface-to-volume mechanism for CEMA has direct parallels in the synchronization literature, though it is not typically framed in those terms:

**Genetic oscillator chips** (Weitz et al., PNAS 2017): On a microfluidic array of coupled synthetic gene oscillators, boundary compartments desynchronize first because they have fewer coupled neighbors. Phase-shift stripe patterns nucleate at the edges and propagate inward. This is the surface-to-volume mechanism applied to coupled oscillators on a lattice — the closest published analog to the embryo group rescue system.

**Somitogenesis critical size**: The vertebrate segmentation clock (coupled Notch/Wnt oscillators in the presomitic mesoderm) stops functioning when the tissue shrinks below a critical size at the end of development. The clock arrests not because individual cells stop oscillating, but because the tissue becomes too small to sustain collective synchronization — a direct demonstration of the surface-to-volume principle in developmental biology.

**RF resonator arrays** (Pogorzelski, JPL Monograph): In coupled oscillator antenna arrays, edge elements show significantly larger control voltage variation than interior elements. Engineers compensate by adding extra coupling at the boundary.

**Phonon edge modes** (Phys. Rev. B 81, 174117, 2010): In finite lattices, the fraction of boundary-influenced modes scales as the perimeter-to-area ratio. Larger lattices suppress edge-mode influence on bulk dynamics. This provides the most developed theoretical framework in condensed matter physics.

### 25.7 The Diffusive Field as a Virtual Governor (Cybernetics)

In cybernetics, a **virtual governor** is a system that exhibits regulatory behavior without an explicit controller — the regulation emerges from the structure of interactions (Ashby, *Design for a Brain*, 1952). The diffusive stress field with absorbing BCs is a concrete instance:

- **No embryo "knows" about the group.** Each simply secretes stress and responds to local concentration. There is no cell that measures group size, computes a threshold, or sends corrective instructions.
- **Yet the system regulates.** Below a critical group size, all embryos fail. Above it, rescue propagates from the interior outward in a pattern precisely predicted by lattice geometry.
- **The "governor" is the Laplacian eigenstructure.** The fundamental eigenmode sin(πi/(L+1))·sin(πj/(L+1)) determines which embryos see enough field to cross threshold. This mode is not computed by anyone — it is a consequence of diffusion + boundaries + collective emission.

This connects to three cybernetic principles:

1. **Ashby's Requisite Variety (at minimum).** The governor has one degree of freedom (field amplitude ∝ L²), which suffices because the regulatory task is also one-dimensional: partition the lattice into rescued/failed along the eigenmode profile.

2. **Structural Good Regulator (Conant-Ashby).** The steady-state field profile *is* a model of the lattice geometry — encoding distance from boundaries, dimensionality, and group size — constructed automatically by the physics of diffusion.

3. **Regulation by constraint, not computation.** The absorbing BC is the constraint that enables regulation. Without it (periodic BCs), the field would be spatially uniform and no differential regulation would occur. The boundary creates the non-uniform eigenmode that creates the interior/exterior distinction.

The critical group size N_crit is where the virtual governor "turns on." Below it, embryos are independent failing units (a governor without authority). Above it, they are a regulated collective with differential spatial outcomes. See [FIELD_RESCUE_ANALYSIS.md Section 8.6](FIELD_RESCUE_ANALYSIS.md#86-the-diffusive-field-as-a-virtual-governor) for a detailed analysis including connections to Levin's morphogenetic fields and the limits of the autopoietic interpretation.

4. **Topoiesis — production by place (not autopoiesis).** The system produces a functional boundary/bulk distinction from homogeneous initial conditions, but this is not autopoietic mutual production: all embryos emit at the same constant rate regardless of rescue status, so there is no causal feedback from rescue outcome to field generation. The boundary and bulk are co-effects of the field geometry, not mutual causes. We term this **topoiesis** (τόπος + ποίησις) — differentiation produced entirely by position in a geometric structure, without feedback, symmetry-breaking instability, or source asymmetry. The only symmetry-breaking element is the boundary condition. See [FIELD_RESCUE_ANALYSIS.md Section 8.6](FIELD_RESCUE_ANALYSIS.md#86-the-diffusive-field-as-a-virtual-governor) for detailed analysis including connections to Wolpert's positional information and Turing morphogenesis.

### 25.8 Summary: The Unifying Mathematical Structure

All of these phenomena — physical, biological, and cybernetic — share the same PDE and the same resolution:

```
D∇²F − γF + S = 0,    F|_boundary = 0

⟨F⟩ ≈ S / (γ + C·D/L²)
```

The term `C·D/L²` is the **geometric leakage** — it dominates for small domains and vanishes for large ones. Whether the field is neutron flux, body heat, autoinducer concentration, acoustic energy, or embryo stress signal, the physics is identical: **interior sources accumulate when the surface-to-volume ratio is small enough that boundary leakage can't drain them**.

| Analogy | Emission | Field | Boundary loss | Critical condition |
|---------|----------|-------|---------------|-------------------|
| Nuclear reactor | Fission neutrons | Neutron flux Φ | Surface leakage | k_eff > 1 |
| Penguin huddle | Metabolic heat | Temperature field | Radiative/convective loss | Body temp > survival threshold |
| Bacterial quorum | Autoinducer | AI concentration | Diffusion to exterior | [AI] > quorum threshold |
| KiSS ecology | Reproduction | Population density | Emigration to hostile zone | Growth > boundary loss |
| Resonant cavity | Vibration energy | Acoustic field | Radiation at edges | Gain > radiation loss |
| **CEMA embryos** | **Stress signal** | **Diffusive field F** | **Absorbing BC leakage** | **α·⟨F⟩ > 1** |

In every case, the same PDE acts as a **virtual governor** (Section 25.7): no individual unit computes the collective state, yet the system regulates which units exceed threshold and which do not. The eigenstructure of the boundary-value problem encodes the regulatory outcome before dynamics begin — a governor latent in the geometry, activated when the system is assembled.

#### The alternative route: bistable reaction-diffusion with reflecting BCs

The ATP reaction-diffusion model (`cellularFieldNetwork.computeATPRate`) achieves group-size-dependent rescue through a fundamentally different mechanism — **critical nucleus size** with reflecting BCs — requiring no absorbing boundary conditions at all. With a bistable cubic reaction term, group-size dependence arises from transient dynamics: whether a localized perturbation can nucleate and propagate as a traveling wave before diffusion dilutes it. The competition is governed by the Damköhler number Da = |R′(u*)| · L²/D, where rescue requires Da ~ π². See [FIELD_RESCUE_ANALYSIS.md Sections 8.7–8.8](FIELD_RESCUE_ANALYSIS.md#87-the-alternative-route-bistable-reaction-diffusion-with-reflecting-bcs) for detailed analysis including the nucleation mechanism, a comparison table of the two routes, and the unified Morphogenetic Damköhler framework.

#### Generalisation: the Morphogenetic Damköhler number

Despite their different mechanisms, the stress field and ATP models are limiting cases of a single general equation:

```
∂u/∂t = D∇²u + f(u),    D∂ₙu + hu = 0 at boundary
```

where h → ∞ gives absorbing BCs (this section's framework) and h → 0 gives reflecting BCs (the bistable route). Both routes share the same critical-size scaling, governed by the **Morphogenetic Damköhler number**:

```
Da_m = L² · κ / D ~ O(10) at threshold
```

where κ is the effective local activation rate (= S/θ − γ for the linear model; = |R′(u*)| for the bistable model). The critical domain size is L_c ∝ √(D/κ) in both cases — identical scaling, different mechanisms. The absorbing BC can be understood as an effective nonlinearity concentrated at the boundary, while bistable kinetics concentrate the nonlinearity in the bulk. Real biology presumably uses both: partially leaky boundaries and cooperative intracellular dynamics. See [FIELD_RESCUE_ANALYSIS.md Section 8.7](FIELD_RESCUE_ANALYSIS.md#87-unified-framework-the-morphogenetic-damk%C3%B6hler-number) for detailed analysis including testable predictions, connections to FKPP/Allen-Cahn theory, and limitations of the unification.

## 26. The Super-Embryo: Collective Agency and Emergent Wholeness

### 26.1 From Individuals to Collective Entity

The absorbing-BC CEMA model predicts a sharp transition: below the critical group size `L* = 4D_F/(α·e − γ_F)`, embryos are isolated individuals unable to rescue. Above it, the collective field sustains rescue that no individual could achieve alone. This transition has precedent at every biological scale:

| Scale | System | Critical threshold | Emergent capability |
|-------|--------|-------------------|-------------------|
| Molecular | Bacterial colony | Quorum concentration | Biofilm, virulence, bioluminescence |
| Cellular | Xenopus mesoderm (Gurdon 1988) | ~100 cells | Sustained muscle differentiation |
| Cellular | Dictyostelium | Starvation + density | Slug: phototaxis, thermotaxis, fruiting body |
| Organism | Xenopus embryos (CEMA) | ~25–100 embryos | Collective teratogen resistance |
| Colony | Ant/bee colonies | Colony size threshold | Thermoregulation, division of labor |

The structure is identical at every scale:

```
Individual units  --exceed critical size-->  Collective entity
     ↓                                           ↓
Cannot maintain                            Emergent capability
state alone                                unavailable to individuals
```

### 26.2 The Gurdon Community Effect: Cell-Level Twin of CEMA

Gurdon (1988) showed that isolated Xenopus mesoderm cells, even given the correct inductive signal, cannot maintain muscle differentiation — they revert. But groups of ≥100 cells sustain differentiation through paracrine positive feedback. The mathematical analysis (Saka et al., BMC Systems Biology, 2011) proved this is a bistability problem with a critical group-size saddle point: below the threshold, the differentiated state is unstable; above it, it's stable.

This is formally identical to the CEMA rescue condition `α·⟨F⟩ > 1`: below the critical group size, the rescue signal is too weak to shift the damping sigmoid past the tipping point.

Key references:
- Gurdon (1988), "A community effect in animal development," Nature (PMID: 3205305)
- Saka et al. (2011), "Theoretical basis of the community effect," BMC Systems Biology
- Lowell (2020), "Community effects in biology," Nature Reviews MCB

### 26.3 Evidence for the Hyper-Embryo

The CEMA paper (Tung et al., Nature Communications, 2024) provides three lines of evidence that the embryo group constitutes a genuinely new level of biological organization:

1. **Group-specific transcriptome**: RNA-seq revealed genes expressed only in large cohorts — not present in singletons or small groups. The group has its own molecular identity that individual embryos do not possess.

2. **Non-linear threshold**: Survival jumps from ~0% (5 embryos) to ~98% (300 embryos) — a phase transition, not gradual improvement.

3. **Morphogenetic specificity**: Embryos stressed by different teratogens don't help each other — only embryos facing the same challenge collectively rescue. This rules out generic buffering and implies instructive morphogenetic information flows between embryos (cf. the resonance/frequency-matching analogy, Section 26.5).

Michael Levin uses the term **"hyper-embryo"** for this collective entity and frames it within a "hyper-developmental biology" where groups of embryos solve problems in morphospace that individual embryos cannot.

Key reference:
- Tung et al. (2024), "Embryos assist morphogenesis of others through calcium and ATP signaling mechanisms in collective teratogen resistance," Nature Communications (PMID: 38233424)

### 26.4 Proposed Metrics for Measuring Collective Wholeness

Drawing on information-theoretic metrics from the flocking literature (Maisto, Nuzzi & Pezzulo, "What the flock knows that the birds do not," arXiv:2511.10835), we propose the following metrics for quantifying the degree to which an embryo group becomes a "super-embryo," ranked by implementation complexity:

#### 26.4.1 Pattern Coherence (Easy)

Measures how aligned neighboring embryos' Vmem patterns are:

```
ℋ = (1/|edges|) · Σ_{(i,j) neighbors} cos_similarity(Vmem_i, Vmem_j)
```

ℋ → 1 when all embryos converge to similar healthy patterns (integrated collective). ℋ → 0 when patterns are uncorrelated (mere aggregation). Computable from a single simulation run using the existing Vmem vectors and adjacency matrix.

#### 26.4.2 Rescue Rate (Easy — Already Implemented)

```
rescue_rate(t) = |{i : similarity_i(t) > threshold}| / N
```

See Section 20.8. The rescue rate is a binary collective outcome measure: how many embryos achieve the healthy state that none could achieve individually (in a damaged group).

#### 26.4.3 Algebraic Connectivity λ₂ (Medium)

Build a functional connectivity graph where edge weight = correlation between Vmem timeseries of embryos i and j. The **Fiedler eigenvalue** (second-smallest eigenvalue of the graph Laplacian) measures how tightly integrated the collective is:

```
L = D − W    (weighted graph Laplacian)
λ₂ = second smallest eigenvalue of L
```

λ₂ = 0 means disconnected components (no collective). λ₂ large means tightly integrated whole. The corresponding **Fiedler eigenvector** identifies which embryos are functionally integrated vs. which are "outsiders" — analogous to identifying the Markov blanket of the collective.

#### 26.4.4 Synergistic Information (Hard — Requires Ensemble)

From the partial information decomposition (PID) framework, **synergy** measures what the group knows that individuals don't:

```
S_ij = I(X_i, X_j; Y) − I(X_i; Y) − I(X_j; Y) + min{I(X_i; Y), I(X_j; Y)}
```

where X_i = embryo i's Vmem state and Y = rescue outcome (binary). S > 0 means rescue predictability exists only when examining embryos jointly, not individually. This is the most theoretically rigorous "wholeness" metric but requires an ensemble of simulation runs with different random seeds to estimate the mutual information terms.

**Prediction**: Below the critical group size, S ≈ 0 (each embryo's fate is independent). Above it, S > 0 (the collective field creates correlations that make rescue predictable only from the joint state).

#### 26.4.5 The Super-Embryo Transition

The transition from "collection of struggling individuals" to "super-embryo" would be visible as:

- Below critical size: ℋ ≈ 0, λ₂ ≈ 0, rescue_rate ≈ 0, S ≈ 0
- Above critical size: ℋ → 1, λ₂ jumps, rescue_rate → 1, S > 0

The critical group size `L* = 4D_F/(α·e − γ_F)` is the embryo-scale analog of the critical mass, the quorum threshold, and the Gurdon minimum cell number. Below L*, embryos are isolated individuals. Above L*, they are a collective entity with emergent rescue capability.

### 26.5 Gap in the Literature

No published paper explicitly connects:
1. Surface-to-volume ratio (nuclear physics / geometric buckling)
2. Boundary-leakage-driven synchronization failure (coupled oscillator physics)
3. Critical group size for morphogenetic rescue (CEMA developmental biology)

into a unified framework. The absorbing-BC field model with `γ̄_eff = γ_F + 4D_F/L` provides this unification — a single formula explaining why small groups fail and large groups succeed, with the mechanism being geometric boundary leakage rather than any change in individual properties. The closest published work in each domain:

- **Nuclear physics**: Geometric buckling formalism (textbooks; nuclear-power.com)
- **Quorum sensing**: Redfield (2002), diffusion sensing; Hense et al. (2007), efficiency sensing
- **Synchronization**: Weitz et al. (PNAS 2017), genetic oscillator boundary effects
- **Developmental biology**: Gurdon (1988), community effect; Tung et al. (2024), CEMA
- **Ecology**: Kierstead-Slobodkin-Skellam (1953), critical patch size

The CEMA model may represent one of the clearest biological demonstrations of the geometric leakage principle, bridging the gap between the nuclear physics framing and developmental biology observations.
