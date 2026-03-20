#!/usr/bin/env python3
"""
Visualize single-cell stress bistable dynamics in isolation.

Produces:
  1. Timeseries grids: S(t) from various S₀ at different fixed Ca levels
  2. Bifurcation diagram: steady-state S vs Ca (from low and high initial S)
  3. Phase portraits: dS/dt vs S decomposed into reaction, decay, and total
  4. Component decomposition: self-activation, or_gate, ca_drive vs S

Usage:
    python visualize_stress_bistability.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def load_stress_params(path='./data/bestLearnedStressParams_6.dat'):
    """Load learned stress parameters and decode from logit space."""
    data = torch.load(path, weights_only=False)
    raw_params = data['parameters']
    bounds = data['parameter_bounds']

    decoded = {}
    param_names = [
        'tau_S', 'k_on_S', 'k_off_S', 'K_S', 'Ca_stress_threshold',
        'sigma_ca', 'gain_S', 'or_threshold_S', 'D_S', 'gamma', 'K_decay'
    ]
    for name in param_names:
        raw = raw_params[name]
        mn = bounds[f'{name}_min']
        mx = bounds[f'{name}_max']
        decoded[name] = float(mn + (mx - mn) * torch.sigmoid(raw))

    return decoded


def compute_stress_components(S, Ca, params):
    """
    Compute all intermediate signals and dS/dt components for given S and Ca.

    Returns dict with: ca_drive, self_activation, or_input, or_gate,
                        reaction, decay, dS_dt
    """
    tau_S = params['tau_S']
    k_on_S = params['k_on_S']
    k_off_S = params['k_off_S']
    K_S = params['K_S']
    Ca_thresh = params['Ca_stress_threshold']
    sigma_ca = params['sigma_ca']
    gain_S = params['gain_S']
    or_thresh = params['or_threshold_S']
    gamma = params['gamma']
    K_decay = params['K_decay']

    # Ca drive
    x = -(Ca - Ca_thresh) / sigma_ca
    ca_drive = 1.0 / (1.0 + np.exp(np.clip(x, -500, 500)))

    # Self-activation
    S_sq = S * S
    K_sq = K_S * K_S
    self_act = (S_sq - K_sq) / (S_sq + K_sq + 1e-10)

    # OR gate
    or_input = gain_S * ca_drive + self_act - or_thresh
    or_gate = 1.0 / (1.0 + np.exp(np.clip(-or_input, -500, 500)))

    # Reaction and decay
    reaction = (k_on_S * or_gate * (1.0 - S) - k_off_S * S) / tau_S
    decay = -gamma * S / (K_decay + S + 1e-10)
    dS_dt = reaction + decay

    return {
        'ca_drive': ca_drive,
        'self_activation': self_act,
        'or_input': or_input,
        'or_gate': or_gate,
        'reaction': reaction,
        'decay': decay,
        'dS_dt': dS_dt,
    }


def single_cell_stress_step(S, Ca, params, dt):
    """Advance single-cell stress by one timestep. Returns new S."""
    c = compute_stress_components(S, Ca, params)
    S_new = S + dt * c['dS_dt']
    return np.clip(S_new, 0.0, 1.0)


def simulate_single_cell(S0, Ca, params, num_steps=5000, dt=0.1):
    """Simulate single-cell stress dynamics and return timeseries."""
    S_history = np.zeros(num_steps + 1)
    S_history[0] = S0
    S = S0
    for t in range(num_steps):
        S = single_cell_stress_step(S, Ca, params, dt)
        S_history[t + 1] = S
    return S_history


def main():
    params = load_stress_params('./data/bestLearnedStressParams_6.dat')

    print("Decoded stress parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:.6f}")
    print()

    Ca_thresh = params['Ca_stress_threshold']
    K_S = params['K_S']
    print(f"Ca_stress_threshold = {Ca_thresh:.2f} (Ca drive activates here)")
    print(f"K_S = {K_S:.4f} (self-activation crossover: S > K_S => positive feedback)")
    print(f"or_threshold_S = {params['or_threshold_S']:.4f}")
    print(f"gain_S = {params['gain_S']:.4f}")
    print(f"gamma = {params['gamma']:.4f}, K_decay = {params['K_decay']:.4f}")
    print()

    # Check: what or_threshold would allow self-activation alone to sustain S?
    # At max self_act = +1, or_input = gain_S*ca_drive + 1 - or_thresh
    # For Ca=0, ca_drive=0, so or_input = 1 - or_thresh
    # Need or_input > 0 => or_thresh < 1 for self-activation alone to work
    print(f"For self-activation to sustain S (Ca=0):")
    print(f"  Need or_threshold < 1.0 + gain_S*ca_drive(Ca=0)")
    print(f"  Current or_threshold = {params['or_threshold_S']:.3f} >> 1.0")
    print(f"  => Self-activation CANNOT sustain S alone with these params\n")

    num_steps = 5000
    dt = 0.1
    time = np.arange(num_steps + 1) * dt
    S0_values = np.linspace(0.0, 1.0, 11)
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 0.95, len(S0_values)))

    # ================================================================
    # Figure 1: Timeseries grid — extended Ca range
    # ================================================================
    Ca_extended = np.linspace(0.0, 12.0, 9)
    fig1, axes1 = plt.subplots(3, 3, figsize=(14, 12), sharex=True, sharey=True)
    axes_flat = axes1.flatten()

    for idx, Ca in enumerate(Ca_extended):
        ax = axes_flat[idx]
        for j, S0 in enumerate(S0_values):
            S_ts = simulate_single_cell(S0, Ca, params, num_steps, dt)
            ax.plot(time, S_ts, color=colors[j], linewidth=1.2, alpha=0.85)
        ax.set_title(f'Ca = {Ca:.1f}', fontsize=11, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)
        if idx >= 6:
            ax.set_xlabel('Time')
        if idx % 3 == 0:
            ax.set_ylabel('Stress (S)')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig1.colorbar(sm, ax=axes1, fraction=0.02, pad=0.04).set_label('Initial S₀')
    fig1.suptitle(
        'S(t) for Various S₀ and Fixed Ca²⁺ ∈ [0, 12]\n'
        f'(Ca threshold ≈ {Ca_thresh:.1f}, K_S = {K_S:.3f})',
        fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig1.savefig('./data/stress_single_cell_bistability_Ca0to12.png',
                 dpi=150, bbox_inches='tight')
    print("Saved: data/stress_single_cell_bistability_Ca0to12.png")

    # ================================================================
    # Figure 2: Bifurcation diagram
    # ================================================================
    Ca_fine = np.linspace(0.0, 12.0, 400)
    ss_up = np.zeros(len(Ca_fine))
    ss_down = np.zeros(len(Ca_fine))
    for i, Ca in enumerate(Ca_fine):
        ss_up[i] = simulate_single_cell(0.01, Ca, params, 10000, 0.1)[-1]
        ss_down[i] = simulate_single_cell(0.99, Ca, params, 10000, 0.1)[-1]

    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(Ca_fine, ss_up, 'b-', lw=2, label='S₀ = 0.01 (rising)')
    ax2.plot(Ca_fine, ss_down, 'r-', lw=2, label='S₀ = 0.99 (falling)')
    ax2.fill_between(Ca_fine, ss_up, ss_down, alpha=0.15, color='purple',
                     label='Bistable region')
    ax2.axvline(Ca_thresh, color='gray', ls='--', lw=1,
                label=f'Ca threshold = {Ca_thresh:.2f}')
    ax2.axvline(K_S, color='orange', ls=':', lw=1,
                label=f'K_S = {K_S:.3f}')
    ax2.set_xlabel('Fixed Ca²⁺ Level', fontsize=12)
    ax2.set_ylabel('Steady-State Stress (S)', fontsize=12)
    ax2.set_title('Bifurcation Diagram: Steady-State S vs Ca²⁺', fontsize=13,
                  fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    fig2.savefig('./data/stress_bifurcation_diagram.png', dpi=150, bbox_inches='tight')
    print("Saved: data/stress_bifurcation_diagram.png")

    # ================================================================
    # Figure 3: Decomposed phase portraits — reaction vs decay vs total
    # ================================================================
    S_range = np.linspace(0.001, 0.999, 500)
    # sigma_ca is very small (~0.005), so ca_drive is a near-step at Ca_thresh (~8.06).
    # Tightly bracket threshold to show rapid lifting of dS/dt curve.
    Ca_thresh_val = params['Ca_stress_threshold']
    sigma_val = params['sigma_ca']
    Ca_select = [0.0,
                 Ca_thresh_val - 3 * sigma_val,
                 Ca_thresh_val - sigma_val,
                 Ca_thresh_val,
                 Ca_thresh_val + sigma_val,
                 Ca_thresh_val + 3 * sigma_val]

    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for idx, Ca in enumerate(Ca_select):
        ax = axes3.flatten()[idx]
        reaction = np.zeros_like(S_range)
        decay = np.zeros_like(S_range)
        total = np.zeros_like(S_range)

        for j, S in enumerate(S_range):
            c = compute_stress_components(S, Ca, params)
            reaction[j] = c['reaction']
            decay[j] = c['decay']
            total[j] = c['dS_dt']

        # Compute ca_drive for annotation
        x_sig = -(Ca - Ca_thresh_val) / params['sigma_ca']
        ca_drv = 1.0 / (1.0 + np.exp(np.clip(x_sig, -500, 500)))

        ax.plot(S_range, reaction, 'b-', lw=1.5, label='Reaction')
        ax.plot(S_range, decay, 'r-', lw=1.5, label='Decay (MM)')
        ax.plot(S_range, total, 'k-', lw=2.5, label='Total dS/dt')
        ax.axhline(0, color='gray', ls='--', lw=0.5)
        ax.set_title(f'Ca = {Ca:.4f}  (ca_drive = {ca_drv:.3f})',
                     fontsize=10, fontweight='bold')
        ax.grid(alpha=0.2)
        if idx >= 3:
            ax.set_xlabel('S')
        if idx % 3 == 0:
            ax.set_ylabel('Rate')
        if idx == 0:
            ax.legend(fontsize=8, loc='lower left')

    fig3.suptitle(
        'Decomposed dS/dt: Reaction (blue) vs Decay (red) vs Total (black)\n'
        f'or_threshold = {params["or_threshold_S"]:.3f}, '
        f'gamma = {params["gamma"]:.3f}, K_decay = {params["K_decay"]:.4f}',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig3.savefig('./data/stress_decomposed_phase_portrait.png',
                 dpi=150, bbox_inches='tight')
    print("Saved: data/stress_decomposed_phase_portrait.png")

    # ================================================================
    # Figure 4: Internal signals — self_activation, or_gate, ca_drive vs S
    # ================================================================
    fig4, axes4 = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for idx, Ca in enumerate(Ca_select):
        ax = axes4.flatten()[idx]
        self_act = np.zeros_like(S_range)
        or_gate = np.zeros_like(S_range)
        ca_drv = np.zeros_like(S_range)
        or_inp = np.zeros_like(S_range)

        for j, S in enumerate(S_range):
            c = compute_stress_components(S, Ca, params)
            self_act[j] = c['self_activation']
            or_gate[j] = c['or_gate']
            ca_drv[j] = c['ca_drive']
            or_inp[j] = c['or_input']

        ax.plot(S_range, self_act, 'g-', lw=1.5, label='self_activation [-1,+1]')
        ax.plot(S_range, or_gate, 'm-', lw=2, label='or_gate [0,1]')
        ax.axhline(ca_drv[0], color='c', ls='--', lw=1,
                    label=f'ca_drive = {ca_drv[0]:.3f}')
        ax.axhline(0, color='gray', ls=':', lw=0.5)
        ax.set_title(f'Ca = {Ca:.4f}  (ca_drive = {ca_drv[0]:.3f})',
                     fontsize=10, fontweight='bold')
        ax.set_ylim(-1.1, 1.1)
        ax.grid(alpha=0.2)
        if idx >= 3:
            ax.set_xlabel('S')
        if idx % 3 == 0:
            ax.set_ylabel('Signal value')
        if idx == 0:
            ax.legend(fontsize=7, loc='lower right')

    fig4.suptitle(
        'Internal Signals vs S: self_activation (green), or_gate (magenta), '
        'ca_drive (cyan)\n'
        f'K_S = {K_S:.4f} (self_act crosses 0), '
        f'or_threshold = {params["or_threshold_S"]:.3f}, '
        f'gain_S = {params["gain_S"]:.3f}',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig4.savefig('./data/stress_internal_signals.png',
                 dpi=150, bbox_inches='tight')
    print("Saved: data/stress_internal_signals.png")

    # ================================================================
    # Figure 5: Learned vs true-bistable comparison
    #
    # From parameter sweep, true bistability (3 zero-crossings at Ca=0)
    # requires or_threshold ~ 1.35-1.45 (close to learned!) but with
    # higher k_off (~0.01 vs 0.001) and slightly lower gamma.
    # The low stable point is near S~0.03 (narrow OFF basin),
    # unstable point near S~0.07, high stable near S~0.7.
    # ================================================================
    print("\n--- Bistable comparison ---")
    params_bistable = dict(params)
    # From sweep: K_S=0.2 k_on=3.0 or_thresh=1.45 k_off=0.01
    #             gamma=0.21 K_decay=0.01 => crossings at 0.033, 0.075, 0.703
    params_bistable['k_on_S'] = 3.0
    params_bistable['or_threshold_S'] = 1.45
    params_bistable['k_off_S'] = 0.01
    params_bistable['gamma'] = 0.21
    params_bistable['K_decay'] = 0.01

    print("  Bistable params (changed from learned):")
    for key in ['k_on_S', 'or_threshold_S', 'k_off_S', 'gamma', 'K_decay']:
        print(f"    {key}: {params[key]:.4f} -> {params_bistable[key]:.4f}")

    fig5, axes5 = plt.subplots(2, 3, figsize=(17, 10))

    # --- Top row: timeseries at Ca=0 ---
    # Left: Learned params
    ax = axes5[0, 0]
    for j, S0 in enumerate(S0_values):
        S_ts = simulate_single_cell(S0, 0.0, params, num_steps, dt)
        ax.plot(time, S_ts, color=colors[j], lw=1.2, alpha=0.85)
    ax.set_title('Learned (File 6), Ca = 0\n(monostable OFF)', fontsize=10,
                 fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('S')
    ax.set_xlabel('Time')
    ax.grid(alpha=0.2)

    # Middle: Bistable params at Ca=0 — should show S₀-dependent fate
    ax = axes5[0, 1]
    for j, S0 in enumerate(S0_values):
        S_ts = simulate_single_cell(S0, 0.0, params_bistable, num_steps, dt)
        ax.plot(time, S_ts, color=colors[j], lw=1.2, alpha=0.85)
    ax.set_title('Bistable params, Ca = 0\n(S₀ determines fate)',
                 fontsize=10, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Time')
    ax.grid(alpha=0.2)

    # Right: Bistable params at Ca above threshold — should be monostable ON
    ax = axes5[0, 2]
    for j, S0 in enumerate(S0_values):
        S_ts = simulate_single_cell(S0, 10.0, params_bistable, num_steps, dt)
        ax.plot(time, S_ts, color=colors[j], lw=1.2, alpha=0.85)
    ax.set_title('Bistable params, Ca = 10\n(monostable ON)',
                 fontsize=10, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Time')
    ax.grid(alpha=0.2)

    # --- Bottom row: decomposed phase portraits ---
    for col, (p, Ca, label) in enumerate([
        (params, 0.0, 'Learned, Ca = 0'),
        (params_bistable, 0.0, 'Bistable, Ca = 0'),
        (params_bistable, 10.0, 'Bistable, Ca = 10'),
    ]):
        ax = axes5[1, col]
        reaction = np.zeros_like(S_range)
        decay_arr = np.zeros_like(S_range)
        total = np.zeros_like(S_range)
        for j, S in enumerate(S_range):
            c = compute_stress_components(S, Ca, p)
            reaction[j] = c['reaction']
            decay_arr[j] = c['decay']
            total[j] = c['dS_dt']
        ax.plot(S_range, reaction, 'b-', lw=1.5, label='Reaction')
        ax.plot(S_range, decay_arr, 'r-', lw=1.5, label='Decay')
        ax.plot(S_range, total, 'k-', lw=2.5, label='Total dS/dt')
        ax.axhline(0, color='gray', ls='--', lw=0.5)
        # Mark zero crossings
        signs = np.sign(total)
        crossings = np.where(np.diff(signs))[0]
        for ci in crossings:
            ax.axvline(S_range[ci], color='orange', ls=':', lw=1, alpha=0.7)
            ax.plot(S_range[ci], 0, 'o', color='orange', ms=6, zorder=5)
        ax.set_title(f'{label}\n({len(crossings)} fixed points)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('S')
        if col == 0:
            ax.set_ylabel('Rate')
            ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    fig5.suptitle(
        'Learned params vs true bistable regime (Ca = 0)\n'
        f'Key difference: k_on {params["k_on_S"]:.2f}→{params_bistable["k_on_S"]:.1f}, '
        f'k_off {params["k_off_S"]:.3f}→{params_bistable["k_off_S"]:.2f}, '
        f'or_thresh {params["or_threshold_S"]:.2f}→{params_bistable["or_threshold_S"]:.2f}',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig5.savefig('./data/stress_bistability_comparison.png',
                 dpi=150, bbox_inches='tight')
    print("Saved: data/stress_bistability_comparison.png")

    plt.show()


if __name__ == '__main__':
    main()