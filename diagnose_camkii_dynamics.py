"""
Diagnostic script to understand why CaMKII bistability isn't developing spatial features.

This script tests the dynamics at a single timestep to understand parameter sensitivities.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

def analyze_dynamics():
    """Analyze CaMKII dynamics with current learned parameters."""

    # Load learned parameters
    data = torch.load('./data/bestLearnedCaMKIIParams_0.dat')
    params_raw = data['parameters']
    bounds = data['parameter_bounds']

    # Extract actual parameter values
    def apply_sigmoid_constraint(raw_val, min_val, max_val):
        return min_val + (max_val - min_val) * torch.sigmoid(raw_val)

    k_on = apply_sigmoid_constraint(params_raw['k_on'], bounds['k_on_min'], bounds['k_on_max']).item()
    k_off = apply_sigmoid_constraint(params_raw['k_off'], bounds['k_off_min'], bounds['k_off_max']).item()
    V_half = apply_sigmoid_constraint(params_raw['V_half'], bounds['V_half_min'], bounds['V_half_max']).item()
    k_vmem = apply_sigmoid_constraint(params_raw['k_vmem'], bounds['k_vmem_min'], bounds['k_vmem_max']).item()
    or_threshold = apply_sigmoid_constraint(params_raw['or_threshold'], bounds['or_threshold_min'], bounds['or_threshold_max']).item()
    or_sharpness = apply_sigmoid_constraint(params_raw['or_sharpness'], bounds['or_sharpness_min'], bounds['or_sharpness_max']).item()

    print("=" * 70)
    print("CAMKII DYNAMICS ANALYSIS")
    print("=" * 70)
    print(f"\nLearned Parameters:")
    print(f"  k_on = {k_on:.4f}")
    print(f"  k_off = {k_off:.4f}")
    print(f"  V_half = {V_half:.4f} V")
    print(f"  k_vmem = {k_vmem:.4f} V")
    print(f"  or_threshold = {or_threshold:.4f}")
    print(f"  or_sharpness = {or_sharpness:.4f}")

    # Test Vmem range from stigmergic simulation
    vmem_depolarized = -0.01  # Feature regions (depolarized)
    vmem_hyperpolarized = -0.05  # Background (hyperpolarized)

    print(f"\n" + "=" * 70)
    print("VMEM SIGNAL ANALYSIS")
    print("=" * 70)

    # Compute vmem_signal for both regions
    vmem_sig_depol = torch.sigmoid(torch.tensor((vmem_depolarized - V_half) / k_vmem)).item()
    vmem_sig_hyper = torch.sigmoid(torch.tensor((vmem_hyperpolarized - V_half) / k_vmem)).item()

    print(f"\nDepolarized regions (Vmem = {vmem_depolarized:.4f} V):")
    print(f"  vmem_signal = {vmem_sig_depol:.6f}")

    print(f"\nHyperpolarized regions (Vmem = {vmem_hyperpolarized:.4f} V):")
    print(f"  vmem_signal = {vmem_sig_hyper:.6f}")

    print(f"\nSpatial contrast in vmem_signal: {vmem_sig_depol - vmem_sig_hyper:.6f}")

    if abs(vmem_sig_depol - vmem_sig_hyper) < 0.1:
        print("  ⚠️  WARNING: Very low spatial contrast! Vmem signal is nearly uniform.")
        print(f"     k_vmem={k_vmem:.4f} is too small, making sigmoid too steep.")
        print(f"     Vmem range is ~{abs(vmem_depolarized - vmem_hyperpolarized):.4f}V")
        print(f"     Recommend k_vmem ~ {abs(vmem_depolarized - vmem_hyperpolarized)/4:.4f}V for good contrast")

    print(f"\n" + "=" * 70)
    print("BISTABILITY ANALYSIS")
    print("=" * 70)

    # Test equilibrium at different CaMKII values
    CaMKII_values = np.linspace(0, 1, 11)

    print(f"\nWith vmem_signal = {vmem_sig_depol:.4f} (depolarized):")
    print(f"{'CaMKII':<10} {'activation':<12} {'dCaMKII/dt':<15} {'Equilibrium?':<15}")
    print("-" * 70)

    for CaMKII_val in CaMKII_values:
        combined = or_sharpness * (vmem_sig_depol + CaMKII_val - or_threshold)
        activation = max(0, combined) / or_sharpness
        dCaMKII_dt = k_on * activation - k_off * CaMKII_val

        is_eq = "YES ✓" if abs(dCaMKII_dt) < 0.01 else ""
        print(f"{CaMKII_val:<10.2f} {activation:<12.4f} {dCaMKII_dt:<15.4f} {is_eq:<15}")

    print(f"\nWith vmem_signal = {vmem_sig_hyper:.4f} (hyperpolarized):")
    print(f"{'CaMKII':<10} {'activation':<12} {'dCaMKII/dt':<15} {'Equilibrium?':<15}")
    print("-" * 70)

    for CaMKII_val in CaMKII_values:
        combined = or_sharpness * (vmem_sig_hyper + CaMKII_val - or_threshold)
        activation = max(0, combined) / or_sharpness
        dCaMKII_dt = k_on * activation - k_off * CaMKII_val

        is_eq = "YES ✓" if abs(dCaMKII_dt) < 0.01 else ""
        print(f"{CaMKII_val:<10.2f} {activation:<12.4f} {dCaMKII_dt:<15.4f} {is_eq:<15}")

    # Find exact equilibria
    print(f"\n" + "=" * 70)
    print("EQUILIBRIUM POINTS")
    print("=" * 70)

    def find_equilibrium(vmem_sig):
        """Find CaMKII equilibrium for given vmem_signal."""
        # At equilibrium: k_on * activation(CaMKII) = k_off * CaMKII
        # activation = relu(or_sharpness * (vmem_sig + CaMKII - or_threshold)) / or_sharpness

        # Try to find equilibria numerically
        CaMKII_test = np.linspace(-0.5, 2.0, 1000)
        equilibria = []

        for CaMKII_val in CaMKII_test:
            combined = or_sharpness * (vmem_sig + CaMKII_val - or_threshold)
            activation = max(0, combined) / or_sharpness
            dCaMKII_dt = k_on * activation - k_off * CaMKII_val

            if abs(dCaMKII_dt) < 0.001:
                equilibria.append((CaMKII_val, dCaMKII_dt))

        return equilibria

    eq_depol = find_equilibrium(vmem_sig_depol)
    eq_hyper = find_equilibrium(vmem_sig_hyper)

    print(f"\nDepolarized regions (vmem_signal = {vmem_sig_depol:.4f}):")
    if eq_depol:
        for CaMKII_eq, dCdt in eq_depol:
            in_range = " ✓ in [0,1]" if 0 <= CaMKII_eq <= 1 else f" ✗ OUTSIDE [0,1]"
            print(f"  CaMKII_eq = {CaMKII_eq:.4f}{in_range}")
    else:
        print(f"  No equilibrium found in tested range")

    print(f"\nHyperpolarized regions (vmem_signal = {vmem_sig_hyper:.4f}):")
    if eq_hyper:
        for CaMKII_eq, dCdt in eq_hyper:
            in_range = " ✓ in [0,1]" if 0 <= CaMKII_eq <= 1 else f" ✗ OUTSIDE [0,1]"
            print(f"  CaMKII_eq = {CaMKII_eq:.4f}{in_range}")
    else:
        print(f"  No equilibrium found in tested range")

    # Check k_on/k_off ratio
    print(f"\n" + "=" * 70)
    print("PARAMETER RATIO ANALYSIS")
    print("=" * 70)

    ratio = k_on / k_off
    print(f"\nk_on / k_off = {ratio:.2f}")

    if ratio > 5:
        print(f"  ⚠️  WARNING: High ratio pushes equilibrium toward saturation")
        print(f"     With full activation, CaMKII_eq ≈ {ratio:.2f} (>> 1)")
        print(f"     System will saturate at upper bound")

    # Plot nullclines
    print(f"\n" + "=" * 70)
    print("GENERATING PHASE PORTRAIT")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    CaMKII_range = np.linspace(-0.2, 1.5, 200)

    for idx, (vmem_sig, title) in enumerate([(vmem_sig_depol, "Depolarized"),
                                               (vmem_sig_hyper, "Hyperpolarized")]):
        ax = axes[idx]

        # Compute dCaMKII/dt
        dCaMKII_dt_vals = []
        for CaMKII_val in CaMKII_range:
            combined = or_sharpness * (vmem_sig + CaMKII_val - or_threshold)
            activation = max(0, combined) / or_sharpness
            dCaMKII_dt = k_on * activation - k_off * CaMKII_val
            dCaMKII_dt_vals.append(dCaMKII_dt)

        ax.plot(CaMKII_range, dCaMKII_dt_vals, 'b-', linewidth=2, label='dCaMKII/dt')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3, label='Nullcline')
        ax.axvline(0, color='gray', linestyle=':', alpha=0.3)
        ax.axvline(1, color='gray', linestyle=':', alpha=0.3)
        ax.axvspan(0, 1, alpha=0.1, color='green', label='[0,1] range')

        ax.set_xlabel('CaMKII', fontsize=12)
        ax.set_ylabel('dCaMKII/dt', fontsize=12)
        ax.set_title(f'{title} (vmem_signal={vmem_sig:.4f})', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Mark equilibria
        eq = find_equilibrium(vmem_sig)
        for CaMKII_eq, _ in eq:
            color = 'green' if 0 <= CaMKII_eq <= 1 else 'red'
            ax.plot(CaMKII_eq, 0, 'o', color=color, markersize=10,
                   markeredgecolor='black', markeredgewidth=2)

    plt.tight_layout()
    plt.savefig('camkii_phase_portrait.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Phase portrait saved to: camkii_phase_portrait.png")

    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nKey findings:")
    print(f"1. Vmem spatial contrast: {abs(vmem_sig_depol - vmem_sig_hyper):.4f}")
    if abs(vmem_sig_depol - vmem_sig_hyper) < 0.1:
        print(f"   → PROBLEM: Too low! Need larger k_vmem")

    print(f"2. k_on/k_off ratio: {ratio:.2f}")
    if ratio > 5:
        print(f"   → PROBLEM: Too high! Equilibria outside [0,1]")

    print(f"3. Number of equilibria in [0,1]:")
    eq_depol_in_range = [eq for eq in eq_depol if 0 <= eq[0] <= 1]
    eq_hyper_in_range = [eq for eq in eq_hyper if 0 <= eq[0] <= 1]
    print(f"   Depolarized: {len(eq_depol_in_range)}")
    print(f"   Hyperpolarized: {len(eq_hyper_in_range)}")
    if len(eq_depol_in_range) == 0 or len(eq_hyper_in_range) == 0:
        print(f"   → PROBLEM: No stable equilibria in valid range!")

    print(f"\n" + "=" * 70)

if __name__ == "__main__":
    analyze_dynamics()
