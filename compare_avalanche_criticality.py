# Side-by-side comparison of avalanche criticality:
# Stigmergic bioelectric model vs Schnakenberg Turing system
#
# Loads saved results from both analyses and produces a comparison figure.

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from avalanche_analysis import fit_power_law_mle, compute_ccdf

def main():
    # Load stigmergic results
    stig = torch.load('data/stigmergic_avalanche_results.dat', weights_only=False)
    stig_rbt = stig['results_by_time']
    stig_times = stig['perturb_times']

    # Load Turing results (re-run extraction since npz doesn't store dicts well)
    # We'll just re-extract from the last run's console output embedded in the script
    # Actually, let's load what we saved and also compute from the raw data
    turing_npz = np.load('data/turing_avalanche_results.npz', allow_pickle=True)

    # Since the npz doesn't store the results_by_time dict well, let's
    # re-run the Turing analysis inline to get the numbers
    from turing_avalanche_analysis import (
        run_schnakenberg, build_laplacian_matrix
    )
    from avalanche_analysis import compute_avalanche_metrics

    lattice_dims = (11, 11)
    num_cells = 121
    a, b, Du, Dv, dt = 0.1, 0.9, 0.01, 1.0, 0.01
    turing_times = [3000, 5000, 7000, 8000, 9000]
    response_window = 500
    num_total_iters = max(turing_times) + response_window

    print("Running Turing reference simulation for comparison...")
    np.random.seed(42)
    ref_u_ts, ref_v_ts, _, _ = run_schnakenberg(
        lattice_dims, a, b, Du, Dv, dt, num_total_iters, record_every=1)

    np.random.seed(42)
    kick_locations = np.arange(num_cells)
    kick_amplitude = 0.01

    turing_rbt = {}
    for t_perturb in turing_times:
        remaining = min(response_window, num_total_iters - t_perturb)
        ref_window = ref_u_ts[t_perturb:t_perturb + remaining]
        sizes, durations, spatial_extents, branching_ratios = [], [], [], []

        for cell in kick_locations:
            u_init = ref_u_ts[t_perturb].copy()
            v_init = ref_v_ts[t_perturb].copy()
            u_init[cell] += kick_amplitude
            pert_u_ts, _, _, _ = run_schnakenberg(
                lattice_dims, a, b, Du, Dv, dt, remaining,
                u_init=u_init, v_init=v_init, record_every=1)
            delta_u = pert_u_ts - ref_window[:len(pert_u_ts)]
            m = compute_avalanche_metrics(delta_u, lattice_dims)
            if m['size'] > 0:
                sizes.append(m['size'])
                durations.append(m['duration'])
                spatial_extents.append(m['spatial_extent'])
                if not np.isnan(m['branching_ratio']):
                    branching_ratios.append(m['branching_ratio'])

        turing_rbt[t_perturb] = {
            'sizes': np.array(sizes),
            'durations': np.array(durations),
            'spatial_extents': np.array(spatial_extents),
            'branching_ratios': np.array(branching_ratios),
        }
        print(f"  t={t_perturb}: {len(sizes)} avalanches, "
              f"BR={np.mean(branching_ratios):.4f}")

    # ============================================================
    # COMPARISON FIGURE
    # ============================================================
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

    # --- Row 1: Size CCDF comparison ---
    ax_stig_size = fig.add_subplot(gs[0, 0])
    ax_turing_size = fig.add_subplot(gs[0, 1])

    stig_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(stig_times)))
    turing_colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(turing_times)))

    for i, t in enumerate(stig_times):
        sizes = stig_rbt[t]['sizes']
        if len(sizes) >= 5:
            x, ccdf = compute_ccdf(sizes)
            ax_stig_size.loglog(x, ccdf, 'o', markersize=3, alpha=0.7,
                                color=stig_colors[i], label=f't={t}')
            alpha, sigma, xmin, _ = fit_power_law_mle(sizes)
            if np.isfinite(alpha):
                x_fit = np.logspace(np.log10(xmin), np.log10(sizes.max()), 50)
                ccdf_fit = (x_fit / xmin) ** (1 - alpha)
                scale = np.sum(sizes >= xmin) / len(sizes)
                ax_stig_size.loglog(x_fit, ccdf_fit * scale, '--',
                                    color=stig_colors[i], alpha=0.5)

    ax_stig_size.set_xlabel('Avalanche Size')
    ax_stig_size.set_ylabel('P(S >= s)')
    ax_stig_size.set_title('Stigmergic Model: Size CCDF')
    ax_stig_size.legend(fontsize=7)
    ax_stig_size.grid(True, alpha=0.3, which='both')

    for i, t in enumerate(turing_times):
        sizes = turing_rbt[t]['sizes']
        if len(sizes) >= 5:
            x, ccdf = compute_ccdf(sizes)
            ax_turing_size.loglog(x, ccdf, 'o', markersize=3, alpha=0.7,
                                  color=turing_colors[i], label=f't={t}')
            alpha, sigma, xmin, _ = fit_power_law_mle(sizes)
            if np.isfinite(alpha):
                x_fit = np.logspace(np.log10(xmin), np.log10(sizes.max()), 50)
                ccdf_fit = (x_fit / xmin) ** (1 - alpha)
                scale = np.sum(sizes >= xmin) / len(sizes)
                ax_turing_size.loglog(x_fit, ccdf_fit * scale, '--',
                                      color=turing_colors[i], alpha=0.5)

    ax_turing_size.set_xlabel('Avalanche Size')
    ax_turing_size.set_ylabel('P(S >= s)')
    ax_turing_size.set_title('Turing (Schnakenberg): Size CCDF')
    ax_turing_size.legend(fontsize=7)
    ax_turing_size.grid(True, alpha=0.3, which='both')

    # --- Row 2: Branching ratio comparison ---
    ax_br = fig.add_subplot(gs[1, 0])

    # Stigmergic
    stig_br_means = [np.mean(stig_rbt[t]['branching_ratios'])
                     if len(stig_rbt[t]['branching_ratios']) > 0 else np.nan
                     for t in stig_times]
    stig_br_stds = [np.std(stig_rbt[t]['branching_ratios'])
                    if len(stig_rbt[t]['branching_ratios']) > 0 else 0
                    for t in stig_times]
    # Normalize times to fraction of total for comparison
    ax_br.errorbar(stig_times, stig_br_means, yerr=stig_br_stds,
                   fmt='o-', capsize=4, markersize=6, color='firebrick',
                   label='Stigmergic')

    # Turing
    turing_br_means = [np.mean(turing_rbt[t]['branching_ratios'])
                       if len(turing_rbt[t]['branching_ratios']) > 0 else np.nan
                       for t in turing_times]
    turing_br_stds = [np.std(turing_rbt[t]['branching_ratios'])
                      if len(turing_rbt[t]['branching_ratios']) > 0 else 0
                      for t in turing_times]
    ax_br2 = ax_br.twiny()
    ax_br2.errorbar(turing_times, turing_br_means, yerr=turing_br_stds,
                    fmt='s-', capsize=4, markersize=6, color='steelblue',
                    label='Turing')
    ax_br2.set_xlabel('Turing perturbation time', color='steelblue', fontsize=9)
    ax_br2.tick_params(axis='x', labelcolor='steelblue', labelsize=8)

    ax_br.axhline(y=1.0, color='k', linestyle='--', alpha=0.4, label='Critical (BR=1)')
    ax_br.set_xlabel('Stigmergic perturbation time', color='firebrick', fontsize=9)
    ax_br.tick_params(axis='x', labelcolor='firebrick', labelsize=8)
    ax_br.set_ylabel('Branching Ratio')
    ax_br.set_title('Branching Ratio Comparison')
    # Combined legend
    lines1, labels1 = ax_br.get_legend_handles_labels()
    lines2, labels2 = ax_br2.get_legend_handles_labels()
    ax_br.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax_br.grid(True, alpha=0.3)

    # --- Row 2 right: Size variability comparison ---
    ax_var = fig.add_subplot(gs[1, 1])
    stig_cvs = []
    for t in stig_times:
        s = stig_rbt[t]['sizes']
        if len(s) > 0:
            stig_cvs.append(np.std(s) / np.mean(s))
        else:
            stig_cvs.append(np.nan)
    turing_cvs = []
    for t in turing_times:
        s = turing_rbt[t]['sizes']
        if len(s) > 0:
            turing_cvs.append(np.std(s) / np.mean(s))
        else:
            turing_cvs.append(np.nan)

    x_pos = np.arange(max(len(stig_times), len(turing_times)))
    bar_width = 0.35
    ax_var.bar(x_pos[:len(stig_cvs)] - bar_width / 2, stig_cvs,
               bar_width, color='firebrick', alpha=0.7, label='Stigmergic')
    ax_var.bar(x_pos[:len(turing_cvs)] + bar_width / 2, turing_cvs,
               bar_width, color='steelblue', alpha=0.7, label='Turing')
    ax_var.set_xlabel('Perturbation time index')
    ax_var.set_ylabel('Coefficient of Variation (size)')
    ax_var.set_title('Size Variability: CV = std/mean')
    ax_var.legend(fontsize=8)
    ax_var.grid(True, alpha=0.3, axis='y')

    # --- Row 3: Power-law exponent comparison ---
    ax_alpha = fig.add_subplot(gs[2, 0])
    stig_alphas = []
    stig_alpha_errs = []
    for t in stig_times:
        s = stig_rbt[t]['sizes']
        if len(s) >= 5:
            alpha, sigma, _, _ = fit_power_law_mle(s)
            stig_alphas.append(alpha if np.isfinite(alpha) else np.nan)
            stig_alpha_errs.append(sigma if np.isfinite(sigma) else 0)
        else:
            stig_alphas.append(np.nan)
            stig_alpha_errs.append(0)

    turing_alphas = []
    turing_alpha_errs = []
    for t in turing_times:
        s = turing_rbt[t]['sizes']
        if len(s) >= 5:
            alpha, sigma, _, _ = fit_power_law_mle(s)
            turing_alphas.append(alpha if np.isfinite(alpha) else np.nan)
            turing_alpha_errs.append(sigma if np.isfinite(sigma) else 0)
        else:
            turing_alphas.append(np.nan)
            turing_alpha_errs.append(0)

    ax_alpha.errorbar(stig_times, stig_alphas, yerr=stig_alpha_errs,
                      fmt='o-', capsize=4, markersize=6, color='firebrick',
                      label='Stigmergic')
    ax_alpha2 = ax_alpha.twiny()
    ax_alpha2.errorbar(turing_times, turing_alphas, yerr=turing_alpha_errs,
                       fmt='s-', capsize=4, markersize=6, color='steelblue',
                       label='Turing')
    ax_alpha2.set_xlabel('Turing perturbation time', color='steelblue', fontsize=9)
    ax_alpha2.tick_params(axis='x', labelcolor='steelblue', labelsize=8)
    ax_alpha.set_xlabel('Stigmergic perturbation time', color='firebrick', fontsize=9)
    ax_alpha.tick_params(axis='x', labelcolor='firebrick', labelsize=8)
    ax_alpha.set_ylabel(r'Power-law exponent $\alpha$')
    ax_alpha.set_title(r'Size Distribution Exponent $\alpha$')
    lines1, labels1 = ax_alpha.get_legend_handles_labels()
    lines2, labels2 = ax_alpha2.get_legend_handles_labels()
    ax_alpha.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax_alpha.grid(True, alpha=0.3)

    # --- Row 3 right: Summary table ---
    ax_table = fig.add_subplot(gs[2, 1])
    ax_table.axis('off')

    # Aggregate stats
    all_stig_sizes = np.concatenate([stig_rbt[t]['sizes'] for t in stig_times])
    all_stig_brs = np.concatenate([stig_rbt[t]['branching_ratios'] for t in stig_times])
    all_stig_spatial = np.concatenate([stig_rbt[t]['spatial_extents'] for t in stig_times])
    all_stig_durs = np.concatenate([stig_rbt[t]['durations'] for t in stig_times])

    all_turing_sizes = np.concatenate([turing_rbt[t]['sizes'] for t in turing_times])
    all_turing_brs = np.concatenate([turing_rbt[t]['branching_ratios'] for t in turing_times])
    all_turing_spatial = np.concatenate([turing_rbt[t]['spatial_extents'] for t in turing_times])
    all_turing_durs = np.concatenate([turing_rbt[t]['durations'] for t in turing_times])

    stig_alpha_all, stig_sigma_all, _, _ = fit_power_law_mle(all_stig_sizes)
    turing_alpha_all, turing_sigma_all, _, _ = fit_power_law_mle(all_turing_sizes)

    table_data = [
        ['Metric', 'Stigmergic', 'Turing'],
        ['Grid size', '11x11', '11x11'],
        ['Kick amplitude', '5 mV', '0.01 (u)'],
        ['N avalanches', str(len(all_stig_sizes)), str(len(all_turing_sizes))],
        ['Size range',
         f'{all_stig_sizes.min():.2e} - {all_stig_sizes.max():.2e}',
         f'{all_turing_sizes.min():.2e} - {all_turing_sizes.max():.2e}'],
        ['Size CV',
         f'{np.std(all_stig_sizes)/np.mean(all_stig_sizes):.2f}',
         f'{np.std(all_turing_sizes)/np.mean(all_turing_sizes):.2f}'],
        [r'$\alpha$ (pooled)',
         f'{stig_alpha_all:.2f} +/- {stig_sigma_all:.2f}' if np.isfinite(stig_alpha_all) else 'N/A',
         f'{turing_alpha_all:.2f} +/- {turing_sigma_all:.2f}' if np.isfinite(turing_alpha_all) else 'N/A'],
        ['Branching ratio',
         f'{np.mean(all_stig_brs):.4f} +/- {np.std(all_stig_brs):.4f}',
         f'{np.mean(all_turing_brs):.4f} +/- {np.std(all_turing_brs):.4f}'],
        ['Spatial extent',
         f'{np.mean(all_stig_spatial):.1f} +/- {np.std(all_stig_spatial):.1f}',
         f'{np.mean(all_turing_spatial):.1f} +/- {np.std(all_turing_spatial):.1f}'],
        ['Duration range',
         f'{all_stig_durs.min()} - {all_stig_durs.max()}',
         f'{all_turing_durs.min()} - {all_turing_durs.max()}'],
    ]

    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    # Style header row
    for j in range(3):
        table[0, j].set_facecolor('#d4d4d4')
        table[0, j].set_text_props(fontweight='bold')
    # Color-code model columns
    for i in range(1, len(table_data)):
        table[i, 1].set_facecolor('#ffe0e0')
        table[i, 2].set_facecolor('#e0e8ff')

    ax_table.set_title('Summary Comparison', fontsize=11, fontweight='bold', pad=20)

    plt.suptitle('Criticality Comparison: Stigmergic Bioelectric vs Schnakenberg Turing',
                 fontsize=14, fontweight='bold')
    plt.savefig('data/criticality_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: data/criticality_comparison.png")

    # Print text summary
    print("\n" + "=" * 70)
    print("CRITICALITY COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Stigmergic':>20} {'Turing':>20}")
    print("-" * 65)
    print(f"{'Branching ratio':<25} {np.mean(all_stig_brs):>17.4f} {np.mean(all_turing_brs):>17.4f}")
    print(f"{'BR std':<25} {np.std(all_stig_brs):>17.4f} {np.std(all_turing_brs):>17.4f}")
    print(f"{'|BR - 1|':<25} {abs(np.mean(all_stig_brs)-1):>17.4f} {abs(np.mean(all_turing_brs)-1):>17.4f}")
    if np.isfinite(stig_alpha_all) and np.isfinite(turing_alpha_all):
        print(f"{'Alpha (pooled)':<25} {stig_alpha_all:>17.2f} {turing_alpha_all:>17.2f}")
    print(f"{'Size CV':<25} {np.std(all_stig_sizes)/np.mean(all_stig_sizes):>17.2f} {np.std(all_turing_sizes)/np.mean(all_turing_sizes):>17.2f}")
    print(f"{'Size range (decades)':<25} {np.log10(all_stig_sizes.max()/all_stig_sizes.min()):>17.2f} {np.log10(all_turing_sizes.max()/all_turing_sizes.min()):>17.2f}")
    print(f"{'Mean spatial extent':<25} {np.mean(all_stig_spatial):>17.1f} {np.mean(all_turing_spatial):>17.1f}")


if __name__ == '__main__':
    main()