"""Plot the complexity-addressability trade-off across field action range, for both lattices."""
import argparse, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec

parser = argparse.ArgumentParser()
parser.add_argument('--inputs', type=str,
                    default="[('11x11','data/cxAddr11x11Dense_metricEvaluation.npz'),"
                            "('30x30','data/cxAddr30x30Dense_metricEvaluation.npz')]")
parser.add_argument('--output', type=str, default='data/cxAddrTradeoff.png')
args = parser.parse_args()
inputs = eval(args.inputs)

figure = plt.figure(figsize=(7.6 * len(inputs), 24.5))
grid = gridspec.GridSpec(7, len(inputs), figure=figure, hspace=0.42, wspace=0.26)

for column, (label, path) in enumerate(inputs):
    archive = np.load(path, allow_pickle=True)
    records = archive['records'].item(); ranges = list(archive['ranges'])
    # Plot against reach, not the parameter: fieldScreenSize quantises onto the extracellular
    # grid, so sizes 1 and 2 are the same condition and the step from 2 to 3 nearly quadruples
    # the neighbourhood. Duplicate reaches are dropped so a repeated point cannot read as a plateau.
    reach = [records[r]['reach'] for r in ranges]
    keep = [i for i, x in enumerate(reach) if x not in reach[:i]]
    ranges = [ranges[i] for i in keep]; reach = [reach[i] for i in keep]
    shellSizes = list(archive['shellSizes'])
    get = lambda k: np.array([records[r][k] for r in ranges], dtype=float)

    # 1. Whole interior: how much readable structure exists, and how much is controlled.
    axis = figure.add_subplot(grid[0, column])
    axis.plot(reach, get('CX_variance'), 'o-', color='steelblue', label='CX  readable variance')
    axis.plot(reach, get('ADDR_variance'), 's-', color='crimson', label='ADDR  controlled variance')
    axis.set_yscale('symlog', linthresh=1)
    axis.set_ylabel('variance (mV$^2$)'); axis.legend(fontsize=8)
    axis.set_xlabel('reach (grid points per cell)')
    axis.set_title(f'{label}   whole interior', fontsize=11)

    # 2. Depth-fair: every shell weighted equally, so the fringe cannot carry the score.
    axis = figure.add_subplot(grid[1, column])
    axis.plot(reach, get('fair_CX_perCell'), 'o-', color='steelblue', label='CX  arithmetic mean')
    axis.plot(reach, get('fair_ADDR_perCell'), 's-', color='crimson', label='ADDR  arithmetic mean')
    axis.plot(reach, get('geo_CX_perCell'), 'o--', color='steelblue', alpha=0.55, label='CX  geometric mean')
    axis.plot(reach, get('geo_ADDR_perCell'), 's--', color='crimson', alpha=0.55, label='ADDR  geometric mean')
    axis.set_yscale('symlog', linthresh=0.01)
    axis.set_ylabel('variance per cell (mV$^2$)'); axis.legend(fontsize=8)
    axis.set_xlabel('reach (grid points per cell)')
    axis.set_title('depth-fair: arithmetic tolerates concentration, geometric does not', fontsize=10)

    # 3. Coverage: how much of the tissue's depth carries any readable structure at all. Neither
    # CX form answers this -- an average cannot distinguish structure spread across every shell
    # from the same total piled into one -- so it is plotted in its own right.
    axis = figure.add_subplot(grid[2, column])
    axis.plot(reach, get('liveShells'), 'o-', color='rebeccapurple')
    axis.axhline(get('shellCount')[0], color='0.6', linestyle=':', linewidth=1.2)
    axis.text(reach[len(reach)//2], get('shellCount')[0] * 0.94, 'every shell patterned',
              fontsize=8, color='0.45')
    axis.set_ylabel('depth shells carrying\nreadable structure')
    axis.set_xlabel('reach (grid points per cell)')
    axis.set_ylim(0, get('shellCount')[0] * 1.12)
    axis.set_title('spatial extent of patterning', fontsize=11)

    # 4. The fraction under control, both weightings. The gap between them is fringe concentration.
    axis = figure.add_subplot(grid[3, column])
    axis.plot(reach, get('ADDR_fraction'), 'o-', color='darkorange', label='whole interior')
    axis.plot(reach, get('fair_ADDR_fraction'), 's--', color='seagreen',
              label='depth-fair (arithmetic mean)')
    axis.set_ylabel('fraction of readable\nvariance controlled'); axis.legend(fontsize=8)
    axis.set_xlabel('reach (grid points per cell)')
    axis.set_title('controlled fraction', fontsize=11); axis.set_ylim(bottom=0)

    # 4. The trade-off itself, as a path through the CX-ADDR plane.
    axis = figure.add_subplot(grid[4, column])
    arithCx, arithAddr = get('fair_CX_perCell'), get('fair_ADDR_perCell')
    cx, addr = get('geo_CX_perCell'), get('geo_ADDR_perCell')
    axis.plot(arithCx, arithAddr, '-', color='0.82', linewidth=1, zorder=0)
    axis.scatter(arithCx, arithAddr, c='0.82', s=26, zorder=0, label='arithmetic mean')
    axis.plot(cx, addr, '-', color='0.6', linewidth=1, zorder=1)
    onFront = [i for i in range(len(cx))
               if not any(cx[j] >= cx[i] and addr[j] >= addr[i] and
                          (cx[j] > cx[i] or addr[j] > addr[i]) for j in range(len(cx)))]
    axis.scatter(cx[onFront], addr[onFront], s=230, facecolor='none', edgecolor='crimson',
                 linewidth=1.8, zorder=3, label='Pareto frontier')
    scatter = axis.scatter(cx, addr, c=reach, cmap='viridis', s=70, zorder=2, edgecolor='k', linewidth=0.4)
    axis.legend(fontsize=8, loc='lower right')
    for r, x, y in zip(reach, cx, addr):
        axis.annotate(f'{r:.0f}', (x, y), fontsize=7, xytext=(4, 3), textcoords='offset points')
    figure.colorbar(scatter, ax=axis, label='reach (grid points per cell)', fraction=0.045)
    axis.set_xscale('symlog', linthresh=1); axis.set_yscale('symlog', linthresh=0.01)
    axis.set_xlabel('CX  (geometric mean over shells, mV$^2$)')
    axis.set_ylabel('ADDR  (geometric mean\nover shells, mV$^2$)')
    axis.set_title('the trade-off under geometric aggregation; grey shows the arithmetic path',
                   fontsize=10)

    # 6. Capacity: how much the boundary writes, in bits.
    axis = figure.add_subplot(grid[5, column])
    axis.plot(reach, get('ADDR_bits'), 'o-', color='rebeccapurple', label='whole interior')
    axis.plot(reach, get('fair_ADDR_bits'), 's--', color='teal',
              label='depth-fair (arithmetic mean)')
    axis.set_ylabel('capacity (bits)'); axis.legend(fontsize=8)
    axis.set_xlabel('reach (grid points per cell)'); axis.set_ylim(bottom=0)
    axis.set_title('information the boundary writes into the interior', fontsize=11)

    # 7. Where in the tissue the control actually lives.
    axis = figure.add_subplot(grid[6, column])
    profile = np.array([records[r]['profile_ADDR_fraction'] for r in ranges], dtype=float)
    image = axis.imshow(profile.T, aspect='auto', cmap='magma', origin='lower', vmin=0, vmax=1,
                        extent=[-0.5, len(ranges) - 0.5, 0.5, len(shellSizes) + 0.5])
    axis.set_xticks(range(len(ranges)))
    axis.set_xticklabels([f'{x:.0f}' for x in reach], fontsize=7)
    axis.set_xlabel('reach (grid points per cell)'); axis.set_ylabel('depth shell\n(1 = touching boundary)')
    figure.colorbar(image, ax=axis, label='fraction controlled', fraction=0.045)
    axis.set_title('controlled fraction by depth', fontsize=11)

# Labels are set per panel above rather than by a sweep over figure.axes: colorbars are axes too,
# and they carry a y label and a subplotspec, so any blanket rule writes the axis title across them.
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"Saved {args.output}")
