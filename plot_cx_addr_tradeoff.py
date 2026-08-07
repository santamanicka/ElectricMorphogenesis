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

figure = plt.figure(figsize=(7.6 * len(inputs), 21.0))
grid = gridspec.GridSpec(6, len(inputs), figure=figure, hspace=0.42, wspace=0.26)

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
    axis.set_title(f'{label}   whole interior', fontsize=11)

    # 2. Depth-fair: every shell weighted equally, so the fringe cannot carry the score.
    axis = figure.add_subplot(grid[1, column])
    axis.plot(reach, get('fair_CX_perCell'), 'o-', color='steelblue', label='CX  per cell, shell-averaged')
    axis.plot(reach, get('fair_ADDR_perCell'), 's-', color='crimson', label='ADDR  per cell, shell-averaged')
    axis.set_yscale('symlog', linthresh=0.01)
    axis.set_ylabel('variance per cell (mV$^2$)'); axis.legend(fontsize=8)
    axis.set_title('depth-fair (equal weight per shell)', fontsize=11)

    # 3. The fraction under control, both weightings. The gap between them is fringe concentration.
    axis = figure.add_subplot(grid[2, column])
    axis.plot(reach, get('ADDR_fraction'), 'o-', color='darkorange', label='whole interior')
    axis.plot(reach, get('fair_ADDR_fraction'), 's--', color='seagreen', label='depth-fair')
    axis.set_ylabel('fraction of readable\nvariance controlled'); axis.legend(fontsize=8)
    axis.set_title('controlled fraction', fontsize=11); axis.set_ylim(bottom=0)

    # 4. The trade-off itself, as a path through the CX-ADDR plane.
    axis = figure.add_subplot(grid[3, column])
    cx, addr = get('fair_CX_perCell'), get('fair_ADDR_perCell')
    axis.plot(cx, addr, '-', color='0.6', linewidth=1, zorder=1)
    scatter = axis.scatter(cx, addr, c=reach, cmap='viridis', s=70, zorder=2, edgecolor='k', linewidth=0.4)
    for r, x, y in zip(reach, cx, addr):
        axis.annotate(f'{r:.0f}', (x, y), fontsize=7, xytext=(4, 3), textcoords='offset points')
    figure.colorbar(scatter, ax=axis, label='reach (grid points per cell)', fraction=0.045)
    axis.set_xscale('symlog', linthresh=1); axis.set_yscale('symlog', linthresh=0.01)
    axis.set_xlabel('CX  (depth-fair variance per cell, mV$^2$)')
    axis.set_ylabel('ADDR  (depth-fair, mV$^2$)')
    axis.set_title('the trade-off, labelled by reach', fontsize=11)

    # 5. Capacity: how much the boundary writes, in bits.
    axis = figure.add_subplot(grid[4, column])
    axis.plot(reach, get('ADDR_bits'), 'o-', color='rebeccapurple', label='whole interior')
    axis.plot(reach, get('fair_ADDR_bits'), 's--', color='teal', label='depth-fair')
    axis.set_ylabel('capacity (bits)'); axis.legend(fontsize=8)
    axis.set_xlabel('reach (grid points per cell)'); axis.set_ylim(bottom=0)
    axis.set_title('information the boundary writes into the interior', fontsize=11)

    # 6. Where in the tissue the control actually lives.
    axis = figure.add_subplot(grid[5, column])
    profile = np.array([records[r]['profile_ADDR_fraction'] for r in ranges], dtype=float)
    image = axis.imshow(profile.T, aspect='auto', cmap='magma', origin='lower', vmin=0, vmax=1,
                        extent=[-0.5, len(ranges) - 0.5, 0.5, len(shellSizes) + 0.5])
    axis.set_xticks(range(len(ranges)))
    axis.set_xticklabels([f'{x:.0f}' for x in reach], fontsize=7)
    axis.set_xlabel('reach (grid points per cell)'); axis.set_ylabel('depth shell\n(1 = touching boundary)')
    figure.colorbar(image, ax=axis, label='fraction controlled', fraction=0.045)
    axis.set_title('controlled fraction by depth', fontsize=11)

for axis in figure.axes:
    if axis.get_xlabel() == '' and axis.get_ylabel() != '':
        axis.set_xlabel('reach (grid points per cell)')
plt.savefig(args.output, dpi=150, bbox_inches='tight')
print(f"Saved {args.output}")
