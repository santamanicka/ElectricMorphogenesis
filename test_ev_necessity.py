"""
Test which state variables at the pre-pattern time (t=clampEndIter+1) carry
the attractor-selection information.

Conditions tested per model:
  A) Full simulation with clamp (ground truth)
  B) Restart from Vmem_pre + eV_pre + initial G_pol  (previous test — fails)
  C) Restart from Vmem_pre + eV=0  + G_pol_pre       (G_pol restored, eV dropped)
  D) Restart from Vmem_pre + eV_pre + G_pol_pre       (full state minus G_dep)
  E) Restart from Vmem_pre + eV=0  + G_pol_pre + G_dep_pre  (complete state, eV=0)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from embryo import model

torch.set_grad_enabled(False)

MODELS = [
    ('Stigmergic', 'data/StigmergicModelParameters.dat'),
    ('ap_band',    'data/bestModelParameters_fieldVector_ap_band_1.dat'),
    ('stripes',    'data/bestModelParameters_fieldVector_stripes_1.dat'),
]

def load(path):
    p = torch.load(path, weights_only=False)
    p['ATPParameters'] = None
    p['latticePeriodicBoundaryGJ'] = False
    iv = p['simParameters']['initialValues']
    numSamples = p['simParameters']['numSamples']
    numCells = p['latticeDims'][0] * p['latticeDims'][1]
    if 'ligandConc' not in iv:
        iv['ligandConc'] = torch.zeros((numSamples, numCells, 1), dtype=torch.float64)
    return p

def run_full(path):
    """Full simulation with clamp. Returns vmem_final and timeseries vmem_pre."""
    p = load(path)
    numSamples = p['simParameters']['numSamples']
    m = model(p, numSamples)
    m.setExperimentalConditions((p['simParameters']['initialValues'], numSamples))
    clamp = p['clampParameters']
    m.simulate(externalInputs=p['simParameters']['externalInputs'],
               clampParameters=clamp, perturbation=None,
               numSimIters=p['simParameters']['numSimIters'])
    pre_idx = clamp['clampEndIter'] + 1
    return (m.electricNetwork.Vmem.clone(),
            m.timeseriesVmem[pre_idx].clone(),
            pre_idx,
            p['simParameters']['numSimIters'] - pre_idx,
            p['latticeDims'])

def capture_state_at(path, stop_iter):
    """Run simulation for stop_iter steps; return (eV, G_pol, G_dep) at end."""
    p = load(path)
    numSamples = p['simParameters']['numSamples']
    m = model(p, numSamples)
    m.setExperimentalConditions((p['simParameters']['initialValues'], numSamples))
    c = m.electricNetwork
    m.simulate(externalInputs=p['simParameters']['externalInputs'],
               clampParameters=p['clampParameters'], perturbation=None,
               numSimIters=stop_iter)
    return c.eV.clone(), c.G_pol.clone(), c.G_dep.clone()

def run_from(path, vmem_init, ev_init, gpol_init, gdep_init, numSteps):
    """Restart from specified state, no clamp."""
    p = load(path)
    numSamples = p['simParameters']['numSamples']
    m = model(p, numSamples)
    m.setExperimentalConditions((p['simParameters']['initialValues'], numSamples))
    c = m.electricNetwork

    iv_ov = dict(p['simParameters']['initialValues'])
    iv_ov['Vmem'] = vmem_init.clone().double()
    iv_ov['eV']   = ev_init.clone().double()
    c.initVariables(iv_ov)

    # Override conductance state directly
    if gpol_init is not None:
        c.G_pol = gpol_init.clone().double()
    if gdep_init is not None:
        c.G_dep = gdep_init.clone().double()

    m.simulate(externalInputs=p['simParameters']['externalInputs'],
               clampParameters=None, perturbation=None, numSimIters=numSteps)
    return c.Vmem.clone()


fig, axes = plt.subplots(len(MODELS), 5, figsize=(20, 4 * len(MODELS)))
col_titles = ['A: full sim\n(ground truth)',
              'B: Vmem+eV\ninitial G_pol',
              'C: Vmem eV=0\nG_pol_pre',
              'D: Vmem+eV\nG_pol_pre',
              'E: Vmem eV=0\nG_pol+G_dep pre']

for row, (name, path) in enumerate(MODELS):
    print(f"\n{'='*60}")
    print(f"Model: {name}")

    vmem_full, vmem_pre, pre_idx, numFree, (numRows, numCols) = run_full(path)
    ev_pre, gpol_pre, gdep_pre = capture_state_at(path, pre_idx)
    ev_zero = torch.zeros_like(ev_pre)

    # Load initial G_pol/G_dep (what the model starts with before clamping)
    p_ref = load(path)
    numSamples = p_ref['simParameters']['numSamples']
    m_ref = model(p_ref, numSamples)
    m_ref.setExperimentalConditions((p_ref['simParameters']['initialValues'], numSamples))
    gpol_init = m_ref.electricNetwork.G_pol.clone()
    gdep_init = m_ref.electricNetwork.G_dep.clone()

    # Check how much G_pol actually changed during clamping
    gpol_change = (gpol_pre - gpol_init).abs()
    print(f"  G_pol change during clamp: max={gpol_change.max().item():.4e}, "
          f"mean={gpol_change.mean().item():.4e}")
    gdep_change = (gdep_pre - gdep_init).abs()
    print(f"  G_dep change during clamp: max={gdep_change.max().item():.4e}, "
          f"mean={gdep_change.mean().item():.4e}")

    results = {
        'A': vmem_full,
        'B': run_from(path, vmem_pre, ev_pre,  None,      None,      numFree),  # initial G_pol
        'C': run_from(path, vmem_pre, ev_zero, gpol_pre,  None,      numFree),
        'D': run_from(path, vmem_pre, ev_pre,  gpol_pre,  None,      numFree),
        'E': run_from(path, vmem_pre, ev_zero, gpol_pre,  gdep_pre,  numFree),
    }

    vrange = (vmem_full.max() - vmem_full.min()).item()
    for key in ['B', 'C', 'D', 'E']:
        diff = (results['A'] - results[key]).abs().max().item()
        print(f"  Max diff A vs {key}: {diff*1000:.3f} mV  ({100*diff/vrange:.1f}% of range)")

    # Shared colorscale across all conditions
    all_v = torch.cat(list(results.values()))
    vmin, vmax = all_v.min().item(), all_v.max().item()

    for col, key in enumerate('ABCDE'):
        v = results[key][0,:,0].detach().numpy().reshape(numRows, numCols)
        im = axes[row, col].imshow(v, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        title = f'{name}\n{col_titles[col]}'
        axes[row, col].set_title(title, fontsize=8)
        plt.colorbar(im, ax=axes[row, col], fraction=0.046)

plt.tight_layout()
outfile = 'data/ev_necessity_test.png'
plt.savefig(outfile, dpi=150)
plt.close()
print(f"\nFigure saved to {outfile}")
