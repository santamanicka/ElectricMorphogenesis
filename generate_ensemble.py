"""
Generate an ensemble of random-clamp simulations using fixed Stigmergic tissue parameters.

Each simulation uses a different random oscillatory clamp following the
fieldDomeTwoFoldSymmetry convention: left-half field dome is parameterized
independently, then mirrored to the right half.

Outputs (saved to data/):
  ensemble_gpol_prepatterns.npy  — (N, numCells) G_pol at t = clampEndIter + 1
  ensemble_vmem_final.npy        — (N, numCells) Vmem at t = numSimIters - 1
  ensemble_clamp_params.npz      — frequencies, phases, amplitudes (left half only)
"""

import gc
import argparse
import torch
import numpy as np
from embryo import model
import utilities

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=200, help='Number of ensemble samples')
parser.add_argument('--source', type=str, default='data/StigmergicModelParameters.dat')
parser.add_argument('--output_prefix', type=str, default='data/ensemble')
parser.add_argument('--num_sim_iters', type=int, default=1000)
parser.add_argument('--clamp_duration_prop', type=float, default=0.1)
parser.add_argument('--clamp_iters', type=int, default=None,
                    help='absolute clamp duration, overriding clamp_duration_prop. Preferred when '
                         'comparing lattice sizes, since it holds the write budget fixed while '
                         'only the free-evolution window changes.')
parser.add_argument('--readout_iters', type=int, default=1,
                    help='number of final iterations averaged for the Vmem readout. Defaults to 1 '
                         '(a single snapshot), which is what the published 11x11 results used. '
                         'The 30x30 protocol uses 200: that tissue does not reach a fixed point, '
                         'so a snapshot samples the phase of a fluctuating field and would enter '
                         'the flicker into the PCA as spurious dimensions.')
parser.add_argument('--freq_range', type=str, default='(100.0,1000.0)')
parser.add_argument('--amp_range', type=str, default='(-1.0,1.0)')
parser.add_argument('--fieldScreenSize', type=int, default=None,
                    help='override the field action range in the source .dat; the sweep uses this '
                         'to vary reach while holding the clamp set fixed')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

import ast
N = args.n
SOURCE_DAT = args.source
OUTPUT_PREFIX = args.output_prefix
NUM_SIM_ITERS = args.num_sim_iters
CLAMP_DURATION_PROP = args.clamp_duration_prop
min_freq, max_freq = ast.literal_eval(args.freq_range)
min_amp, max_amp = ast.literal_eval(args.amp_range)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# --- Load tissue parameters (fixed across all simulations) ---
params_base = torch.load(SOURCE_DAT, weights_only=False)
params_base['ATPParameters'] = None
params_base['latticePeriodicBoundaryGJ'] = False
if args.fieldScreenSize is not None:
    params_base['fieldParameters']['fieldScreenSize'] = args.fieldScreenSize
    print(f"field action range overridden to {args.fieldScreenSize}")
iv = params_base['simParameters']['initialValues']
numRows, numCols = params_base['latticeDims']
numCells = numRows * numCols
if 'ligandConc' not in iv:
    iv['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)

utils_obj = utilities.utilities()

# Build a reference model once to derive fixed geometry (clamp indices)
m_ref = model(params_base, 1)
circuit_ref = m_ref.electricNetwork

# Fixed clamp geometry: left-half field dome → mirror to right half
fieldDomeLeftHalfIndices = utils_obj.computeDomeIndices(circuit_ref, mode='field', region='leftHalf')
numClampPoints_half = len(fieldDomeLeftHalfIndices)

verticalReflectedIndices = utils_obj.computeSymmetricalIndices(
    circuit_ref, fieldDomeLeftHalfIndices, mode='field', symmetry='twofold'
)
full_indices = np.concatenate((fieldDomeLeftHalfIndices, verticalReflectedIndices))
_, unique_idx = np.unique(full_indices, return_index=True)
clampPointIndices = full_indices[unique_idx]  # sorted unique indices into field grid
numClampPoints_full = len(clampPointIndices)

sampleIndices = np.zeros(numClampPoints_full, dtype=int)
clampIndices = (sampleIndices, clampPointIndices)

clampStartIter = 0
clampEndIter = args.clamp_iters if args.clamp_iters is not None else int(CLAMP_DURATION_PROP * NUM_SIM_ITERS)
numClampIters = clampEndIter - clampStartIter + 1
pre_idx = clampEndIter + 1
timeIndices = torch.linspace(0, 0.5, numClampIters).view(-1, 1)  # (T, 1)

del m_ref, circuit_ref  # free reference model
gc.collect()

print(f"Ensemble generator")
print(f"  Source:          {SOURCE_DAT}")
print(f"  N samples:       {N}")
print(f"  numSimIters:     {NUM_SIM_ITERS}")
print(f"  clampEndIter:    {clampEndIter}  (pre_idx = {pre_idx})")
print(f"  readout:         mean Vmem over the last {args.readout_iters} iterations")
print(f"  Left-half clamp points: {numClampPoints_half}")
print(f"  Full clamp points (after mirror): {numClampPoints_full}")

# Output arrays
gpol_ensemble = np.zeros((N, numCells), dtype=np.float64)
vmem_ensemble = np.zeros((N, numCells), dtype=np.float64)
all_freqs  = np.zeros((N, numClampPoints_half), dtype=np.float64)
all_phases = np.zeros((N, numClampPoints_half), dtype=np.float64)
all_amps   = np.zeros((N, numClampPoints_half), dtype=np.float64)

for i in range(N):
    # Random clamp parameters for the left half only
    freqs  = torch.rand(numClampPoints_half, dtype=torch.double) * (max_freq - min_freq) + min_freq
    phases = torch.rand(numClampPoints_half, dtype=torch.double) * 2 * torch.pi
    amps   = torch.rand(numClampPoints_half, dtype=torch.double) * (max_amp - min_amp) + min_amp

    # Tile to cover full (left + mirrored right) set, then select unique indices
    freqs_full  = torch.tile(freqs,  (2,))
    phases_full = torch.tile(phases, (2,))
    amps_full   = torch.tile(amps,   (2,))

    # clampValues: (numClampIters, numClampPoints_full)
    clampValues = torch.cos(timeIndices * freqs_full + phases_full) * amps_full
    clampValues = clampValues[:, unique_idx]

    clampParameters = {
        'clampMode':     'fieldDomeTwoFoldSymmetry',
        'clampIndices':  clampIndices,
        'clampValues':   clampValues,
        'clampStartIter': clampStartIter,
        'clampEndIter':   clampEndIter,
    }

    # Fresh model each iteration (tissue params fixed)
    m = model(params_base, 1)
    m.setExperimentalConditions((params_base['simParameters']['initialValues'], 1))
    m.simulate(
        externalInputs=params_base['simParameters']['externalInputs'],
        clampParameters=clampParameters,
        perturbation=None,
        numSimIters=NUM_SIM_ITERS,
        # Only these two are needed downstream. Recording everything would allocate
        # timeseriesGij at numSimIters x numCells^2, which is 16 GB on a 30x30 lattice.
        storeVariables=('Vmem', 'Gpol'),
    )

    gpol_ensemble[i] = m.timeseriesGpol[pre_idx][0, :, 0].detach().numpy()
    # timeseriesVmem records at the top of each iteration, so its last row is the state *before*
    # the final update; electricNetwork.Vmem is the state after it. The readout therefore takes
    # the true final state plus however many recorded frames precede it, which keeps
    # readout_iters=1 exactly equal to the original single-snapshot behaviour.
    finalVmem = m.electricNetwork.Vmem[0, :, 0].detach().numpy()
    if args.readout_iters <= 1:
        vmem_ensemble[i] = finalVmem
    else:
        precedingVmem = m.timeseriesVmem[-(args.readout_iters - 1):, 0, :, 0].detach().numpy()
        vmem_ensemble[i] = (precedingVmem.sum(axis=0) + finalVmem) / args.readout_iters

    all_freqs[i]  = freqs.numpy()
    all_phases[i] = phases.numpy()
    all_amps[i]   = amps.numpy()

    del m
    gc.collect()

    if (i + 1) % 25 == 0 or i == 0:
        print(f"  [{i+1:4d}/{N}] G_pol range: [{gpol_ensemble[i].min():.3e}, {gpol_ensemble[i].max():.3e}]  "
              f"Vmem range: [{vmem_ensemble[i].min()*1000:.1f}, {vmem_ensemble[i].max()*1000:.1f}] mV")

np.save(f'{OUTPUT_PREFIX}_gpol_prepatterns.npy', gpol_ensemble)
np.save(f'{OUTPUT_PREFIX}_vmem_final.npy',        vmem_ensemble)
np.savez(f'{OUTPUT_PREFIX}_clamp_params.npz',
         frequencies=all_freqs, phases=all_phases, amplitudes=all_amps)

print(f"\nSaved {N} samples:")
print(f"  {OUTPUT_PREFIX}_gpol_prepatterns.npy  shape {gpol_ensemble.shape}")
print(f"  {OUTPUT_PREFIX}_vmem_final.npy         shape {vmem_ensemble.shape}")
print(f"  {OUTPUT_PREFIX}_clamp_params.npz")
