"""
Merge the per-task chunks written by runGenerateEnsemble30x30.sh into one ensemble.

The array job splits the ensemble across tasks, each with its own clamp seed, so the chunks
have to be concatenated in a defined order before PCA. This also checks the two things that
would silently corrupt the result: a chunk missing because its task failed, and duplicate
clamps arising from a seed collision, which would show up in PCA as spurious zero-variance
directions rather than as an error.

Outputs the same three files generate_ensemble.py produces, so the downstream analysis scripts
take the merged ensemble without modification.

  python merge_ensemble_chunks.py --chunkDir data/ensemble30x30 --outputPrefix data/ensemble30x30
"""

import argparse
import glob
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--chunkDir',     type=str, default='data/ensemble30x30')
parser.add_argument('--chunkPattern', type=str, default='chunk*')
parser.add_argument('--outputPrefix', type=str, default='data/ensemble30x30')
parser.add_argument('--expectedChunks', type=int, default=None,
                    help='fail if this many chunks are not present, so a silently failed array '
                         'task cannot shrink the ensemble unnoticed')
args = parser.parse_args()

prefixes = sorted({path[:-len('_gpol_prepatterns.npy')] for path in
                   glob.glob(os.path.join(args.chunkDir, args.chunkPattern + '_gpol_prepatterns.npy'))})
if len(prefixes) == 0:
    raise SystemExit(f"no chunks matching {args.chunkPattern} in {args.chunkDir}")
if args.expectedChunks is not None and len(prefixes) != args.expectedChunks:
    found = [os.path.basename(p) for p in prefixes]
    raise SystemExit(f"expected {args.expectedChunks} chunks, found {len(prefixes)}: {found}")

gpolChunks, vmemChunks = [], []
frequencyChunks, phaseChunks, amplitudeChunks = [], [], []
chunkOfSample = []
for chunkIndex, prefix in enumerate(prefixes):
    gpol = np.load(f'{prefix}_gpol_prepatterns.npy')
    vmem = np.load(f'{prefix}_vmem_final.npy')
    clamp = np.load(f'{prefix}_clamp_params.npz')
    if gpol.shape[0] != vmem.shape[0]:
        raise SystemExit(f"{prefix}: {gpol.shape[0]} G_pol samples but {vmem.shape[0]} Vmem samples")
    gpolChunks.append(gpol)
    vmemChunks.append(vmem)
    frequencyChunks.append(clamp['frequencies'])
    phaseChunks.append(clamp['phases'])
    amplitudeChunks.append(clamp['amplitudes'])
    chunkOfSample.extend([chunkIndex] * gpol.shape[0])
    print(f"  {os.path.basename(prefix):<16} {gpol.shape[0]:>4} samples, {gpol.shape[1]} cells")

gpol = np.concatenate(gpolChunks)
vmem = np.concatenate(vmemChunks)
frequencies = np.concatenate(frequencyChunks)
phases = np.concatenate(phaseChunks)
amplitudes = np.concatenate(amplitudeChunks)
chunkOfSample = np.array(chunkOfSample)

# A seed collision would duplicate whole clamps across chunks. Checking the clamp parameters
# rather than the outcomes catches it at the source.
clampSignatures = np.concatenate([frequencies, phases, amplitudes], axis=1)
_, uniqueIndices = np.unique(clampSignatures, axis=0, return_index=True)
numDuplicates = len(clampSignatures) - len(uniqueIndices)

print(f"\nmerged {len(prefixes)} chunks -> {gpol.shape[0]} samples x {gpol.shape[1]} cells")
print(f"  G_pol range: [{gpol.min():.3e}, {gpol.max():.3e}]")
# Stored in volts, as generate_ensemble.py writes it and the analysis scripts expect it.
print(f"  Vmem  range: [{vmem.min()*1000:.1f}, {vmem.max()*1000:.1f}] mV")
print(f"  duplicate clamps: {numDuplicates}" + ("  <-- seed collision" if numDuplicates else ""))
if numDuplicates:
    raise SystemExit("duplicate clamps found; rerun the affected tasks with distinct seeds")

# PCA rank is capped by sample count, so record whether the ensemble is large enough to resolve
# the participation ratios it is meant to measure.
print(f"  PCA rank ceiling: {gpol.shape[0] - 1} components")

np.save(f'{args.outputPrefix}_gpol_prepatterns.npy', gpol)
np.save(f'{args.outputPrefix}_vmem_final.npy', vmem)
np.savez(f'{args.outputPrefix}_clamp_params.npz',
         frequencies=frequencies, phases=phases, amplitudes=amplitudes,
         chunkOfSample=chunkOfSample)
print(f"\nSaved {args.outputPrefix}_gpol_prepatterns.npy, _vmem_final.npy, _clamp_params.npz")
