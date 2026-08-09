"""How far each training target sits from anything the tissue has been seen to produce.

Distance to the nearest of 400 reachable patterns means nothing on its own, because in 900 dimensions
every point is far from every other, so it is read against the distance a genuinely reachable pattern
sits from the rest of the ensemble. A target beyond that null's maximum is further from the reachable
set than any reachable pattern is.

The face target is loaded from a trained model rather than rebuilt: it is the border ring plus the
eyes, nose and mouth, while faceFeatureIndices returns the interior features alone, and rebuilding it
from that function alone drops 116 cells and moves both the mean level and the distance enough to
reverse which target ranks closest. The other three targets carry no border, which is how the
training script builds them.

Four hundred samples cover a set whose principal components need a hundred dimensions for eighty
percent of the variance, so this bounds what the sampling has seen rather than what the tissue can do.
"""
import numpy as np
import torch
from scipy.spatial.distance import squareform, pdist

faceModel = 'data/bestModelParameters_fieldVector_24.dat'
ensembleFile = 'data/fieldRangeSweepDense/screen04_vmem_final.npy'

face = torch.load(faceModel, map_location='cpu', weights_only=False)
face = face['trainParameters']['targetVmem'].detach().numpy().reshape(-1)*1000

src = open('learnCellularFieldNetwork.py').read()
namespace = {'np': np}
exec(src[src.index('def rowColumnBlock'):src.index('def verifyTargetsAgainstReference')], namespace)
targets = {'face (border + features)': face}
for name, builder in (('ap_band','apBandIndices'), ('stripes','stripeIndices'),
                      ('triangular_wave','triangularWaveIndices')):
    t = np.full(900, -9.2e-3)
    t[namespace[builder](30, 30)] = -0.06
    targets[name] = t*1000

ensemble = np.load(ensembleFile)*1000
numCells = ensemble.shape[1]
distances = squareform(pdist(ensemble))/np.sqrt(numCells)
np.fill_diagonal(distances, np.inf)
nearestNeighbour = distances.min(axis=1)
meanLevels = ensemble.mean(axis=1)

print(f"  null: reachable patterns sit a median {np.median(nearestNeighbour):.2f} mV from their "
      f"nearest neighbour (max {nearestNeighbour.max():.2f})")
print(f"  ensemble mean level {meanLevels.mean():.2f} mV, "
      f"range [{meanLevels.min():.2f}, {meanLevels.max():.2f}]\n")
print(f"  {'target':26s} {'cells@-60':>10s} {'mean mV':>9s} {'nearest':>9s} {'percentile':>11s}")
for name, t in targets.items():
    nearest = float(np.sqrt(((ensemble - t)**2).mean(axis=1)).min())
    percentile = 100.0*(nearestNeighbour < nearest).mean()
    print(f"  {name:26s} {int((t < -30).sum()):10d} {t.mean():9.2f} {nearest:9.2f} {percentile:10.1f}%")
