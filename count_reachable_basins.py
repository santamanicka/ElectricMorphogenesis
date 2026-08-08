"""How many distinct outcomes can the boundary actually select? Counted, not fitted.

For a switching system the useful notion of control is a count of reachable basins. The ensembles
already hold it: 400 random clamps per condition, each an independent draw from the address space.
Clustering their interior patterns at a resolution a cell could distinguish counts the outcomes
the boundary reached, with no regression involved.
"""
import numpy as np, torch, gc, utilities
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from embryo import model
torch.set_grad_enabled(False)

def setup(dat):
    p = torch.load(dat, weights_only=False); nR, nC = p['latticeDims']; n = nR*nC
    p['ATPParameters'] = None; p['latticePeriodicBoundaryGJ'] = False
    v = p['simParameters']['initialValues']
    if 'ligandConc' not in v: v['ligandConc'] = torch.zeros((1,n,1), dtype=torch.float64)
    m = model(p,1); d = utilities.utilities().computeDomeIndices(m.electricNetwork, mode='tissue')
    b = np.zeros(n,bool); b[d]=True; del m; gc.collect(); return ~b

conditions = [('11x11 reach 14.6', 'data/StigmergicModelParameters.dat',
               'data/fieldRangeSweep11x11Dense/screen03', 0.408),
              ('30x30 reach 4.0',  'data/StigmergicModelParameters_30x30.dat',
               'data/fieldRangeSweepDense/screen02', 0.533),
              ('30x30 reach 15.5', 'data/StigmergicModelParameters_30x30.dat',
               'data/fieldRangeSweepDense/screen03', 0.091),
              ('30x30 reach 22.9', 'data/StigmergicModelParameters_30x30.dat',
               'data/fieldRangeSweepDense/screen04', 0.007)]
print(f"{'condition':>18}{'CT frac':>9}{'clusters at 1mV':>17}{'at 5mV':>9}{'at 20mV':>9}"
      f"{'largest cluster':>17}{'nearest pair':>14}")
for label, dat, prefix, ctf in conditions:
    interior = setup(dat)
    patterns = np.load(f'{prefix}_vmem_final.npy')[:, interior] * 1000.0
    patterns = patterns - patterns.mean(axis=1, keepdims=True)
    perCell = pdist(patterns) / np.sqrt(patterns.shape[1])       # per-cell RMS separation, mV
    tree = linkage(perCell, method='average')
    counts = {t: len(np.unique(fcluster(tree, t=t, criterion='distance'))) for t in (1.0, 5.0, 20.0)}
    biggest = max(np.bincount(fcluster(tree, t=5.0, criterion='distance')))
    print(f"{label:>18}{ctf:>9.3f}{counts[1.0]:>17}{counts[5.0]:>9}{counts[20.0]:>9}"
          f"{biggest:>16} {perCell.min():>13.2f}")
print()
print("  Clusters are counted on per-cell RMS separation, so the threshold is the voltage")
print("  difference a downstream cell would have to resolve to tell two outcomes apart.")
