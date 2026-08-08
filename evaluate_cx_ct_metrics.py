"""Score CX/CT metrics on split-half stability and curve smoothness before trusting any curve."""
import argparse, ast, gc
import numpy as np, torch, utilities
from embryo import model
import cx_ct_metrics as M

parser = argparse.ArgumentParser()
parser.add_argument('--sweepDir',     type=str, default='data/fieldRangeSweep')
parser.add_argument('--ranges',       type=str, default='[2,3,4,5,6,8,10,11,13,15,17]')
parser.add_argument('--sourceDat',    type=str, default='data/StigmergicModelParameters_30x30.dat')
parser.add_argument('--floor',        type=float, default=M.FLOOR_MV)
parser.add_argument('--maxSamples',   type=int, default=0,
                    help='truncate every condition to this many clamps; 0 uses the smallest '
                         'available. Capacity in bits grows with sample count -- range 2 reads '
                         '14.0 bits at N=200 and 16.7 at N=400 -- so conditions must be compared '
                         'at equal N or a short ensemble reads as a poorer tissue')
parser.add_argument('--keepUniformShift', action='store_true',
                    help='count the whole-interior voltage level as pattern (off by default)')
parser.add_argument('--outputPrefix', type=str, default='data/cxCt30x30')
args = parser.parse_args()
ranges = ast.literal_eval(args.ranges)
torch.set_grad_enabled(False)


def screenDirName(value):
    """Directory for a screen size, integer or fractional (2.5 -> screen02p5)."""
    return (f'screen{int(value):02d}' if float(value).is_integer()
            else f'screen{int(value):02d}p{round((float(value) % 1) * 10):.0f}')


def actionReach(sourceDat, screenSize):
    """Grid points per cell the field actually reaches, which is what the parameter stands for.

    fieldScreenSize is a distance threshold on a discrete extracellular grid, so it quantises:
    sizes 1 and 2 both give a 4-point neighbourhood and are the same condition, while 2 to 3
    jumps from 4 to 15.5. Plotting against the parameter would show a cliff that is partly the
    discretisation rather than the tissue, so the reach is recorded and plotted instead.
    """
    parameters = torch.load(sourceDat, weights_only=False)
    parameters['ATPParameters'] = None
    parameters['latticePeriodicBoundaryGJ'] = False
    parameters['fieldParameters']['fieldScreenSize'] = screenSize
    cells = parameters['latticeDims'][0] * parameters['latticeDims'][1]
    values = parameters['simParameters']['initialValues']
    if 'ligandConc' not in values:
        values['ligandConc'] = torch.zeros((1, cells, 1), dtype=torch.float64)
    instance = model(parameters, 1)
    reach = float(instance.electricNetwork.fieldScreenMatrix.sum().item()) / instance.electricNetwork.numCells
    del instance; gc.collect()
    return reach

p = torch.load(args.sourceDat, weights_only=False)
numRows, numCols = p['latticeDims']; numCells = numRows * numCols
p['ATPParameters'] = None; p['latticePeriodicBoundaryGJ'] = False
v = p['simParameters']['initialValues']
if 'ligandConc' not in v: v['ligandConc'] = torch.zeros((1, numCells, 1), dtype=torch.float64)
m = model(p, 1)
dome = utilities.utilities().computeDomeIndices(m.electricNetwork, mode='tissue')
boundary = np.zeros(numCells, bool); boundary[dome] = True; interior = ~boundary; del m; gc.collect()
rows, cols = np.divmod(np.arange(numCells), numCols)
bR, bC = rows[boundary], cols[boundary]
depth = np.array([np.min(np.maximum(np.abs(bR - r), np.abs(bC - c))) for r, c in zip(rows, cols)])
shells = M.depthShells(depth, interior)
print(f"{numRows}x{numCols}: {interior.sum()} interior cells, {len(shells)} shells "
      f"(sizes {[int(c.sum()) for _, _, c in shells]}), floor {args.floor} mV")

available = [len(np.load(f'{args.sweepDir}/{screenDirName(r)}_vmem_final.npy', mmap_mode='r'))
             for r in ranges]
commonSamples = args.maxSamples if args.maxSamples > 0 else min(available)
if len(set(available)) > 1:
    print(f"  sample counts differ across conditions {sorted(set(available))}; "
          f"truncating all to {commonSamples} so capacity is comparable")

records = {}
for r in ranges:
    code = np.load(f'{args.sweepDir}/{screenDirName(r)}_gpol_prepatterns.npy')[:, boundary]
    vmem = np.load(f'{args.sweepDir}/{screenDirName(r)}_vmem_final.npy')[:commonSamples] * 1000
    code = code[:commonSamples]
    # A tissue that stays perfectly uniform and merely shifts its whole voltage level with the
    # code is not patterning, but principal components see that shift as a mode and multiply its
    # variance by every cell in the region: a synthetic ensemble with no spatial structure at all
    # scores 19061 mV^2 and 4.98 bits. Removing each sample's interior-wide mean deletes exactly
    # that one degenerate mode. It is taken over the whole interior rather than per shell, so
    # radial structure -- shells genuinely sitting at different levels -- is preserved.
    # The uniform level is reported as its own channel; each region is centred on its own mean
    # inside metricsForRegion, so nothing is subtracted from the array here.
    uniformLevel = vmem[:, interior].mean(axis=1)
    levelBits = float(-0.5 * np.log2(1 - np.clip(
        M.crossValidatedR2(code, uniformLevel)[0], 0.0, 0.999)))
    record = dict(M.metricsForRegion(code, vmem[:, interior], floor=args.floor))
    record.update(M.depthFairMetrics(code, vmem, shells, floor=args.floor))
    half = len(vmem) // 2
    halves = [M.metricsForRegion(code[i:i + half], vmem[i:i + half][:, interior],
                                 floor=args.floor, seed=s) for s, i in enumerate((0, half))]
    record['splitHalf'] = {k: (halves[0][k], halves[1][k]) for k in halves[0]}
    record['numSamples'] = len(vmem)
    record['uniformShift_bits'] = levelBits
    record['uniformShift_std'] = float(uniformLevel.std())
    record['reach'] = actionReach(args.sourceDat, r)
    records[r] = record
    print(f"  range {r:>4} (reach {record['reach']:>5.1f}): CX {record['CX_variance']:>9.0f}  CT {record['CT_variance']:>8.0f}  "
          f"frac {record['CT_fraction']:>5.3f} | fair CX {record['fair_CX_perCell']:>7.2f}  "
          f"fair CT {record['fair_CT_perCell']:>7.3f}  bits {record['CT_bits']:>5.1f}  "
          f"| uniform shift {record['uniformShift_std']:>5.2f} mV, {levelBits:>4.1f} bits")

def jaggedness(values):
    values = np.asarray(values, float)
    span = values.max() - values.min()
    return float(np.abs(np.diff(values, 2)).mean() / span) if span > 0 else 0.0

print(f"\n{'metric':>22} {'split-half':>11} {'jaggedness':>11}")
keys = [k for k in records[ranges[0]] if not k.startswith(('profile_', 'splitHalf', 'numSamples'))]
summary = {}
for k in keys:
    curve = [records[r][k] for r in ranges]
    span = max(curve) - min(curve)
    if k in records[ranges[0]]['splitHalf'] and span > 0:
        pairs = [records[r]['splitHalf'][k] for r in ranges]
        discrepancy = float(np.mean([abs(a - b) for a, b in pairs]) / span)
    else:
        discrepancy = np.nan
    summary[k] = (discrepancy, jaggedness(curve))
    print(f"{k:>22} {discrepancy:>11.3f} {jaggedness(curve):>11.3f}")
np.savez(f'{args.outputPrefix}_metricEvaluation.npz', records=records, ranges=ranges,
         shellSizes=[int(c.sum()) for _, _, c in shells], summary=summary, floor=args.floor)
print(f"Saved {args.outputPrefix}_metricEvaluation.npz")
