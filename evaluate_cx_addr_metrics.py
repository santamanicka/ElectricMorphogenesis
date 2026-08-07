"""Score CX/ADDR metrics on split-half stability and curve smoothness before trusting any curve."""
import argparse, ast, gc
import numpy as np, torch, utilities
from embryo import model
import cx_addr_metrics as M

parser = argparse.ArgumentParser()
parser.add_argument('--sweepDir',     type=str, default='data/fieldRangeSweep')
parser.add_argument('--ranges',       type=str, default='[2,3,4,5,6,8,10,11,13,15,17]')
parser.add_argument('--sourceDat',    type=str, default='data/StigmergicModelParameters_30x30.dat')
parser.add_argument('--floor',        type=float, default=M.FLOOR_MV)
parser.add_argument('--outputPrefix', type=str, default='data/cxAddr30x30')
args = parser.parse_args()
ranges = ast.literal_eval(args.ranges)
torch.set_grad_enabled(False)

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

records = {}
for r in ranges:
    code = np.load(f'{args.sweepDir}/screen{r:02d}_gpol_prepatterns.npy')[:, boundary]
    vmem = np.load(f'{args.sweepDir}/screen{r:02d}_vmem_final.npy') * 1000
    record = dict(M.metricsForRegion(code, vmem[:, interior], floor=args.floor))
    record.update(M.depthFairMetrics(code, vmem, shells, floor=args.floor))
    half = len(vmem) // 2
    halves = [M.metricsForRegion(code[i:i + half], vmem[i:i + half][:, interior],
                                 floor=args.floor, seed=s) for s, i in enumerate((0, half))]
    record['splitHalf'] = {k: (halves[0][k], halves[1][k]) for k in halves[0]}
    record['numSamples'] = len(vmem)
    records[r] = record
    print(f"  range {r:>2}: CX {record['CX_variance']:>9.0f}  ADDR {record['ADDR_variance']:>8.0f}  "
          f"frac {record['ADDR_fraction']:>5.3f} | fair CX {record['fair_CX_perCell']:>7.2f}  "
          f"fair ADDR {record['fair_ADDR_perCell']:>7.3f}  fair frac {record['fair_ADDR_fraction']:>5.3f}")

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
