"""Check the numpy lattice port against the *recorded* result in data/StigmergicModelParameters.dat.

VALIDATION ONLY.  The recorded run used an oscillatory boundary-field clamp, which Potentia
deliberately does not implement; the clamp is replayed here (copied from model.simulate +
cellularFieldNetwork.updateExtracellularVoltage('eVClamp')) purely to confirm that the port
reproduces the stored bestLoss and final Vmem.
"""
import sys, os
import numpy as np
import torchfree_io as tio
from lattice import Lattice

p = tio.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'StigmergicModelParameters.dat'))
fp, cp, tp = p['fieldParameters'], p['clampParameters'], p['trainParameters']
L = Lattice(dims=p['latticeDims'], fieldScreenSize=fp['fieldScreenSize'], fieldStrength=fp['fieldStrength'],
            gain=fp['fieldTransductionGain'], weight=float(fp['fieldTransductionWeight'][0]),
            bias=float(fp['fieldTransductionBias'][0]), tau=float(fp['fieldTransductionTimeConstant'][0]),
            GJStrength=p['GJParameters']['GJStrength'], fieldVector=fp['fieldVector'],
            fieldRangeSymmetric=fp['fieldRangeSymmetric'],
            leak=(float(os.environ['LEAK']) if 'LEAK' in os.environ else None))
nsim = p['simParameters']['numSimIters']
clamp_idx = cp['clampIndices'][1]
vals = cp['clampValues']
free = np.setdiff1d(np.arange(L.xf.size), clamp_idx)
# clamp-source geometry: free points <- clamp points, *no* screening radius (as in the repo)
dxx = L.xf[free][:, None] - L.xf[clamp_idx][None, :]
dyy = L.yf[free][:, None] - L.yf[clamp_idx][None, :]
rinv = 1.0 / np.sqrt(dxx ** 2 + dyy ** 2)

V, G = L.initial()
ts = []
for it in range(nsim):
    ts.append(V.copy())
    V, G, eV = L.step(V, G, return_eV=True)
    if cp['clampStartIter'] <= it <= cp['clampEndIter']:
        eV = eV.copy()
        eV[clamp_idx] = vals[it]
        Q = L.C * eV[clamp_idx]
        ex = L.fc * ((rinv ** 2) * dxx) @ Q
        ey = L.fc * ((rinv ** 2) * dyy) @ Q
        eV[free] += np.sqrt(ex ** 2 + ey ** 2)
        G = L.update_G(G, L.perceive(eV))
        V = V + L.dt * L.current(V, G) / L.C

ts = np.array(ts)
nev = int(tp['evalDurationProp'] * nsim)
target = tp['targetVmem'].reshape(-1)
loss = np.sqrt(((target[None, :] - ts[-nev:]) ** 2).sum())
rec_loss = tp['bestLoss']
rec_V = tp['actualVmem'].reshape(-1)
print(f'recorded bestLoss     = {rec_loss:.6f}')
print(f'port loss (last {nev} it) = {loss:.6f}    |diff| = {abs(loss-rec_loss):.2e}')
print(f'final Vmem: max|port - recorded| = {np.abs(V-rec_V).max()*1e3:.4f} mV  (recorded range {rec_V.min()*1e3:.1f}..{rec_V.max()*1e3:.1f} mV)')
ok = abs(loss - rec_loss) / rec_loss < 1e-3 and np.abs(V - rec_V).max() < 1e-4
print('PORT MATCHES RECORDED RESULT' if ok else 'PORT DOES NOT MATCH RECORDED RESULT')
sys.exit(0 if ok else 1)
