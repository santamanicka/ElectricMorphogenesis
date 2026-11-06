"""The GRN provision is inert by default and wired correctly when supplied."""
import sys, numpy as np
from potentia import Potentia
import torchfree_io as tio
assert 'geneRegulatoryNetwork' not in sys.modules and 'torch' not in sys.modules
P0 = Potentia(size_cells=(9, 9), m=2)
stub = dict(state=np.zeros((18, 18, 2)), biases=np.zeros(2), toVmemWeights=np.zeros(2))
def pair(leak):
    A = Potentia(size_cells=(9, 9), m=2, leak=leak); B = Potentia(size_cells=(9, 9), m=2, leak=leak, grn=dict(stub))
    V0, g0 = A.initial(); V1, g1 = V0.copy(), g0.copy()
    for _ in range(300):
        V0, g0 = A.step(V0, g0); V1, g1 = B.step(V1, g1)
    return np.abs(V0 - V1).max(), np.abs(g0 - g1).max(), g0
dV, dg, g0 = pair(0.0)
print('zero-weight hook, leak=0: bit-identical to no hook:', dV == 0 and dg == 0)
dV, dg, g0 = pair(None)
print(f'zero-weight hook, as-implemented leak: differs only by the extra 1e-9 leak step (max|dg|={dg:.1e}):', dV < 1e-7 and dg < 1e-7)
P0 = Potentia(size_cells=(9, 9), m=2)
V0, g0 = P0.initial()
for _ in range(300): V0, g0 = P0.step(V0, g0)
P2 = Potentia(size_cells=(9, 9), m=2, grn=dict(state=np.ones((18, 18, 2)), biases=np.zeros(2), toVmemWeights=np.array([50., 0.])))
V2, g2 = P0.initial()
for _ in range(300): V2, g2 = P2.step(V2, g2)
print('non-zero gene coupling changes g (hook is live):', np.abs(g2 - g0).max() > 0.1)
p = tio.load('../data/StigmergicModelParameters.dat')
S = Potentia.from_repo_parameters(p, m=2)
print('from_repo_parameters: size', S.size_cells, 'R_s', S.R_s, 'bias', S.bias, 'GJ', S.GJ, 'GRNEnabled', S.GRNEnabled, '| imported torch/GRN module:', 'torch' in sys.modules or 'geneRegulatoryNetwork' in sys.modules)
