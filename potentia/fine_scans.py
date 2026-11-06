"""Fine 1-D scans through the Stigmergic point with jitter replicates (chaotic dynamics => single runs are noisy).

 F1: bias / bias_*  = 2^(k/6), k = -9..10 (20 values) at R_s = 4;   replicates: weight x (1, 1.003, 0.997)
 F2: R_s = 3.0 .. 5.0 step 0.25 (9 values) at bias = bias_*;       replicates: bias  x (1, 1.003, 0.997)
Run for original (lattice) and Potentia m=2, m=4.  usage: python fine_scans.py <F1|F2> <model>
"""
import sys
import numpy as np
import atlas
from sweeps import STIG

B0, W0 = STIG['bias'], STIG['weight']
atlas.SWEEPS['F1'] = dict(axes={'bias': [B0 * 2 ** (k / 6) for k in range(-9, 11)], 'weight': [W0, W0 * 1.003, W0 * 0.997]}, params={})
atlas.SWEEPS['F2'] = dict(axes={'fieldScreenSize': [3.0 + 0.25 * i for i in range(9)], 'bias': [B0, B0 * 1.003, B0 * 0.997]}, params={})

if __name__ == '__main__':
    sweep, model = sys.argv[1], sys.argv[2]
    atlas.run(sweep, 11, model, T=300.0)
