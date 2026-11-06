"""20-point sub-grid of S1 (R_s in {2,4,6,8} x bias/bias_* in {1/4,1/2,1,2,4}) on a 25x25 tissue for the original (lattice),
which is ~8x slower than Potentia at this size.  usage: python sub_scans.py <lattice|potentia:2>"""
import sys
import atlas
from sweeps import STIG
B0 = STIG['bias']
atlas.SWEEPS['S1sub'] = dict(axes={'fieldScreenSize': [2, 4, 6, 8], 'bias': [B0 * 2 ** (k / 2) for k in (-4, -2, 0, 2, 4)]}, params={})
if __name__ == '__main__':
    atlas.run('S1sub', 25, sys.argv[1], T=300.0)
