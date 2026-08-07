"""How many dimensions of interior pattern can the boundary code write, at readable amplitude?

The addressability index is a rank statistic and the R^2 of a regression is a variance ratio.
Both are scale-free, so both certify a perfectly controlled signal regardless of whether that
signal is 60 mV or a nanovolt. That is not a hypothetical failure: at action range 2 the tissue
deeper than four cells from the boundary is uniform to about 1e-6 mV, and on that numerical
residue the rank index scores +0.387 at z=+17.9 and this regression finds 7.18 controlled
dimensions. Both are real in float64 and meaningless in a cell.

So dimensions are counted only where the mode's amplitude clears a resolution floor -- the
smallest voltage difference a downstream reader could act on. With the floor applied, range 2
retains 12.96 addressable dimensions across the interior as a whole and exactly 0.00 in its
core, which is the correct answer: it writes a rich pattern into a shell about four cells deep
and nothing at all beyond that.

Because the shell's thickness is set by the field action range rather than by the tissue, this
is also why the result does not scale. At 11x11 a four-cell shell is nearly the whole tissue; at
30x30 it is the periphery, leaving 400 of 784 interior cells untouched.
"""
import numpy as np, torch, gc, utilities
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from embryo import model
torch.set_grad_enabled(False)
FLOOR = 0.1   # mV a downstream reader must be able to resolve

p = torch.load('data/StigmergicModelParameters_30x30.dat', weights_only=False)
numRows, numCols = p['latticeDims']; numCells = numRows*numCols
p['ATPParameters']=None; p['latticePeriodicBoundaryGJ']=False
v=p['simParameters']['initialValues']
if 'ligandConc' not in v: v['ligandConc']=torch.zeros((1,numCells,1),dtype=torch.float64)
m=model(p,1); dome=utilities.utilities().computeDomeIndices(m.electricNetwork,mode='tissue')
b=np.zeros(numCells,bool); b[dome]=True; del m; gc.collect()
rows,cols=np.divmod(np.arange(numCells),numCols); bR,bC=rows[b],cols[b]
depth=np.array([np.min(np.maximum(np.abs(bR-r),np.abs(bC-c))) for r,c in zip(rows,cols)])

print(f"{'range':>6} {'region':>9} {'raw dims':>9} {'dims above':>11} {'largest mode':>13}")
print(f"{'':>6} {'':>9} {'(no floor)':>9} {f'{FLOOR} mV':>11} {'amplitude':>13}")
for s in (2,5,11,15,17):
    gpol=np.load(f'data/fieldRangeSweep/screen{s:02d}_gpol_prepatterns.npy')
    vm=np.load(f'data/fieldRangeSweep/screen{s:02d}_vmem_final.npy')
    X=StandardScaler().fit_transform(gpol[:,b])
    for label,mask in [('all',~b),('deep >4',depth>4)]:
        pat=vm[:,mask]*1000
        pca=PCA(n_components=40).fit(pat-pat.mean(0)); sc=pca.transform(pat-pat.mean(0))
        amp=sc.std(axis=0)
        r2=[]
        for j in range(sc.shape[1]):
            pred=cross_val_predict(RidgeCV(alphas=np.logspace(-2,6,25)),X,sc[:,j],
                                   cv=KFold(5,shuffle=True,random_state=0))
            r2.append(1-np.sum((sc[:,j]-pred)**2)/np.sum((sc[:,j]-sc[:,j].mean())**2))
        r2=np.clip(np.array(r2),0,None)
        print(f"{s:>6} {label:>9} {r2.sum():>9.2f} {r2[amp>FLOOR].sum():>11.2f} {amp.max():>10.2e} mV")
    print()
