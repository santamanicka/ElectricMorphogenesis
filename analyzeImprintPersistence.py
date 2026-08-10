"""Does the clamp's imprint survive after the clamp is released?

One model showed the Vmem spread collapse from 47 mV to 5 and later regrow, which read as the tissue
erasing what the clamp wrote and producing something of its own. Measured across configurations that
turns out to be true of that model alone: every other run, at either lattice size, either horizon,
and with or without the transduction bias learned, holds 45 to 48 mV from the release of the clamp
through iteration 2400. The one exception had its bias driven to 0.28 of default, far enough to
quench the tissue, so the collapse is that quenching seen over time rather than anything about
stigmergy.

Amplitude is all this measures. A spread that holds says nothing about whether the arrangement inside
it holds, which is the question the persistence figure addresses.
"""
import numpy as np, torch, sys
from embryo import model
torch.set_grad_enabled(False)
models = [('data/bestModelParameters_fieldVector_30x30_511.dat','bias learned, 2500',30),
          ('data/bestModelParameters_fieldVector_30x30_612.dat','clamp only, 2500 (612)',30),
          ('data/bestModelParameters_fieldVector_30x30_616.dat','clamp only, 2500 (616)',30),
          ('data/bestModelParameters_fieldVector_24.dat','bias learned, 1000 (task 24)',30),
          ('data/bestModelParameters_fieldVector_30x30_708.dat','clamp only, 1000 (708)',30),
          ('data/StigmergicModelParameters.dat','11x11 reference',11)]
print(f"  Vmem span (max-min, mV) through the run; clamp is released at iteration 100")
print(f"  {'model':32s} {'i200':>7s} {'i600':>7s} {'i1000':>7s} {'i1400':>7s} {'i2000':>7s} {'i2400':>7s} {'min':>7s}")
for f, label, n in models:
    try:
        p = torch.load(f, weights_only=False)
    except Exception as e:
        print(f"  {label:32s} (unavailable)"); continue
    cells=n*n; ns=p['simParameters']['numSamples']; iv=p['simParameters']['initialValues']
    if 'ligandConc' not in iv: iv['ligandConc']=torch.zeros((ns,cells,1),dtype=torch.float64)
    p['latticePeriodicBoundaryGJ']=False; p['ATPParameters']=None
    m=model(p,ns); m.setExperimentalConditions((iv,ns))
    m.simulate(clampParameters=p['clampParameters'],fieldModulation=True,numSimIters=2500,
               storeVariables=['Vmem'],storeStride=100)
    V=torch.stack(list(m.timeseriesVmem)).reshape(-1,cells).numpy()*1000
    span=V.max(axis=1)-V.min(axis=1)
    vals=[span[i] for i in (2,6,10,14,20,24)]
    mid=span[2:22].min()
    print(f"  {label:32s} " + " ".join(f"{v:7.2f}" for v in vals) + f" {mid:7.2f}")
