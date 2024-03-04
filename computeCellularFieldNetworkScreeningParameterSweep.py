import numpy as np
import torch
from itertools import chain
from cellularFieldNetwork import cellularFieldNetwork
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

circuitRows,circuitCols = 5,5
circuitDims = (circuitRows,circuitCols)  # (rows,columns) of lattice
fieldResolution = 1
fieldStrength = 10.0
# GapJunctionStrength = 1.0
maxNumBoundingSquares = 2*max(circuitDims) - 1  # Max value of numBoundingSquares so the field will permeate the entire tissue = 2(l-1)+1, where l is the max of circuitDims
# numBoundingSquares = 4
eVBias = torch.DoubleTensor([0.0214])  # 0.0214
eVWeight = torch.DoubleTensor([9.4505])  # 9.4505
evTimeConstant = torch.DoubleTensor([10.0])
numSamples = 1
numSimIters = 5000
numCells = circuitRows * circuitCols
BlockGapJunctions = False
AmplifyGapJunctions = False

VmemBins = np.linspace(-0.0, -0.1, 3)

fieldParameters = (fieldResolution,fieldStrength,(eVBias,eVWeight,evTimeConstant))

initialValues = dict()
initVmem = list(chain([-9.2e-3] * numSamples))
initialValues['Vmem'] = torch.repeat_interleave(torch.DoubleTensor(initVmem),numCells,0).view(numSamples,numCells,1)
initialValues['G_pol'] = dict()
initialValues['G_pol']['cells'] = [[[0]]] * numSamples
initialValues['G_pol']['values'] = [torch.DoubleTensor([1.0])] * numSamples  # bistable
initialValues['G_dep'] = dict()
initialValues['G_dep']['cells'] = []
initialValues['G_dep']['values'] = torch.DoubleTensor([])

def generateTimeSeriesMovie(data,numBoundingSquares,GapJunctionStrength):
    dims = circuitRows,circuitCols
    data = data[:,0,:,0].reshape(-1,*dims)

    fig, ax = plt.subplots()

    heatmap = ax.pcolormesh(data[0], cmap='hot', vmin=-0.06, vmax=0.0)

    def animate(t):
        # Update the heatmap
        heatmap.set_array(data[t])

        # Return the updated heatmap
        return heatmap,

    ani = FuncAnimation(fig, animate, frames=np.arange(0,data.shape[0],100), interval=0.1, blit=True)
    Writer = animation.writers['ffmpeg']
    mywriter = Writer(fps=100, metadata=dict(artist='Me'))
    duration = int(numSimIters/1000)
    ani.save('./data/VmemSpatial_' + str(duration) + 'K_' + str(circuitRows) + 'x' + str(circuitCols) +
             '_ScreenSize' + str(numBoundingSquares) + '_GJ' + str(GapJunctionStrength) + '.mp4', writer=mywriter)

for GapJunctionStrength in [0.05,0.5,1.0]:
    for numBoundingSquares in range(1,maxNumBoundingSquares+1,2):
        print('GJStrength = ', GapJunctionStrength, "numBoundingSquares = ",numBoundingSquares)
        fieldParameters = (fieldResolution,fieldStrength,(eVBias,eVWeight,evTimeConstant))
        circuit = cellularFieldNetwork(circuitDims, GRNParameters=(None, None, None, None),
                                       fieldParameters=fieldParameters, numSamples=numSamples)
        circuit.initVariables(initialValues)
        circuit.initParameters(initialValues)
        circuit.G_0 = GapJunctionStrength * circuit.G_ref
        inputs = {'gene':None}
        fieldScreenParameters = {'numBoundingSquares':numBoundingSquares}
        circuit.simulate(inputs=inputs,fieldEnabled=True,fieldClampParameters=None,fieldScreenParameters=None,
                     perturbationParameters=None,numSimIters=numSimIters,stochasticIonChannels=False,saveData=True)
        data = circuit.timeseriesVmem.detach().numpy()
        generateTimeSeriesMovie(data,numBoundingSquares,GapJunctionStrength)
