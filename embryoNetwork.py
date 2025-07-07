import torch
import numpy as np
from embryo import model
import utilities

class embryoNetwork():

    # arguments will be passed by the GA or by any program that wants to call the simulation module
    def __init__(self,parameters=None,nsamples=1,niters=100):
        self.utils = utilities.utilities()
        self.dims = parameters['dims']
        self.modelNumEmbryo = parameters['modelNumEmbryo']
        self.modelNumATP = parameters['modelNumATP']
        self.teratogenExposed = parameters['teratogenExposure']
        self.nrows, self.ncols = self.dims
        self.numEmbryos = self.nrows * self.ncols
        self.nsamples = nsamples
        self.niters = niters
        self.grid = [[self.instantiateEmbryo(modelNum=self.modelNumEmbryo) for j in range(self.ncols)] for i in range(self.nrows)]

    def instantiateEmbryo(self,row=0,col=0,modelNum=0):
        Sfx = '_fieldVector_Ligand' + '_GRN'
        parameterfilename = './data/bestModelParameters' + Sfx + '_' + str(modelNum) + '.dat'
        embryoParameters = torch.load(parameterfilename)
        embryoParameters['GRNParameters']['GRNWeights'] /= 11.5
        embryoParameters['GRNParameters']['InterGRNWeights'] /= 11.5
        embryoParameters['ATPParameters'] = dict()
        embryoParameters['ATPParameters']['ATPEnabled'] = True
        embryoParameters['ATPParameters']['ATPReactionStrength'] = 1.0
        embryoParameters['ATPParameters']['ATPDiffusionStrength'] = 10.0
        embryoParameters['ATPParameters']['tissueConnectivity'] = \
            self.utils.computeLatticeAdjacencyMatrix(latticeDims=embryoParameters['latticeDims'],periodicBoundary=False)
        embryoinstance = model(embryoParameters)
        return embryoinstance

    def simulateEmbryo(self,sampleNumber=0,row=0,col=0,save=False):
        # embryoinstance = self.grid[row][col]
        embryoinstance = self.grid[0][0]
        initialValues = embryoinstance.parameters['simParameters']['initialValues']
        if self.teratogenExposed:
            initialValues['ATPConc'] = torch.ones((self.nsamples,embryoinstance.numCells,1),dtype=torch.float64) * 2.5
        else:
            initialValues['ATPConc'] = torch.ones((self.nsamples,embryoinstance.numCells,1),dtype=torch.float64) * 11.5
        embryoinstance.setExperimentalConditions((initialValues, 1))
        clampParameters = embryoinstance.parameters['clampParameters']
        numCells = embryoinstance.numCells
        circuit = embryoinstance.electricNetwork
        boundaryCells = self.utils.computeDomeIndices(circuit,mode='tissue')
        externalInputs = dict()
        externalInputs['ATP'] = torch.zeros((self.nsamples,self.niters,numCells,1),dtype=torch.float64)
        ATPCurrentEmbryo = self.ATPCurrent[:,:,row,col].unsqueeze(2).repeat(1,1,len(boundaryCells))  # shape = (nsamples,niters,nboundary)
        externalInputs['ATP'][:,:,boundaryCells,0] = ATPCurrentEmbryo
        embryoinstance.simulate(externalInputs=externalInputs,clampParameters=clampParameters,numSimIters=self.niters)
        if save:
            self.savedSims[sampleNumber][row,col] = embryoinstance.timeseriesVmem[-1,0,:,0].numpy()
        del embryoinstance, self.grid[0][0]  # saves memory

    def simulateATPFlow(self,modelNum=0):
        params = torch.load('./data/survival_'+str(modelNum)+'.dat')
        parameterNames = params['bestParameters'].keys()
        for parameter in parameterNames:
            value = params['bestParameters'][parameter].numpy()
            # exec(f"{parameter} = value")
            setattr(self,parameter,value)
        Adjacency = self.utils.computeLatticeAdjacencyMatrix(self.dims,periodicBoundary=False,dtype='numpy')
        Degree = np.diag(np.sum(Adjacency,1))
        Laplacian = Degree - Adjacency
        minDim, maxDim, minNoise = 1, 10, 0.0
        if self.teratogenExposed:
            unstableEquilibrium = 2.5
        else:
            unstableEquilibrium = 11.5
        std = lambda dim: (((dim-minDim)/(maxDim-minDim))*(self.maxNoise-minNoise))+minNoise
        self.ATPCurrent = np.zeros((self.nsamples,self.niters,self.numEmbryos))
        self.ATPConcs = np.random.normal(unstableEquilibrium, std(self.nrows),(self.nsamples,self.numEmbryos,1))
        for iter in range(self.niters):
            diffusionCurrent = self.w1 * np.matmul(Laplacian,self.ATPConcs)
            dATP = ((self.a*pow(self.ATPConcs+self.xoff,3)) + (self.b*pow(self.ATPConcs+self.xoff,2)) +
                    (self.c*(self.ATPConcs+self.xoff)) + self.d + diffusionCurrent)
            self.ATPConcs = self.ATPConcs + (dATP * 0.01)
            self.ATPCurrent[:,iter] = diffusionCurrent.squeeze(2)
        self.ATPCurrent = self.ATPCurrent.reshape((self.nsamples,self.niters,self.nrows,self.ncols))
        self.ATPCurrent = torch.DoubleTensor(self.ATPCurrent)

    def simulate(self,sampleNumber=0,save=False):
        # simulate single multi-embryo ATP network
        self.simulateATPFlow(modelNum=self.modelNumATP)
        # simulate multiple single-embryo patterning networks
        for row in range(self.nrows):
            for col in range(self.ncols):
                self.simulateEmbryo(sampleNumber,row,col,save=save)
            del self.grid[0]  # save memory



