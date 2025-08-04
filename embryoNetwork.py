import torch
import numpy as np
from embryo import model
import utilities

class embryoNetwork():

    # arguments will be passed by the GA or by any program that wants to call the simulation module
    def __init__(self,parameters=None,nsamples=1,niters=100):
        self.utils = utilities.utilities()
        self.dims = parameters['dims']
        self.modelNameEmbryo = parameters['modelNameEmbryo']
        self.modelNameATP = parameters['modelNameATP']
        self.fieldModulation = parameters['fieldModulation']
        self.LigandEnabled = parameters['LigandEnabled']
        self.GRNEnabled = parameters['GRNEnabled']
        self.teratogenExposed = parameters['teratogenExposure']
        self.boundaryAssistance = parameters['boundaryAssistance']
        self.nrows, self.ncols = self.dims
        self.numEmbryos = self.nrows * self.ncols
        self.nsamples = nsamples
        self.niters = niters
        self.grid = [[self.instantiateEmbryo(modelNum=self.modelNameEmbryo,GRNEnabled=self.GRNEnabled,LigandEnabled=self.LigandEnabled,
                                             fieldModulation=self.fieldModulation) for j in range(self.ncols)] for i in range(self.nrows)]

    def instantiateEmbryo(self,modelNum=0,GRNEnabled=False,LigandEnabled=False,fieldModulation=False):
        Sfx = '_fieldVector'
        if LigandEnabled:
            Sfx += '_Ligand'
        if GRNEnabled:
            Sfx += '_GRN'
        parameterfilename = './data/bestModelParameters' + Sfx + '_' + modelNum + '.dat'
        embryoParameters = torch.load(parameterfilename)
        if GRNEnabled:
            embryoParameters['GRNParameters']['GRNWeights'] /= 11.5  # assuming ATP model = 262
            embryoParameters['GRNParameters']['InterGRNWeights'] /= 11.5
        if fieldModulation and (not GRNEnabled) and (not LigandEnabled):
            # embryoParameters['fieldParameters']['fieldModulation'] = True
            embryoParameters['fieldParameters']['fieldTransductionGain'] /= 11.5  # assuming ATP model = 262
        else:
            embryoParameters['fieldParameters']['fieldModulation'] = False
        embryoParameters['ATPParameters'] = dict()
        embryoParameters['ATPParameters']['ATPEnabled'] = True
        embryoParameters['ATPParameters']['ATPReactionStrength'] = 1.0
        embryoParameters['ATPParameters']['ATPDiffusionStrength'] = 10.0
        embryoParameters['ATPParameters']['tissueConnectivity'] = \
            self.utils.computeLatticeAdjacencyMatrix(latticeDims=embryoParameters['latticeDims'],periodicBoundary=False)
        embryoinstance = model(embryoParameters)
        return embryoinstance

    def simulateEmbryo(self,row=0,col=0,save=False):
        # embryoinstance = self.grid[row][col]
        embryoinstance = self.grid[0][0]  # this instance will be deleted at the end
        numCells = embryoinstance.numCells
        initialValues = embryoinstance.parameters['simParameters']['initialValues']
        if self.teratogenExposed:
            initialValues['ATPConc'] = torch.ones((self.nsamples,embryoinstance.numCells,1),dtype=torch.float64)
            initialValues['ATPConc'][:,:,0] = self.ATPConcsInit[:,row,col].unsqueeze(1).repeat(1,numCells)  # makes sense only if nsamples=1
        else:
            initialValues['ATPConc'] = torch.ones((self.nsamples,embryoinstance.numCells,1),dtype=torch.float64) * 11.5
        embryoinstance.setExperimentalConditions((initialValues, 1))
        clampParameters = embryoinstance.parameters['clampParameters']
        circuit = embryoinstance.electricNetwork
        boundaryCells = self.utils.computeDomeIndices(circuit,mode='tissue')
        externalInputs = dict()
        externalInputs['ATP'] = torch.zeros((self.nsamples,self.niters,numCells,1),dtype=torch.float64)
        if self.boundaryAssistance:
            ATPCurrentEmbryo = self.ATPCurrent[:,:,row,col].unsqueeze(2).repeat(1,1,len(boundaryCells))  # shape = (nsamples,niters,nboundary)
            externalInputs['ATP'][:,:,boundaryCells, 0] = ATPCurrentEmbryo
        else:  # full-embryo assistance
            ATPCurrentEmbryo = self.ATPCurrent[:,:,row,col].unsqueeze(2).repeat(1,1,numCells)  # shape = (nsamples,niters,nboundary)
            externalInputs['ATP'][:,:,:,0] = ATPCurrentEmbryo
        embryoinstance.simulate(externalInputs=externalInputs,clampParameters=clampParameters,numSimIters=self.niters,fieldModulation=self.fieldModulation)
        if save:
            self.savedSims[row,col] = embryoinstance.timeseriesVmem[-1,0,:,0].numpy()
        del embryoinstance, self.grid[0][0]  # saves memory

    def simulateATPFlow(self,modelName='0'):
        params = torch.load('./data/survival_'+str(modelName)+'.dat')
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
        self.ATPConcs = np.random.normal(unstableEquilibrium,std(self.nrows),(self.nsamples,self.numEmbryos,1))
        self.ATPConcsInit = self.ATPConcs.copy()
        for iter in range(self.niters):
            diffusionCurrent = self.w1 * np.matmul(Laplacian,self.ATPConcs)
            dATP = ((self.a*pow(self.ATPConcs+self.xoff,3)) + (self.b*pow(self.ATPConcs+self.xoff,2)) +
                    (self.c*(self.ATPConcs+self.xoff)) + self.d + diffusionCurrent)
            self.ATPConcs = self.ATPConcs + (dATP * 0.01)
            self.ATPCurrent[:,iter] = diffusionCurrent.squeeze(2)
        self.ATPCurrent = self.ATPCurrent.reshape((self.nsamples,self.niters,self.nrows,self.ncols))
        self.ATPCurrent = torch.DoubleTensor(self.ATPCurrent)
        self.ATPConcsInit = self.ATPConcsInit.reshape((self.nsamples,self.nrows,self.ncols))
        self.ATPConcsInit = torch.DoubleTensor(self.ATPConcsInit)

    def simulate(self,save=False):
        # simulate single multi-embryo ATP network
        self.simulateATPFlow(modelName=self.modelNameATP)
        # simulate multiple single-embryo patterning networks
        for row in range(self.nrows):
            for col in range(self.ncols):
                self.simulateEmbryo(row,col,save=save)
            del self.grid[0]  # saves memory



