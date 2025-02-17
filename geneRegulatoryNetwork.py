# A model of the gene regulatory network that every cell contains

# Notes:
# 1) Learned parameters and shapes:
#    a) GRNWeights of shape = (numGenes,numGenes)
#    b) InterGRNWeights of shape = (numGenes,numGenes)
#    c) VmemToGRNWeights of shape = (1,numGenes)
#    d) VmemGain of shape = (1,1)
#    e) GRNBiases of shape = (1,numGenes)  # shared with neuralPlateCircuit
#    f) VmemBias of shape = (1,1)
#    g) GRNTimeconstants of shape = (1,numGenes)
#    h) InterGRNWeightsTimeconstant of shape = (1,1)
#    h) VmemToGRNWeightsTimeconstant of shape = (1,1)
# 2) Other shapes:
#    a) tissueConnectivity = (numCells,numCells)
#    b) externalInputs = (numSamples,numCells,1)
#    c) tissueExternalInputs = (numSamples,numVariables,1)
#    d) tissueGRNWeights = (numVariables,numVariables)
#    e) tissueVmemToGRNWeights = (numVariables,1)
#    f) tissueGRNBias = (numVariables,1)
#    g) state = (numSamples,numVariables,1)
#    where, numVariables = numGenes * numCells
# 3) Parameters that need to be passed to this program in the same order as follows:
#    tissueConnectivity, GRNWeights, InterGRNWeights, VmemToGRNWeights, VmemGain, GRNBiases, VmemBias
# 4) Model parameters: Network weights and biases (structural); and external inputs to some nodes (dynamical -- can be modified via interactions with Vmem)
#    We plan to have a constant population of models whose individual network structures and parameters would be updated by a GA.
#    Therefore, there should be the following functions: modifyNetwork() and modifyParameters()
# 4) External inputs: Although we treat Vmem as "external inputs", we don't treat in the conventional CTRNN sense. Specifically, we treat
#    Vmem as a regular input (weighted sigmoid), although with an additional weight representing its gain.

import torch
from itertools import chain

class geneRegulatoryNetwork():

    def __init__(self,parameters=None,numSamples=1):
        self.parameters = parameters
        self.numSamples = numSamples
        self.defineParameters()
        self.defineVariables()
        self.composeTissueLevelGRN()
        self.timestep = 0.01

    # define parameters (weights, biases and external inputs) and populate them with default values
    def defineParameters(self):
        self.tissueConnectivity = self.parameters['tissueConnectivity']
        self.LatticeDimensions = self.parameters['latticeDims']
        self.numRows, self.numCols = self.LatticeDimensions
        self.AsymmetricInterGRN = self.parameters['GRNParameters']['AsymmetricInterGRN']  # if True it would imply that there are (4) PCP genes: left, right, top, bottom
        self.PCPAxes = self.parameters['GRNParameters']['PCPAxes']  # options: '2D', 'Horizontal'
        self.GRNWeights = self.parameters['GRNParameters']['GRNWeights']
        self.numGenes = self.parameters['GRNParameters']['GRNNumGenes']
        self.numCells = self.tissueConnectivity.shape[0]
        self.numVariables = self.numCells * self.numGenes
        self.InterGRNWeights = self.parameters['GRNParameters']['InterGRNWeights']
        self.VmemToGRNWeights = self.parameters['GRNParameters']['VmemToGRNWeights']  # NOTE: We conceived this as an Adj rather than weights matrix since we were thinking of Vmem as external inputs (with no weights by CTRNN convention)
        self.VmemGain = self.parameters['GRNParameters']['VmemGain']
        self.GRNGains = self.parameters['GRNParameters']['GRNGains']
        self.GRNBiases = self.parameters['GRNParameters']['GRNBiases']
        self.VmemBias = self.parameters['GRNParameters']['VmemBias']
        self.GRNTimeconstants = self.parameters['GRNParameters']['GRNTimeconstants']
        self.InterGRNWeightsTimeconstant = self.parameters['GRNParameters']['InterGRNWeightsTimeconstant']
        self.VmemToGRNWeightsTimeconstant = self.parameters['GRNParameters']['VmemToGRNWeightsTimeconstant']
        if self.InterGRNWeightsTimeconstant == None:
            self.InterGRNWeightsTimeconstant = torch.ones(1,1)
        if self.VmemToGRNWeightsTimeconstant == None:
            self.VmemToGRNWeightsTimeconstant = torch.ones(1,1)
        if self.GRNWeights == None:
            self.GRNWeights = torch.zeros(self.numGenes,self.numGenes)
        if self.InterGRNWeights == None:
            self.InterGRNWeights = torch.zeros(self.numGenes,self.numGenes)
        else:
            self.InterGRNWeights = self.InterGRNWeights / self.InterGRNWeightsTimeconstant
        if self.VmemToGRNWeights == None:
            self.VmemToGRNWeights = torch.zeros(1,self.numGenes)
            self.VmemToGRNWeightsTimeconstant = torch.ones(1,1)
        else:
            self.VmemToGRNWeights = self.VmemToGRNWeights / self.VmemToGRNWeightsTimeconstant
        if self.VmemGain == None:
            self.VmemGain = torch.zeros(1,1)
        if self.GRNGains == None:
            self.GRNGains = torch.ones(1,self.numGenes)
        if self.GRNBiases == None:
            self.GRNBiases = torch.zeros(1,self.numGenes)
        if self.VmemBias == None:
            self.VmemBias = torch.zeros(1,1)
        if self.GRNTimeconstants == None:
            self.GRNTimeconstants = torch.ones(1,self.numGenes)

     # Full internetwork of grn networks including the inter-grn network.
    # We assume that this network follows the same connectivity as the tissue since both are lattices;
    # this assumption could change if the tissue network does not follow a lattice structure.
    def composeTissueLevelGRN(self):
        self.tissueGRNWeights = torch.kron(torch.eye(self.numCells,self.numCells),self.GRNWeights) + \
                                torch.kron(self.tissueConnectivity,self.InterGRNWeights)  # assumes no self-loops in tissue connectivity
        # Note: the ordering of the genes warrants the use of tile, not repeat_interleave
        self.tissueVmemToGRNWeights = torch.tile(self.VmemToGRNWeights,(self.numCells,)).view(self.numVariables,1)
        self.tissueGRNGain = torch.tile(self.GRNGains, (self.numCells,)).view(self.numVariables, 1)
        self.tissueGRNBias = torch.tile(self.GRNBiases,(self.numCells,)).view(self.numVariables,1)
        self.tissueGRNTimeconstants = torch.tile(self.GRNTimeconstants,(self.numCells,)).view(self.numVariables,1)

    # initialize parameters with special values
    def initParameters(self, initialValues):
        pass

    # create arrays of genetic variables with default values
    def defineVariables(self):
        self.dstate = torch.zeros(self.numSamples,self.numVariables,1,dtype=torch.float64)
        self.state = torch.zeros(self.numSamples,self.numVariables,1,dtype=torch.float64)
        self.tissueExternalInputs = torch.zeros(self.numSamples,self.numVariables,1,dtype=torch.float64)

    # initialize variables with special values
    def initVariables(self, initialValues):
        pass

    # the interface through which the interaction with Vmem would modify the dynamic grn parameters (e.g., external inputs to the genes)
    def updateDynamicalParameters(self,externalInputs=None):
        if externalInputs == None:
            self.tissueExternalInputs = torch.zeros(self.numSamples,self.numVariables,1,dtype=torch.float64)
            self.VmemToGRNWeights = torch.zeros(1,self.numGenes,dtype=torch.float64)
        else:  # Note: the ordering of the genes warrants the use of repeat_interleave, not tile
            self.tissueExternalInputs = torch.repeat_interleave(externalInputs,repeats=self.numGenes,dim=1).view(self.numSamples,self.numVariables,1)

    def updateState(self):
        self.dstate = -self.state + torch.matmul(self.tissueGRNWeights, torch.sigmoid(self.tissueGRNGain * (self.state + self.tissueGRNBias))) + \
             self.tissueVmemToGRNWeights * (2*torch.sigmoid(torch.exp(self.VmemGain) * self.tissueExternalInputs + self.VmemBias) - 1)
        self.dstate = self.dstate / self.tissueGRNTimeconstants
        self.state = self.state + (self.timestep * self.dstate)

    def simulate(self,electricNetworkState=None,numSimIters=1):
        for iter in range(numSimIters):
            self.updateDynamicalParameters(externalInputs=electricNetworkState)
            self.updateState()

