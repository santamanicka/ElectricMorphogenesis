# The overall model that manages the submodels namely neuralPlateCircuit and geneRegulatoryNetwork
import torch
import numpy as np
import cellularFieldNetwork as cfn
import geneRegulatoryNetwork as grn
import copy
from itertools import chain

# Notes:
# 1) Learned parameters for the neural plate bioelectric circuit:
#    a) GRNtoVmemWeights of shape = (1,numGenes)
#    b) GRNBiases of shape = (1,numGenes)  # same as (e) in the genetic network parameters
#    c) GRNtoVmemWeightsTimeconstant = (1,1)
# 2) Learned parameters for the genetic network:
#    a) GRNWeights of shape = (numGenes,numGenes)
#    b) InterGRNWeights of shape = (numGenes,numGenes)
#    c) VmemToGRNWeights of shape = (1,numGenes)
#    d) VmemGain of shape = (1,1)
#    e) GRNBiases of shape = (1,numGenes)  # same as (b) in the neural plate circuit parameters
#    f) VmemBias of shape = (1,1)
#    g) GRNTimeconstants of shape = (1,numGenes)
#    h) InterGRNWeightsTimeconstant of shape = (1,1)
#    h) VmemToGRNWeightsTimeconstant of shape = (1,1)
# 3) Total number of learned parameters = 2*numGenes^2 + 3*numGenes + 5
# 4) Miscellaneous:
#    a) The InterVmemGRNNetwork is relatively simpler since for there's only one sender and receiver node on the side of the Vmem, which is itself

class model():

    # arguments will be passed by the GA or by any program that wants to call the simulation module
    def __init__(self,parameters=None,numBasicSamples=1,numNoisySamples=1):
        self.parameters = parameters
        self.numBasicSamples = numBasicSamples
        self.numNoisySamples = numNoisySamples
        self.numSamples = self.numBasicSamples * self.numNoisySamples
        if self.parameters is not None:
            self.electricNetwork = cfn.cellularFieldNetwork(latticeDims=parameters['latticeDims'],parameters=parameters,
                                                            numSamples=self.numSamples)
            self.parameters['tissueConnectivity'] = self.electricNetwork.Adjacency
            if self.parameters['GRNParameters'] is not None:
                if self.parameters['GRNParameters']['GRNEnabled']:
                    self.geneNetwork = grn.geneRegulatoryNetwork(parameters=parameters,numSamples=self.numSamples)
                    self.GRNEnabled = True
                else:
                    self.GRNEnabled = False
            else:
                self.GRNEnabled = False

    def setExperimentalConditions(self,experimentalConditions=None):
        self.experimentalConditions = experimentalConditions
        if experimentalConditions != None:
            self.initialValues, self.numSamples = experimentalConditions
            self.electricNetwork.initVariables(self.initialValues)
            self.electricNetwork.initParameters(self.initialValues)
        else:
            self.initialValues, self.numSamples = None, 1

    def saveModel(self):
        self.savedModelCopy = copy.deepcopy(self)

    # For the sake of simplicity, the electric and grn layers are processed independently and sequentially.
    # The assumption is that the character of the information-processing strategies doesn't depend on whether
    # the layers are sequentially or parallely updated.
    def simulate(self,clampParameters=None,perturbation=None,numSimIters=1):
        numCells = self.electricNetwork.numCells
        if self.GRNEnabled:
            numGenes = self.geneNetwork.numGenes
            numVariables = numGenes * numCells
            self.timeseriesGRN = torch.DoubleTensor([-999]*numSimIters*self.numSamples*numGenes*numCells).view(numSimIters,self.numSamples,numGenes*numCells,1)
            self.timeseriesGRNExternalInputs = torch.DoubleTensor([-999]*numSimIters*self.numSamples*numVariables).view(numSimIters,self.numSamples,numVariables,1)
        self.timeseriesVmem = torch.DoubleTensor([-999]*numSimIters*self.numSamples*numCells).view(numSimIters,self.numSamples,numCells,1)
        self.timeseriesGpol = torch.DoubleTensor([-999]*numSimIters*self.numSamples*numCells).view(numSimIters,self.numSamples,numCells,1)
        self.timeseriesdGpol = torch.DoubleTensor([-999]*numSimIters*self.numSamples*numCells).view(numSimIters,self.numSamples,numCells,1)
        self.timeseriesIncurrent = torch.DoubleTensor([-999]*numSimIters*self.numSamples*numCells).view(numSimIters,self.numSamples,numCells,1)
        self.timeseriesOutcurrent = torch.DoubleTensor([-999]*numSimIters*self.numSamples*numCells).view(numSimIters,self.numSamples,numCells,1)
        self.timeseriesGij = torch.DoubleTensor([-999]*numSimIters*self.numSamples*numCells*numCells).view(numSimIters,self.numSamples,numCells,numCells)
        self.timeseriesGJcurrent = torch.DoubleTensor([-999]*numSimIters*self.numSamples*numCells).view(numSimIters,self.numSamples,numCells,1)
        if clampParameters is not None:
            clampMode = clampParameters['clampMode']
            clampIndices = clampParameters['clampIndices']
            clampValues = clampParameters['clampValues']
            clampStartIter =  clampParameters['clampStartIter']
            clampEndIter = clampParameters['clampEndIter']
            sampleIndices, clampPointIndices = clampIndices
            # Compute the field distance matrix consisting of the pairwise distances between the clamp points and extracellular coordinates
            # shape = (numSamples,numClampPoints,numFieldGridPoints)
            if 'field' in clampMode:
                self.electricNetwork.fieldClampSampleIndices = sampleIndices
                self.electricNetwork.fieldClampPointIndices1D = clampPointIndices
                self.electricNetwork.numFieldClampPoints = int(len(self.electricNetwork.fieldClampPointIndices1D)/self.numSamples)
                self.electricNetwork.clampFieldPointCoordinates = (self.electricNetwork.extracellularCoordinates[0][:,self.electricNetwork.fieldClampPointIndices1D].view(self.numSamples,self.electricNetwork.numFieldClampPoints),
                                                                    self.electricNetwork.extracellularCoordinates[1][:,self.electricNetwork.fieldClampPointIndices1D].view(self.numSamples,self.electricNetwork.numFieldClampPoints))
                # NOTE: The setdiff would have to be done separately for each set of clamp points
                self.electricNetwork.fieldClampPointIndices2D = self.electricNetwork.fieldClampPointIndices1D.reshape(self.numSamples,self.electricNetwork.numFieldClampPoints)
                self.electricNetwork.freeFieldPointIndices1D = np.concatenate([np.setdiff1d(range(self.electricNetwork.numFieldGridPoints),indices)
                                                                 for indices in self.electricNetwork.fieldClampPointIndices2D])
                self.electricNetwork.freeFieldPointCoordinates = (self.electricNetwork.extracellularCoordinates[0][:,self.electricNetwork.freeFieldPointIndices1D].view(self.numSamples,-1),
                                                  self.electricNetwork.extracellularCoordinates[1][:,self.electricNetwork.freeFieldPointIndices1D].view(self.numSamples,-1))  # shape = (numSamples,numFreeFieldPoints)
                self.electricNetwork.fieldClampDistanceMatrix = (self.electricNetwork.utils.computePairwiseDistances(self.electricNetwork.clampFieldPointCoordinates,self.electricNetwork.freeFieldPointCoordinates).double()
                                                 .view(self.numSamples,-1,self.electricNetwork.numFieldClampPoints))
                self.electricNetwork.numFreeFieldPoints = self.electricNetwork.numFieldGridPoints - self.electricNetwork.numFieldClampPoints
                self.electricNetwork.fieldFreeSampleIndices = np.repeat(range(self.numSamples),self.electricNetwork.numFreeFieldPoints)
            elif 'tissue' in clampMode:
                sampleIndices, clampPointIndices = clampIndices
        else:
            clampMode, sampleIndices, clampPointIndices, clampValues, clampStartIter, clampEndIter = None, None, None, None, 0, -1
        if perturbation is not None:
            perturbStartIter, perturbEndIter = perturbation['time']
        else:
            perturbStartIter, perturbEndIter = 0, -1
        for iter in range(numSimIters):
            if self.GRNEnabled:
                self.timeseriesGRN[iter] = self.geneNetwork.state
                self.timeseriesGRNExternalInputs[iter] = self.geneNetwork.tissueExternalInputs
            self.timeseriesVmem[iter] = self.electricNetwork.Vmem
            # the below are recorded for debugging purpose only
            self.timeseriesGpol[iter] = self.electricNetwork.G_pol
            self.timeseriesdGpol[iter] = self.electricNetwork.dG_pol
            # self.timeseriesIncurrent[iter] = self.electricNetwork.InCurrent
            # self.timeseriesOutcurrent[iter] = self.electricNetwork.OutCurrent
            self.timeseriesGij[iter] = self.electricNetwork.G_ij
            self.timeseriesGJcurrent[iter] = self.electricNetwork.GapJunctionCurrent
            if self.GRNEnabled:
                externalInputs = {'gene':self.geneNetwork.state}
            else:
                externalInputs = {'gene':None}
            self.electricNetwork.simulate(externalInputs=externalInputs,numSimIters=1,stochasticIonChannels=False,
                                          setGradient=False,retainGradients=False,saveData=False)  # shape = (numSamples,numGenes*numCells,1)
            if self.GRNEnabled:
                self.geneNetwork.simulate(self.electricNetwork.Vmem,numSimIters=1)  # shape = (numSamples,numCells,1)
            if (iter >= perturbStartIter) and (iter <= perturbEndIter):
                self.electricNetwork.perturb(perturbation=perturbation,currentIter=iter)
            if (iter >= clampStartIter) and (iter <= clampEndIter):
                if ('field' in clampMode) and self.electricNetwork.fieldEnabled:
                    self.electricNetwork.eV[sampleIndices,clampPointIndices,0] = clampValues[iter,:]  # clamped points act like field sources themselves
                    self.electricNetwork.updateExtracellularVoltage(source='eVClamp')
                    self.electricNetwork.updateIonChannelConductance(inputSource='field',stochasticIonChannels=False,fieldAggregation=self.electricNetwork.fieldAggregation,perturbation=None)
                    if self.electricNetwork.ligandEnabled:
                        self.electricNetwork.updateLigandConcentration(source='Vmem')
                        self.electricNetwork.updateLigandConcentration(source='ligand')
                        # self.updateIonChannelConductance(inputSource='ligand',stochasticIonChannels=stochasticIonChannels,perturbation=None)
                        self.electricNetwork.updateFieldSensitivity(inputSource='ligand')
                    self.electricNetwork.updateCurrent()
                    self.electricNetwork.updateVmem()
                elif 'Vmem' in clampMode:
                    self.electricNetwork.Vmem[sampleIndices,clampPointIndices,0] = clampValues[iter,:]
                elif ('Ligand' in clampMode) and self.electricNetwork.ligandEnabled:
                    self.electricNetwork.ligandConc[sampleIndices,clampPointIndices,0] = clampValues[iter,:]
                    self.electricNetwork.updateLigandConcentration(source='ligand')
                    # self.updateIonChannelConductance(inputSource='ligand',stochasticIonChannels=stochasticIonChannels,perturbation=None)
                    self.electricNetwork.updateFieldSensitivity(inputSource='ligand')
                    self.electricNetwork.updateCurrent()
                    self.electricNetwork.updateVmem()
                elif 'Gpol' in clampMode:
                    self.electricNetwork.G_pol[sampleIndices,clampPointIndices,0] = clampValues[iter,:] * self.electricNetwork.G_ref
                    self.electricNetwork.updateCurrent()
                    self.electricNetwork.updateVmem()


# # test
# latticeDimensions = (2,2)
# numGenes = 4
# GRNtoVmemWeights = torch.FloatTensor(range(numGenes)).view(1,numGenes)
# GRNBiases = torch.FloatTensor(range(numGenes)).view(1,numGenes)
# GRNtoVmemWeightsTimeconstant = torch.FloatTensor([4.5])
# GRNWeights = torch.FloatTensor(range(numGenes**2)).view(numGenes,numGenes)
# InterGRNWeights = torch.FloatTensor(range(numGenes**2)).view(numGenes,numGenes)
# # InterGRNWeights = torch.zeros(numGenes,numGenes)
# VmemToGRNWeights = torch.FloatTensor(range(numGenes)).view(1,numGenes)
# VmemGain = torch.FloatTensor([2.5])
# VmemBias = torch.FloatTensor([-1.2])
# GRNTimeconstants = torch.FloatTensor(range(1,numGenes+1)).view(1,numGenes)
# InterGRNWeightsTimeconstant = torch.FloatTensor([3.7])
# VmemToGRNWeightsTimeconstant = torch.FloatTensor([5.1])
# parameters = (latticeDimensions,GRNtoVmemWeights,GRNBiases,GRNtoVmemWeightsTimeconstant,
#               GRNWeights,InterGRNWeights,VmemToGRNWeights,VmemGain,VmemBias,
#               GRNTimeconstants,InterGRNWeightsTimeconstant,VmemToGRNWeightsTimeconstant)
# model = model(parameters=parameters)
# model.simulate(numSimIters=100)
# numCells = latticeDimensions[0] * latticeDimensions[1]
# print(model.electricNetwork.Vmem.view(1,numCells),model.geneNetwork.state.view(numCells,numGenes))

