# The overall model that manages the submodels namely cellularFieldNetwork and geneRegulatoryNetwork
import torch
import numpy as np
import cellularFieldNetwork as cfn
import copy
import utilities

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
        numRows, numCols = parameters['latticeDims'][0], parameters['latticeDims'][1]
        self.numCells = numRows * numCols
        self.utils = utilities.utilities()
        if self.parameters is not None:
            if 'latticePeriodicBoundaryGJ' in parameters.keys():
                latticePeriodicBoundaryGJ = parameters['latticePeriodicBoundaryGJ']
            else:
                latticePeriodicBoundaryGJ = False
            if 'latticePeriodicBoundaryLigand' in parameters.keys():
                latticePeriodicBoundaryLigand = parameters['latticePeriodicBoundaryLigand']
            else:
                latticePeriodicBoundaryLigand = False
            if 'boundaryEdgeDiffusionStrengthLigand' in self.parameters.keys():
                boundaryEdgeDiffusionStrengthLigand = parameters['boundaryEdgeDiffusionStrengthLigand']
            else:
                boundaryEdgeDiffusionStrengthLigand = None
            self.parameters['ligandParameters']['tissueConnectivity'] = self.utils.computeLatticeAdjacencyMatrix(latticeDims=parameters['latticeDims'],periodicBoundary=latticePeriodicBoundaryLigand)
            if latticePeriodicBoundaryLigand and boundaryEdgeDiffusionStrengthLigand is not None:
                tissueConnectivityCoeffs = parameters['ligandParameters']['tissueConnectivity'] * 1.0
                boundaryEdges = np.array([(cell,neighbor.item()) for cell in range(self.numCells)
                                          for neighbor in torch.where(tissueConnectivityCoeffs[cell,:]==1)[0]
                                          if ((cell-neighbor).abs()==(numCols-1)) or ((cell-neighbor).abs()==((numRows-1)*numCols))])
                tissueConnectivityCoeffs[boundaryEdges[:,0],boundaryEdges[:,1]] = boundaryEdgeDiffusionStrengthLigand
                self.parameters['ligandParameters']['tissueConnectivity'] = self.parameters['ligandParameters']['tissueConnectivity'] * tissueConnectivityCoeffs
            self.electricNetwork = cfn.cellularFieldNetwork(latticeDims=parameters['latticeDims'],latticePeriodicBoundary=latticePeriodicBoundaryGJ,
                                                            parameters=parameters,numSamples=self.numSamples)
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

    def simulate(self,externalInputs=dict(),clampParameters=None,perturbation=None,fieldModulation=False,numSimIters=1,outerIter=0,alignmentParameters=None):
        """
        Simulate the embryo model for a specified number of iterations.

        Args:
            externalInputs: External inputs to the model (dict)
            clampParameters: Parameters for voltage/field clamping
            perturbation: Perturbation parameters
            fieldModulation: Whether to modulate field by ion channels
            numSimIters: Number of simulation iterations to run
            outerIter: Starting iteration number for this simulation call (default: 0).
                      Use this when calling simulate() multiple times iteratively to ensure
                      time-varying clamps and perturbations use correct global iteration index.
            alignmentParameters: Parameters for field alignment forcing (passed to electricNetwork.simulate())
        """
        numFieldGridPoints = self.electricNetwork.numFieldGridPoints
        self.timeseriesVmem = torch.DoubleTensor([-999]*numSimIters*self.numSamples*self.numCells).view(numSimIters,self.numSamples,self.numCells,1)
        self.timeseriesdVmem = torch.DoubleTensor([-999]*numSimIters*self.numSamples*self.numCells).view(numSimIters,self.numSamples,self.numCells,1)
        self.timeserieseV = torch.DoubleTensor([-999]*numSimIters*self.numSamples*numFieldGridPoints).view(numSimIters,self.numSamples,numFieldGridPoints,1)
        self.timeserieseVforceVector = torch.DoubleTensor([-999]*2*numSimIters*self.numSamples*numFieldGridPoints).view(numSimIters,2,self.numSamples,numFieldGridPoints,1)
        self.timeseriesGpol = torch.DoubleTensor([-999]*numSimIters*self.numSamples*self.numCells).view(numSimIters,self.numSamples,self.numCells,1)
        self.timeseriesdGpol = torch.DoubleTensor([-999]*numSimIters*self.numSamples*self.numCells).view(numSimIters,self.numSamples,self.numCells,1)
        self.timeseriesIncurrent = torch.DoubleTensor([-999]*numSimIters*self.numSamples*self.numCells).view(numSimIters,self.numSamples,self.numCells,1)
        self.timeseriesOutcurrent = torch.DoubleTensor([-999]*numSimIters*self.numSamples*self.numCells).view(numSimIters,self.numSamples,self.numCells,1)
        self.timeseriesGij = torch.DoubleTensor([-999]*numSimIters*self.numSamples*self.numCells*self.numCells).view(numSimIters,self.numSamples,self.numCells,self.numCells)
        self.timeseriesGJcurrent = torch.DoubleTensor([-999]*numSimIters*self.numSamples*self.numCells).view(numSimIters,self.numSamples,self.numCells,1)
        self.timeseriesLigandConc = torch.DoubleTensor([-999]*numSimIters*self.numSamples*self.numCells).view(numSimIters,self.numSamples,self.numCells,1)
        self.timeseriesFieldTransductionWeight = torch.DoubleTensor([-999]*numSimIters*self.numSamples*self.numCells).view(numSimIters,self.numSamples,self.numCells,1)
        if clampParameters is not None:
            clampMode = clampParameters['clampMode']
            clampIndices = clampParameters['clampIndices'] #.int()
            clampValues = clampParameters['clampValues']
            clampStartIter =  clampParameters['clampStartIter']
            clampEndIter = clampParameters['clampEndIter']
            sampleIndices, clampPointIndices = clampIndices
            # Compute the field distance matrix consisting of the pairwise distances between the clamp points and extracellular coordinates
            # shape = (numSamples,numClampPoints,numFieldGridPoints)
            if 'field' in clampMode:
                self.electricNetwork.fieldClampSampleIndices = sampleIndices #.int()
                self.electricNetwork.fieldClampPointIndices1D = clampPointIndices #.int()
                self.electricNetwork.numFieldClampPoints = int(len(self.electricNetwork.fieldClampPointIndices1D)/self.numSamples)
                # self.electricNetwork.numFieldClampPoints = int(len(self.electricNetwork.fieldClampPointIndices1D))
                self.electricNetwork.clampFieldPointCoordinates = (self.electricNetwork.extracellularCoordinates[0][:,self.electricNetwork.fieldClampPointIndices1D].view(self.numSamples,self.electricNetwork.numFieldClampPoints),
                                                                    self.electricNetwork.extracellularCoordinates[1][:,self.electricNetwork.fieldClampPointIndices1D].view(self.numSamples,self.electricNetwork.numFieldClampPoints))
                # NOTE: The setdiff would have to be done separately for each set of clamp points
                self.electricNetwork.fieldClampPointIndices2D = self.electricNetwork.fieldClampPointIndices1D.reshape(self.numSamples,self.electricNetwork.numFieldClampPoints)
                self.electricNetwork.freeFieldPointIndices1D = np.concatenate([np.setdiff1d(range(self.electricNetwork.numFieldGridPoints),indices)
                                                                 for indices in self.electricNetwork.fieldClampPointIndices2D])
                self.electricNetwork.freeFieldPointIndices2D = self.electricNetwork.freeFieldPointIndices1D.reshape(self.numSamples,-1)
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
            self.timeseriesVmem[iter] = self.electricNetwork.Vmem
            self.timeseriesdVmem[iter] = self.electricNetwork.dVmem
            self.timeserieseV[iter] = self.electricNetwork.eV
            self.timeserieseVforceVector[iter,0] = self.electricNetwork.eVforceVector[0]
            self.timeserieseVforceVector[iter,1] = self.electricNetwork.eVforceVector[1]
            # the below are recorded for debugging purpose only
            self.timeseriesGpol[iter] = self.electricNetwork.G_pol
            self.timeseriesdGpol[iter] = self.electricNetwork.dG_pol
            # self.timeseriesIncurrent[iter] = self.electricNetwork.InCurrent
            # self.timeseriesOutcurrent[iter] = self.electricNetwork.OutCurrent
            self.timeseriesGij[iter] = self.electricNetwork.G_ij
            self.timeseriesGJcurrent[iter] = self.electricNetwork.GapJunctionCurrent
            self.timeseriesLigandConc[iter] = self.electricNetwork.ligandConc
            self.timeseriesFieldTransductionWeight[iter] = self.electricNetwork.fieldTransductionWeight
            externalInputs['gene'] = None
            self.electricNetwork.simulate(externalInputs=externalInputs,numSimIters=1,outerIter=outerIter+iter,stochasticIonChannels=False,fieldModulation=fieldModulation,
                                          setGradient=False,retainGradients=False,saveData=False,alignmentParameters=alignmentParameters)
            if (iter >= perturbStartIter) and (iter <= perturbEndIter):
                self.electricNetwork.perturb(perturbation=perturbation,currentIter=iter)
            # Use global iteration (outerIter+iter) for clamp timing, not local iter
            global_iter = outerIter + iter
            if (global_iter >= clampStartIter) and (global_iter <= clampEndIter):
                if ('field' in clampMode) and self.electricNetwork.fieldEnabled:
                    self.electricNetwork.eV[sampleIndices,clampPointIndices,0] = clampValues[global_iter,:]  # clamped points act like field sources themselves
                    self.electricNetwork.updateExtracellularVoltage(source='eVClamp')
                    self.electricNetwork.updateIonChannelConductance(inputSource='field',stochasticIonChannels=False,fieldModulation=fieldModulation,
                                                                     fieldAggregation=self.electricNetwork.fieldAggregation,perturbation=None)
                    if self.electricNetwork.ligandEnabled:
                        self.electricNetwork.updateLigandConcentration(source='Vmem')
                        self.electricNetwork.updateLigandConcentration(source='ligand')
                        # self.updateIonChannelConductance(inputSource='ligand',stochasticIonChannels=stochasticIonChannels,perturbation=None)
                        self.electricNetwork.updateFieldSensitivity(inputSource='ligand')
                    self.electricNetwork.updateCurrent()
                    self.electricNetwork.updateVmem()
                elif 'Vmem' in clampMode:
                    self.electricNetwork.Vmem[sampleIndices,clampPointIndices,0] = clampValues[global_iter,:]
                elif ('Ligand' in clampMode) and self.electricNetwork.ligandEnabled:
                    self.electricNetwork.ligandConc[sampleIndices,clampPointIndices,0] = clampValues[global_iter,:]
                    self.electricNetwork.updateLigandConcentration(source='ligand')
                    # self.updateIonChannelConductance(inputSource='ligand',stochasticIonChannels=stochasticIonChannels,perturbation=None)
                    self.electricNetwork.updateFieldSensitivity(inputSource='ligand')
                    self.electricNetwork.updateCurrent()
                    self.electricNetwork.updateVmem()
                elif 'Gpol' in clampMode:
                    self.electricNetwork.G_pol[sampleIndices,clampPointIndices,0] = clampValues[global_iter,:] * self.electricNetwork.G_ref
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

