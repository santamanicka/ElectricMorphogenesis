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

    # Timeseries that simulate() can record, and the electricNetwork attribute each one reads.
    # InCurrent/OutCurrent are deliberately absent: they were allocated but never written
    # (InCurrent does not exist until updateIonChannelCurrent runs), and nothing consumes them.
    TIMESERIES_SOURCES = {'Vmem':                    'Vmem',
                          'dVmem':                   'dVmem',
                          'eV':                      'eV',
                          'eVforceVector':           'eVforceVector',
                          'Gpol':                    'G_pol',
                          'dGpol':                   'dG_pol',
                          'Gij':                     'G_ij',
                          'GJcurrent':               'GapJunctionCurrent',
                          'LigandConc':              'ligandConc',
                          'FieldTransductionWeight': 'fieldTransductionWeight'}
    TIMESERIES_VARIABLES = tuple(TIMESERIES_SOURCES.keys())

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

    # Row of the timeseries tensors holding a given simulation iteration. Only needed when
    # simulate() was called with storeStride > 1 or storeIters; otherwise row == iteration.
    def storedIndex(self,iteration):
        row = int(np.searchsorted(self.storedIters,iteration))
        if (row >= len(self.storedIters)) or (self.storedIters[row] != iteration):
            raise ValueError('iteration ' + str(iteration) + ' was not recorded; pass it in storeIters '
                             'or use a storeStride that includes it')
        return row

    def simulate(self,externalInputs=dict(),clampParameters=None,perturbation=None,fieldModulation=False,numSimIters=1,outerIter=0,alignmentParameters=None,
                 storeVariables=None,storeStride=1,storeIters=None):
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
            storeVariables: Names of timeseries to record (see TIMESERIES_VARIABLES); None records
                      all of them. Unrecorded timeseries are set to None rather than left stale.
                      Storage is the binding constraint on a large lattice: Gij alone is
                      numSimIters x numSamples x numCells^2, which at 30x30 over 5000 iterations
                      is 32 GB, so an analysis needing only Vmem and Gpol should ask for those.
            storeStride: Record every storeStride-th iteration instead of every one. The final
                      iteration is always recorded.
            storeIters: Additional iterations to record regardless of stride -- pass the ones an
                      analysis depends on (e.g. the pre-pattern step clampEndIter+1), since a
                      stride that skips them would otherwise discard exactly what is needed.

        With the defaults (stride 1, no extra iterations) a timeseries row index is the iteration
        index, which is what every existing caller assumes. Under any other setting, map an
        iteration to its row with storedIndex(); self.storedIters lists what was recorded.

        Gradients are unaffected by which variables are stored -- the recorded tensors are written
        in place and stay in the autograd graph, so a loss over a stored timeseries backpropagates
        exactly as before. A stride is a different matter and should not be used while training:
        code that slices by position rather than by iteration (computeLoss() in
        learnCellularFieldNetwork.py takes timeseriesVmem[-evalDuration:]) would then read the last
        evalDuration *recorded* frames, which are spread over a longer stretch of simulated time.
        That silently optimises a different loss rather than raising.
        """
        numFieldGridPoints = self.electricNetwork.numFieldGridPoints
        if storeVariables is None:
            storeVariables = self.TIMESERIES_VARIABLES
        unknownVariables = [name for name in storeVariables if name not in self.TIMESERIES_SOURCES]
        if len(unknownVariables) > 0:
            raise ValueError('unknown storeVariables ' + str(unknownVariables) + '; choose from ' + str(self.TIMESERIES_VARIABLES))
        if storeStride < 1:
            raise ValueError('storeStride must be >= 1, got ' + str(storeStride))
        storedIterations = set(range(0,numSimIters,storeStride))
        storedIterations.add(numSimIters-1)
        if storeIters is not None:
            storedIterations.update(int(iteration) for iteration in storeIters if 0 <= iteration < numSimIters)
        self.storedIters = np.array(sorted(storedIterations))
        self.storeStride = storeStride
        rowOfIteration = np.full(numSimIters,-1,dtype=np.int64)
        rowOfIteration[self.storedIters] = np.arange(len(self.storedIters))
        numStored = len(self.storedIters)
        cellShape = (numStored,self.numSamples,self.numCells,1)
        fieldShape = (numStored,self.numSamples,numFieldGridPoints,1)
        # torch.full rather than torch.DoubleTensor([-999]*n): the list form materialises n Python
        # floats before the tensor, which is fatal for Gij on a large lattice (30x30 over 5000
        # iterations is 4e9 elements).
        timeseriesShapes = {'Vmem':                    cellShape,
                            'dVmem':                   cellShape,
                            'eV':                      fieldShape,
                            'eVforceVector':           (numStored,2,self.numSamples,numFieldGridPoints,1),
                            'Gpol':                    cellShape,
                            'dGpol':                   cellShape,
                            'Gij':                     (numStored,self.numSamples,self.numCells,self.numCells),
                            'GJcurrent':               cellShape,
                            'LigandConc':              cellShape,
                            'FieldTransductionWeight': cellShape}
        activeStores = []
        for name in self.TIMESERIES_VARIABLES:
            timeseries = None
            if name in storeVariables:
                timeseries = torch.full(timeseriesShapes[name],-999,dtype=torch.float64)
                activeStores.append((name,timeseries,self.TIMESERIES_SOURCES[name]))
            setattr(self,'timeseries'+name,timeseries)
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
            row = rowOfIteration[iter]
            if row >= 0:
                for name, timeseries, sourceName in activeStores:
                    if name == 'eVforceVector':
                        timeseries[row,0] = self.electricNetwork.eVforceVector[0]
                        timeseries[row,1] = self.electricNetwork.eVforceVector[1]
                    else:
                        timeseries[row] = getattr(self.electricNetwork,sourceName)
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

