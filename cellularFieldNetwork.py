# A model of the multicellular (tissue) bioelectric network that's responsible for establishing the Vmem pattern

# Notes:
# 1) Learned parameters and shapes:
#    a) GRNtoVmemWeights of shape = (1,numGenes)
#    b) GRNBiases of shape = (1,numGenes)  # shared with geneRegulatoryNetwork
#    c) GRNtoVmemWeightsTimeconstant = (1,1)
#    d) numGenes = (1,1)
# 2) Other shapes:
#    a) Vmem = (numSamples,numCells,1)
#    b) externalInputs = (numSamples,numCells,numGenes)
# 3) Model parameters: ion channel conductances G_pol (fixed) and G_dep (can be modified via interactions with GRN);
#    and max gap junction conductance G_0 (assumed to be fixed since that's the assumption in Pai et al. 2020)
# 4) We assume that the network structure of the bioelectric layer is fixed and is a lattice, since it models a patch of non-neural tissue.
#    a) Specifically, the 'LatticeDims' is fixed throughout and is not learned

import torch
import numpy as np
import utilities

# Assumptions:
# connectivity = lattice
class cellularFieldNetwork():

    # Constants and symbols were adopted from:
    # Pai, V. P., et al. (2020). "HCN2 Channel-Induced Rescue of Brain Teratogenesis via Local and Long-Range Bioelectric Repair." Front Cell Neurosci 14: 136.
    # Main equation was adopted from:
    # Fig.3 of Cervera, J., et al. (2016). "The interplay between genetic and bioelectrical signaling permits a spatial regionalisation of membrane potentials in model multicellular ensembles." Sci Rep 6: 35201.
    def __init__(self,LatticeDims=(3,3),GRNParameters=None,fieldParameters=None,numSamples=1):
        self.Z = 3 # valence
        self.V_th = -27e-3  # threshold voltage (mV)
        self.V_T = 27e-3  # thermal potential (mV); assumed = -V_th
        self.V_0 = 12e-3  # gap junction gating potential width (mV); other possible values = 18
        self.C = 0.1e-9  # cell capacitance (nF)
        self.E_pol = -55e-3  # reversal potential of the hyperpolarizing (inward-rectifying) channel (mV)
        self.E_dep = -5e-3  # reversal potential of the depolarizing (outward-rectifying) channel (mV)
        self.G_ref = 1.0e-9  # reference value of GJ conductance for scaling (nS; S for siemens)
        self.G_0 = 0.05 * self.G_ref  # maximum conductance of the gap junction; NOTE: original value was 0.5
        self.G_res = 0 * self.G_0  # residual gap junction conductance while "closed"; other possible values = 0.05*G_0
        self.cell_radius = 5.0e-6   # radius of single cell (m)
        self.k_e = 8.987e9 # Coulomb constant (N.m^2.C^-2)
        self.relativePermittivity = 10**(7) # static relative permittivity of cytoplasm (dimensionless)
        self.LatticeDims = LatticeDims
        self.fieldResolution, self.fieldStrength, self.fieldTransductionParameters = fieldParameters
        self.timestep = 0.01
        self.GRNParameters = GRNParameters
        self.numSamples = numSamples
        self.numCells = np.prod(self.LatticeDims)
        self.utils = utilities.utilities()
        self.defineCellularNetwork()
        self.defineFieldConstants()
        self.defineParameters()
        self.defineVariables()

    # create connectivity matrices with appropriate values defined in init()
    def defineCellularNetwork(self):
        self.Adjacency = self.utils.computeLatticeAdjacencyMatrix(self.LatticeDims)
        self.i, self.j = torch.where(torch.triu(self.Adjacency) == 1)  #ordered indices of connected node-pairs

    def defineFieldConstants(self):
        # Compute the coordinates of the cellular and extracellular grids
        xc, yc = self.utils.computeCellularCoordinates(self.LatticeDims,self.cell_radius)
        self.cellularCoordinates = (xc.reshape(1,-1),yc.reshape(1,-1))  # dim 0 added to match the 'samples' dim in other variables
        xec, yec = self.utils.computeExtracellularCoordinates(self.LatticeDims,self.cell_radius,self.fieldResolution)
        self.extracellularCoordinates = (xec.reshape(1,-1),yec.reshape(1,-1))  # dim 0 added to match the 'samples' dim in other variables
        # Compute the field distance matrix consisting of the pairwise distances between the cellular and the extracellular coordinates
        # shape = (numExtracellularGridPoints,numCells)
        self.fieldCellDistanceMatrix = self.utils.computePairwiseDistances(self.cellularCoordinates,self.extracellularCoordinates)
        distanceThreshold = self.cell_radius * np.sqrt(2) * 1.001  # length of half diagonal of a square of side equal to cell diameter; 0.1% extra to accommodate numerical precision
        self.fieldCellNeighborhoodBitmap = self.utils.defineFieldCellNeighborhoodMap(self.fieldCellDistanceMatrix,distanceThreshold=distanceThreshold)
        self.numFieldNeighbors = self.fieldCellNeighborhoodBitmap.sum(0).sum(0)[0].item()  # first sum is over the 'samples' dim though numSamples=1 for this variable
        self.numExtracellularGridPoints = self.extracellularCoordinates[0].shape[1]

    # Create arrays of bioelectric variables with default values
    def defineVariables(self):
        self.InCurrent = torch.zeros(self.numSamples,self.numCells,1,dtype=torch.float64)
        self.OutCurrent = torch.zeros(self.numSamples,self.numCells,1,dtype=torch.float64)
        # self.G_dep = torch.zeros(self.numSamples,self.numCells,1)
        self.GapJunctionCurrent = torch.zeros(self.numSamples,self.numCells,1,dtype=torch.float64)
        self.Current = torch.zeros(self.numSamples,self.numCells,1,dtype=torch.float64)
        self.Vmem = torch.zeros(self.numSamples,self.numCells,1,dtype=torch.float64)
        self.eV = torch.zeros(self.numSamples, self.numExtracellularGridPoints,1,dtype=torch.float64)
        self.G_ij = torch.zeros(self.numSamples,self.numCells,self.numCells,dtype=torch.float64)

    # Initialize arrays of bioelectric variables with (mandatory) values passed by the user in a dictionary
    # There's no point in initializing current and G_ij, since they will be overwritten by the corresponding update() methods.
    def initVariables(self, initialValues):
        self.Vmem.set_(initialValues['Vmem'])

    # Define parameters and populate them with default values
    def defineParameters(self):
        self.G_pol = torch.DoubleTensor([1.0 * self.G_ref] * self.numSamples * self.numCells).view(self.numSamples,self.numCells,1)  # maximum conductance of the inward-rectifying channel (favors hyperpolarization) in the control condition
        self.G_dep = torch.DoubleTensor([1.5 * self.G_ref] * self.numSamples * self.numCells).view(self.numSamples,self.numCells,1)  # maximum conductance of the outward-rectifying channel (favors depolarization) in the control condition
        self.numGenes = self.GRNParameters[3]
        if self.numGenes == None:
            self.numGenes = 0
        self.GRNtoVmemWeights = self.GRNParameters[0]
        # the interface through which the interaction with the grn would modify the dynamical bioelectric parameters
        # (e.g., the ratio G_pol/G_dep or just G_pol/G_dep while the other would be fixed)
        if self.GRNtoVmemWeights == None:
            self.GRNtoVmemWeights = torch.zeros(self.numGenes,1)
        else:
            self.GRNtoVmemWeights = self.GRNtoVmemWeights.t()  # shape = (numGenes,1)
        self.GRNBiases = self.GRNParameters[1]
        if self.GRNBiases == None:
            self.GRNBiases = torch.zeros(1,self.numGenes)
        self.GRNtoVmemWeightsTimeconstant = self.GRNParameters[2]
        if self.GRNtoVmemWeightsTimeconstant == None:
            self.GRNtoVmemWeightsTimeconstant = 1
        else:
            self.GRNtoVmemWeights = self.GRNtoVmemWeights / self.GRNtoVmemWeightsTimeconstant
        if self.fieldTransductionParameters == None:
            self.eVBias, self.eVWeight, self.evTimeConstant = torch.inf, torch.DoubleTensor([0]), torch.DoubleTensor([1])
        else:
            self.eVBias, self.eVWeight, self.evTimeConstant = self.fieldTransductionParameters

    # Selectively update parameters with (optional) values passed by the user in a dictionary
    # Examples of such "variable" parameters include maximum ion channel conductance
    # (e.g., neural patch in the control tissue, nicotine modulation to the control tissue, etc.)
    def initParameters(self, initialValues):
        for sample in range(self.numSamples):
            initCells = initialValues['G_pol']['cells'][sample]
            if len(initCells) > 0:
                # initCells must be a list of two lists, corresponding to Depolarized and Hyperpolarized cells in that order
                for i in range(len(initCells)):
                    if len(initCells[i]) > 0:  # one of the two sublists could be empty if all cells are of the same Vmem
                        self.G_pol[sample,initCells[i]] = initialValues['G_pol']['values'][sample][i].view(-1,1) * self.G_ref
            initCells = initialValues['G_dep']['cells']
            if len(initCells) > 0:
                self.G_dep[sample,initCells] = initialValues['G_dep']['values'][sample] * self.G_ref

    # The interface through which the interaction with the GRN would modify the dynamical bioelectric parameters
    # (e.g., the ratio G_pol/G_dep or just G_pol or G_dep while the other would be fixed).
    # Here, only G_dep is updated. The reason for this choice is that realistic Vmems are negative, hence if there
    # are forces that tend to make the Vmem even more negative then the depolarizing channel could be amplified to balance it.
    def updateIonChannelConductance(self,inputState=None,inputSource=None,stochastic=False,perturbation=None):
        # ODE for updating G_dep
        dp = 0
        if inputSource == 'gene':
            geneState = inputState.view(self.numSamples,self.numCells,self.numGenes)
            dp = (-self.G_pol + 2*torch.matmul(torch.sigmoid(geneState + self.GRNBiases)-1, self.GRNtoVmemWeights))
        elif inputSource == 'field':
            self.eVneighborsMean = (self.eV * self.fieldCellNeighborhoodBitmap).sum(1) / self.numFieldNeighbors  # shape = (numSamples,numCells)
            self.eVneighborsMean = self.eVneighborsMean.unsqueeze(2)  # shape = (numSamples,numCells,1)
            dp = self.fieldStrength * (-self.G_pol + (2*torch.sigmoid(self.eVneighborsMean + self.eVBias)-1) * self.eVWeight) / self.evTimeConstant
        if stochastic:
            dp = dp + (torch.randn(*dp.shape)/8)  # max abs delta ~ 0.5
        if perturbation is not None:
            perturbSampleIndices, perturbPointIndices = perturbation
            numPerturbPoints = len(perturbPointIndices)
            dp = torch.zeros(*self.G_pol.shape,dtype=torch.float64)
            delta = torch.randn(numPerturbPoints,1,dtype=torch.float64)
            dp[perturbSampleIndices, perturbPointIndices] = delta
        dp = dp * self.G_ref  # not scaling by G_ref would lead to dramatic changes in all the variables
        self.G_pol = self.G_pol + (self.timestep * dp)
        self.G_pol[self.G_pol < 0] = 0  # this truncation could potentially cause numerical instability issues

    # Compute currents through voltage-gated ion channels
    def updateIonChannelCurrent(self):
        self.InChannelPermeability = 1 / (1 + torch.exp(self.Z * (self.Vmem - self.V_th) / self.V_T))  # voltage-gating
        self.InCurrent = -self.G_pol * (self.Vmem - self.E_pol) * self.InChannelPermeability
        self.OutChannelPermeability = 1 / (1 + torch.exp(-self.Z * (self.Vmem - self.V_th) / self.V_T))  # voltage-gating
        self.OutCurrent = -self.G_dep * (self.Vmem - self.E_dep) * self.OutChannelPermeability
        self.IonChannelCurrent = self.InCurrent + self.OutCurrent

    # Compute currents through voltage-gated gap junctions
    def updateGapJunctionConductance(self):
        Vi = self.Vmem[:,self.i]
        Vj = self.Vmem[:,self.j]
        Gij_values = self.G_res + (2 * self.G_0 / (1 + torch.cosh((Vi - Vj)/self.V_0)))  # symmetric function about 0
        self.G_ij[:,self.i, self.j] = Gij_values.squeeze(2)
        self.G_ij[:,self.j, self.i] = Gij_values.squeeze(2)  # Gij is symmetrical by virtue of the symmetry of cosh around 0

    def updateGapJunctionCurrent(self):
        self.updateGapJunctionConductance()
        DegreeMatrix = torch.diag_embed(self.G_ij.sum(2))  # although it's called 'degree matrix' it's really a sum of row-wise of G_ij values
        Laplacian = self.G_ij - DegreeMatrix  # negative Laplacian for clarity (so diffusion coefficient doesn't have to be negative)
        self.GapJunctionCurrent = torch.matmul(Laplacian,self.Vmem)

    def updateCurrent(self):
        self.updateIonChannelCurrent()
        if self.numCells > 1:
            self.updateGapJunctionCurrent()
        self.Current = self.IonChannelCurrent + self.GapJunctionCurrent

    def updateVmem(self):
        dVmem = self.Current / self.C
        self.Vmem = self.Vmem + (dVmem * self.timestep)

    # Two ways to compute charge: 1) Q=C*V; 2) dQ=I*dt (since Q=It)
    # Method (1) will be more appropriate here since (2) requires specifying an initial value for Q.
    def computeCharge(self,V):
        Q = self.C * V  # shape = (numSamples,numCells or numExtracellularGridPoints,1)
        return Q

    # Given the charge distribution of the circuit, compute the field values (extracellular Vmem) at the field coordinates
    # using Coulomb's law of electrostatics
    def updateExtracellularVoltage(self,source='Vmem'):
        if source == 'Vmem':  # Vmem fully determines eV (overwrites current eV)
            Q = self.computeCharge(V=self.Vmem)  # shape = (numSamples,numCells,1)
            r = (1 / self.fieldCellDistanceMatrix) #.float()  # shape = (numExtracellularGridPoints,numCells)
            self.eV = self.fieldStrength * (self.k_e / self.relativePermittivity) * torch.matmul(r,Q)  # shape = (numSamples,numExtracellularGridPoints,1)
        elif source == 'eVClamp':  # clamped eV adds to current eV (if there's no clamping of eV then there will be no updates)
            Q = self.computeCharge(V=self.eV)  # shape = (numSamples,numExtracellularGridPoints,1)
            Q = Q[self.clampSampleIndices,self.clampPointIndices1D,:].view(self.numSamples,-1,1)  # shape = (numSamples,numFreeFieldPoints,1)
            r = (1 / self.clampFieldDistanceMatrix).float()  # shape = (numSamples,numExtracellularGridPoints,numClampPoints)
            deV = (self.fieldStrength * (self.k_e / self.relativePermittivity) * torch.matmul(r,Q)).view(-1,1)  # shape = (numFreeFieldPoints*numSamples,1)
            self.eV[self.freeSampleIndices,self.freeFieldPointIndices1D,:] = self.eV[self.freeSampleIndices,self.freeFieldPointIndices1D,:] + deV

    def updateFieldEffects(self,source='Vmem',stochastic=False,perturbation=None):
        self.updateExtracellularVoltage(source=source)
        self.updateIonChannelConductance(inputSource='field',stochastic=stochastic,perturbation=perturbation)

    def simulate(self,inputs=None,clampParameters=None,screenParameters=None,perturbationParameters=None,numSimIters=1,
                 stochastic=False,saveData=False):
        if saveData:
            self.timeseriesVmem = torch.DoubleTensor([-999]*numSimIters*self.numSamples*self.numCells).view(numSimIters,self.numSamples,self.numCells,1)
            self.timeserieseV = torch.DoubleTensor([-999]*numSimIters*self.numSamples*self.numExtracellularGridPoints).view(numSimIters,self.numSamples,self.numExtracellularGridPoints,1)
            if clampParameters is not None:
                clampMode, clampIndices, clampVoltage, clampDurationPercent = clampParameters
                sampleIndices, clampPointIndices = clampIndices
                clampIters = clampDurationPercent * numSimIters
                # Compute the field distance matrix consisting of the pairwise distances between the clamp points and extracellular coordinates
                # shape = (numSamples,numClampPoints,numExtracellularGridPoints)
                if (clampMode == 'field') or (clampMode == 'fieldDome'):
                    self.clampSampleIndices = sampleIndices
                    self.clampPointIndices1D = clampPointIndices
                    self.numClampFieldPoints = int(len(self.clampPointIndices1D)/self.numSamples)
                    self.clampFieldPointCoordinates = (self.extracellularCoordinates[0][:,self.clampPointIndices1D].view(self.numSamples,self.numClampFieldPoints),
                                                       self.extracellularCoordinates[1][:,self.clampPointIndices1D].view(self.numSamples,self.numClampFieldPoints))
                    # NOTE: The setdiff would have to be done separately for each set of clamp points
                    self.clampPointIndices2D = self.clampPointIndices1D.reshape(self.numSamples,self.numClampFieldPoints)
                    self.freeFieldPointIndices1D = np.concatenate([np.setdiff1d(range(self.numExtracellularGridPoints),indices)
                                                           for indices in self.clampPointIndices2D])
                    self.freeFieldPointCoordinates = (self.extracellularCoordinates[0][:,self.freeFieldPointIndices1D].view(self.numSamples,-1),
                                                      self.extracellularCoordinates[1][:,self.freeFieldPointIndices1D].view(self.numSamples,-1))  # shape = (numSamples,numFreeFieldPoints)
                    self.clampFieldDistanceMatrix = (self.utils.computePairwiseDistances(self.clampFieldPointCoordinates,self.freeFieldPointCoordinates)
                                                     .view(self.numSamples,-1,self.numClampFieldPoints))
                    self.numFreeFieldPoints = self.numExtracellularGridPoints - self.numClampFieldPoints
                    self.freeSampleIndices = np.repeat(range(self.numSamples),self.numFreeFieldPoints)
            else:
                clampMode, sampleIndices, clampPointIndices, clampVoltage, clampDurationPercent, clampIters = None, None, None, None, 0, 0
            # Specify the extent to which the field is constrained (beyond which it's suppressed);
            # default is there's no screening, meaning the field permeates the entire tissue.
            if screenParameters is not None:
                # numBoundingSquares: the length of the side of the square around a cell to which the field's reach will be limited;
                # Square of size 1 refers to the circumscribing square.
                # Max value of numBoundingSquares so the field will permeate the entire tissue = 2(l-1)+1, where l is the longest side of the 2D lattice
                numBoundingSquares = screenParameters['numBoundingSquares']
                distanceThreshold = self.cell_radius * np.sqrt(2) * (numBoundingSquares + .001)  # length of half diagonal of a square of side equal to (cell diameter * screenNeighborhoodSize); 0.1% extra to accommodate numerical precision
                self.fieldScreenMatrix = self.utils.defineFieldCellNeighborhoodMap(self.fieldCellDistanceMatrix,distanceThreshold=distanceThreshold)
                self.fieldCellDistanceMatrix = self.fieldCellDistanceMatrix * self.fieldScreenMatrix
                self.fieldCellDistanceMatrix[self.fieldCellDistanceMatrix == 0.0] = torch.inf
            if perturbationParameters is not None:
                perturbStartIter, perturbEndIter, perturbIndices = perturbationParameters
                perturbSampleIndices, perturbPointIndices = perturbIndices
            else:
                perturbStartIter, perturbEndIter, perturbIndices = -1, -1, None
        for iter in range(numSimIters):
            if saveData:
                self.timeseriesVmem[iter] = self.Vmem
                self.timeserieseV[iter] = self.eV
            if inputs != None:
                geneInputs = inputs['gene']
                if (geneInputs != None) and (self.GRNtoVmemWeights != None):
                    self.updateIonChannelConductance(inputState=geneInputs,inputType='gene')
            self.updateCurrent()
            self.updateVmem()
            if (iter >= perturbStartIter) and (iter <= perturbEndIter):
                perturbation = (perturbSampleIndices, perturbPointIndices)
            else:
                perturbation = None
            self.updateFieldEffects(source='Vmem',stochastic=stochastic,perturbation=perturbation)
            if iter < clampIters:
                if (clampMode == 'field') or (clampMode == 'fieldDome'):
                    self.eV[self.clampSampleIndices,self.clampPointIndices1D,0] = clampVoltage
                    self.updateFieldEffects(source='eVClamp',stochastic=stochastic)  # permeate the field of the clamped points into the tissue
                elif (clampMode == 'tissue') or (clampMode == 'tissueDome'):
                    self.Vmem[self.clampSampleIndices,self.clampPointIndices1D,0] = clampVoltage
