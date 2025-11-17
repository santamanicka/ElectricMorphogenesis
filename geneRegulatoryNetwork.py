# A hierarchical two-tier gene regulatory network model
# Tier 1 (Upstream): Neural Crest GRN - specialized developmental control
# Tier 2 (Downstream): Generic GRN - general cellular processes

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
import torch.nn.functional as F
from itertools import chain

class NeuralCrestGRN:
    """Neural Crest Gene Regulatory Network - Upstream Tier"""

    def __init__(self, device='cpu'):
        self.device = device
        self.gene_names = ['Pax3', 'Zic1', 'Msx1', 'Sox9', 'FoxD3', 'Snail2', 'Sox10']
        self.num_nc_genes = len(self.gene_names)

        # Default parameters converted to torch tensors
        self.params = {
            'k_pax3': torch.tensor(2.0, device=device), 'k_zic1': torch.tensor(1.8, device=device),
            'k_msx1': torch.tensor(1.5, device=device), 'k_sox9': torch.tensor(1.2, device=device),
            'k_foxd3': torch.tensor(1.5, device=device), 'k_snail2': torch.tensor(1.0, device=device),
            'k_sox10': torch.tensor(0.8, device=device),
            'd_pax3': torch.tensor(0.2, device=device), 'd_zic1': torch.tensor(0.2, device=device),
            'd_msx1': torch.tensor(0.2, device=device), 'd_sox9': torch.tensor(0.15, device=device),
            'd_foxd3': torch.tensor(0.15, device=device), 'd_snail2': torch.tensor(0.18, device=device),
            'd_sox10': torch.tensor(0.15, device=device),
            'n_pax3_wnt': torch.tensor(2.5, device=device), 'n_pax3_bmp': torch.tensor(2.0, device=device),
            'n_zic1_bmp': torch.tensor(2.0, device=device), 'n_msx1_bmp': torch.tensor(2.5, device=device),
            'n_sox9_pax3': torch.tensor(2.0, device=device), 'n_foxd3_pax3': torch.tensor(2.0, device=device),
            'n_foxd3_msx1': torch.tensor(2.0, device=device), 'n_snail2_sox9': torch.tensor(3.0, device=device),
            'n_snail2_pax3': torch.tensor(2.0, device=device), 'n_sox10_foxd3': torch.tensor(2.0, device=device),
            'n_sox10_snail2': torch.tensor(2.0, device=device),
            'K_pax3_wnt': torch.tensor(0.5, device=device), 'K_pax3_bmp': torch.tensor(0.5, device=device),
            'K_zic1_bmp': torch.tensor(0.4, device=device), 'K_msx1_bmp': torch.tensor(0.6, device=device),
            'K_sox9_pax3': torch.tensor(0.8, device=device), 'K_foxd3_pax3': torch.tensor(0.7, device=device),
            'K_foxd3_msx1': torch.tensor(0.7, device=device), 'K_snail2_sox9': torch.tensor(0.6, device=device),
            'K_snail2_pax3': torch.tensor(0.5, device=device), 'K_sox10_foxd3': torch.tensor(0.7, device=device),
            'K_sox10_snail2': torch.tensor(0.8, device=device),
            'wnt_baseline': torch.tensor(0.5, device=device), 'wnt_peak': torch.tensor(1.5, device=device),
            'wnt_time': torch.tensor(5.0, device=device), 'bmp_baseline': torch.tensor(0.4, device=device),
            'bmp_peak': torch.tensor(1.2, device=device), 'bmp_time': torch.tensor(4.0, device=device),
        }

        self.timestep = 0.01

    def wnt_signal(self, t):
        p = self.params
        return p['wnt_baseline'] + (p['wnt_peak'] - p['wnt_baseline']) / (1 + torch.exp(-0.8 * (t - p['wnt_time'])))

    def bmp_signal(self, t):
        p = self.params
        return p['bmp_baseline'] + (p['bmp_peak'] - p['bmp_baseline']) * torch.exp(-0.15 * (t - p['bmp_time'])**2)

    def hill_activation(self, x, K, n):
        """Hill activation function"""
        return (x**n) / (K**n + x**n)

    def cooperative_activation(self, x1, K1, n1, x2, K2, n2):
        """Cooperative Hill activation"""
        return self.hill_activation(x1, K1, n1) * self.hill_activation(x2, K2, n2)

    def update_state(self, state, t):
        """Update neural crest GRN state"""
        pax3, zic1, msx1, sox9, foxd3, snail2, sox10 = state[..., 0], state[..., 1], state[..., 2], state[..., 3], state[..., 4], state[..., 5], state[..., 6]
        p = self.params
        wnt = self.wnt_signal(t)
        bmp = self.bmp_signal(t)

        # Neural crest GRN dynamics
        pax3_act = self.cooperative_activation(wnt, p['K_pax3_wnt'], p['n_pax3_wnt'], bmp, p['K_pax3_bmp'], p['n_pax3_bmp'])
        dpax3_dt = p['k_pax3'] * pax3_act - p['d_pax3'] * pax3

        zic1_act = self.hill_activation(bmp, p['K_zic1_bmp'], p['n_zic1_bmp'])
        dzic1_dt = p['k_zic1'] * zic1_act - p['d_zic1'] * zic1

        msx1_act = self.hill_activation(bmp, p['K_msx1_bmp'], p['n_msx1_bmp'])
        dmsx1_dt = p['k_msx1'] * msx1_act - p['d_msx1'] * msx1

        sox9_act = self.hill_activation(pax3, p['K_sox9_pax3'], p['n_sox9_pax3'])
        dsox9_dt = p['k_sox9'] * sox9_act - p['d_sox9'] * sox9

        foxd3_act = self.cooperative_activation(pax3, p['K_foxd3_pax3'], p['n_foxd3_pax3'], msx1, p['K_foxd3_msx1'], p['n_foxd3_msx1'])
        dfoxd3_dt = p['k_foxd3'] * foxd3_act - p['d_foxd3'] * foxd3

        snail2_act = self.cooperative_activation(sox9, p['K_snail2_sox9'], p['n_snail2_sox9'], pax3, p['K_snail2_pax3'], p['n_snail2_pax3'])
        dsnail2_dt = p['k_snail2'] * snail2_act - p['d_snail2'] * snail2

        sox10_act = self.cooperative_activation(foxd3, p['K_sox10_foxd3'], p['n_sox10_foxd3'], snail2, p['K_sox10_snail2'], p['n_sox10_snail2'])
        dsox10_dt = p['k_sox10'] * sox10_act - p['d_sox10'] * sox10

        # Stack derivatives
        dstate_dt = torch.stack([dpax3_dt, dzic1_dt, dmsx1_dt, dsox9_dt, dfoxd3_dt, dsnail2_dt, dsox10_dt], dim=-1)

        return state + self.timestep * dstate_dt

    def get_downstream_regulation(self, nc_state):
        """Compute how neural crest genes regulate downstream generic GRN"""
        # Key neural crest outputs that regulate downstream processes
        pax3, zic1, msx1, sox9, foxd3, snail2, sox10 = [nc_state[..., i] for i in range(7)]

        # Create regulatory signals for downstream GRN
        # These represent how NC genes control generic cellular processes
        proliferation_signal = self.hill_activation(pax3 + sox9, torch.tensor(0.5, device=self.device), torch.tensor(2.0, device=self.device))
        migration_signal = self.hill_activation(snail2 + foxd3, torch.tensor(0.6, device=self.device), torch.tensor(2.5, device=self.device))
        differentiation_signal = self.hill_activation(sox10 + foxd3, torch.tensor(0.7, device=self.device), torch.tensor(2.0, device=self.device))

        return torch.stack([proliferation_signal, migration_signal, differentiation_signal], dim=-1)

class geneRegulatoryNetwork():
    """Hierarchical Two-Tier Gene Regulatory Network - Downstream Generic GRN"""

    def __init__(self,parameters=None,numSamples=1):
        self.parameters = parameters
        self.numSamples = numSamples
        self.defineParameters()
        self.defineVariables()
        self.composeTissueLevelGRN()
        self.timestep = 0.01

        # Initialize Neural Crest GRN (upstream tier)
        self.neuralCrestGRN = NeuralCrestGRN(device='cpu')
        self.nc_state = torch.zeros(self.numSamples, self.numCells, 7, dtype=torch.float64)
        self.current_time = 0.0

        # Hill function parameters for downstream GRN
        self.hill_params = {
            'K_values': torch.ones(1, self.numGenes, dtype=torch.float64) * 0.5,  # Half-saturation constants
            'n_values': torch.ones(1, self.numGenes, dtype=torch.float64) * 2.0,  # Hill coefficients
        }

    # define parameters (weights, biases and external inputs) and populate them with default values
    def defineParameters(self):
        self.LatticeDimensions = self.parameters['latticeDims']
        self.numRows, self.numCols = self.LatticeDimensions
        self.tissueConnectivity = self.parameters['GRNParameters']['tissueConnectivity']
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

    def hill_activation(self, x, K, n):
        """Hill activation function for downstream GRN"""
        return (x**n) / (K**n + x**n)

    def updateState(self, ATPConc=None):
        """Update both neural crest GRN (upstream) and generic GRN (downstream) states"""

        # Update neural crest GRN state
        for cell_idx in range(self.numCells):
            self.nc_state[:, cell_idx, :] = self.neuralCrestGRN.update_state(self.nc_state[:, cell_idx, :], self.current_time)

        # Get downstream regulation signals from neural crest GRN
        nc_regulation = self.neuralCrestGRN.get_downstream_regulation(self.nc_state)  # Shape: (numSamples, numCells, 3)

        # Reshape regulation signals for tissue-level application
        nc_regulation_tissue = nc_regulation.repeat_interleave(self.numGenes // 3, dim=-1)  # Distribute to downstream genes
        if nc_regulation_tissue.shape[-1] < self.numGenes:
            # Pad if needed
            padding = torch.zeros(self.numSamples, self.numCells, self.numGenes - nc_regulation_tissue.shape[-1], dtype=torch.float64)
            nc_regulation_tissue = torch.cat([nc_regulation_tissue, padding], dim=-1)
        nc_regulation_flat = nc_regulation_tissue.reshape(self.numSamples, self.numVariables, 1)

        # Update downstream generic GRN with ATP modulation
        if ATPConc is not None:
            self.ATPConc = ATPConc.repeat_interleave(self.numGenes,dim=1)  # shape = (1,numCells*numGenes,1)
            self.W = ((self.tissueGRNWeights * self.ATPConc) + (self.tissueGRNWeights * self.ATPConc.transpose(1,2))).squeeze(0)  # shape = (numCells*numGenes,numCells*numGenes)
            self.W = self.W / 2
        else:
            self.W = self.tissueGRNWeights

        # Replace sigmoid functions with Hill functions
        # Prepare Hill function parameters
        K_tissue = torch.tile(self.hill_params['K_values'], (self.numCells,)).view(self.numVariables, 1)
        n_tissue = torch.tile(self.hill_params['n_values'], (self.numCells,)).view(self.numVariables, 1)

        # Hill activation for gene-gene interactions
        gene_input = self.tissueGRNGain * (self.state + self.tissueGRNBias)
        gene_hill_activation = self.hill_activation(torch.clamp(gene_input, min=0), K_tissue, n_tissue)

        # Hill activation for Vmem input
        vmem_input = torch.exp(self.VmemGain) * self.tissueExternalInputs + self.VmemBias
        vmem_hill_activation = 2 * self.hill_activation(torch.clamp(vmem_input, min=0), torch.tensor(0.5), torch.tensor(2.0)) - 1

        # Include neural crest regulation as additional input
        self.dstate = -self.state + torch.matmul(self.W, gene_hill_activation) + \
                     self.tissueVmemToGRNWeights * vmem_hill_activation + \
                     0.5 * nc_regulation_flat  # Neural crest upstream control

        self.dstate = self.dstate / self.tissueGRNTimeconstants
        self.state = self.state + (self.timestep * self.dstate)

        # Increment time for neural crest GRN
        self.current_time += self.timestep

    def simulate(self,electricNetworkState=None,ATPConc=None,numSimIters=1):
        for iter in range(numSimIters):
            self.updateDynamicalParameters(externalInputs=electricNetworkState)
            self.updateState(ATPConc=ATPConc)


class FacialGRN:
    """Craniofacial Patterning Gene Regulatory Network - Morphogen-based pattern formation

    Compatible with geneRegulatoryNetwork parameter structure for integration with existing framework.
    Can be used in two modes:
    1. Standalone mode: FacialGRN(grid_size=40, device='cpu')
    2. Framework mode: FacialGRN(parameters=params_dict, numSamples=1)
    """

    def __init__(self, parameters=None, numSamples=1, grid_size=None, device='cpu'):
        """Initialize FacialGRN with either parameters dict or simple arguments

        Args:
            parameters: Dict with structure similar to geneRegulatoryNetwork (optional)
            numSamples: Number of parallel samples (for batch processing)
            grid_size: Grid size (only used if parameters=None)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.numSamples = numSamples
        self.current_time = 0

        # Gene and morphogen names
        self.morphogen_names = ['shh', 'fgf8', 'edn1']
        self.gene_names = ['rx', 'six3', 'pax6', 'lhx2', 'alx', 'dlx', 'hand2', 'runx2']
        self.feature_names = ['undifferentiated', 'eye', 'nose', 'jaw', 'bone']
        self.numGenes = len(self.gene_names)

        # Initialize morphogen-specific parameters FIRST (before defineParameters)
        self.params = {
            'shhStrength': torch.tensor(1.0, device=device, dtype=torch.float64),
            'fgf8Strength': torch.tensor(1.0, device=device, dtype=torch.float64),
            'edn1Strength': torch.tensor(1.0, device=device, dtype=torch.float64),
            'diffusionRate': torch.tensor(0.1, device=device, dtype=torch.float64),
            'degradationRate': torch.tensor(0.05, device=device, dtype=torch.float64),
            'inhibitionStrength': torch.tensor(0.3, device=device, dtype=torch.float64),
            'geneActivationRate': torch.tensor(0.05, device=device, dtype=torch.float64),
            'geneDegradationRate': torch.tensor(0.02, device=device, dtype=torch.float64),
        }

        # Determine initialization mode
        if parameters is not None:
            # Framework mode - compatible with geneRegulatoryNetwork
            self.parameters = parameters
            self.defineParameters()
        else:
            # Standalone mode - simple initialization
            self.grid_size = grid_size if grid_size is not None else 40
            self.numCells = self.grid_size * self.grid_size
            self.numRows, self.numCols = self.grid_size, self.grid_size
            self.numVariables = self.numCells * self.numGenes

            # Set default facial-specific parameters
            self.InterGRNWeights = None  # Cell-autonomous by default
            self.tissueConnectivity = None

        # Initialize variables
        self.defineVariables()

        # Initialize grid data structures
        self.initialize_grid()
        self.timestep = 0.01

        # Bioelectric coupling placeholders
        self.face_set_point = None
        self.face_snap_strength = 0.0
        self.bioelectric_targets = None
        self.bioelectric_weight = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        self.bioelectric_feature_mask = None
        self.bioelectric_prepattern_enabled = True
        self.voltage_input_gain = torch.tensor(0.05, dtype=torch.float64, device=self.device)
        self.voltage_input = None
        self.voltage_detail = None
        self.voltage_lowpass = 0.8

    def defineParameters(self):
        """Define parameters from parameters dict (compatible with geneRegulatoryNetwork)"""
        self.LatticeDimensions = self.parameters['latticeDims']
        self.numRows, self.numCols = self.LatticeDimensions
        self.grid_size = self.numRows  # Assume square grid for facial patterning

        # Tissue connectivity (optional for FacialGRN)
        if 'GRNParameters' in self.parameters:
            grn_params = self.parameters['GRNParameters']
            self.tissueConnectivity = grn_params.get('tissueConnectivity', None)
            self.InterGRNWeights = grn_params.get('InterGRNWeights', None)
            self.numCells = self.tissueConnectivity.shape[0] if self.tissueConnectivity is not None else self.numRows * self.numCols

            # Override morphogen parameters if specified
            if 'facialParams' in grn_params:
                facial_params = grn_params['facialParams']
                for key in ['shhStrength', 'fgf8Strength', 'edn1Strength', 'diffusionRate',
                           'degradationRate', 'inhibitionStrength', 'geneActivationRate',
                           'geneDegradationRate']:
                    if key in facial_params:
                        self.params[key] = torch.tensor(facial_params[key], device=self.device, dtype=torch.float64)
        else:
            self.tissueConnectivity = None
            self.InterGRNWeights = None
            self.numCells = self.numRows * self.numCols

        self.numVariables = self.numCells * self.numGenes

        # Enforce cell-autonomous behavior (no InterGRNWeights)
        if self.InterGRNWeights is not None:
            print("Warning: FacialGRN is cell-autonomous. Setting InterGRNWeights=None.")
            self.InterGRNWeights = None

    def defineVariables(self):
        """Create arrays of genetic variables with default values"""
        # State for compatibility with geneRegulatoryNetwork interface
        self.dstate = torch.zeros(self.numSamples, self.numVariables, 1, dtype=torch.float64, device=self.device)
        self.state = torch.zeros(self.numSamples, self.numVariables, 1, dtype=torch.float64, device=self.device)
        self.tissueExternalInputs = torch.zeros(self.numSamples, self.numVariables, 1, dtype=torch.float64, device=self.device)

    def initialize_grid(self):
        """Initialize morphogen gradients and gene expression grids"""
        gs = self.grid_size
        mid_x = gs / 2.0

        # Initialize all grids
        self.grid = {
            'shh': torch.zeros(gs, gs, dtype=torch.float64, device=self.device),
            'fgf8': torch.zeros(gs, gs, dtype=torch.float64, device=self.device),
            'edn1': torch.zeros(gs, gs, dtype=torch.float64, device=self.device),
            'rx': torch.zeros(gs, gs, dtype=torch.float64, device=self.device),
            'six3': torch.zeros(gs, gs, dtype=torch.float64, device=self.device),
            'pax6': torch.zeros(gs, gs, dtype=torch.float64, device=self.device),
            'lhx2': torch.zeros(gs, gs, dtype=torch.float64, device=self.device),
            'alx': torch.zeros(gs, gs, dtype=torch.float64, device=self.device),
            'dlx': torch.zeros(gs, gs, dtype=torch.float64, device=self.device),
            'hand2': torch.zeros(gs, gs, dtype=torch.float64, device=self.device),
            'feature': torch.zeros(gs, gs, dtype=torch.float64, device=self.device),
            'runx2': torch.zeros(gs, gs, dtype=torch.float64, device=self.device),
        }

        # Create coordinate grids
        # Normalized coordinate grids (0 at dorsal/left, 1 at ventral/right)
        y_coords = torch.linspace(0.0, 1.0, gs, dtype=torch.float64, device=self.device).view(gs, 1).expand(gs, gs)
        x_coords = torch.linspace(0.0, 1.0, gs, dtype=torch.float64, device=self.device).view(1, gs).expand(gs, gs)
        dist_from_midline = torch.abs(x_coords - 0.5)

        # Widths expressed as fractions so they adapt to small grids (e.g., 11x11)
        sigma_med = 0.18
        sigma_lat = 0.15
        sigma_jaw = 0.25

        # Shh: medial ridge, slightly stronger anteriorly
        shh_medial = torch.exp(-(dist_from_midline ** 2) / (2 * sigma_med ** 2))
        shh_anterior = 0.7 + 0.3 * (1.0 - y_coords)
        self.grid['shh'] = self.params['shhStrength'] * shh_medial * shh_anterior

        # Fgf8: lateral bands (eyes) diminished at the very midline
        fgf8_lateral = 1.0 - torch.exp(-(dist_from_midline ** 2) / (2 * sigma_lat ** 2))
        fgf8_anterior = 0.7 + 0.3 * (1.0 - y_coords)
        self.grid['fgf8'] = self.params['fgf8Strength'] * fgf8_lateral * fgf8_anterior

        # Edn1: posterior/ventral gradient with slight medial bias for jaw
        edn1_gradient = torch.clamp((y_coords - 0.25) / 0.75, min=0.0, max=1.0)
        jaw_focus = torch.exp(-(dist_from_midline ** 2) / (2 * sigma_jaw ** 2))
        self.grid['edn1'] = self.params['edn1Strength'] * edn1_gradient * (0.6 + 0.4 * jaw_focus)

        # Reset time
        self.current_time = 0

    def hill_activation(self, x, k, n):
        """Hill activation function"""
        return (x**n) / (k**n + x**n)

    def inhibit(self, x, k, n):
        """Inhibition function"""
        return 1.0 / (1.0 + (x / k)**n)

    def update_morphogens(self):
        """Update morphogen gradients with diffusion and mutual inhibition"""
        gs = self.grid_size

        # Create new grids for updates
        new_shh = self.grid['shh'].clone()
        new_fgf8 = self.grid['fgf8'].clone()
        new_edn1 = self.grid['edn1'].clone()

        # Update interior points (excluding boundaries)
        for y in range(1, gs - 1):
            for x in range(1, gs - 1):
                # Diffusion via Laplacian
                shh_laplacian = (
                    self.grid['shh'][y-1, x] + self.grid['shh'][y+1, x] +
                    self.grid['shh'][y, x-1] + self.grid['shh'][y, x+1] -
                    4 * self.grid['shh'][y, x]
                )

                fgf8_laplacian = (
                    self.grid['fgf8'][y-1, x] + self.grid['fgf8'][y+1, x] +
                    self.grid['fgf8'][y, x-1] + self.grid['fgf8'][y, x+1] -
                    4 * self.grid['fgf8'][y, x]
                )

                # Mutual inhibition between Shh and Fgf8
                shh_inhibition = self.inhibit(self.grid['fgf8'][y, x], torch.tensor(0.5), torch.tensor(2.0))
                fgf8_inhibition = self.inhibit(self.grid['shh'][y, x], torch.tensor(0.5), torch.tensor(2.0))

                # Update Shh
                new_shh[y, x] = self.grid['shh'][y, x] + \
                    self.params['diffusionRate'] * shh_laplacian - \
                    self.params['degradationRate'] * self.grid['shh'][y, x] * (1.0 - shh_inhibition) + \
                    self.params['inhibitionStrength'] * self.grid['shh'][y, x] * (1.0 - self.grid['fgf8'][y, x])

                # Update Fgf8
                new_fgf8[y, x] = self.grid['fgf8'][y, x] + \
                    self.params['diffusionRate'] * fgf8_laplacian - \
                    self.params['degradationRate'] * self.grid['fgf8'][y, x] * (1.0 - fgf8_inhibition) + \
                    self.params['inhibitionStrength'] * self.grid['fgf8'][y, x] * (1.0 - self.grid['shh'][y, x])

                # Edn1 doesn't diffuse much - maintain posterior expression
                new_edn1[y, x] = self.grid['edn1'][y, x]

                # Clamp values to [0, 1]
                new_shh[y, x] = torch.clamp(new_shh[y, x], min=0.0, max=1.0)
                new_fgf8[y, x] = torch.clamp(new_fgf8[y, x], min=0.0, max=1.0)
                new_edn1[y, x] = torch.clamp(new_edn1[y, x], min=0.0, max=1.0)

        # Update grids
        self.grid['shh'] = new_shh
        self.grid['fgf8'] = new_fgf8
        self.grid['edn1'] = new_edn1

    def update_genes(self):
        """Update gene expression based on morphogen levels"""
        gs = self.grid_size

        # Vectorized gene expression updates
        shh = self.grid['shh']
        fgf8 = self.grid['fgf8']
        edn1 = self.grid['edn1']

        # Eye pathway: High Fgf8, Low Shh, Low Edn1
        target_rx = self.hill_activation(fgf8, torch.tensor(0.3), torch.tensor(2.0)) * \
                    self.inhibit(shh, torch.tensor(0.4), torch.tensor(2.0)) * \
                    self.inhibit(edn1, torch.tensor(0.2), torch.tensor(2.0))
        target_six3 = self.hill_activation(self.grid['rx'], torch.tensor(0.3), torch.tensor(2.0))
        target_pax6 = self.hill_activation(self.grid['six3'], torch.tensor(0.3), torch.tensor(2.0))
        target_lhx2 = self.hill_activation(self.grid['pax6'], torch.tensor(0.3), torch.tensor(2.0))

        # Nose pathway: High Shh, Low Fgf8, Low Edn1
        target_alx = self.hill_activation(shh, torch.tensor(0.5), torch.tensor(2.0)) * \
                     self.inhibit(fgf8, torch.tensor(0.4), torch.tensor(2.0)) * \
                     self.inhibit(edn1, torch.tensor(0.2), torch.tensor(2.0))

        # Jaw pathway: High Edn1, Moderate Shh
        target_dlx = self.hill_activation(edn1, torch.tensor(0.3), torch.tensor(2.0)) * \
                     self.hill_activation(shh, torch.tensor(0.15), torch.tensor(1.5))
        target_hand2 = self.hill_activation(self.grid['dlx'], torch.tensor(0.3), torch.tensor(2.0))

        # Gradual dynamics for gene expression
        self.grid['rx'] += self.params['geneActivationRate'] * (target_rx - self.grid['rx']) - \
                          self.params['geneDegradationRate'] * self.grid['rx']
        self.grid['six3'] += self.params['geneActivationRate'] * (target_six3 - self.grid['six3']) - \
                            self.params['geneDegradationRate'] * self.grid['six3']
        self.grid['pax6'] += self.params['geneActivationRate'] * (target_pax6 - self.grid['pax6']) - \
                            self.params['geneDegradationRate'] * self.grid['pax6']
        self.grid['lhx2'] += self.params['geneActivationRate'] * (target_lhx2 - self.grid['lhx2']) - \
                            self.params['geneDegradationRate'] * self.grid['lhx2']
        self.grid['alx'] += self.params['geneActivationRate'] * (target_alx - self.grid['alx']) - \
                           self.params['geneDegradationRate'] * self.grid['alx']
        self.grid['dlx'] += self.params['geneActivationRate'] * (target_dlx - self.grid['dlx']) - \
                           self.params['geneDegradationRate'] * self.grid['dlx']
        self.grid['hand2'] += self.params['geneActivationRate'] * (target_hand2 - self.grid['hand2']) - \
                             self.params['geneDegradationRate'] * self.grid['hand2']

        if self.bioelectric_targets is not None:
            for gene_name, target in self.bioelectric_targets.items():
                self.grid[gene_name] += self.bioelectric_weight * (target - self.grid[gene_name])

        if self.voltage_input is not None:
            detail_map = self.voltage_detail if self.voltage_detail is not None else (self.voltage_input - 0.5)
            detail_map = detail_map[0]
            gain = float(self.voltage_input_gain.item())
            base = 0.5
            eye_drive = torch.clamp(-detail_map, min=0.0, max=1.0)
            jaw_drive = torch.clamp(detail_map, min=0.0, max=1.0)
            nose_drive = torch.clamp(1.0 - detail_map.abs(), min=0.0, max=1.0)
            bone_drive = torch.clamp(0.7 - detail_map.abs(), min=0.0, max=1.0)
            for gene in ['rx', 'six3', 'pax6', 'lhx2']:
                self.grid[gene] += gain * (eye_drive - base)
            self.grid['alx'] += 0.5 * gain * (nose_drive - base)
            self.grid['dlx'] += gain * (jaw_drive - base)
            self.grid['hand2'] += gain * (jaw_drive - base)
            self.grid['runx2'] += 0.3 * gain * (bone_drive - base)

        eye_score = self.grid['pax6'] * self.grid['lhx2']
        nose_score = self.grid['alx']
        jaw_score = self.grid['hand2']

        if self.bioelectric_feature_mask is not None:
            bone_prepattern = (self.bioelectric_feature_mask == 0).to(dtype=torch.float64, device=self.device)
        else:
            bone_prepattern = ((eye_score < 0.15) & (nose_score < 0.15) & (jaw_score < 0.15)).to(torch.float64)
        self.grid['runx2'] += self.params['geneActivationRate'] * (bone_prepattern - self.grid['runx2']) - \
                              self.params['geneDegradationRate'] * self.grid['runx2']

        # Clamp gene expression values to [0, 1]
        for gene in ['rx', 'six3', 'pax6', 'lhx2', 'alx', 'dlx', 'hand2', 'runx2']:
            self.grid[gene] = torch.clamp(self.grid[gene], min=0.0, max=1.0)

        # Feature assignment based on gene expression

        # Determine dominant feature at each position
        self.grid['feature'] = torch.zeros_like(self.grid['feature'])

        # Eye (feature = 1)
        eye_mask = (eye_score > nose_score) & (eye_score > jaw_score) & (eye_score > 0.15)
        self.grid['feature'][eye_mask] = 1.0

        # Nose (feature = 2)
        nose_mask = (nose_score > jaw_score) & (nose_score > 0.15) & (~eye_mask)
        self.grid['feature'][nose_mask] = 2.0

        # Jaw (feature = 3)
        jaw_mask = (jaw_score > 0.15) & (~eye_mask) & (~nose_mask)
        self.grid['feature'][jaw_mask] = 3.0

        bone_mask = (~eye_mask) & (~nose_mask) & (~jaw_mask)
        if self.bioelectric_feature_mask is not None:
            bone_mask = bone_mask | (self.bioelectric_feature_mask == 0)
        bone_mask = bone_mask | (self.grid['runx2'] > 0.2)
        self.grid['feature'][bone_mask] = 0.0

    def get_feature_map(self):
        return self.grid['feature'].clone()

    def sync_grid_to_state(self):
        """Sync grid representation to state vector (for framework compatibility)"""
        # Flatten grid genes into state vector format (numSamples, numVariables, 1)
        # Order: all genes for cell 0, all genes for cell 1, etc.
        for sample_idx in range(self.numSamples):
            for gene_idx, gene_name in enumerate(self.gene_names):
                gene_grid = self.grid[gene_name].flatten()  # Flatten 2D grid to 1D
                # Place into state vector
                for cell_idx in range(self.numCells):
                    var_idx = cell_idx * self.numGenes + gene_idx
                    self.state[sample_idx, var_idx, 0] = gene_grid[cell_idx]

    def sync_state_to_grid(self):
        """Sync state vector to grid representation (for framework compatibility)"""
        # Extract grid genes from state vector
        for sample_idx in range(self.numSamples):
            for gene_idx, gene_name in enumerate(self.gene_names):
                gene_data = torch.zeros(self.numCells, dtype=torch.float64, device=self.device)
                for cell_idx in range(self.numCells):
                    var_idx = cell_idx * self.numGenes + gene_idx
                    gene_data[cell_idx] = self.state[sample_idx, var_idx, 0]
                # Reshape to 2D grid
                self.grid[gene_name] = gene_data.reshape(self.grid_size, self.grid_size)

    def updateDynamicalParameters(self, externalInputs=None):
        """Interface for external input (Vmem) - compatible with geneRegulatoryNetwork"""
        # For FacialGRN, external inputs could modulate morphogen production
        # Currently not used, but maintained for interface compatibility
        if externalInputs is not None:
            self.tissueExternalInputs = torch.repeat_interleave(
                externalInputs, repeats=self.numGenes, dim=1
            ).view(self.numSamples, self.numVariables, 1)
        else:
            self.tissueExternalInputs = torch.zeros(
                self.numSamples, self.numVariables, 1,
                dtype=torch.float64, device=self.device
            )

        if externalInputs is not None:
            vmem = externalInputs
            if vmem.dim() == 2:
                vmem = vmem.unsqueeze(2)
            vmem = vmem.view(self.numSamples, self.numCells, -1)[..., 0]
            vmem_grid = vmem.view(self.numSamples, self.numRows, self.numCols)
            vmin = vmem_grid.amin(dim=(1, 2), keepdim=True)
            vmax = vmem_grid.amax(dim=(1, 2), keepdim=True)
            norm = torch.clamp((vmem_grid - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
            if self.voltage_input is None:
                self.voltage_input = norm
            else:
                alpha = self.voltage_lowpass
                self.voltage_input = (alpha * self.voltage_input) + ((1 - alpha) * norm)
            blur = F.avg_pool2d(self.voltage_input.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
            self.voltage_detail = self.voltage_input - blur
        else:
            self.voltage_input = None
            self.voltage_detail = None

    def register_face_set_point(self, set_point, snap_strength=0.3):
        """Store bioelectric face targets and register them as a prepattern."""
        self.face_set_point = set_point
        self.face_snap_strength = snap_strength
        if self.bioelectric_prepattern_enabled:
            self.register_bioelectric_prepattern(set_point, weight=snap_strength)
            if snap_strength > 0:
                self.apply_face_set_point(force=True)

    def register_bioelectric_prepattern(self, set_point, weight=0.3):
        """Convert bioelectric gene targets into a slow-acting prepattern."""
        targets_grid = set_point['gene_targets_grid']
        if targets_grid.ndim == 4:
            targets_grid = targets_grid[0]
        if self.bioelectric_prepattern_enabled:
            self.bioelectric_targets = {
                gene: targets_grid[:, :, idx].to(dtype=torch.float64, device=self.device)
                for idx, gene in enumerate(self.gene_names)
            }
            self.bioelectric_weight = torch.tensor(weight, dtype=torch.float64, device=self.device)
            feature_mask = set_point.get('feature_mask_grid', None)
            if feature_mask is not None:
                if feature_mask.ndim == 3:
                    feature_mask = feature_mask[0]
                self.bioelectric_feature_mask = feature_mask.to(dtype=torch.float64, device=self.device)

    def apply_face_set_point(self, force=False):
        """Blend current gene expression toward the bioelectric set point."""
        if self.face_set_point is None:
            return
        snap = 1.0 if force else self.face_snap_strength
        if snap <= 0:
            return
        targets_grid = self.face_set_point['gene_targets_grid']
        feature_grid = self.face_set_point['feature_mask_grid']
        if targets_grid.shape[0] > 1:
            target_idx = 0  # FacialGRN grids are shared across samples; use first sample
        else:
            target_idx = 0
        for gene_idx, gene_name in enumerate(self.gene_names):
            target = targets_grid[target_idx, :, :, gene_idx]
            self.grid[gene_name] = ((1 - snap) * self.grid[gene_name]) + (snap * target)
        self.grid['feature'] = feature_grid[target_idx]

    def updateState(self, ATPConc=None):
        """Update state - compatible with geneRegulatoryNetwork interface

        This is the framework-compatible version called by simulation loops.
        Updates both morphogens and genes, then syncs to state vector.
        """
        # Update morphogen gradients and gene expression
        self.update_morphogens()
        self.update_genes()
        self.apply_face_set_point()

        # Sync grid to state vector for framework compatibility
        self.sync_grid_to_state()

        # Update time
        self.current_time += 1

    def update_state(self):
        """Update both morphogens and gene expression for one timestep (standalone)

        This is the simple version for standalone usage.
        """
        self.updateState()  # Call framework-compatible version

    def simulate(self, electricNetworkState=None, ATPConc=None, numSimIters=None, num_steps=None):
        """Run simulation - compatible with both standalone and framework modes

        Args:
            electricNetworkState: External inputs (Vmem) - framework mode
            ATPConc: ATP concentration - framework mode (not used in FacialGRN)
            numSimIters: Number of iterations - framework mode
            num_steps: Number of steps - standalone mode

        Usage:
            Standalone: grn.simulate(num_steps=100)
            Framework: grn.simulate(electricNetworkState=vmem, numSimIters=100)
        """
        # Determine number of iterations
        if numSimIters is not None:
            iterations = numSimIters
        elif num_steps is not None:
            iterations = num_steps
        else:
            iterations = 100  # Default

        # Run simulation loop
        for iter in range(iterations):
            self.updateDynamicalParameters(externalInputs=electricNetworkState)
            self.updateState(ATPConc=ATPConc)

    def reset(self):
        """Reset simulation to initial state"""
        self.initialize_grid()
        self.current_time = 0
        self.sync_grid_to_state()

    def get_state(self):
        """Get current state of all grids (standalone interface)"""
        return {
            'morphogens': {k: self.grid[k].clone() for k in self.morphogen_names},
            'genes': {k: self.grid[k].clone() for k in self.gene_names},
            'features': self.grid['feature'].clone(),
            'time': self.current_time
        }

    def set_parameters(self, **kwargs):
        """Update model parameters (standalone interface)"""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = torch.tensor(value, device=self.device, dtype=torch.float64)
        # Reinitialize grid with new parameters
        self.initialize_grid()
