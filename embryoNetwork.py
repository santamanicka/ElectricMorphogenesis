import torch
import numpy as np
from embryo import model
import utilities
from scipy.integrate import odeint
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

class embryoNetwork():

    # arguments will be passed by the GA or by any program that wants to call the simulation module
    def __init__(self,parameters=None,nsamples=1,niters=100,use_parallel=True,force_cpu=False,parallel_threshold=50):
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
        self.use_parallel = use_parallel
        self.parallel_threshold = parallel_threshold

        # Detect available device (CUDA, MPS for Mac, or CPU)
        # Note: MPS support is disabled by default due to float64 compatibility issues
        # in the underlying simulation code. Use force_cpu=False explicitly to enable MPS.
        if force_cpu:
            self.device = torch.device('cpu')
            self.dtype = torch.float64
            print("Using CPU device (forced)")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.dtype = torch.float64  # CUDA supports float64
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Default to CPU on MPS due to dtype compatibility issues
            # The internal simulation code has many hardcoded float64 tensors
            print("MPS device detected but using CPU (safer for this codebase)")
            print("Note: MPS doesn't support float64 which this simulation requires")
            self.device = torch.device('cpu')
            self.dtype = torch.float64
        else:
            self.device = torch.device('cpu')
            self.dtype = torch.float64  # CPU supports float64
            print("Using CPU device")

        self.grid = [[self.instantiateEmbryo(modelNum=self.modelNameEmbryo,GRNEnabled=self.GRNEnabled,LigandEnabled=self.LigandEnabled,
                                             fieldModulation=self.fieldModulation) for j in range(self.ncols)] for i in range(self.nrows)]

    def instantiateEmbryo(self,modelNum=0,GRNEnabled=False,LigandEnabled=False,fieldModulation=False):
        Sfx = '_fieldVector'
        if LigandEnabled:
            Sfx += '_Ligand'
        if GRNEnabled:
            Sfx += '_GRN'
        parameterfilename = './data/bestModelParameters' + Sfx + '_' + modelNum + '.dat'
        embryoParameters = torch.load(parameterfilename,weights_only=False)
        if GRNEnabled:
            if self.teratogenExposed:
                embryoParameters['GRNParameters']['GRNWeights'] /= 11.5  # assuming ATP model = 262
                embryoParameters['GRNParameters']['InterGRNWeights'] /= 11.5
        if fieldModulation and (not GRNEnabled) and (not LigandEnabled):
            if self.teratogenExposed:
                embryoParameters['fieldParameters']['fieldTransductionGain'] /= 9.6  # assuming ATP model = 262
        embryoParameters['ATPParameters'] = dict()
        embryoParameters['ATPParameters']['ATPEnabled'] = True
        embryoParameters['ATPParameters']['ATPReactionStrength'] = 1.0
        embryoParameters['ATPParameters']['ATPDiffusionStrength'] = 10.0
        embryoParameters['ATPParameters']['tissueConnectivity'] = \
            self.utils.computeLatticeAdjacencyMatrix(latticeDims=embryoParameters['latticeDims'],periodicBoundary=False)
        embryoParameters['ATPParameters']['ATPModelNum'] = self.modelNameATP
        embryoinstance = model(embryoParameters)
        embryoinstance.electricNetwork.timepointsATP = np.linspace(0,20,self.niters)
        return embryoinstance

    def move_embryo_to_device(self, embryoinstance):
        """Move all tensors in an embryo instance to the selected device and dtype."""
        # Move electricNetwork tensors - iterate through all attributes
        if hasattr(embryoinstance, 'electricNetwork'):
            for attr_name in dir(embryoinstance.electricNetwork):
                if not attr_name.startswith('_'):  # Skip private attributes
                    try:
                        attr = getattr(embryoinstance.electricNetwork, attr_name, None)
                        if isinstance(attr, torch.Tensor):
                            # Only convert dtype for floating point tensors, not integer/bool tensors
                            if attr.is_floating_point():
                                setattr(embryoinstance.electricNetwork, attr_name,
                                       attr.to(device=self.device, dtype=self.dtype))
                            else:
                                # Just move to device, keep original dtype for indices
                                setattr(embryoinstance.electricNetwork, attr_name,
                                       attr.to(device=self.device))
                    except (AttributeError, RuntimeError):
                        # Skip attributes that can't be accessed or converted
                        pass

        # Move geneNetwork tensors if GRN is enabled
        if hasattr(embryoinstance, 'geneNetwork') and embryoinstance.geneNetwork is not None:
            for attr_name in dir(embryoinstance.geneNetwork):
                if not attr_name.startswith('_'):  # Skip private attributes
                    try:
                        attr = getattr(embryoinstance.geneNetwork, attr_name, None)
                        if isinstance(attr, torch.Tensor):
                            # Only convert dtype for floating point tensors, not integer/bool tensors
                            if attr.is_floating_point():
                                setattr(embryoinstance.geneNetwork, attr_name,
                                       attr.to(device=self.device, dtype=self.dtype))
                            else:
                                # Just move to device, keep original dtype for indices
                                setattr(embryoinstance.geneNetwork, attr_name,
                                       attr.to(device=self.device))
                    except (AttributeError, RuntimeError):
                        # Skip attributes that can't be accessed or converted
                        pass

    def simulateEmbryo(self,row=0,col=0,save=False):
        """Simulate a single embryo at grid position (row, col)."""
        embryoinstance = self.grid[row][col]

        # Move embryo tensors to device
        self.move_embryo_to_device(embryoinstance)

        numCells = embryoinstance.numCells
        initialValues = embryoinstance.parameters['simParameters']['initialValues']

        if self.teratogenExposed:
            initialValues['ATPConc'] = torch.ones((self.nsamples,embryoinstance.numCells,1),dtype=self.dtype,device=self.device)
            atp_init = self.ATPConcsInit[:,row,col].unsqueeze(1).repeat(1,numCells).to(dtype=self.dtype)
            initialValues['ATPConc'][:,:,0] = atp_init
        else:
            initialValues['ATPConc'] = torch.ones((self.nsamples,embryoinstance.numCells,1),dtype=self.dtype,device=self.device) * 11.5

        embryoinstance.setExperimentalConditions((initialValues, 1))
        clampParameters = embryoinstance.parameters['clampParameters']

        # Convert clampParameters tensors if they exist
        if clampParameters is not None and 'clampVmem' in clampParameters:
            if isinstance(clampParameters['clampVmem'], torch.Tensor):
                clampParameters['clampVmem'] = clampParameters['clampVmem'].to(device=self.device, dtype=self.dtype)

        circuit = embryoinstance.electricNetwork
        boundaryCells = self.utils.computeDomeIndices(circuit,mode='tissue')

        # Ensure boundaryCells is a proper integer tensor/list for indexing
        if isinstance(boundaryCells, torch.Tensor):
            boundaryCells = boundaryCells.long().cpu().tolist()
        elif isinstance(boundaryCells, np.ndarray):
            boundaryCells = boundaryCells.astype(int).tolist()

        externalInputs = dict()
        externalInputs['ATP'] = torch.zeros((self.nsamples,self.niters,numCells,1),dtype=self.dtype,device=self.device)

        if self.boundaryAssistance:
            ATPCurrentEmbryo = self.ATPCurrent[:,:,row,col].unsqueeze(2).repeat(1,1,len(boundaryCells)).to(dtype=self.dtype)
            externalInputs['ATP'][:,:,boundaryCells, 0] = ATPCurrentEmbryo
        else:  # full-embryo assistance
            ATPCurrentEmbryo = self.ATPCurrent[:,:,row,col].unsqueeze(2).repeat(1,1,numCells).to(dtype=self.dtype)
            externalInputs['ATP'][:,:,:,0] = ATPCurrentEmbryo

        try:
            embryoinstance.simulate(externalInputs=externalInputs,clampParameters=clampParameters,
                                  numSimIters=self.niters,fieldModulation=self.fieldModulation)
        except (RuntimeError, TypeError) as e:
            if "float64" in str(e) or "double" in str(e) or "float32" in str(e):
                print(f"\n{'='*60}")
                print(f"ERROR: dtype mismatch detected at embryo ({row}, {col})")
                print(f"Error message: {e}")
                print(f"\nThis typically means a tensor was created with the wrong dtype")
                print(f"inside the simulation. Current device: {self.device}, dtype: {self.dtype}")
                print(f"\nTo work around this, you can:")
                print(f"1. Run on CPU with float64: use_parallel=False, or")
                print(f"2. Force CPU mode by setting PYTORCH_ENABLE_MPS_FALLBACK=1")
                print(f"{'='*60}\n")
            raise

        result = {
            'row': row,
            'col': col,
            'Vmem': embryoinstance.timeseriesVmem[-1,0,:,0].cpu().numpy() if save else None,
            'ATP': embryoinstance.timeseriesATPConc[-1,0,:,0].cpu().numpy() if save else None
        }

        return result

    def ATPRate(self,ATPConc,t=0,externalATP=0):
        dATP = (2.0*((self.a*pow(ATPConc+self.xoff,3)) + (self.b*pow(ATPConc+self.xoff,2)) + (self.c*(ATPConc+self.xoff)) + self.d)
                + externalATP)
        # dATP = ((-self.k * ((ATPConc-self.a) * (ATPConc-self.b-self.unstableEquilOffset) * (ATPConc-self.c) + self.d))
        #         + externalATP)
        return dATP

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
            initialATP = params['unstableEquilibrium']
        else:
            initialATP = 9.6
        std = lambda dim: (((dim-minDim)/(maxDim-minDim))*(self.maxNoise-minNoise))+minNoise
        self.ATPCurrent = np.zeros((self.nsamples,self.niters,self.numEmbryos))
        self.ATPConcs = np.random.normal(initialATP,std(self.nrows),(self.nsamples,self.numEmbryos,1))
        self.ATPConcsInit = self.ATPConcs.copy()
        # self.timeseriesATP = np.array([-999]*self.niters*self.nsamples*self.numEmbryos,dtype=float).reshape(self.niters,self.nsamples,self.numEmbryos,1)
        timepoints = np.linspace(0,20,self.niters)
        for iter in range(self.niters):
            diffusionCurrent = self.w1 * np.matmul(Laplacian,self.ATPConcs)
            # dATP = ((self.a*pow(self.ATPConcs+self.xoff,3)) + (self.b*pow(self.ATPConcs+self.xoff,2)) +
            #         (self.c*(self.ATPConcs+self.xoff)) + self.d + diffusionCurrent)
            # self.ATPConcs = self.ATPConcs + (dATP * 0.01)
            updatedATP = odeint(self.ATPRate, self.ATPConcs.reshape(self.nsamples*self.numEmbryos,), timepoints[iter:(iter+2)],
                          rtol=1e-8, atol=1e-8, args=(diffusionCurrent.reshape(self.nsamples*self.numEmbryos,),))
            self.ATPConcs = updatedATP[-1].reshape(self.nsamples,self.numEmbryos,1)
            self.ATPCurrent[:,iter] = diffusionCurrent.squeeze(2)  # save only the diffusion current for the individual embryo simulations
            # self.timeseriesATP[iter] = self.ATPConcs
        self.ATPCurrent = self.ATPCurrent.reshape((self.nsamples,self.niters,self.nrows,self.ncols))
        self.ATPCurrent = torch.tensor(self.ATPCurrent, dtype=self.dtype, device=self.device)
        self.ATPConcsInit = self.ATPConcsInit.reshape((self.nsamples,self.nrows,self.ncols))
        self.ATPConcsInit = torch.tensor(self.ATPConcsInit, dtype=self.dtype, device=self.device)

    def simulate(self, save=False):
        """
        Run simulation for all embryos in the grid.

        Args:
            save: If True, save the final Vmem and ATP states
        """
        # Simulate single multi-embryo ATP network
        print("Simulating ATP flow across embryo network...")
        self.simulateATPFlow(modelName=self.modelNameATP)

        # Initialize saved results if needed
        if save:
            numCells = self.grid[0][0].numCells
            self.savedSims = {
                'Vmem': np.zeros((self.nrows, self.ncols, numCells)),
                'ATP': np.zeros((self.nrows, self.ncols, numCells))
            }

        # Simulate multiple single-embryo patterning networks
        # Only use parallel processing if we have enough embryos to justify the overhead
        use_parallel = self.use_parallel and self.numEmbryos >= self.parallel_threshold

        if use_parallel:
            # Limit workers to avoid oversubscription
            max_workers = min(mp.cpu_count(), self.numEmbryos, 8)
            print(f"Simulating {self.numEmbryos} embryos in parallel using {max_workers} workers...")
            self._simulate_parallel(save=save, max_workers=max_workers)
        else:
            if self.use_parallel and self.numEmbryos < self.parallel_threshold:
                print(f"Simulating {self.numEmbryos} embryos sequentially (grid too small for parallel speedup, threshold={self.parallel_threshold})...")
            else:
                print(f"Simulating {self.numEmbryos} embryos sequentially...")
            self._simulate_sequential(save=save)

        print("Simulation complete!")

    def _simulate_sequential(self, save=False):
        """Sequential simulation of all embryos."""
        for row in range(self.nrows):
            for col in range(self.ncols):
                result = self.simulateEmbryo(row, col, save=save)
                if save and result['Vmem'] is not None:
                    self.savedSims['Vmem'][row, col] = result['Vmem']
                    self.savedSims['ATP'][row, col] = result['ATP']

    def _simulate_parallel(self, save=False, max_workers=None):
        """Parallel simulation of all embryos using ThreadPoolExecutor."""
        # Create list of all (row, col) positions
        positions = [(row, col) for row in range(self.nrows) for col in range(self.ncols)]

        # Use ThreadPoolExecutor for parallel execution
        # Threads work well with PyTorch GPU operations
        if max_workers is None:
            max_workers = min(mp.cpu_count(), self.numEmbryos, 8)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all simulation tasks
            future_to_pos = {
                executor.submit(self.simulateEmbryo, row, col, save): (row, col)
                for row, col in positions
            }

            # Collect results as they complete
            for future in as_completed(future_to_pos):
                row, col = future_to_pos[future]
                try:
                    result = future.result()
                    if save and result['Vmem'] is not None:
                        self.savedSims['Vmem'][row, col] = result['Vmem']
                        self.savedSims['ATP'][row, col] = result['ATP']
                except Exception as exc:
                    print(f'Embryo at ({row}, {col}) generated an exception: {exc}')



