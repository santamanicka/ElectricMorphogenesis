"""
Test module for field alignment with external electric fields.

This module exposes a single trained (Stigmergic) embryo model to external
electric fields of arbitrary resolutions and visualizes the alignment dynamics.

Usage:
    python test_field_alignment.py --atp_conc 11.5 --resolution 4 --alignment_strength 0.01
    python test_field_alignment.py --atp_conc 9.6 --resolution 1 --field_type radial
    python test_field_alignment.py --resolution 12 --field_type uniform --field_direction 45
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse

from embryo import model
import utilities
from fieldAlignment import (
    FieldCoarseGrainer,
    apply_field_alignment,
    apply_field_alignment_normalized,
    extract_field_2d,
    create_uniform_external_field,
    create_radial_external_field,
    create_gradient_external_field,
)


def compute_field_delta(local_field, external_field, alignment_strength, dt, preserve_magnitude):
    """
    Pre-compute the field delta for alignment.

    This computes the change (delta) that should be applied to the local field
    to align it toward the external field.

    Args:
        local_field: Current field (2, H, W) tensor
        external_field: Target external field (2, H, W) tensor
        alignment_strength: Coupling strength for alignment
        dt: Time step for alignment dynamics
        preserve_magnitude: If True, preserve local field magnitude during alignment

    Returns:
        field_delta: (2, H, W) tensor of field changes to apply
    """
    # Apply alignment dynamics to compute target field
    if preserve_magnitude:
        aligned_field = apply_field_alignment_normalized(
            local_field, external_field, alignment_strength, dt
        )
    else:
        aligned_field = apply_field_alignment(
            local_field, external_field, alignment_strength, dt
        )

    # Compute delta: difference between aligned and current field
    field_delta = aligned_field - local_field

    return field_delta


def load_stigmergic_model(atp_conc=11.5, num_samples=1, num_iters=1000, enable_atp=False,
                          enable_atp_diffusion=True):
    """
    Load a trained Stigmergic embryo model with optional ATP.

    Args:
        atp_conc: ATP concentration level (default 11.5 for healthy)
        num_samples: Number of samples to simulate
        num_iters: Number of simulation iterations (for ATP input setup)
        enable_atp: If True, enable ATP dynamics; if False, disable ATP (default: False)
        enable_atp_diffusion: If True, enable ATP diffusion; if False, only local reactions (default: True)

    Returns:
        embryo_model: Initialized embryo model instance
        parameters: Model parameters dictionary
    """
    # Load Stigmergic model parameters
    parameterfilename = './data/StigmergicModelParameters.dat'

    try:
        parameters = torch.load(parameterfilename, weights_only=False)
    except FileNotFoundError:
        print(f"Error: Could not find {parameterfilename}")
        raise

    # Ensure ligand and GRN are disabled (standard Stigmergic configuration)
    if 'ligandParameters' in parameters and parameters['ligandParameters'] is not None:
        parameters['ligandParameters']['ligandEnabled'] = False
    if 'GRNParameters' in parameters and parameters['GRNParameters'] is not None:
        parameters['GRNParameters']['GRNEnabled'] = False

    # Setup ATP parameters only if requested
    if enable_atp:
        utils = utilities.utilities()
        parameters['ATPParameters'] = dict()
        parameters['ATPParameters']['ATPEnabled'] = True
        parameters['ATPParameters']['ATPReactionStrength'] = 1.0

        # Control ATP diffusion: set to 0 to disable diffusion between cells
        if enable_atp_diffusion:
            parameters['ATPParameters']['ATPDiffusionStrength'] = 10.0
            parameters['ATPParameters']['tissueConnectivity'] = \
                utils.computeLatticeAdjacencyMatrix(latticeDims=parameters['latticeDims'], periodicBoundary=False)
        else:
            # Disable ATP diffusion: only local reactions within each cell
            parameters['ATPParameters']['ATPDiffusionStrength'] = 0.0
            # Use zero matrix (no connections) instead of None
            num_cells = parameters['latticeDims'][0] * parameters['latticeDims'][1]
            parameters['ATPParameters']['tissueConnectivity'] = torch.zeros((num_cells, num_cells))

        parameters['ATPParameters']['ATPModelNum'] = '262'

        # Create model
        embryo_model = model(parameters, numBasicSamples=num_samples, numNoisySamples=1)
        num_cells = embryo_model.numCells

        # Set initial ATP concentration
        initial_values = parameters['simParameters']['initialValues']
        initial_values['ATPConc'] = torch.ones((num_samples, num_cells, 1), dtype=torch.float64) * atp_conc
        embryo_model.setExperimentalConditions((initial_values, 1))

        # Initialize ATP timepoints (required for ATP ODE integration)
        embryo_model.electricNetwork.timepointsATP = np.linspace(0, 20, num_iters)

        # Setup ATP external inputs (required when ATP is enabled)
        # Load ATP input data from model 262
        try:
            atp_data = torch.load('./data/Current_dims4,6,10,15_262.dat')
            # Use the 4th current profile (index 3) as in simulateTrainedModel.py
            atp_inputs = torch.DoubleTensor(atp_data[3]).unsqueeze(1).repeat(1, num_cells)
            # Truncate or extend to match num_iters
            if atp_inputs.shape[0] >= num_iters:
                atp_inputs = atp_inputs[:num_iters]
            else:
                # Repeat the last value to extend
                last_val = atp_inputs[-1:].repeat(num_iters - atp_inputs.shape[0], 1)
                atp_inputs = torch.cat([atp_inputs, last_val], dim=0)
            # Reshape to (numSamples, numSimIters, numCells, 1)
            external_inputs = parameters['simParameters'].get('externalInputs', dict())
            if external_inputs is None:
                external_inputs = dict()
            external_inputs['ATP'] = torch.zeros((num_samples, num_iters, num_cells, 1), dtype=torch.float64)
            external_inputs['ATP'][0, :, :, 0] = atp_inputs
            parameters['simParameters']['externalInputs'] = external_inputs
            print(f"Loaded ATP inputs from model 262 (shape: {external_inputs['ATP'].shape})")
        except FileNotFoundError:
            print("Warning: Could not load ATP input file, using constant ATP")
            external_inputs = parameters['simParameters'].get('externalInputs', dict())
            if external_inputs is None:
                external_inputs = dict()
            external_inputs['ATP'] = torch.ones((num_samples, num_iters, num_cells, 1), dtype=torch.float64) * atp_conc
            parameters['simParameters']['externalInputs'] = external_inputs
    else:
        # Disable ATP (default behavior matching simulateTrainedModel.py)
        parameters['ATPParameters'] = None

        # Create model
        embryo_model = model(parameters, numBasicSamples=num_samples, numNoisySamples=1)

        # Set experimental conditions
        initial_values = parameters['simParameters']['initialValues']
        embryo_model.setExperimentalConditions((initial_values, num_samples))

    return embryo_model, parameters


def run_simulation_with_field_alignment(
    embryo_model,
    parameters,
    external_field=None,
    coarse_resolution=None,
    alignment_strength=0.01,
    num_iters=1000,
    preserve_magnitude=True,
    apply_alignment=False,
    alignment_interval=10,
    same_initial_conditions=False,
    bidirectional_alignment=False,
    perturb_vmem=None,
    perturb_field=None,
    perturb_seed=None,
    coarsen_mode='average',
    upscale_mode='nearest',
    alignment_mode='pre',
):
    """
    Run embryo simulation and optionally apply/analyze field alignment to external field.

    This can either:
    1. Run full simulation then analyze alignment (apply_alignment=False)
    2. Run iterative simulation with active alignment application (apply_alignment=True)

    Args:
        embryo_model: Initialized embryo model
        parameters: Model parameters
        external_field: External field (2, H, W) at full resolution (None to skip alignment)
        coarse_resolution: tuple (h, w) for coarse-graining resolution (None to skip alignment)
        alignment_strength: Strength of alignment forcing/coupling
        num_iters: Number of simulation iterations
        preserve_magnitude: If True, preserve local field magnitude during alignment
        apply_alignment: If True, actively apply field alignment during simulation
        alignment_interval: Apply alignment every N iterations (only when apply_alignment=True)
        same_initial_conditions: If True, use same initial conditions for reference embryo (sanity check)
        bidirectional_alignment: If True, align both embryos to each other (mutual influence)
        perturb_vmem: Standard deviation of Gaussian noise to add to Vmem after clamping ends (None to skip)
        perturb_field: Standard deviation of Gaussian noise to add to electric field after clamping ends (None to skip)
        perturb_seed: Random seed for perturbation (for reproducibility)

    Returns:
        history: Dictionary with simulation history
    """
    circuit = embryo_model.electricNetwork
    field_shape = circuit.extracellularIndexGrid.shape

    # Store perturbation parameters on embryo_model for access in simulation loop
    embryo_model._perturb_vmem = perturb_vmem
    embryo_model._perturb_field = perturb_field
    embryo_model._perturb_seed = perturb_seed

    # Check if external field is provided
    has_external_field = (external_field is not None) and (coarse_resolution is not None)

    # History tracking
    history = {
        'vmem': [],
        'field_x': [],
        'field_y': [],
        'alignment_angle': [],  # Tissue-level (coarse-grained) angle
        'alignment_angle_raw': [],  # Cellular-level (raw) angle
        'magnitude_difference': [],
        'vmem_mse': [],  # MSE between main and reference Vmem patterns
        'iterations': [],
        'field_shape': field_shape,
        'perturbation_vmem_main': None,  # Main embryo Vmem at perturbation time
        'perturbation_vmem_ref': None,   # Ref embryo Vmem at perturbation time
        'perturbation_field_main': None, # Main embryo field at perturbation time
        'perturbation_field_ref': None,  # Ref embryo field at perturbation time
        'perturbation_iter': None,       # Iteration when perturbation was applied
    }

    if has_external_field:
        # Initialize coarse-grainer
        coarsener = FieldCoarseGrainer(field_shape)

        # For static fields, coarse-grain upfront
        # For embryo fields, this happens dynamically in the loop
        is_embryo_field = hasattr(external_field, 'electricNetwork')
        if not is_embryo_field:
            # Coarse-grain and upscale static external field
            external_coarse = coarsener.coarsen(external_field, coarse_resolution, mode=coarsen_mode)
            external_upscaled = coarsener.upscale(external_coarse, field_shape, mode=upscale_mode)

            history['external_field'] = external_field
            history['external_coarse'] = external_coarse
            history['external_upscaled'] = external_upscaled
        else:
            # For embryo fields, store None (will be computed dynamically)
            history['external_field'] = None
            history['external_coarse'] = None
            external_upscaled = None  # Will be handled in alignment loop

        history['coarse_resolution'] = coarse_resolution

        if apply_alignment:
            print(f"Running simulation WITH active field alignment forcing:")
            print(f"  - Alignment interval: every {alignment_interval} iterations")
            if bidirectional_alignment:
                print(f"  - Bidirectional: both embryos align to each other")
            else:
                print(f"  - Unidirectional: main embryo aligns to reference only")
        else:
            print(f"Running simulation with field alignment analysis (passive):")
        print(f"  - Resolution: {coarse_resolution}")
        print(f"  - Alignment strength: {alignment_strength}")
    else:
        print(f"Running simulation without field alignment:")

    # Get simulation parameters
    clamp_parameters = parameters['clampParameters']
    external_inputs = parameters['simParameters'].get('externalInputs', dict())
    if external_inputs is None:
        external_inputs = dict()

    print(f"  - Iterations: {num_iters}")

    # Run simulation: either iterative with alignment or all at once
    if has_external_field and apply_alignment:
        # Iterative simulation with active alignment forcing via alignmentParameters
        print("  Running iterative simulation with field alignment...")
        print("  Using alignmentParameters to apply field forcing within simulation loop")

        # Check if we're aligning to another embryo
        is_embryo_alignment = hasattr(external_field, 'electricNetwork')

        # For embryo-to-embryo alignment, we need to handle coarse-graining dynamically
        if is_embryo_alignment:
            # Create a coarse-grainer for the reference embryo's field
            coarsener = FieldCoarseGrainer(field_shape)

            # Create wrapper functions that extract and coarse-grain fields for alignment
            # (embryos are stepped separately in the loop before alignment is applied externally)
            def get_coarse_grained_ref_field():
                """Extract and coarse-grain reference embryo's current field (no stepping)."""
                try:
                    # Extract its current field (already stepped in loop before this is called)
                    ref_field = extract_field_2d(external_field.electricNetwork, sample_idx=0)
                    # Coarse-grain and upscale
                    ref_coarse = coarsener.coarsen(ref_field, coarse_resolution, mode=coarsen_mode)
                    ref_upscaled = coarsener.upscale(ref_coarse, field_shape, mode=upscale_mode)
                    return ref_upscaled
                except Exception as e:
                    print(f"ERROR in get_coarse_grained_ref_field: {e}")
                    import traceback
                    traceback.print_exc()
                    return None

            # For bidirectional alignment: create reverse callable
            if bidirectional_alignment:
                def get_coarse_grained_main_field():
                    """Extract and coarse-grain main embryo's current field for reverse alignment."""
                    try:
                        main_field = extract_field_2d(embryo_model.electricNetwork, sample_idx=0)
                        main_coarse = coarsener.coarsen(main_field, coarse_resolution, mode=coarsen_mode)
                        main_upscaled = coarsener.upscale(main_coarse, field_shape, mode=upscale_mode)
                        return main_upscaled
                    except Exception as e:
                        print(f"ERROR in get_coarse_grained_main_field: {e}")
                        import traceback
                        traceback.print_exc()
                        return None

        for iter_idx in range(num_iters):
            # Integrated alignment: alignment happens INSIDE simulate() via alignmentParameters
            # Embryos step with alignment applied at alignment_interval frequency

            # Apply Vmem/field perturbation right after clamping ends (once only)
            # Get clamp_end_iter from parameters
            clamp_end_iter = clamp_parameters.get('clampEndIter', 100)
            if iter_idx == (clamp_end_iter+1):
                # Check if Vmem perturbation is requested
                if hasattr(embryo_model, '_perturb_vmem') and embryo_model._perturb_vmem is not None:
                    perturb_std = embryo_model._perturb_vmem
                    perturb_seed = getattr(embryo_model, '_perturb_seed', None)

                    # Set random seed if provided
                    if perturb_seed is not None:
                        torch.manual_seed(perturb_seed)
                        np.random.seed(perturb_seed)

                    # Apply Gaussian noise to Vmem
                    noise = torch.randn_like(circuit.Vmem) * perturb_std
                    circuit.Vmem += noise

                    print(f"  [Iter {iter_idx}] Applied Vmem perturbation: std={perturb_std:.4f}, seed={perturb_seed}")
                    print(f"                   Noise range: [{noise.min():.4f}, {noise.max():.4f}]")

                    # Store Vmem snapshots at perturbation time
                    history['perturbation_iter'] = iter_idx
                    history['perturbation_vmem_main'] = circuit.Vmem[0, :, 0].detach().cpu().numpy().copy()
                    if is_embryo_alignment:
                        history['perturbation_vmem_ref'] = external_field.electricNetwork.Vmem[0, :, 0].detach().cpu().numpy().copy()

                # Check if field perturbation is requested
                # Perturb field vectors directly, then propagate through: field → eV → ion channels → currents → Vmem
                if hasattr(embryo_model, '_perturb_field') and embryo_model._perturb_field is not None:
                    perturb_std = embryo_model._perturb_field
                    perturb_seed = getattr(embryo_model, '_perturb_seed', None)

                    # Set random seed if provided
                    if perturb_seed is not None:
                        torch.manual_seed(perturb_seed)
                        np.random.seed(perturb_seed)

                    # Apply Gaussian noise to electric field (both x and y components)
                    # This perturbs both magnitude and direction
                    noise_x = torch.randn_like(circuit.eVforceVector[0]) * perturb_std
                    noise_y = torch.randn_like(circuit.eVforceVector[1]) * perturb_std
                    circuit.eVforceVector[0] += noise_x
                    circuit.eVforceVector[1] += noise_y

                    print(f"  [Iter {iter_idx}] Applied field perturbation: std={perturb_std:.4f}, seed={perturb_seed}")
                    print(f"                   Noise range X: [{noise_x.min():.4f}, {noise_x.max():.4f}]")
                    print(f"                   Noise range Y: [{noise_y.min():.4f}, {noise_y.max():.4f}]")

                    # Update eV magnitude from perturbed field vectors
                    # eV = sqrt(eVx^2 + eVy^2), consistent with how it's computed in updateExtracellularVoltage
                    eVforce = (circuit.eVforceVector[0]**2 + circuit.eVforceVector[1]**2)
                    circuit.eV = torch.pow(eVforce + circuit.epsilon, 0.5)

                    # Now propagate through bioelectric dynamics: field → ion channels → currents → Vmem
                    circuit.updateIonChannelConductance(inputSource='field', stochasticIonChannels=False,
                                                       fieldModulation=True,
                                                       fieldAggregation=circuit.fieldAggregation,
                                                       perturbation=None)
                    circuit.updateCurrent()
                    circuit.updateVmem()

                    # Store field snapshots at perturbation time (after perturbation and propagation)
                    history['perturbation_iter'] = iter_idx
                    # Extract field as (2, H, W) tensor
                    main_field_snapshot = extract_field_2d(circuit, sample_idx=0)
                    history['perturbation_field_main'] = main_field_snapshot.detach().cpu().numpy().copy()
                    if is_embryo_alignment:
                        ref_field_snapshot = extract_field_2d(external_field.electricNetwork, sample_idx=0)
                        history['perturbation_field_ref'] = ref_field_snapshot.detach().cpu().numpy().copy()

            # Determine if alignment should be applied on this iteration
            should_align = (iter_idx + 1) % alignment_interval == 0

            # Construct alignment parameters for this iteration (if needed)
            # Format depends on alignment_mode:
            #   Pre mode:  ('pre', field_delta_2d, sample_idx)
            #   Post mode: ('post', reference_field_callable, sample_idx, alignment_strength, dt, preserve_magnitude)
            main_alignment_params = None
            ref_alignment_params = None

            if should_align:
                if alignment_mode == 'pre':
                    # ===== Pre mode: compute delta externally =====
                    if bidirectional_alignment and is_embryo_alignment:
                        # For bidirectional alignment, extract BOTH fields before any stepping
                        # to ensure symmetric coupling (both align to each other's pre-alignment state)
                        ref_current_field = extract_field_2d(external_field.electricNetwork, sample_idx=0)
                        main_current_field = extract_field_2d(embryo_model.electricNetwork, sample_idx=0)

                        # Debug: check field difference before alignment
                        if (iter_idx + 1) % 100 == 0:
                            field_diff_pre = (ref_current_field - main_current_field).abs().max().item()
                            print(f"  [Iter {iter_idx+1}] Current embryos: ref vs main = {field_diff_pre:.6e}")

                        # Pre-compute deltas: main aligns to ref, ref aligns to main
                        main_delta = compute_field_delta(
                            main_current_field, ref_current_field, alignment_strength, 1.0, preserve_magnitude
                        )
                        ref_delta = compute_field_delta(
                            ref_current_field, main_current_field, alignment_strength, 1.0, preserve_magnitude
                        )

                        # Package deltas as alignment parameters: (mode, delta, sample_idx)
                        main_alignment_params = ('pre', main_delta, 0)
                        ref_alignment_params = ('pre', ref_delta, 0)

                    elif is_embryo_alignment:
                        # Unidirectional: main aligns to reference
                        main_current_field = extract_field_2d(embryo_model.electricNetwork, sample_idx=0)
                        ref_target_field = get_coarse_grained_ref_field()

                        # Pre-compute delta
                        main_delta = compute_field_delta(
                            main_current_field, ref_target_field, alignment_strength, 1.0, preserve_magnitude
                        )
                        main_alignment_params = ('pre', main_delta, 0)

                    else:
                        # Static field alignment: main aligns to external static field
                        main_current_field = extract_field_2d(embryo_model.electricNetwork, sample_idx=0)

                        # Pre-compute delta against static field
                        main_delta = compute_field_delta(
                            main_current_field, external_upscaled, alignment_strength, 1.0, preserve_magnitude
                        )
                        main_alignment_params = ('pre', main_delta, 0)

                elif alignment_mode == 'post':
                    # ===== Post mode: pass callable reference field =====
                    if bidirectional_alignment and is_embryo_alignment:
                        # For bidirectional alignment, create callables that return current fields
                        # Main embryo aligns to reference field
                        main_alignment_params = ('post', get_coarse_grained_ref_field, 0, alignment_strength, 1.0, preserve_magnitude)
                        # Reference embryo aligns to main field
                        ref_alignment_params = ('post', get_coarse_grained_main_field, 0, alignment_strength, 1.0, preserve_magnitude)

                        # Debug: check field difference before alignment
                        if (iter_idx + 1) % 100 == 0:
                            ref_current_field = extract_field_2d(external_field.electricNetwork, sample_idx=0)
                            main_current_field = extract_field_2d(embryo_model.electricNetwork, sample_idx=0)
                            field_diff_pre = (ref_current_field - main_current_field).abs().max().item()
                            print(f"  [Iter {iter_idx+1}] Current embryos: ref vs main = {field_diff_pre:.6e}")

                    elif is_embryo_alignment:
                        # Unidirectional: main aligns to reference (callable returns coarse-grained ref field)
                        main_alignment_params = ('post', get_coarse_grained_ref_field, 0, alignment_strength, 1.0, preserve_magnitude)

                    else:
                        # Static field alignment: create callable that returns static field
                        def get_static_field():
                            return external_upscaled
                        main_alignment_params = ('post', get_static_field, 0, alignment_strength, 1.0, preserve_magnitude)

            # 1. Step reference embryo (if embryo-to-embryo alignment) with its alignment parameters
            if is_embryo_alignment:
                external_field.simulate(
                    externalInputs=external_inputs,
                    clampParameters=clamp_parameters,
                    numSimIters=1,
                    fieldModulation=False,
                    outerIter=iter_idx,
                    alignmentParameters=ref_alignment_params  # Alignment applied inside simulate()
                )

            # 2. Step main embryo with its alignment parameters
            embryo_model.simulate(
                externalInputs=external_inputs,
                clampParameters=clamp_parameters,
                numSimIters=1,
                fieldModulation=False,
                outerIter=iter_idx,
                alignmentParameters=main_alignment_params  # Alignment applied inside simulate()
            )

            # Debug output (alignment now happens inside simulate())
            if should_align and (iter_idx + 1) % 100 == 0:
                if is_embryo_alignment:
                    main_after = extract_field_2d(embryo_model.electricNetwork, sample_idx=0)
                    ref_after = extract_field_2d(external_field.electricNetwork, sample_idx=0)
                    diff_after = (main_after - ref_after).abs().max().item()
                    print(f"  [Iter {iter_idx+1}] Field diff AFTER integrated alignment (RAW): {diff_after:.6e}")

            # Record history at alignment points
            if (iter_idx + 1) % alignment_interval == 0:
                local_field_raw = extract_field_2d(circuit, sample_idx=0)
                history['vmem'].append(circuit.Vmem[0, :, 0].detach().cpu().numpy().copy())
                history['field_x'].append(local_field_raw[0].detach().cpu().numpy().copy())
                history['field_y'].append(local_field_raw[1].detach().cpu().numpy().copy())
                history['iterations'].append(iter_idx + 1)

                # Debug: check field difference AFTER alignment
                if (iter_idx + 1) % 100 == 0 and is_embryo_alignment:
                    ref_field_after = extract_field_2d(external_field.electricNetwork, sample_idx=0)
                    field_diff_post_raw = (local_field_raw - ref_field_after).abs().max().item()
                    print(f"  [Iter {iter_idx+1}] Field diff AFTER alignment (RAW): {field_diff_post_raw:.6e}")

                # Compute alignment angle (for embryo fields, get current reference field)
                if is_embryo_alignment:
                    ref_field_raw = extract_field_2d(external_field.electricNetwork, sample_idx=0)

                    # Extract and coarse-grain BOTH embryos' fields for fair comparison
                    local_coarse = coarsener.coarsen(local_field_raw, coarse_resolution, mode=coarsen_mode)
                    local_field = coarsener.upscale(local_coarse, field_shape, mode=upscale_mode)
                    ref_coarse = coarsener.coarsen(ref_field_raw, coarse_resolution, mode=coarsen_mode)
                    ext_field_for_angle = coarsener.upscale(ref_coarse, field_shape, mode=upscale_mode)
                else:
                    local_field = local_field_raw
                    ext_field_for_angle = external_upscaled

                # Compute field magnitudes
                local_mag = torch.sqrt(local_field[0]**2 + local_field[1]**2) # + 1e-10)
                ext_mag = torch.sqrt(ext_field_for_angle[0]**2 + ext_field_for_angle[1]**2) # + 1e-10)
                dot = (local_field[0] * ext_field_for_angle[0] + local_field[1] * ext_field_for_angle[1])

                # Only compute metrics for vectors with sufficient magnitude (avoid numerical issues)
                mag_threshold = 1e-6
                valid_mask = (local_mag > mag_threshold) & (ext_mag > mag_threshold)

                if valid_mask.sum() > 0:
                    # Metric 1: Average angle between normalized vectors
                    cos_angle_valid = dot[valid_mask] / torch.clamp(local_mag[valid_mask] * ext_mag[valid_mask], min=1e-10)

                    # Debug: compare two methods of averaging angles
                    if (iter_idx + 1) % 100 == 0 and is_embryo_alignment:
                        # Method 1 (current): acos(mean(cos))
                        avg_cos = cos_angle_valid.mean().item()
                        method1_angle = torch.acos(torch.clamp(torch.tensor(avg_cos), -1, 1)).item()

                        # Method 2 (correct): mean(acos(cos))
                        angles_all = torch.acos(torch.clamp(cos_angle_valid, -1, 1))
                        method2_angle = angles_all.mean().item()

                        # Show difference between raw and coarse-grained field comparison
                        ref_field_after = extract_field_2d(external_field.electricNetwork, sample_idx=0)
                        field_diff_coarse = (local_field - ext_field_for_angle).abs().max().item()
                        field_diff_raw = (local_field_raw - ref_field_after).abs().max().item()

                        print(f"  [Iter {iter_idx+1}] Multi-scale alignment analysis:")
                        print(f"    Field diff (RAW cellular fields 12x12): {field_diff_raw:.6e}")
                        print(f"    Field diff (COARSE tissue fields {coarse_resolution[0]}x{coarse_resolution[1]}): {field_diff_coarse:.6e}")
                        print(f"    Tissue-level angle (coarse-grained): Will be computed below")
                        print(f"    Cellular-level angle (raw fields): Will be computed below")
                        print(f"    Mean cosine (tissue): {avg_cos:.10f}")
                        print(f"    Valid mask count: {valid_mask.sum().item()}/{valid_mask.numel()}")
                        print(f"    Cosine range: [{cos_angle_valid.min().item():.10f}, {cos_angle_valid.max().item():.10f}]")

                    # Use the mathematically correct method: mean(acos(cos))
                    angles_all = torch.acos(torch.clamp(cos_angle_valid, -1, 1))
                    avg_angle = angles_all.mean().item()

                    # Metric 2: Average magnitude difference between raw vectors
                    mag_diff_valid = (local_mag[valid_mask] - ext_mag[valid_mask]).abs()
                    avg_mag_diff = mag_diff_valid.mean().item()
                else:
                    avg_angle = 0.0  # No valid vectors, assume aligned
                    avg_mag_diff = 0.0

                # Compute cellular-level (raw) angle for embryo alignment
                if is_embryo_alignment:
                    # Compute angle on RAW fields (12x12) without coarse-graining
                    local_mag_raw = torch.sqrt(local_field_raw[0]**2 + local_field_raw[1]**2)
                    ref_mag_raw = torch.sqrt(ref_field_raw[0]**2 + ref_field_raw[1]**2)
                    dot_raw = (local_field_raw[0] * ref_field_raw[0] + local_field_raw[1] * ref_field_raw[1])

                    valid_mask_raw = (local_mag_raw > mag_threshold) & (ref_mag_raw > mag_threshold)

                    if valid_mask_raw.sum() > 0:
                        cos_angle_raw = dot_raw[valid_mask_raw] / torch.clamp(local_mag_raw[valid_mask_raw] * ref_mag_raw[valid_mask_raw], min=1e-10)
                        angles_raw = torch.acos(torch.clamp(cos_angle_raw, -1, 1))
                        avg_angle_raw = angles_raw.mean().item()
                    else:
                        avg_angle_raw = 0.0

                    history['alignment_angle_raw'].append(np.degrees(avg_angle_raw))

                    # Print both angles at debug intervals
                    if (iter_idx + 1) % 100 == 0:
                        print(f"    Tissue-level angle (coarse {coarse_resolution[0]}x{coarse_resolution[1]}): {np.degrees(avg_angle):.6f}°")
                        print(f"    Cellular-level angle (raw 12x12): {np.degrees(avg_angle_raw):.6f}°")
                else:
                    # For non-embryo alignment, raw == coarse (no difference)
                    history['alignment_angle_raw'].append(np.degrees(avg_angle))

                history['alignment_angle'].append(np.degrees(avg_angle))
                history['magnitude_difference'].append(avg_mag_diff)

                # Compute Vmem RMSE for embryo-to-embryo alignment
                if is_embryo_alignment:
                    main_vmem = circuit.Vmem[0, :, 0].detach().cpu().numpy()
                    ref_vmem = external_field.electricNetwork.Vmem[0, :, 0].detach().cpu().numpy()
                    vmem_mse = ((main_vmem - ref_vmem) ** 2).mean()
                    vmem_rmse = np.sqrt(vmem_mse) * 1000  # Convert to mV
                    history['vmem_mse'].append(vmem_rmse)
                else:
                    history['vmem_mse'].append(0.0)

            if (iter_idx + 1) % 100 == 0:
                print(f"    Iteration {iter_idx + 1}/{num_iters}")
        print("  Simulation complete!")
    else:
        # Run the full simulation at once (passive mode or no external field)
        print("  Running full simulation...")
        embryo_model.simulate(
            externalInputs=external_inputs,
            clampParameters=clamp_parameters,
            numSimIters=num_iters,
            fieldModulation=True,
        )
        print("  Simulation complete!")

    # Extract final field
    local_field = extract_field_2d(circuit, sample_idx=0)

    # Record final state (if not already recorded during iterative simulation)
    if not (has_external_field and apply_alignment):
        history['vmem'].append(circuit.Vmem[0, :, 0].detach().cpu().numpy().copy())
        history['field_x'].append(local_field[0].detach().cpu().numpy().copy())
        history['field_y'].append(local_field[1].detach().cpu().numpy().copy())
        history['iterations'].append(num_iters)

        # Compute alignment metrics for passive mode (only compute once at the end)
        if has_external_field:
            local_mag = torch.sqrt(local_field[0]**2 + local_field[1]**2 + 1e-10)
            ext_mag = torch.sqrt(external_upscaled[0]**2 + external_upscaled[1]**2 + 1e-10)
            dot = (local_field[0] * external_upscaled[0] + local_field[1] * external_upscaled[1])

            # Metric 1: Average angle between normalized vectors
            cos_angle = dot / (local_mag * ext_mag + 1e-10)
            avg_angle = torch.acos(torch.clamp(cos_angle.mean(), -1, 1)).item()
            history['alignment_angle'].append(np.degrees(avg_angle))

            # Metric 2: Average magnitude difference between raw vectors
            avg_mag_diff = (local_mag - ext_mag).abs().mean().item()
            history['magnitude_difference'].append(avg_mag_diff)

            print(f"  Mean alignment angle: {np.degrees(avg_angle):.1f} degrees")
            print(f"  Mean magnitude difference: {avg_mag_diff:.4f}")

    history['final_field'] = local_field

    # Store reference embryo data for visualization (if embryo-to-embryo alignment)
    if is_embryo_alignment:
        ref_field_final = extract_field_2d(external_field.electricNetwork, sample_idx=0)
        ref_vmem_final = external_field.electricNetwork.Vmem[0, :, 0].detach().cpu().numpy().copy()
        history['ref_final_field'] = ref_field_final
        history['ref_final_vmem'] = ref_vmem_final
    else:
        history['ref_final_field'] = None
        history['ref_final_vmem'] = None

    # Compute alignment metrics if external field provided (for active mode, just print final)
    if has_external_field and apply_alignment and len(history['alignment_angle']) > 0:
        final_angle = history['alignment_angle'][-1]
        final_mag_diff = history['magnitude_difference'][-1]
        print(f"  Final alignment angle: {final_angle:.1f} degrees")
        print(f"  Final magnitude difference: {final_mag_diff:.4f}")

        # Compute what the aligned field would be (for visualization)
        # For embryo fields, get the final reference field
        if is_embryo_alignment:
            ref_coarse_final = coarsener.coarsen(ref_field_final, coarse_resolution, mode=coarsen_mode)
            ref_upscaled_final = coarsener.upscale(ref_coarse_final, field_shape, mode=upscale_mode)
            ext_field_for_vis = ref_upscaled_final
        else:
            ext_field_for_vis = external_upscaled

        if preserve_magnitude:
            aligned_field = apply_field_alignment_normalized(
                local_field, ext_field_for_vis, alignment_strength, dt=1.0
            )
        else:
            aligned_field = apply_field_alignment(
                local_field, ext_field_for_vis, alignment_strength, dt=1.0
            )

        history['aligned_field'] = aligned_field

    return history


def visualize_results(history, save_path=None):
    """
    Visualize field alignment results.

    Args:
        history: Dictionary from run_simulation_with_field_alignment
        save_path: Optional path to save figure
    """
    field_shape = history['field_shape']
    has_ref_embryo = history.get('ref_final_field') is not None
    # Has alignment if we have external field OR reference embryo data
    has_alignment = (('external_field' in history and history['external_field'] is not None) or
                     has_ref_embryo)

    if has_alignment:
        # 3x4 layout with reference embryo data and alignment metrics
        plt.figure(figsize=(24, 18))
        coarse_res = history.get('coarse_resolution', field_shape)
        Y, X = np.mgrid[0:field_shape[0], 0:field_shape[1]]

        # Row 1: External/Reference field processing
        # 1. External field (full resolution) or Reference embryo field (ALWAYS full resolution 12x12)
        ax1 = plt.subplot(3, 4, 1)
        if has_ref_embryo:
            ref_field = history['ref_final_field']
            ref_mag = torch.sqrt(ref_field[0]**2 + ref_field[1]**2).numpy()
            im1 = ax1.imshow(ref_mag, cmap='viridis', origin='lower')
            plt.colorbar(im1, ax=ax1, label='Field magnitude')
            ax1.quiver(X, Y, ref_field[0].numpy(), ref_field[1].numpy(),
                       color='white', alpha=0.7, scale=ref_mag.max()*15 if ref_mag.max() > 0 else 1)
            ax1.set_title(f'Reference Embryo Field ({field_shape[0]}x{field_shape[1]})')
        else:
            ext_field = history['external_field']
            ext_mag = torch.sqrt(ext_field[0]**2 + ext_field[1]**2).numpy()
            im1 = ax1.imshow(ext_mag, cmap='viridis', origin='lower')
            plt.colorbar(im1, ax=ax1, label='Field magnitude')
            ax1.quiver(X, Y, ext_field[0].numpy(), ext_field[1].numpy(),
                       color='white', alpha=0.7, scale=ext_mag.max()*15 if ext_mag.max() > 0 else 1)
            ax1.set_title(f'External Field ({field_shape[0]}x{field_shape[1]})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # 2. External field (coarse-grained) or Reference embryo Vmem
        ax2 = plt.subplot(3, 4, 2)
        if has_ref_embryo:
            ref_vmem = history['ref_final_vmem']
            grid_size = int(np.sqrt(len(ref_vmem)))
            ref_vmem_2d = ref_vmem.reshape(grid_size, grid_size)
            im = ax2.imshow(ref_vmem_2d * 1000, cmap='RdBu_r', origin='lower')
            plt.colorbar(im, ax=ax2, label='mV')
            ax2.set_title('Reference Embryo Vmem (Final)')
        elif history.get('external_coarse') is not None:
            ext_coarse = history['external_coarse']
            coarse_mag = torch.sqrt(ext_coarse[0]**2 + ext_coarse[1]**2).numpy()
            ax2.imshow(coarse_mag, cmap='viridis', origin='lower')
            Yc, Xc = np.mgrid[0:coarse_res[0], 0:coarse_res[1]]
            ax2.quiver(Xc, Yc, ext_coarse[0].numpy(), ext_coarse[1].numpy(),
                       color='white', alpha=0.7, scale=coarse_mag.max()*10 if coarse_mag.max() > 0 else 1)
            ax2.set_title(f'External Field (Coarse: {coarse_res[0]}x{coarse_res[1]})')
        else:
            ax2.axis('off')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')

        # 3. External field (upscaled) or Field difference
        ax3 = plt.subplot(3, 4, 3)
        if has_ref_embryo:
            # Show field difference between main and reference
            main_field = history['final_field']
            ref_field = history['ref_final_field']
            field_diff = torch.sqrt((main_field[0] - ref_field[0])**2 + (main_field[1] - ref_field[1])**2).numpy()
            im = ax3.imshow(field_diff, cmap='hot', origin='lower')
            plt.colorbar(im, ax=ax3, label='Field difference')
            max_diff = field_diff.max()
            ax3.set_title(f'Field Difference (max: {max_diff:.2e})')
        elif history.get('external_upscaled') is not None:
            ext_up = history['external_upscaled']
            up_mag = torch.sqrt(ext_up[0]**2 + ext_up[1]**2).numpy()
            ax3.imshow(up_mag, cmap='viridis', origin='lower')
            ax3.quiver(X, Y, ext_up[0].numpy(), ext_up[1].numpy(),
                       color='white', alpha=0.7, scale=up_mag.max()*15 if up_mag.max() > 0 else 1)
            ax3.set_title(f'External Field (Upscaled: {field_shape[0]}x{field_shape[1]})')
        else:
            ax3.axis('off')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')

        # 4. Alignment angle time series (Column 4, Row 1)
        ax4_col4 = plt.subplot(3, 4, 4)
        if len(history.get('alignment_angle', [])) > 1:
            # Plot both tissue-level and cellular-level angles over time
            iters = history['iterations']
            tissue_angles = history['alignment_angle']
            ax4_col4.plot(iters, tissue_angles, 'b-', linewidth=2, label='Tissue-level', marker='o', markersize=3)
            if 'alignment_angle_raw' in history and len(history['alignment_angle_raw']) > 0:
                cellular_angles = history['alignment_angle_raw']
                ax4_col4.plot(iters, cellular_angles, 'r-', linewidth=2, label='Cellular-level', marker='s', markersize=3)
            ax4_col4.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
            ax4_col4.set_xlabel('Iteration')
            ax4_col4.set_ylabel('Alignment Angle (degrees)')
            ax4_col4.set_title('Multi-Scale Alignment Dynamics')
            ax4_col4.set_ylim(0, 180)
            ax4_col4.legend(fontsize=8)
            ax4_col4.grid(True, alpha=0.3)
        else:
            ax4_col4.axis('off')

        # Row 2: Main embryo results
        # 5. Final local field (main embryo - ALWAYS full resolution 12x12)
        ax5 = plt.subplot(3, 4, 5)
        final_field = history['final_field']
        final_mag = torch.sqrt(final_field[0]**2 + final_field[1]**2).numpy()
        im5 = ax5.imshow(final_mag, cmap='plasma', origin='lower')
        plt.colorbar(im5, ax=ax5, label='Field magnitude')
        ax5.quiver(X, Y, final_field[0].numpy(), final_field[1].numpy(),
                   color='white', alpha=0.7, scale=final_mag.max()*15 if final_mag.max() > 0 else 1)
        ax5.set_title(f'Main Embryo Field ({field_shape[0]}x{field_shape[1]})')
        ax5.set_xlabel('x')
        ax5.set_ylabel('y')

        # 6. Final Vmem (main embryo)
        ax6 = plt.subplot(3, 4, 6)
        if len(history['vmem']) > 0:
            final_vmem = history['vmem'][-1]
            grid_size = int(np.sqrt(len(final_vmem)))
            vmem_2d = final_vmem.reshape(grid_size, grid_size)
            im = ax6.imshow(vmem_2d * 1000, cmap='RdBu_r', origin='lower')
            plt.colorbar(im, ax=ax6, label='mV')
            ax6.set_title('Main Embryo Vmem (Final)')
            ax6.set_xlabel('x')
            ax6.set_ylabel('y')

        # 7. Vmem difference (if reference embryo available)
        ax7_r2 = plt.subplot(3, 4, 7)
        if has_ref_embryo and len(history['vmem']) > 0:
            main_vmem = history['vmem'][-1]
            ref_vmem = history['ref_final_vmem']
            vmem_diff = np.abs(main_vmem - ref_vmem) * 1000  # mV
            grid_size = int(np.sqrt(len(vmem_diff)))
            vmem_diff_2d = vmem_diff.reshape(grid_size, grid_size)
            im = ax7_r2.imshow(vmem_diff_2d, cmap='hot', origin='lower')
            plt.colorbar(im, ax=ax7_r2, label='mV')
            max_diff_mv = vmem_diff.max()
            ax7_r2.set_title(f'Vmem Difference (max: {max_diff_mv:.3f} mV)')
            ax7_r2.set_xlabel('x')
            ax7_r2.set_ylabel('y')
        else:
            ax7_r2.axis('off')
            ax7_r2.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20, color='gray')

        # 8. Vmem RMSE time series (Column 4, Row 2)
        ax8_r2 = plt.subplot(3, 4, 8)
        if has_ref_embryo and len(history.get('vmem_mse', [])) > 1:
            iters = history['iterations']
            vmem_rmse = history['vmem_mse']  # Actually RMSE now
            ax8_r2.plot(iters, vmem_rmse, 'g-', linewidth=2, marker='o', markersize=3)
            ax8_r2.set_xlabel('Iteration')
            ax8_r2.set_ylabel('Vmem RMSE (mV)')
            ax8_r2.set_title('Vmem Pattern Divergence')
            ax8_r2.grid(True, alpha=0.3)
            # Add final RMSE value as text
            final_rmse = vmem_rmse[-1]
            ax8_r2.text(0.98, 0.98, f'Final: {final_rmse:.2f} mV',
                       transform=ax8_r2.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax8_r2.axis('off')

        # Row 3: Coarse-grained field visualization
        # 7. Reference embryo coarse-grained field (Column 1, Row 3)
        ax7 = plt.subplot(3, 4, 9)
        if has_ref_embryo and 'coarse_resolution' in history:
            coarse_res = history['coarse_resolution']
            # Get coarse-grained versions of final fields
            from fieldAlignment import FieldCoarseGrainer
            coarsener = FieldCoarseGrainer(field_shape)

            ref_field = history['ref_final_field']
            ref_coarse = coarsener.coarsen(ref_field, coarse_res)
            ref_coarse_mag = torch.sqrt(ref_coarse[0]**2 + ref_coarse[1]**2).numpy()

            im = ax7.imshow(ref_coarse_mag, cmap='viridis', origin='lower', interpolation='nearest')
            plt.colorbar(im, ax=ax7, label='Field mag')
            # Add vector field arrows
            Yc, Xc = np.mgrid[0:coarse_res[0], 0:coarse_res[1]]
            ax7.quiver(Xc, Yc, ref_coarse[0].numpy(), ref_coarse[1].numpy(),
                       color='white', alpha=0.7, scale=ref_coarse_mag.max()*8 if ref_coarse_mag.max() > 0 else 1)
            ax7.set_title(f'Ref Coarse Field ({coarse_res[0]}x{coarse_res[1]})')
            ax7.set_xlabel('x (coarse)')
            ax7.set_ylabel('y (coarse)')
        else:
            ax7.axis('off')
            ax7.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, color='lightgray')

        # 8. Main embryo coarse-grained field (Column 2, Row 3)
        ax8 = plt.subplot(3, 4, 10)
        if has_ref_embryo and 'coarse_resolution' in history:
            coarse_res = history['coarse_resolution']
            from fieldAlignment import FieldCoarseGrainer
            coarsener = FieldCoarseGrainer(field_shape)

            main_field = history['final_field']
            main_coarse = coarsener.coarsen(main_field, coarse_res)
            main_coarse_mag = torch.sqrt(main_coarse[0]**2 + main_coarse[1]**2).numpy()

            im = ax8.imshow(main_coarse_mag, cmap='plasma', origin='lower', interpolation='nearest')
            plt.colorbar(im, ax=ax8, label='Field mag')
            # Add vector field arrows
            Yc, Xc = np.mgrid[0:coarse_res[0], 0:coarse_res[1]]
            ax8.quiver(Xc, Yc, main_coarse[0].numpy(), main_coarse[1].numpy(),
                       color='white', alpha=0.7, scale=main_coarse_mag.max()*8 if main_coarse_mag.max() > 0 else 1)
            ax8.set_title(f'Main Coarse Field ({coarse_res[0]}x{coarse_res[1]})')
            ax8.set_xlabel('x (coarse)')
            ax8.set_ylabel('y (coarse)')
        else:
            ax8.axis('off')
            ax8.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, color='lightgray')

        # 9. Coarse-grained field difference (Column 3, Row 3)
        ax9 = plt.subplot(3, 4, 11)
        if has_ref_embryo and 'coarse_resolution' in history:
            coarse_res = history['coarse_resolution']
            from fieldAlignment import FieldCoarseGrainer
            coarsener = FieldCoarseGrainer(field_shape)

            main_field = history['final_field']
            ref_field = history['ref_final_field']

            # Coarse-grain both fields
            main_coarse = coarsener.coarsen(main_field, coarse_res)
            ref_coarse = coarsener.coarsen(ref_field, coarse_res)

            # Compute magnitude difference at coarse level
            coarse_diff = torch.sqrt((main_coarse[0] - ref_coarse[0])**2 + (main_coarse[1] - ref_coarse[1])**2).numpy()

            im = ax9.imshow(coarse_diff, cmap='hot', origin='lower', interpolation='nearest')
            plt.colorbar(im, ax=ax9, label='Field diff')
            max_coarse_diff = coarse_diff.max()
            ax9.set_title(f'Coarse Field Diff ({coarse_res[0]}x{coarse_res[1]})\nmax: {max_coarse_diff:.2e}')
            ax9.set_xlabel('x (coarse)')
            ax9.set_ylabel('y (coarse)')

            # Add statistics text below the plot
            stats_text = "Multi-Scale Alignment:\n\n"
            if len(history['alignment_angle']) > 0:
                tissue_angle = history['alignment_angle'][-1]
                stats_text += f"Tissue: {tissue_angle:.2f}°  "
            if len(history.get('alignment_angle_raw', [])) > 0:
                cellular_angle = history['alignment_angle_raw'][-1]
                stats_text += f"Cell: {cellular_angle:.2f}°"
            # Add text annotation below the image
            ax9.text(0.5, -0.15, stats_text, transform=ax9.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='center', family='monospace')
        else:
            ax9.axis('off')
            ax9.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=16, color='lightgray')

        # 10. Perturbation snapshots (Column 4, Row 3)
        ax10 = plt.subplot(3, 4, 12)
        if history.get('perturbation_iter') is not None:
            perturb_iter = history['perturbation_iter']

            # Check if field perturbation data is available
            main_field_perturb = history.get('perturbation_field_main')
            ref_field_perturb = history.get('perturbation_field_ref')

            # Check if Vmem perturbation data is available
            main_vmem_perturb = history.get('perturbation_vmem_main')
            ref_vmem_perturb = history.get('perturbation_vmem_ref')

            # Prioritize field perturbation visualization if available
            if main_field_perturb is not None and ref_field_perturb is not None:
                # Show field magnitude difference at perturbation time
                # Fields are (2, H, W) numpy arrays
                main_mag = np.sqrt(main_field_perturb[0]**2 + main_field_perturb[1]**2)
                ref_mag = np.sqrt(ref_field_perturb[0]**2 + ref_field_perturb[1]**2)
                mag_diff = main_mag - ref_mag

                im = ax10.imshow(mag_diff, cmap='RdBu_r', origin='lower')
                plt.colorbar(im, ax=ax10, label='Field mag')
                max_diff = np.abs(mag_diff).max()
                ax10.set_title(f'Field Mag Diff at Perturb\n(Iter {perturb_iter}, max: {max_diff:.2e})')

            elif main_field_perturb is not None:
                # Only main field available
                main_mag = np.sqrt(main_field_perturb[0]**2 + main_field_perturb[1]**2)
                im = ax10.imshow(main_mag, cmap='plasma', origin='lower')
                plt.colorbar(im, ax=ax10, label='Field mag')
                ax10.set_title(f'Main Field at Perturb\n(Iter {perturb_iter})')

            elif main_vmem_perturb is not None and ref_vmem_perturb is not None:
                # Fall back to Vmem perturbation if field not available
                grid_size = int(np.sqrt(len(main_vmem_perturb)))
                main_2d = main_vmem_perturb.reshape(grid_size, grid_size) * 1000  # Convert to mV
                ref_2d = ref_vmem_perturb.reshape(grid_size, grid_size) * 1000
                diff_2d = main_2d - ref_2d

                im = ax10.imshow(diff_2d, cmap='RdBu_r', origin='lower')
                plt.colorbar(im, ax=ax10, label='mV')
                max_diff = np.abs(diff_2d).max()
                ax10.set_title(f'Vmem Diff at Perturb\n(Iter {perturb_iter}, max: {max_diff:.2f} mV)')

            elif main_vmem_perturb is not None:
                # Only main Vmem available
                grid_size = int(np.sqrt(len(main_vmem_perturb)))
                main_2d = main_vmem_perturb.reshape(grid_size, grid_size) * 1000
                im = ax10.imshow(main_2d, cmap='RdBu_r', origin='lower')
                plt.colorbar(im, ax=ax10, label='mV')
                ax10.set_title(f'Main Vmem at Perturb\n(Iter {perturb_iter})')

            ax10.set_xlabel('x')
            ax10.set_ylabel('y')
        else:
            ax10.axis('off')
            ax10.text(0.5, 0.5, 'No Perturbation', ha='center', va='center',
                     fontsize=12, color='gray', style='italic')

        plt.tight_layout()
    else:
        # Simpler 1x2 layout without alignment analysis
        plt.figure(figsize=(12, 5))
        Y, X = np.mgrid[0:field_shape[0], 0:field_shape[1]]

        # 1. Final local field
        ax1 = plt.subplot(1, 2, 1)
        final_field = history['final_field']
        final_mag = torch.sqrt(final_field[0]**2 + final_field[1]**2).numpy()
        ax1.imshow(final_mag, cmap='plasma', origin='lower')
        ax1.quiver(X, Y, final_field[0].numpy(), final_field[1].numpy(),
                   color='white', alpha=0.7, scale=final_mag.max()*15 if final_mag.max() > 0 else 1)
        ax1.set_title('Final Electric Field')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # 2. Final Vmem
        ax2 = plt.subplot(1, 2, 2)
        if len(history['vmem']) > 0:
            final_vmem = history['vmem'][-1]
            grid_size = int(np.sqrt(len(final_vmem)))
            vmem_2d = final_vmem.reshape(grid_size, grid_size)
            im = ax2.imshow(vmem_2d * 1000, cmap='RdBu_r', origin='lower')
            plt.colorbar(im, ax=ax2, label='mV')
            ax2.set_title('Final Vmem Pattern')
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def test_coarse_graining(field_shape=(12, 12)):
    """
    Test coarse-graining at various resolutions.

    Args:
        field_shape: Shape of field grid
    """
    print("=" * 60)
    print("Testing Coarse-Graining at Various Resolutions")
    print("=" * 60)

    coarsener = FieldCoarseGrainer(field_shape)
    valid_resolutions = coarsener.get_valid_resolutions()

    print(f"\nField shape: {field_shape}")
    print(f"Valid resolutions: {valid_resolutions}")

    # Create test field (radial)
    test_field = create_radial_external_field(field_shape, magnitude=1.0)
    print(f"\nTest field shape: {test_field.shape}")

    # Test each valid resolution
    _, axes = plt.subplots(2, len(valid_resolutions), figsize=(3*len(valid_resolutions), 6))

    for idx, res in enumerate(valid_resolutions):
        # Coarse-grain
        coarse = coarsener.coarsen(test_field, res)

        # Upscale
        upscaled = coarsener.upscale(coarse, field_shape)

        # Plot coarse
        ax1 = axes[0, idx] if len(valid_resolutions) > 1 else axes[0]
        coarse_mag = torch.sqrt(coarse[0]**2 + coarse[1]**2).numpy()
        ax1.imshow(coarse_mag, cmap='viridis', origin='lower')
        ax1.set_title(f'Coarse {res[0]}x{res[1]}')

        # Plot upscaled
        ax2 = axes[1, idx] if len(valid_resolutions) > 1 else axes[1]
        up_mag = torch.sqrt(upscaled[0]**2 + upscaled[1]**2).numpy()
        ax2.imshow(up_mag, cmap='viridis', origin='lower')
        ax2.set_title(f'Upscaled {field_shape[0]}x{field_shape[1]}')

        print(f"  Resolution {res}: coarse shape {coarse.shape}, upscaled shape {upscaled.shape}")

    plt.tight_layout()
    plt.savefig('coarse_graining_test.png', dpi=150)
    print("\nSaved coarse_graining_test.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Test field alignment with external electric fields'
    )
    parser.add_argument('--enable-atp', action='store_true', dest='enable_atp',
                        help='Enable ATP dynamics using model 262 (disabled by default)')
    parser.add_argument('--atp_conc', type=float, default=None,
                        help='ATP concentration (absolute value). Default: 11.5 for healthy if neither --atp_conc nor --atp-delta specified')
    parser.add_argument('--atp-delta', type=float, default=None, dest='atp_delta',
                        help='ATP concentration as delta from unstable equilibrium (2.5). Example: +9 gives 11.5 (healthy), -2 gives 0.5 (unhealthy)')
    parser.add_argument('--ref-atp-conc', type=float, default=None, dest='ref_atp_conc',
                        help='Reference embryo ATP concentration (absolute value). If not specified, uses same as main embryo')
    parser.add_argument('--ref-atp-delta', type=float, default=None, dest='ref_atp_delta',
                        help='Reference embryo ATP delta from equilibrium. If not specified, uses same as main embryo')
    parser.add_argument('--disable-atp-diffusion', action='store_true', dest='disable_atp_diffusion',
                        help='Disable ATP diffusion between cells (keep only local reactions)')
    parser.add_argument('--resolution', type=int, default=4,
                        help='Coarse-graining resolution (must divide 12: 1,2,3,4,6,12)')
    parser.add_argument('--alignment_strength', type=float, default=0.01,
                        help='Alignment coupling strength (default: 0.01)')
    parser.add_argument('--num_iters', type=int, default=1000,
                        help='Number of simulation iterations')
    parser.add_argument('--field_type', type=str, default='uniform',
                        choices=['uniform', 'radial', 'gradient', 'embryo'],
                        help='Type of external field (use "embryo" for another embryo\'s time-varying field)')
    parser.add_argument('--field_direction', type=float, default=45.0,
                        help='Field direction in degrees (for uniform/gradient)')
    parser.add_argument('--field_magnitude', type=float, default=1.0,
                        help='Field magnitude')
    parser.add_argument('--test_coarsening', action='store_true',
                        help='Run coarse-graining test only')
    parser.add_argument('--apply-alignment', action='store_true', dest='apply_alignment',
                        help='Actively apply field alignment during simulation (requires --field_type)')
    parser.add_argument('--alignment-interval', type=int, default=10, dest='alignment_interval',
                        help='Apply alignment every N iterations (default: 10, only with --apply-alignment)')
    parser.add_argument('--allow-magnitude-change', action='store_false', dest='preserve_magnitude',
                        help='Allow field magnitude to change during alignment (default: preserve magnitude)')
    parser.add_argument('--same-initial-conditions', action='store_true', dest='same_initial_conditions',
                        help='Use same initial conditions for reference embryo (sanity check for embryo-to-embryo alignment)')
    parser.add_argument('--bidirectional', action='store_true', dest='bidirectional_alignment',
                        help='Enable bidirectional alignment (both embryos align to each other)')
    parser.add_argument('--coarsen-mode', type=str, default='average', dest='coarsen_mode',
                        choices=['average', 'sample'],
                        help='Coarse-graining mode: "average" (block averaging) or "sample" (corner sampling with interpolation)')
    parser.add_argument('--upscale-mode', type=str, default='nearest', dest='upscale_mode',
                        choices=['nearest', 'interpolate'],
                        help='Upscaling mode: "nearest" (blocky) or "interpolate" (smooth bilinear). Auto-set to "interpolate" if coarsen-mode is "sample".')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save results figure')
    parser.add_argument('--perturb-vmem', type=float, default=None, dest='perturb_vmem',
                        help='Add random Gaussian noise to main embryo Vmem with specified std dev (e.g., 0.1 for 10%% noise). Applied once at iteration after clamping ends.')
    parser.add_argument('--perturb-field', type=float, default=None, dest='perturb_field',
                        help='Add random Gaussian noise to main embryo electric field with specified std dev. Perturbs both magnitude and direction. Applied once at iteration after clamping ends.')
    parser.add_argument('--perturb-seed', type=int, default=None, dest='perturb_seed',
                        help='Random seed for Vmem/field perturbation (for reproducibility)')
    parser.add_argument('--alignment-mode', type=str, default='pre', dest='alignment_mode',
                        choices=['pre', 'post'],
                        help='Alignment computation mode: "pre" (delta computed before simulate()) or "post" (delta computed inside updateExtracellularVoltage())')

    args = parser.parse_args()

    # Auto-set upscale mode to interpolate if coarsen mode is sample
    if args.coarsen_mode == 'sample' and args.upscale_mode == 'nearest':
        args.upscale_mode = 'interpolate'
        print("Note: Auto-setting upscale mode to 'interpolate' (coarsen mode is 'sample')")

    # Test coarse-graining only
    if args.test_coarsening:
        test_coarse_graining()
        return

    # ATP unstable equilibrium (from data/survival_262.dat)
    ATP_UNSTABLE_EQUILIBRIUM = 2.5

    # Compute ATP concentrations from deltas or absolute values
    # Main embryo ATP
    if args.atp_delta is not None:
        main_atp_conc = ATP_UNSTABLE_EQUILIBRIUM + args.atp_delta
    elif args.atp_conc is not None:
        main_atp_conc = args.atp_conc
    else:
        # Default: healthy (9 above equilibrium)
        main_atp_conc = 11.5

    # Reference embryo ATP
    if args.ref_atp_delta is not None:
        ref_atp_conc = ATP_UNSTABLE_EQUILIBRIUM + args.ref_atp_delta
    elif args.ref_atp_conc is not None:
        ref_atp_conc = args.ref_atp_conc
    else:
        # Default: same as main embryo
        ref_atp_conc = main_atp_conc

    # Determine if alignment features should be enabled
    # Alignment is enabled if apply_alignment is set (requires field params to be meaningful)
    enable_alignment = args.apply_alignment

    print("=" * 60)
    if enable_alignment:
        print("Field Alignment Test with Stigmergic Embryo Model")
    else:
        print("Stigmergic Embryo Simulation")
    print("=" * 60)
    print(f"ATP enabled: {args.enable_atp}")
    if args.enable_atp:
        print(f"ATP Model: 262")
        print(f"ATP unstable equilibrium: {ATP_UNSTABLE_EQUILIBRIUM}")
        print(f"Main embryo ATP concentration: {main_atp_conc:.2f} (delta: {main_atp_conc - ATP_UNSTABLE_EQUILIBRIUM:+.2f})")
        if enable_alignment and args.field_type == 'embryo':
            print(f"Reference embryo ATP concentration: {ref_atp_conc:.2f} (delta: {ref_atp_conc - ATP_UNSTABLE_EQUILIBRIUM:+.2f})")
        print(f"ATP diffusion: {'ENABLED' if not args.disable_atp_diffusion else 'DISABLED (local reactions only)'}")
    if enable_alignment:
        print(f"Coarse-graining resolution: {args.resolution}x{args.resolution}")
        print(f"Alignment strength: {args.alignment_strength}")
        print(f"External field type: {args.field_type}")
        print(f"Apply alignment: {args.apply_alignment}")
        if args.apply_alignment:
            print(f"Alignment interval: {args.alignment_interval}")
    if args.perturb_vmem is not None:
        print(f"Vmem perturbation: std={args.perturb_vmem:.4f} (applied once after clamping ends)")
        if args.perturb_seed is not None:
            print(f"Perturbation seed: {args.perturb_seed}")

    # Load model
    print("\nLoading main embryo...")
    embryo_model, parameters = load_stigmergic_model(
        atp_conc=main_atp_conc,
        num_iters=args.num_iters,
        enable_atp=args.enable_atp,
        enable_atp_diffusion=not args.disable_atp_diffusion
    )

    # Get field shape
    field_shape = embryo_model.electricNetwork.extracellularIndexGrid.shape
    print(f"Field grid shape: {field_shape}")

    # Create external field if alignment is enabled
    external_field = None
    coarse_resolution = None
    reference_embryo = None

    if enable_alignment:
        direction_rad = np.radians(args.field_direction)
        if args.field_type == 'uniform':
            external_field = create_uniform_external_field(
                field_shape, direction_rad, args.field_magnitude
            )
        elif args.field_type == 'radial':
            external_field = create_radial_external_field(
                field_shape, magnitude=args.field_magnitude
            )
        elif args.field_type == 'gradient':
            external_field = create_gradient_external_field(
                field_shape, direction_rad, (0.0, args.field_magnitude)
            )
        elif args.field_type == 'embryo':
            # Create a reference embryo with fixed initial conditions from parameter file
            print("\nCreating reference embryo...")
            reference_embryo, params_ref = load_stigmergic_model(
                atp_conc=ref_atp_conc,
                num_iters=args.num_iters,
                enable_atp=args.enable_atp,
                enable_atp_diffusion=not args.disable_atp_diffusion
            )

            if args.same_initial_conditions:
                # Use exact same initial values as main embryo
                print("Using SAME initial conditions as main embryo (sanity check)")
                initial_values_ref = parameters['simParameters']['initialValues']
                print("Expected result: alignment should have minimal effect (embryos evolve identically)")
            else:
                # Use reference embryo's own initial values (loaded from same parameter file)
                print("Using reference embryo's own initial conditions")
                initial_values_ref = params_ref['simParameters']['initialValues']

            reference_embryo.setExperimentalConditions((initial_values_ref, 1))

            # Pass the embryo model itself - will extract field dynamically
            external_field = reference_embryo
            print("Reference embryo field will be extracted during simulation")

        print(f"External field source: {args.field_type}")
        coarse_resolution = (args.resolution, args.resolution)

    # Run simulation with or without alignment
    history = run_simulation_with_field_alignment(
        embryo_model,
        parameters,
        external_field,
        coarse_resolution,
        alignment_strength=args.alignment_strength,
        num_iters=args.num_iters,
        preserve_magnitude=args.preserve_magnitude,
        apply_alignment=args.apply_alignment,
        alignment_interval=args.alignment_interval,
        same_initial_conditions=args.same_initial_conditions if hasattr(args, 'same_initial_conditions') else False,
        bidirectional_alignment=args.bidirectional_alignment if hasattr(args, 'bidirectional_alignment') else False,
        perturb_vmem=args.perturb_vmem,
        perturb_field=args.perturb_field,
        perturb_seed=args.perturb_seed,
        coarsen_mode=args.coarsen_mode,
        upscale_mode=args.upscale_mode,
        alignment_mode=args.alignment_mode,
    )

    # Visualize results
    if args.save:
        save_path = args.save
    else:
        save_path = './data/stigmergic_test.png' if not enable_alignment else './data/field_alignment_test.png'
    visualize_results(history, save_path=save_path)


if __name__ == '__main__':
    main()
