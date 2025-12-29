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
    update_circuit_field_with_alignment,
    create_uniform_external_field,
    create_radial_external_field,
    create_gradient_external_field,
)


def load_stigmergic_model(atp_conc=11.5, num_samples=1, num_iters=1000, enable_atp=False):
    """
    Load a trained Stigmergic embryo model with optional ATP.

    Args:
        atp_conc: ATP concentration level (default 11.5 for healthy)
        num_samples: Number of samples to simulate
        num_iters: Number of simulation iterations (for ATP input setup)
        enable_atp: If True, enable ATP dynamics; if False, disable ATP (default: False)

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
        parameters['ATPParameters']['ATPDiffusionStrength'] = 10.0
        parameters['ATPParameters']['tissueConnectivity'] = \
            utils.computeLatticeAdjacencyMatrix(latticeDims=parameters['latticeDims'], periodicBoundary=False)
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

    Returns:
        history: Dictionary with simulation history
    """
    circuit = embryo_model.electricNetwork
    field_shape = circuit.extracellularIndexGrid.shape

    # Check if external field is provided
    has_external_field = (external_field is not None) and (coarse_resolution is not None)

    # History tracking
    history = {
        'vmem': [],
        'field_x': [],
        'field_y': [],
        'alignment_angle': [],
        'iterations': [],
        'field_shape': field_shape,
    }

    if has_external_field:
        # Initialize coarse-grainer
        coarsener = FieldCoarseGrainer(field_shape)

        # For static fields, coarse-grain upfront
        # For embryo fields, this happens dynamically in the loop
        is_embryo_field = hasattr(external_field, 'electricNetwork')
        if not is_embryo_field:
            # Coarse-grain and upscale static external field
            external_coarse = coarsener.coarsen(external_field, coarse_resolution)
            external_upscaled = coarsener.upscale(external_coarse, field_shape)

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

            # Create a wrapper function that extracts coarse-grained field WITHOUT stepping
            # (reference embryo is stepped separately in the loop for step-step-align pattern)
            def get_coarse_grained_embryo_field(_iter_idx):  # iter_idx required by callable interface but unused
                """Extract and coarse-grain reference embryo's current field (no stepping)."""
                try:
                    # Extract its current field (already stepped in loop before this is called)
                    ref_field = extract_field_2d(external_field.electricNetwork, sample_idx=0)
                    # Coarse-grain and upscale
                    ref_coarse = coarsener.coarsen(ref_field, coarse_resolution)
                    ref_upscaled = coarsener.upscale(ref_coarse, field_shape)
                    return ref_upscaled
                except Exception as e:
                    print(f"ERROR in get_coarse_grained_embryo_field: {e}")
                    import traceback
                    traceback.print_exc()
                    return None

            # Setup alignment parameters for main embryo (aligns to reference)
            alignment_params = {
                'external_field': get_coarse_grained_embryo_field,  # Callable that takes iter_idx
                'alignment_strength': alignment_strength,
                'preserve_magnitude': preserve_magnitude,
                'alignment_interval': alignment_interval,
                'sample_idx': 0
            }

            # Setup bidirectional alignment (reference aligns to main)
            ref_alignment_params = None
            if bidirectional_alignment:
                # Create reverse callable: extract main embryo's field for reference to align to
                def get_coarse_grained_main_field(_iter_idx):
                    """Extract and coarse-grain main embryo's current field for reverse alignment."""
                    try:
                        main_field = extract_field_2d(embryo_model.electricNetwork, sample_idx=0)
                        main_coarse = coarsener.coarsen(main_field, coarse_resolution)
                        main_upscaled = coarsener.upscale(main_coarse, field_shape)
                        return main_upscaled
                    except Exception as e:
                        print(f"ERROR in get_coarse_grained_main_field: {e}")
                        import traceback
                        traceback.print_exc()
                        return None

                ref_alignment_params = {
                    'external_field': get_coarse_grained_main_field,
                    'alignment_strength': alignment_strength,
                    'preserve_magnitude': preserve_magnitude,
                    'alignment_interval': alignment_interval,
                    'sample_idx': 0
                }
        else:
            # Static field - use pre-computed upscaled version
            alignment_params = {
                'external_field': external_upscaled,
                'alignment_strength': alignment_strength,
                'preserve_magnitude': preserve_magnitude,
                'alignment_interval': alignment_interval,
                'sample_idx': 0
            }

        for iter_idx in range(num_iters):
            # Step-step-align pattern: both embryos step, then alignment is applied

            # 1. Step reference embryo FIRST (if embryo-to-embryo alignment)
            if is_embryo_alignment:
                external_field.simulate(
                    externalInputs=external_inputs,
                    clampParameters=clamp_parameters,
                    numSimIters=1,
                    fieldModulation=True,
                    outerIter=iter_idx,
                    alignmentParameters=ref_alignment_params,  # Apply reverse alignment if bidirectional
                )

            # 2. Step main embryo (alignment happens AFTER stepping, using reference's current state)
            embryo_model.simulate(
                externalInputs=external_inputs,
                clampParameters=clamp_parameters,
                numSimIters=1,
                fieldModulation=True,
                outerIter=iter_idx,
                alignmentParameters=alignment_params,
            )

            # Record history at alignment points
            if (iter_idx + 1) % alignment_interval == 0:
                local_field_raw = extract_field_2d(circuit, sample_idx=0)
                history['vmem'].append(circuit.Vmem[0, :, 0].detach().cpu().numpy().copy())
                history['field_x'].append(local_field_raw[0].detach().cpu().numpy().copy())
                history['field_y'].append(local_field_raw[1].detach().cpu().numpy().copy())
                history['iterations'].append(iter_idx + 1)

                # Compute alignment angle (for embryo fields, get current reference field)
                if is_embryo_alignment:
                    # Extract and coarse-grain BOTH embryos' fields for fair comparison
                    local_coarse = coarsener.coarsen(local_field_raw, coarse_resolution)
                    local_field = coarsener.upscale(local_coarse, field_shape)

                    ref_field_raw = extract_field_2d(external_field.electricNetwork, sample_idx=0)
                    ref_coarse = coarsener.coarsen(ref_field_raw, coarse_resolution)
                    ext_field_for_angle = coarsener.upscale(ref_coarse, field_shape)
                else:
                    local_field = local_field_raw
                    ext_field_for_angle = external_upscaled

                # Check if fields are identical (avoids numerical precision issues in angle calculation)
                field_diff_before_angle = (local_field - ext_field_for_angle).abs().max().item()

                if field_diff_before_angle < 1e-12:
                    # Fields are identical, angle is 0
                    avg_angle = 0.0
                    cos_angle = torch.ones_like(local_field[0])  # For debugging
                else:
                    # Fields differ, compute angle normally
                    local_mag = torch.sqrt(local_field[0]**2 + local_field[1]**2 + 1e-10)
                    ext_mag = torch.sqrt(ext_field_for_angle[0]**2 + ext_field_for_angle[1]**2 + 1e-10)
                    dot = (local_field[0] * ext_field_for_angle[0] + local_field[1] * ext_field_for_angle[1])

                    # Only compute angle for vectors with sufficient magnitude (avoid numerical issues)
                    mag_threshold = 1e-6
                    valid_mask = (local_mag > mag_threshold) & (ext_mag > mag_threshold)

                    if valid_mask.sum() > 0:
                        cos_angle_valid = dot[valid_mask] / torch.clamp(local_mag[valid_mask] * ext_mag[valid_mask], min=1e-10)
                        avg_angle = torch.acos(torch.clamp(cos_angle_valid.mean(), -1, 1)).item()
                    else:
                        avg_angle = 0.0  # No valid vectors, assume aligned

                    # For debugging, still compute full cos_angle
                    cos_angle = dot / torch.clamp(local_mag * ext_mag, min=1e-10)
                history['alignment_angle'].append(np.degrees(avg_angle))

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

        # Compute alignment angle for passive mode (only compute once at the end)
        if has_external_field:
            local_mag = torch.sqrt(local_field[0]**2 + local_field[1]**2 + 1e-10)
            ext_mag = torch.sqrt(external_upscaled[0]**2 + external_upscaled[1]**2 + 1e-10)
            dot = (local_field[0] * external_upscaled[0] + local_field[1] * external_upscaled[1])
            cos_angle = dot / (local_mag * ext_mag + 1e-10)
            avg_angle = torch.acos(torch.clamp(cos_angle.mean(), -1, 1)).item()
            history['alignment_angle'].append(np.degrees(avg_angle))
            print(f"  Mean alignment angle: {np.degrees(avg_angle):.1f} degrees")

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
        print(f"  Final alignment angle: {final_angle:.1f} degrees")

        # Compute what the aligned field would be (for visualization)
        # For embryo fields, get the final reference field
        if is_embryo_alignment:
            ref_coarse_final = coarsener.coarsen(ref_field_final, coarse_resolution)
            ref_upscaled_final = coarsener.upscale(ref_coarse_final, field_shape)
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
        # 3x3 layout with reference embryo data (if available)
        plt.figure(figsize=(18, 18))
        coarse_res = history.get('coarse_resolution', field_shape)
        Y, X = np.mgrid[0:field_shape[0], 0:field_shape[1]]

        # Row 1: External/Reference field processing
        # 1. External field (full resolution) or Reference embryo field
        ax1 = plt.subplot(3, 3, 1)
        if has_ref_embryo:
            ref_field = history['ref_final_field']
            ref_mag = torch.sqrt(ref_field[0]**2 + ref_field[1]**2).numpy()
            ax1.imshow(ref_mag, cmap='viridis', origin='lower')
            ax1.quiver(X, Y, ref_field[0].numpy(), ref_field[1].numpy(),
                       color='white', alpha=0.7, scale=ref_mag.max()*15 if ref_mag.max() > 0 else 1)
            ax1.set_title('Reference Embryo Field (Final)')
        else:
            ext_field = history['external_field']
            ext_mag = torch.sqrt(ext_field[0]**2 + ext_field[1]**2).numpy()
            ax1.imshow(ext_mag, cmap='viridis', origin='lower')
            ax1.quiver(X, Y, ext_field[0].numpy(), ext_field[1].numpy(),
                       color='white', alpha=0.7, scale=ext_mag.max()*15 if ext_mag.max() > 0 else 1)
            ax1.set_title(f'External Field (Full: {field_shape[0]}x{field_shape[1]})')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # 2. External field (coarse-grained) or Reference embryo Vmem
        ax2 = plt.subplot(3, 3, 2)
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
        ax3 = plt.subplot(3, 3, 3)
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

        # Row 2: Main embryo results
        # 4. Final local field (main embryo)
        ax4 = plt.subplot(3, 3, 4)
        final_field = history['final_field']
        final_mag = torch.sqrt(final_field[0]**2 + final_field[1]**2).numpy()
        ax4.imshow(final_mag, cmap='plasma', origin='lower')
        ax4.quiver(X, Y, final_field[0].numpy(), final_field[1].numpy(),
                   color='white', alpha=0.7, scale=final_mag.max()*15 if final_mag.max() > 0 else 1)
        ax4.set_title('Main Embryo Field (Final)')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')

        # 5. Final Vmem (main embryo)
        ax5 = plt.subplot(3, 3, 5)
        if len(history['vmem']) > 0:
            final_vmem = history['vmem'][-1]
            grid_size = int(np.sqrt(len(final_vmem)))
            vmem_2d = final_vmem.reshape(grid_size, grid_size)
            im = ax5.imshow(vmem_2d * 1000, cmap='RdBu_r', origin='lower')
            plt.colorbar(im, ax=ax5, label='mV')
            ax5.set_title('Main Embryo Vmem (Final)')
            ax5.set_xlabel('x')
            ax5.set_ylabel('y')

        # 6. Vmem difference (if reference embryo available)
        ax6 = plt.subplot(3, 3, 6)
        if has_ref_embryo and len(history['vmem']) > 0:
            main_vmem = history['vmem'][-1]
            ref_vmem = history['ref_final_vmem']
            vmem_diff = np.abs(main_vmem - ref_vmem) * 1000  # mV
            grid_size = int(np.sqrt(len(vmem_diff)))
            vmem_diff_2d = vmem_diff.reshape(grid_size, grid_size)
            im = ax6.imshow(vmem_diff_2d, cmap='hot', origin='lower')
            plt.colorbar(im, ax=ax6, label='mV')
            max_diff_mv = vmem_diff.max()
            ax6.set_title(f'Vmem Difference (max: {max_diff_mv:.3f} mV)')
            ax6.set_xlabel('x')
            ax6.set_ylabel('y')
        else:
            ax6.axis('off')
            ax6.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=20, color='gray')

        # Row 3: Alignment metrics
        # 7. Alignment angle over time
        ax7 = plt.subplot(3, 3, 7)
        if len(history['alignment_angle']) > 0:
            if len(history['alignment_angle']) > 1:
                # Show alignment angle over time (active alignment mode)
                ax7.plot(history['iterations'], history['alignment_angle'], 'b-', linewidth=2, marker='o')
                ax7.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='Random (90°)')
                ax7.set_xlabel('Iteration')
                ax7.set_ylabel('Mean Alignment Angle (degrees)')
                ax7.set_title('Field Alignment Dynamics')
                ax7.set_ylim(0, 180)
                ax7.legend()
                ax7.grid(True, alpha=0.3)
            else:
                # Show the alignment angle as a bar (passive mode)
                angle = history['alignment_angle'][0]
                ax7.bar(['Final'], [angle], color='steelblue', alpha=0.7)
                ax7.axhline(y=90, color='r', linestyle='--', alpha=0.5, label='Random alignment')
                ax7.set_ylabel('Mean Alignment Angle (degrees)')
                ax7.set_title('Field Alignment Quality')
                ax7.set_ylim(0, 180)
                ax7.legend()
                ax7.grid(True, alpha=0.3, axis='y')
        else:
            ax7.axis('off')

        # 8. Statistics summary
        ax8 = plt.subplot(3, 3, 8)
        ax8.axis('off')
        stats_text = "Simulation Statistics:\n\n"
        if len(history['alignment_angle']) > 0:
            final_angle = history['alignment_angle'][-1]
            stats_text += f"Final Alignment: {final_angle:.2f}°\n"
        if has_ref_embryo:
            main_field = history['final_field']
            ref_field = history['ref_final_field']
            field_diff_max = torch.sqrt((main_field[0] - ref_field[0])**2 + (main_field[1] - ref_field[1])**2).max().item()
            stats_text += f"Max Field Diff: {field_diff_max:.2e}\n\n"
            if len(history['vmem']) > 0:
                vmem_diff_max = np.abs(history['vmem'][-1] - history['ref_final_vmem']).max() * 1000
                stats_text += f"Max Vmem Diff: {vmem_diff_max:.3f} mV\n"
        stats_text += f"\nIterations: {history['iterations'][-1] if len(history['iterations']) > 0 else 'N/A'}"
        ax8.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', family='monospace')

        # 9. Reserved for future use
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        ax9.text(0.5, 0.5, 'Reserved', ha='center', va='center', fontsize=16, color='lightgray')

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
                        help='Enable ATP dynamics (disabled by default)')
    parser.add_argument('--atp_conc', type=float, default=11.5,
                        help='ATP concentration (default: 11.5 for healthy, only used if --enable-atp)')
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
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save results figure')

    args = parser.parse_args()

    # Test coarse-graining only
    if args.test_coarsening:
        test_coarse_graining()
        return

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
        print(f"ATP concentration: {args.atp_conc}")
    if enable_alignment:
        print(f"Coarse-graining resolution: {args.resolution}x{args.resolution}")
        print(f"Alignment strength: {args.alignment_strength}")
        print(f"External field type: {args.field_type}")
        print(f"Apply alignment: {args.apply_alignment}")
        if args.apply_alignment:
            print(f"Alignment interval: {args.alignment_interval}")

    # Load model
    print("\nLoading Stigmergic model...")
    embryo_model, parameters = load_stigmergic_model(
        atp_conc=args.atp_conc,
        num_iters=args.num_iters,
        enable_atp=args.enable_atp
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
                atp_conc=args.atp_conc,
                num_iters=args.num_iters,
                enable_atp=args.enable_atp
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
    )

    # Visualize results
    if args.save:
        save_path = args.save
    else:
        save_path = 'stigmergic_test.png' if not enable_alignment else 'field_alignment_test.png'
    visualize_results(history, save_path=save_path)


if __name__ == '__main__':
    main()
