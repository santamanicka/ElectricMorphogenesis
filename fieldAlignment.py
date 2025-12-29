"""
Field Alignment Module for Inter-Embryo Electric Field Communication

This module implements multi-resolution coarse-graining and alignment of electric
field vectors for inter-embryo signaling. External fields from neighboring embryos
can be coarse-grained to arbitrary resolutions and used to gently align the local
field vectors of a target embryo.

Key concepts:
- Coarse-graining: Partition field grid, average vectors in each partition
- Upscaling: Expand coarse field back to full resolution
- Alignment: Slow relaxation of local field toward external field direction
"""

import torch
import numpy as np


class FieldCoarseGrainer:
    """
    Handles multi-resolution coarse-graining and upscaling of electric field vectors.

    For an 11x11 cell grid with fieldResolution=1, the field grid is 12x12.
    Valid coarse-graining resolutions must evenly divide 12: 1, 2, 3, 4, 6, 12
    """

    def __init__(self, field_grid_shape=(12, 12)):
        """
        Initialize the coarse-grainer.

        Args:
            field_grid_shape: tuple (H, W) of the full-resolution field grid
        """
        self.field_shape = field_grid_shape
        self.H, self.W = field_grid_shape

    def _validate_resolution(self, resolution):
        """Check that resolution evenly divides field dimensions."""
        res_h, res_w = resolution
        if self.H % res_h != 0:
            raise ValueError(f"Resolution height {res_h} must evenly divide field height {self.H}")
        if self.W % res_w != 0:
            raise ValueError(f"Resolution width {res_w} must evenly divide field width {self.W}")

    def coarsen(self, field_2d, target_resolution):
        """
        Coarse-grain a 2D field to target resolution by averaging within partitions.

        Args:
            field_2d: Field vectors, shape (2, H, W) where dim 0 is x/y component
                      Can also be (2, numSamples, H, W) for batched processing
            target_resolution: tuple (rows, cols) for output grid, e.g., (4, 4)

        Returns:
            Coarsened field: shape (2, target_rows, target_cols) or
                            (2, numSamples, target_rows, target_cols) if batched
        """
        self._validate_resolution(target_resolution)

        res_h, res_w = target_resolution

        # Handle batched input
        if field_2d.dim() == 4:
            # Shape: (2, numSamples, H, W)
            batched = True
            num_samples = field_2d.shape[1]
        else:
            # Shape: (2, H, W)
            batched = False
            field_2d = field_2d.unsqueeze(1)  # Add sample dimension
            num_samples = 1

        # Partition sizes
        partition_h = self.H // res_h
        partition_w = self.W // res_w

        # Reshape and average
        # (2, S, H, W) -> (2, S, res_h, partition_h, res_w, partition_w)
        field_reshaped = field_2d.view(
            2, num_samples, res_h, partition_h, res_w, partition_w
        )

        # Average over partition dimensions (3 and 5)
        coarse_field = field_reshaped.mean(dim=(3, 5))  # (2, S, res_h, res_w)

        if not batched:
            coarse_field = coarse_field.squeeze(1)  # Remove sample dimension

        return coarse_field

    def upscale(self, coarse_field, target_shape=None):
        """
        Upscale coarse field back to full resolution using nearest-neighbor.

        Args:
            coarse_field: Coarsened field, shape (2, coarse_H, coarse_W) or
                         (2, numSamples, coarse_H, coarse_W) if batched
            target_shape: tuple (H, W) for output, defaults to self.field_shape

        Returns:
            Upscaled field: shape (2, H, W) or (2, numSamples, H, W)
        """
        if target_shape is None:
            target_shape = self.field_shape

        target_h, target_w = target_shape

        # Handle batched input
        if coarse_field.dim() == 4:
            batched = True
            num_samples = coarse_field.shape[1]
            coarse_h, coarse_w = coarse_field.shape[2], coarse_field.shape[3]
        else:
            batched = False
            coarse_field = coarse_field.unsqueeze(1)
            num_samples = 1
            coarse_h, coarse_w = coarse_field.shape[2], coarse_field.shape[3]

        # Compute repeat factors
        repeat_h = target_h // coarse_h
        repeat_w = target_w // coarse_w

        # Use repeat_interleave for nearest-neighbor upscaling
        upscaled = coarse_field.repeat_interleave(repeat_h, dim=2)
        upscaled = upscaled.repeat_interleave(repeat_w, dim=3)

        if not batched:
            upscaled = upscaled.squeeze(1)

        return upscaled

    def get_valid_resolutions(self):
        """Return list of valid resolution tuples that evenly divide field shape."""
        valid = []
        for h in range(1, self.H + 1):
            for w in range(1, self.W + 1):
                if self.H % h == 0 and self.W % w == 0:
                    valid.append((h, w))
        return valid


def apply_field_alignment(local_field, external_field, alignment_strength, dt=1.0):
    """
    Apply slow alignment forcing of local field toward external field.

    This implements a simple relaxation dynamics:
        d(local_field)/dt = alignment_strength * (external_field - local_field)

    Args:
        local_field: Current field vectors, shape (2, H, W) or (2, S, H, W)
        external_field: Target field vectors (same shape as local_field)
        alignment_strength: Coupling strength (small value, e.g., 0.001-0.1)
        dt: Timestep for integration

    Returns:
        Updated local field (same shape as input)
    """
    delta = alignment_strength * (external_field - local_field) * dt
    # print ((external_field - local_field).unique())
    return local_field + delta


def apply_field_alignment_normalized(local_field, external_field, alignment_strength, dt=1.0):
    """
    Apply alignment forcing that preserves local field magnitude.

    This rotates the local field direction toward the external field direction
    while preserving the local field magnitude.

    Args:
        local_field: Current field vectors, shape (2, H, W)
        external_field: Target field vectors, shape (2, H, W)
        alignment_strength: Coupling strength
        dt: Timestep

    Returns:
        Updated local field with preserved magnitude
    """
    # Compute magnitudes
    local_mag = torch.sqrt(local_field[0]**2 + local_field[1]**2 + 1e-10)
    external_mag = torch.sqrt(external_field[0]**2 + external_field[1]**2 + 1e-10)

    # Normalize external field
    external_norm = external_field / external_mag.unsqueeze(0)

    # Compute alignment delta (in direction space)
    local_norm = local_field / local_mag.unsqueeze(0)
    delta_direction = alignment_strength * (external_norm - local_norm) * dt

    # Update direction and restore magnitude
    new_direction = local_norm + delta_direction
    new_mag = torch.sqrt(new_direction[0]**2 + new_direction[1]**2 + 1e-10)
    new_direction = new_direction / new_mag.unsqueeze(0)

    return new_direction * local_mag.unsqueeze(0)


def extract_field_2d(circuit, sample_idx=0):
    """
    Extract 2D field vectors from a cellularFieldNetwork circuit.

    Args:
        circuit: cellularFieldNetwork instance with computed field
        sample_idx: Which sample to extract (default 0)

    Returns:
        field_2d: shape (2, H, W) where H, W are field grid dimensions
    """
    # Get field vector components
    eVx = circuit.eVforceVector[0, sample_idx, :, 0]  # (numFieldGridPoints,)
    eVy = circuit.eVforceVector[1, sample_idx, :, 0]  # (numFieldGridPoints,)

    # Get grid shape from extracellularIndexGrid
    grid_shape = circuit.extracellularIndexGrid.shape  # (H, W)
    H, W = grid_shape

    # Create 2D field arrays
    field_2d = torch.zeros(2, H, W, dtype=eVx.dtype, device=eVx.device)

    # Map 1D indices to 2D grid
    index_grid = torch.from_numpy(circuit.extracellularIndexGrid.astype(np.int64))

    for i in range(H):
        for j in range(W):
            idx = index_grid[i, j].item()
            if idx >= 0:  # Valid index
                field_2d[0, i, j] = eVx[int(idx)]
                field_2d[1, i, j] = eVy[int(idx)]

    return field_2d


def update_circuit_field_with_alignment(circuit, external_field, alignment_strength, dt=1.0,
                                        sample_idx=0, preserve_magnitude=True):
    """
    Apply alignment dynamics to update the circuit's field in-place.

    This is a convenience function that extracts the current field, applies
    alignment dynamics using apply_field_alignment or apply_field_alignment_normalized,
    and injects the result back into the circuit.

    Args:
        circuit: cellularFieldNetwork instance
        external_field: Target field vectors, shape (2, H, W)
        alignment_strength: Coupling strength for alignment
        dt: Timestep for integration
        sample_idx: Which sample to modify (default 0)
        preserve_magnitude: If True, use apply_field_alignment_normalized; else apply_field_alignment

    Returns:
        None (modifies circuit in-place)
    """
    # Extract current field
    local_field = extract_field_2d(circuit, sample_idx=sample_idx)

    # print("Inside field diff = ",(external_field - local_field).abs().max().item())

    # Apply alignment dynamics
    if preserve_magnitude:
        aligned_field = apply_field_alignment_normalized(
            local_field, external_field, alignment_strength, dt
        )
    else:
        aligned_field = apply_field_alignment(
            local_field, external_field, alignment_strength, dt
        )

    # Inject updated field back into circuit
    inject_field_2d(circuit, aligned_field, sample_idx=sample_idx)


def inject_field_2d(circuit, field_2d, sample_idx=0):
    """
    Inject 2D field vectors back into a cellularFieldNetwork circuit.

    This updates both eVforceVector (the x,y components) and eV (the magnitude).

    Args:
        circuit: cellularFieldNetwork instance
        field_2d: shape (2, H, W) field to inject
        sample_idx: Which sample to modify (default 0)
    """
    H, W = field_2d.shape[1], field_2d.shape[2]
    index_grid = torch.from_numpy(circuit.extracellularIndexGrid.astype(np.int64))

    for i in range(H):
        for j in range(W):
            idx = index_grid[i, j].item()
            if idx >= 0:  # Valid index
                circuit.eVforceVector[0, sample_idx, int(idx), 0] = field_2d[0, i, j]
                circuit.eVforceVector[1, sample_idx, int(idx), 0] = field_2d[1, i, j]

    # Recompute eV (magnitude) from updated eVforceVector components
    # This is critical because Vmem responds to eV magnitude, not just direction
    eVx = circuit.eVforceVector[0, sample_idx, :, :]
    eVy = circuit.eVforceVector[1, sample_idx, :, :]
    eVforce = (eVx**2) + (eVy**2)
    circuit.eV[sample_idx, :, :] = torch.pow(eVforce + circuit.epsilon, 0.5)


def create_uniform_external_field(shape, direction, magnitude=1.0):
    """
    Create a uniform external field pointing in a given direction.

    Args:
        shape: tuple (H, W) for field grid
        direction: angle in radians (0 = +x, pi/2 = +y)
        magnitude: field strength

    Returns:
        field_2d: shape (2, H, W)
    """
    H, W = shape
    field_2d = torch.zeros(2, H, W)
    field_2d[0, :, :] = magnitude * np.cos(direction)
    field_2d[1, :, :] = magnitude * np.sin(direction)
    return field_2d


def create_radial_external_field(shape, center=None, magnitude=1.0, outward=True):
    """
    Create a radial external field pointing toward/away from center.

    Args:
        shape: tuple (H, W) for field grid
        center: tuple (y, x) for center point, defaults to grid center
        magnitude: field strength
        outward: if True, field points away from center

    Returns:
        field_2d: shape (2, H, W)
    """
    H, W = shape
    if center is None:
        center = ((H - 1) / 2, (W - 1) / 2)

    cy, cx = center
    field_2d = torch.zeros(2, H, W)

    for i in range(H):
        for j in range(W):
            dx = j - cx
            dy = i - cy
            r = np.sqrt(dx**2 + dy**2) + 1e-10

            sign = 1.0 if outward else -1.0
            field_2d[0, i, j] = sign * magnitude * dx / r
            field_2d[1, i, j] = sign * magnitude * dy / r

    return field_2d


def create_gradient_external_field(shape, direction, magnitude_range=(0.0, 1.0)):
    """
    Create a gradient external field with magnitude varying along a direction.

    Args:
        shape: tuple (H, W) for field grid
        direction: angle in radians for field direction
        magnitude_range: tuple (min, max) for magnitude gradient

    Returns:
        field_2d: shape (2, H, W)
    """
    H, W = shape
    field_2d = torch.zeros(2, H, W)

    min_mag, max_mag = magnitude_range

    # Gradient along the direction
    for i in range(H):
        for j in range(W):
            # Project position onto direction
            t = (i / (H - 1) * np.sin(direction) + j / (W - 1) * np.cos(direction))
            t = t / (np.abs(np.sin(direction)) + np.abs(np.cos(direction)) + 1e-10)
            t = np.clip(t, 0, 1)

            mag = min_mag + t * (max_mag - min_mag)
            field_2d[0, i, j] = mag * np.cos(direction)
            field_2d[1, i, j] = mag * np.sin(direction)

    return field_2d
