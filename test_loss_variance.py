"""
Test loss variance due to random CaMKII initialization
"""
import torch
import numpy as np
import sys
sys.path.insert(0, '.')

# Import the necessary components
from test_camkii_bistability import (
    SimpleCaMKII, define_target_face_pattern, compute_loss,
    load_stigmergic_parameters, load_learned_parameters
)
from embryo import model
import copy

# Load learned parameters
learned_params = load_learned_parameters('data/bestLearnedCaMKIIParams_0.dat')

# Load stigmergic parameters
params = load_stigmergic_parameters('data/StigmergicModelParameters.dat')
grid_size = params['latticeDims'][0]

# Define target
target_masks = define_target_face_pattern(grid_size)

# Run bioelectric simulation once (this is deterministic)
num_samples = params["simParameters"]["numSamples"]
initial_values = copy.deepcopy(params["simParameters"]["initialValues"])
external_inputs = copy.deepcopy(params["simParameters"]["externalInputs"])
clamp_params = copy.deepcopy(params["clampParameters"])

bio_model = model(params, numBasicSamples=num_samples)
bio_model.setExperimentalConditions((initial_values, num_samples))
initial_vmem_grid = bio_model.electricNetwork.Vmem[0, :, 0].reshape(grid_size, grid_size).clone()

bio_model.simulate(
    externalInputs=external_inputs,
    clampParameters=clamp_params,
    perturbation=None,
    fieldModulation=False,
    numSimIters=1000
)
vmem_final = bio_model.electricNetwork.Vmem[0, :, 0].reshape(grid_size, grid_size).clone()

print("Testing loss variance over multiple random seeds...")
print("=" * 60)

losses = []
for seed in range(10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create CaMKII tracker with learned params
    camkii = SimpleCaMKII(grid_size=grid_size, device='cpu', learned_params=learned_params)
    
    # Run simulation
    dt = 0.01
    for t in range(2000):
        if t < 1000:
            alpha = t / 1000
            vmem_grid = (1 - alpha) * initial_vmem_grid + alpha * vmem_final
        else:
            decay_progress = (t - 1000) / 1000
            vmem_grid = (1 - decay_progress) * vmem_final + decay_progress * initial_vmem_grid
        
        camkii.update(vmem_grid, dt=dt)
    
    # Compute loss at t=2000
    loss_results = compute_loss(camkii.CaMKII_active, target_masks)
    losses.append(loss_results['total_loss'])
    
    print(f"Seed {seed:2d}: loss={loss_results['total_loss']:.4f}, contrast={loss_results['contrast']:.3f}")

losses = np.array(losses)
print("\n" + "=" * 60)
print(f"Loss statistics over 10 random seeds:")
print(f"  Mean: {losses.mean():.4f}")
print(f"  Std:  {losses.std():.4f}")
print(f"  Min:  {losses.min():.4f}")
print(f"  Max:  {losses.max():.4f}")
print(f"\nBest loss during training: 0.2504")
print(f"Your test loss (0.34) is within {(0.34 - losses.mean()) / losses.std():.2f} std from mean")
