"""
Test if Hill function has gradient flow issues
"""

import torch

# Test Hill function gradient flow
k_on = torch.tensor(10.0, requires_grad=True)
k_off = torch.tensor(0.5, requires_grad=True)
K_half = torch.tensor(0.5, requires_grad=True)

# Starting from CaMKII = 0 (like in simulation)
CaMKII = torch.tensor(0.0, requires_grad=False)  # Initial state has no gradient
vmem_signal = torch.tensor(0.8)  # Strong input
or_threshold = torch.tensor(0.5)
or_sharpness = torch.tensor(10.0)

print("Testing Hill function gradient flow")
print("=" * 60)

# Compute one step
CaMKII_sq = CaMKII * CaMKII
K_half_sq = K_half * K_half
self_activation = CaMKII_sq / (K_half_sq + CaMKII_sq)

print(f"CaMKII = {CaMKII.item():.4f}")
print(f"self_activation = {self_activation.item():.4f}")
print(f"self_activation.requires_grad = {self_activation.requires_grad}")
print(f"self_activation.grad_fn = {self_activation.grad_fn}")

combined_signal = or_sharpness * (vmem_signal + self_activation - or_threshold)
activation = torch.relu(combined_signal) / or_sharpness

print(f"\nactivation = {activation.item():.4f}")
print(f"activation.requires_grad = {activation.requires_grad}")
print(f"activation.grad_fn = {activation.grad_fn}")

dCaMKII_dt = k_on * activation - k_off * CaMKII

print(f"\ndCaMKII_dt = {dCaMKII_dt.item():.4f}")
print(f"dCaMKII_dt.requires_grad = {dCaMKII_dt.requires_grad}")
print(f"dCaMKII_dt.grad_fn = {dCaMKII_dt.grad_fn}")

# Try to backprop through it
loss = dCaMKII_dt ** 2
loss.backward()

print(f"\n\nAfter backward:")
print(f"k_on.grad = {k_on.grad}")
print(f"k_off.grad = {k_off.grad}")
print(f"K_half.grad = {K_half.grad}")

print("\n" + "=" * 60)
print("PROBLEM IDENTIFIED:")
print("=" * 60)
print("When CaMKII=0:")
print("  self_activation = 0²/(K_half² + 0²) = 0")
print("  This is CONSTANT zero, independent of K_half!")
print("  So ∂self_activation/∂K_half = 0")
print("\nThe Hill function has NO gradient when x=0.")
print("This is why gradients are zero - system starts at x=0 and can't learn!")
print("\nSOLUTION: Need small initial CaMKII values, not exact zero.")
