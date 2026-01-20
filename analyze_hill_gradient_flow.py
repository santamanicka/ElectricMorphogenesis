"""
Analyze gradient flow through Hill function at different CaMKII values
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# Test gradient flow at different CaMKII values
K_half = torch.tensor(0.5, requires_grad=True)
k_on = torch.tensor(10.0, requires_grad=True)

camkii_values = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

print("=" * 70)
print("HILL FUNCTION GRADIENT ANALYSIS")
print("=" * 70)
print("\nTesting: self_activation = x² / (K_half² + x²)")
print(f"K_half = {K_half.item():.2f}\n")

results = []

for x_val in camkii_values:
    x = torch.tensor(x_val, requires_grad=False)

    # Compute Hill function
    x_sq = x * x
    K_sq = K_half * K_half
    self_activation = x_sq / (K_sq + x_sq)

    # Compute loss (arbitrary)
    loss = (self_activation - 0.5) ** 2

    # Zero gradients
    if K_half.grad is not None:
        K_half.grad.zero_()
    if k_on.grad is not None:
        k_on.grad.zero_()

    # Backprop
    loss.backward(retain_graph=True)

    grad_K = K_half.grad.item() if K_half.grad is not None else 0.0

    results.append({
        'x': x_val,
        'self_activation': self_activation.item(),
        'grad_K_half': grad_K
    })

    print(f"x={x_val:.2f}:  self_act={self_activation.item():.4f}  ∂loss/∂K_half={grad_K:.2e}")

print("\n" + "=" * 70)
print("OBSERVATIONS")
print("=" * 70)

# Identify problem regions
small_grad_region = [r for r in results if abs(r['grad_K_half']) < 1e-3]
print(f"\nGradient < 1e-3 for x in: {[r['x'] for r in small_grad_region]}")
print("→ When CaMKII starts near 0 (x < 0.1), gradients are tiny!")
print("→ This prevents learning K_half parameter")

print("\n" + "=" * 70)
print("ROOT CAUSE")
print("=" * 70)
print("""
The Hill function has two problems:

1. **Flat at LOW x**: When x ≈ 0, self_activation ≈ 0 regardless of K_half
   → ∂self_activation/∂K_half ≈ 0
   → Can't learn K_half!

2. **Flat at HIGH x**: When x ≈ 1, self_activation ≈ 1 regardless of K_half
   → Also can't learn K_half!

3. **Only learns in MIDDLE**: K_half gradients are strong only when x ≈ K_half
   → But optimizer needs to KNOW to get there first!

This is a **chicken-and-egg problem**:
- Need x to grow to ~K_half to learn K_half
- But need proper K_half to allow x to grow!
""")

print("\n" + "=" * 70)
print("SOLUTION OPTIONS")
print("=" * 70)
print("""
Option 1: REMOVE K_half from learning, FIX it
  - Set K_half = 0.5 (not learnable)
  - Learn only k_on, k_off, V_half, k_vmem, or_threshold, or_sharpness
  - This eliminates the gradient bottleneck

Option 2: Better initialization
  - Initialize CaMKII with larger values (0.1-0.3 instead of 0.01)
  - This puts system in region where gradients flow
  - But still fragile

Option 3: Alternative bistability mechanism
  - Use multiplicative gating: activation = input * (1 + α*self)
  - Or additive with threshold: activation = relu(input + β*self - threshold)
  - These have better gradient flow

RECOMMENDATION: Option 1 (fix K_half)
  - Simplest and most robust
  - Hill function cooperativity (n=2) is still present
  - K_half = 0.5 is biologically reasonable
  - Reduces parameters from 7 to 6
""")
