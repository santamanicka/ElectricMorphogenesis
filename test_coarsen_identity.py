"""
Test whether coarsen/upscale operations preserve identity when resolution == field_shape.
"""
import torch
import numpy as np
from fieldAlignment import FieldCoarseGrainer

# Create a simple test field (12x12 grid)
field_shape = (12, 12)
torch.manual_seed(42)
A_raw = torch.randn(2, 12, 12)
B_raw = A_raw.clone()  # Identical copy

print("Initial state:")
print(f"A_raw == B_raw: {torch.allclose(A_raw, B_raw, atol=0)}")
print(f"Max difference: {(A_raw - B_raw).abs().max().item():.2e}")

# Test at full resolution (12x12)
print("\n" + "="*60)
print("Test 1: Full resolution (12x12) - should be identity")
print("="*60)

coarsener = FieldCoarseGrainer(field_shape)
coarse_resolution = (12, 12)

# Process A
A_coarse = coarsener.coarsen(A_raw, coarse_resolution)
A_snap = coarsener.upscale(A_coarse, field_shape)

# Process B
B_coarse = coarsener.coarsen(B_raw, coarse_resolution)
B_snap = coarsener.upscale(B_coarse, field_shape)

print(f"\nAfter coarsen/upscale:")
print(f"A_snap == B_snap: {torch.allclose(A_snap, B_snap, atol=0)}")
print(f"Max difference A_snap - B_snap: {(A_snap - B_snap).abs().max().item():.2e}")

print(f"\nA_raw == A_snap: {torch.allclose(A_raw, A_snap, atol=0)}")
print(f"Max difference A_raw - A_snap: {(A_raw - A_snap).abs().max().item():.2e}")

print(f"\nB_raw == B_snap: {torch.allclose(B_raw, B_snap, atol=0)}")
print(f"Max difference B_raw - B_snap: {(B_raw - B_snap).abs().max().item():.2e}")

# Test at coarse resolution (4x4)
print("\n" + "="*60)
print("Test 2: Coarse resolution (4x4) - expected to differ")
print("="*60)

coarse_resolution = (4, 4)

# Process A
A_coarse = coarsener.coarsen(A_raw, coarse_resolution)
A_snap = coarsener.upscale(A_coarse, field_shape)

# Process B
B_coarse = coarsener.coarsen(B_raw, coarse_resolution)
B_snap = coarsener.upscale(B_coarse, field_shape)

print(f"\nAfter coarsen/upscale:")
print(f"A_snap == B_snap: {torch.allclose(A_snap, B_snap, atol=0)}")
print(f"Max difference A_snap - B_snap: {(A_snap - B_snap).abs().max().item():.2e}")

print(f"\nA_raw == A_snap: {torch.allclose(A_raw, A_snap, atol=1e-6)}")
print(f"Max difference A_raw - A_snap: {(A_raw - A_snap).abs().max().item():.2e}")

# Test with truly identical tensors (no clone, same object)
print("\n" + "="*60)
print("Test 3: Same tensor object at full resolution")
print("="*60)

coarse_resolution = (12, 12)
C_raw = torch.randn(2, 12, 12)

# Process twice
C_snap1 = coarsener.upscale(coarsener.coarsen(C_raw, coarse_resolution), field_shape)
C_snap2 = coarsener.upscale(coarsener.coarsen(C_raw, coarse_resolution), field_shape)

print(f"\nC_snap1 == C_snap2: {torch.allclose(C_snap1, C_snap2, atol=0)}")
print(f"Max difference: {(C_snap1 - C_snap2).abs().max().item():.2e}")
print(f"C_raw == C_snap1: {torch.allclose(C_raw, C_snap1, atol=0)}")
print(f"Max difference C_raw - C_snap1: {(C_raw - C_snap1).abs().max().item():.2e}")
