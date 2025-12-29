"""
Debug why MSE loss = 0.57 when all CaMKII = 1.0
"""

import torch

# Simulate the situation: all CaMKII = 1.0
grid_size = 11
camkii_t2000 = torch.ones(grid_size, grid_size)

# Target pattern (from real code)
# Let's assume 8 eye + 3 nose + 5 mouth = 16 feature cells
# Total cells = 11*11 = 121
num_feature_cells = 16
num_background_cells = 121 - 16
total_cells = 121

# Feature mask
all_features = torch.zeros(grid_size, grid_size, dtype=torch.bool)
all_features.view(-1)[:num_feature_cells] = True  # First 16 cells are features

# Target: 1.0 at features, 0.0 at background
target_pattern = torch.zeros(grid_size, grid_size)
target_pattern[all_features] = 1.0

print("=" * 70)
print("LOSS CALCULATION WHEN ALL CaMKII = 1.0")
print("=" * 70)

print(f"\nSetup:")
print(f"  Total cells: {total_cells}")
print(f"  Feature cells: {num_feature_cells}")
print(f"  Background cells: {num_background_cells}")
print(f"  CaMKII: all = 1.0")
print(f"  Target: features = 1.0, background = 0.0")

# Compute class-balanced weights
weight_feature = total_cells / (2.0 * num_feature_cells)
weight_background = total_cells / (2.0 * num_background_cells)

print(f"\nClass weights:")
print(f"  weight_feature = {weight_feature:.4f}")
print(f"  weight_background = {weight_background:.4f}")

# Separate losses
feature_loss = torch.mean((camkii_t2000[all_features] - target_pattern[all_features]) ** 2)
background_loss = torch.mean((camkii_t2000[~all_features] - target_pattern[~all_features]) ** 2)

print(f"\nSeparate losses:")
print(f"  Feature MSE: (1.0 - 1.0)² = {feature_loss.item():.4f}")
print(f"  Background MSE: (1.0 - 0.0)² = {background_loss.item():.4f}")

# Weighted loss
loss = weight_feature * feature_loss + weight_background * background_loss

print(f"\nWeighted combination:")
print(f"  loss = {weight_feature:.4f} * {feature_loss.item():.4f} + {weight_background:.4f} * {background_loss.item():.4f}")
print(f"  loss = {loss.item():.4f}")

print("\n" + "=" * 70)
print("EXPLANATION")
print("=" * 70)
print(f"""
When all CaMKII = 1.0:
- Features (16 cells): perfectly correct (1.0 vs target 1.0) → MSE = 0.0
- Background (105 cells): completely wrong (1.0 vs target 0.0) → MSE = 1.0

Class-balanced weighting:
- Feature weight = 121/(2*16) = 3.78
- Background weight = 121/(2*105) = 0.576

Final loss = 3.78 * 0.0 + 0.576 * 1.0 = 0.576

This is why loss ≈ 0.57 when everything saturates at 1.0!

THE PROBLEM: Class balancing heavily weights features (3.78x vs 0.576x)
because features are the minority class. When all cells = 1.0:
- Features are PERFECT (no contribution to loss)
- Background is WRONG but down-weighted

This creates a perverse incentive: saturating at 1.0 achieves lower loss
than having proper spatial contrast!

SOLUTION: The class weighting is backwards. We should weight errors equally
in absolute terms, not inverse-frequency. Or remove class balancing entirely
since we care about spatial pattern, not class prediction.
""")

print("\n" + "=" * 70)
print("TESTING ALTERNATIVE: Uniform MSE (no class weighting)")
print("=" * 70)

uniform_loss = torch.mean((camkii_t2000 - target_pattern) ** 2)
print(f"\nUniform MSE (all cells weighted equally):")
print(f"  loss = mean((CaMKII - target)²)")
print(f"  loss = {uniform_loss.item():.4f}")

fraction_correct = num_feature_cells / total_cells
print(f"\nInterpretation:")
print(f"  {fraction_correct:.1%} of cells are correct (features at 1.0)")
print(f"  {1-fraction_correct:.1%} of cells are wrong (background at 1.0 instead of 0.0)")
print(f"  Average squared error = (1-{fraction_correct:.3f}) = {1-fraction_correct:.3f}")
print(f"\nThis makes more sense! Uniform saturation should have HIGH loss.")
