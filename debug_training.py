"""
Debug why training isn't working - trace through one iteration
"""

import torch

# Simulate the learning setup
dtype = torch.float32
device = 'cpu'

# Create learnable parameters (like in the real code)
k_on_raw = torch.randn(1, dtype=dtype).requires_grad_(True)
k_off_raw = torch.randn(1, dtype=dtype).requires_grad_(True)

# Apply sigmoid constraint
def apply_sigmoid_constraint(raw_param, min_val, max_val):
    return min_val + (max_val - min_val) * torch.sigmoid(raw_param)

k_on = apply_sigmoid_constraint(k_on_raw, 1.0, 50.0)
k_off = apply_sigmoid_constraint(k_off_raw, 0.01, 2.0)

print("=" * 70)
print("TRACING GRADIENT FLOW")
print("=" * 70)

print(f"\n1. Parameters:")
print(f"   k_on_raw = {k_on_raw.item():.4f}, requires_grad={k_on_raw.requires_grad}")
print(f"   k_on = {k_on.item():.4f}, requires_grad={k_on.requires_grad}, grad_fn={k_on.grad_fn}")

# Simulate dynamics (simplified)
K_half = torch.tensor(0.5, dtype=dtype)  # Fixed
vmem_signal = torch.tensor(0.8, dtype=dtype)
or_threshold = torch.tensor(0.5, dtype=dtype)
or_sharpness = torch.tensor(10.0, dtype=dtype)

print(f"\n2. Fixed inputs:")
print(f"   K_half = {K_half.item()}, requires_grad={K_half.requires_grad}")
print(f"   vmem_signal = {vmem_signal.item()}, requires_grad={vmem_signal.requires_grad}")

# Initialize CaMKII
CaMKII = torch.rand(5, 5, dtype=dtype) * 0.01
print(f"\n3. Initial CaMKII:")
print(f"   requires_grad={CaMKII.requires_grad}")
print(f"   This is the PROBLEM! CaMKII starts with no gradient tracking!")

# Run a few steps
dt = 0.01
for step in range(3):
    # Hill function
    CaMKII_sq = CaMKII * CaMKII
    K_half_sq = K_half * K_half
    self_activation = CaMKII_sq / (K_half_sq + CaMKII_sq)

    # Activation
    combined_signal = or_sharpness * (vmem_signal + self_activation - or_threshold)
    activation = torch.relu(combined_signal) / or_sharpness

    # Dynamics
    dCaMKII_dt = k_on * activation - k_off * CaMKII

    # Update
    CaMKII = CaMKII + dt * dCaMKII_dt
    CaMKII = torch.clamp(CaMKII, min=0.0, max=1.0)

    print(f"\n   Step {step}: CaMKII mean={CaMKII.mean().item():.4f}, requires_grad={CaMKII.requires_grad}, grad_fn={CaMKII.grad_fn}")

# Compute loss
target = torch.ones(5, 5, dtype=dtype) * 0.5
loss = ((CaMKII - target) ** 2).mean()

print(f"\n4. Loss:")
print(f"   loss = {loss.item():.4f}, requires_grad={loss.requires_grad}, grad_fn={loss.grad_fn}")

# Backprop
loss.backward()

print(f"\n5. After backward:")
print(f"   k_on_raw.grad = {k_on_raw.grad}")
print(f"   k_off_raw.grad = {k_off_raw.grad}")

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)

if k_on_raw.grad is None or k_on_raw.grad.abs().sum() < 1e-10:
    print("""
❌ GRADIENTS ARE ZERO/NONE!

Possible causes:
1. Initial CaMKII has requires_grad=False
   → torch.rand() creates leaf tensor with requires_grad=False by default
   → Gradients don't flow backward through leaf tensors

2. Long backprop path (2000 steps) might have numerical issues

3. Operations might be creating disconnected graph

SOLUTION: CaMKII doesn't need requires_grad=True because it's not a parameter.
The gradients should flow through k_on, k_off in the dCaMKII_dt computation.
Let me check if that's working...
""")
else:
    print(f"✅ Gradients are flowing! k_on_raw.grad = {k_on_raw.grad.item():.2e}")
