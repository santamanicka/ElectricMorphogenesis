#!/usr/bin/env python3
"""
CaMKII Bistability Visualization for Presentations

Visualizes the core concept: Ca²⁺ "drags and drops" CaMKII to states where
feature cells hover just above K_half and background cells just below it.

This creates an animated visualization showing:
1. The bistability landscape with K_half as the decision boundary
2. Ca²⁺ as a force that pushes CaMKII states
3. Feature cells pushed above K_half, background cells below
4. Pattern persistence after Ca²⁺ decays

Usage:
    python visualize_camkii_bistability.py
    python visualize_camkii_bistability.py --save_gif  # Save as animated GIF
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects
import argparse
from scipy.ndimage import zoom
from PIL import Image
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--save_gif', action='store_true', help='Save as animated GIF')
parser.add_argument('--save_mp4', action='store_true', help='Save as MP4 video')
parser.add_argument('--output', type=str, default='camkii_drag_drop', help='Output filename (without extension)')
args = parser.parse_args()

# ============================================================
# Parameters
# ============================================================
K_half = 0.5  # Decision boundary
num_feature_cells = 5
num_background_cells = 5
num_frames = 200

# Time phases
phase1_end = 50    # Ca²⁺ rises
phase2_end = 100   # Ca²⁺ at peak, dragging cells
phase3_end = 150   # Ca²⁺ decays
# phase4: t > 150, autonomous maintenance

# ============================================================
# Define face pattern using exact definition from learn_camkii_bistability.py
# ============================================================
def define_target_face_pattern(grid_size=11):
    """
    Define target face pattern from learn_camkii_bistability.py.
    Returns binary masks for facial features.
    """
    eye_mask = np.zeros((grid_size, grid_size), dtype=bool)
    nose_mask = np.zeros((grid_size, grid_size), dtype=bool)
    mouth_mask = np.zeros((grid_size, grid_size), dtype=bool)

    if grid_size == 11:
        # Eye indices: Two tiny square patches (2x2)
        left_eye_indices = [(2, 2), (2, 3), (3, 2), (3, 3)]
        right_eye_indices = [(2, 7), (2, 8), (3, 7), (3, 8)]

        for (row, col) in left_eye_indices + right_eye_indices:
            eye_mask[row, col] = True

        # Nose indices: Minimal vertical stripe at rows 3-5, col 5 (3 cells)
        nose_indices = [(3, 5), (4, 5), (5, 5)]
        for (row, col) in nose_indices:
            nose_mask[row, col] = True

        # Mouth indices: Narrow horizontal stripe at row 8, cols 3-7 (5 cells)
        mouth_indices = [(8, 3), (8, 4), (8, 5), (8, 6), (8, 7)]
        for (row, col) in mouth_indices:
            mouth_mask[row, col] = True

    return {'eye': eye_mask, 'nose': nose_mask, 'mouth': mouth_mask}

# Define face pattern
grid_size = 11
target_masks = define_target_face_pattern(grid_size)
face_pattern_small = (target_masks['eye'] | target_masks['nose'] | target_masks['mouth']).astype(float)

# ============================================================
# Create figure with three panels
# ============================================================
fig = plt.figure(figsize=(18, 6))
fig.patch.set_facecolor('#1a1a2e')

# Left panel: 3D face blocks
ax1 = fig.add_subplot(131, projection='3d')
ax1.set_facecolor('#1a1a2e')

# Middle panel: Energy landscape view
ax2 = fig.add_subplot(132)
ax2.set_facecolor('#1a1a2e')

# Right panel: Time series view
ax3 = fig.add_subplot(133)
ax3.set_facecolor('#1a1a2e')

# ============================================================
# Energy landscape function
# ============================================================
def energy_landscape(x, K=K_half):
    """
    Bistable energy landscape: two wells at x=0.1 (OFF) and x=1.1 (ON)
    with barrier at K_half
    """
    # Double-well potential with minima at 0.1 and 1.1 (where balls actually settle)
    return 2 * (x - 0.1) ** 2 * (x - 1.1) ** 2 - 0.1 * (x - K) ** 2

def self_activation(x, K=K_half):
    """Competitive self-activation: (x² - K²) / (x² + K²)"""
    return (x**2 - K**2) / (x**2 + K**2 + 1e-10)

# ============================================================
# Initialize cell states
# ============================================================
# Feature cells start low, will be pushed high
feature_states = np.ones(num_feature_cells) * 0.1 + np.random.randn(num_feature_cells) * 0.02
# Background cells start low, will stay low
background_states = np.ones(num_background_cells) * 0.1 + np.random.randn(num_background_cells) * 0.02

# Ca²⁺ levels (feature cells get high Ca, background gets low)
feature_ca_max = 0.9
background_ca_max = 0.2

# History for time series
history = {
    'feature': [feature_states.copy()],
    'background': [background_states.copy()],
    'ca_feature': [0.0],
    'ca_background': [0.0],
    'time': [0]
}

# ============================================================
# Simulation function
# ============================================================
def simulate_step(t, feature_states, background_states):
    """Simulate one time step"""
    # Adaptive dt that slows down dynamics to fill the full animation duration
    # Increase dt in later phases to speed up equilibration
    if t < phase1_end:
        dt = 0.02  # Slow rise
    elif t < phase2_end:
        dt = 0.03  # Moderate dragging
    elif t < phase3_end:
        dt = 0.025  # Slow release
    else:
        dt = 0.015  # Very slow equilibration to fill remaining time

    # Determine Ca²⁺ levels based on phase
    if t < phase1_end:
        # Ca rising
        progress = t / phase1_end
        ca_feature = feature_ca_max * progress
        ca_background = background_ca_max * progress
    elif t < phase2_end:
        # Ca at peak
        ca_feature = feature_ca_max
        ca_background = background_ca_max
    elif t < phase3_end:
        # Ca decaying
        progress = (t - phase2_end) / (phase3_end - phase2_end)
        ca_feature = feature_ca_max * (1 - progress) + 0.1 * progress
        ca_background = background_ca_max * (1 - progress) + 0.1 * progress
    else:
        # Ca at low baseline
        ca_feature = 0.1
        ca_background = 0.1

    # Ca²⁺ signal (sigmoid activation)
    ca_threshold = 0.3
    ca_sensitivity = 0.1

    def ca_signal(ca):
        return 1 / (1 + np.exp(-(ca - ca_threshold) / ca_sensitivity))

    # Update feature cells with smooth dynamics
    for i in range(len(feature_states)):
        x = feature_states[i]

        # Feature cells rise fast and consistently toward 1.0
        feat_target = 1.0

        if x < feat_target and t < phase3_end + 80:
            # Rising phase: Fast, consistent rise - extended duration
            rise_rate = 1.2 if t < phase3_end else 0.9
            dx = rise_rate * (feat_target - x) * dt * 2.5
        else:
            # Maintenance phase: slow equilibration
            sa = self_activation(x)
            ca_drive = ca_signal(ca_feature) * 2.0
            combined = ca_drive + sa - 0.5
            or_gate = 1 / (1 + np.exp(-combined * 5))
            dx = (or_gate * 1.0 - 0.1) * dt * 0.5

        feature_states[i] = np.clip(x + dx + np.random.randn() * 0.003, 0, 1.1)  # Allow slight overshoot

    # Update background cells - they rise initially but peak just below K_half, then shrink
    for i in range(len(background_states)):
        x = background_states[i]

        # Background cells have different dynamics:
        # - Rise more slowly up to just below K_half (K_half - 0.1)
        # - Peak during phase 2
        # - Then actively decay back down

        # Target for background: rise to just below K_half
        bg_target = K_half - 0.1

        if x < bg_target and t < phase2_end + 10:
            # Rising phase: Slower rise driven by Ca²⁺ (50% of feature rate)
            rise_rate = 0.5 if t < phase3_end else 0.3
            # Rise toward bg_target
            dx = rise_rate * (bg_target - x) * dt * 2.5
        else:
            # Decay phase: actively shrink after reaching peak
            # Gradual then strong decay to pull back down
            if t < phase3_end:
                decay_rate = 0.12  # Gentle decay during phase 2-3 transition
            else:
                decay_rate = 0.20  # Stronger decay in phase 4
            dx = -decay_rate * dt

        background_states[i] = np.clip(x + dx + np.random.randn() * 0.003, 0, 1)

    return feature_states, background_states, ca_feature, ca_background

# Pre-compute all frames
all_feature_states = [feature_states.copy()]
all_background_states = [background_states.copy()]
all_ca_feature = [0.0]
all_ca_background = [0.0]

current_feature = feature_states.copy()
current_background = background_states.copy()

for t in range(1, num_frames):
    current_feature, current_background, ca_f, ca_b = simulate_step(
        t, current_feature, current_background
    )
    all_feature_states.append(current_feature.copy())
    all_background_states.append(current_background.copy())
    all_ca_feature.append(ca_f)
    all_ca_background.append(ca_b)

# ============================================================
# Animation function
# ============================================================
def animate(frame):
    ax1.clear()
    ax2.clear()
    ax3.clear()

    ax1.set_facecolor('#1a1a2e')
    ax2.set_facecolor('#1a1a2e')
    ax3.set_facecolor('#1a1a2e')

    # Get current states
    feat_states = all_feature_states[frame]
    bg_states = all_background_states[frame]
    ca_f = all_ca_feature[frame]
    ca_b = all_ca_background[frame]

    # Determine phase for annotation
    if frame < phase1_end:
        phase_text = "Phase 1: Ca²⁺ Rising"
        phase_desc = "Bioelectric signal activates\nvoltage-gated Ca²⁺ channels"
    elif frame < phase2_end:
        phase_text = "Phase 2: Ca²⁺ Dragging"
        phase_desc = "High Ca²⁺ pushes feature cells\npast the K_half threshold"
    elif frame < phase3_end:
        phase_text = "Phase 3: Ca²⁺ Releasing"
        phase_desc = "Ca²⁺ decays, but cells maintain\ntheir positions via self-activation"
    else:
        phase_text = "Phase 4: Bistable Memory"
        phase_desc = "Pattern persists autonomously!\nCa²⁺ no longer needed"

    # ========== LEFT PANEL: 3D Face Blocks ==========
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    ax1.xaxis.pane.set_edgecolor('#30363d')
    ax1.yaxis.pane.set_edgecolor('#30363d')
    ax1.zaxis.pane.set_edgecolor('#30363d')
    ax1.grid(True, color='#30363d', alpha=0.3)

    # Compute average CaMKII activity for feature and background cells
    feat_mean_activity = np.mean(feat_states)
    bg_mean_activity = np.mean(bg_states)

    # Draw decision boundary plane at K_half
    xx, yy = np.meshgrid([0, grid_size], [0, grid_size])
    zz = np.ones_like(xx) * K_half * 2.0  # Scale to match block height scaling
    ax1.plot_surface(xx, yy, zz, alpha=0.6, color='#e74c3c', edgecolor='none')

    # Draw blocks for each cell in the grid
    for i in range(grid_size):
        for j in range(grid_size):
            is_feature = face_pattern_small[i, j] > 0.5

            # Height based on CaMKII activity (synced with balls rolling)
            if is_feature:
                # Feature blocks rise with feature cell CaMKII activity
                height = feat_mean_activity * 2.4  # Increased scaling to rise well above plane
                color = '#2ecc71'  # Green for feature
            else:
                # Background blocks stay low
                height = bg_mean_activity * 2.0  # Same scaling but lower activity
                color = '#3498db'  # Blue for background

            # Draw block as a cuboid
            x, y = j, i

            # Define vertices of the block
            vertices = [
                [x, y, 0], [x+1, y, 0], [x+1, y+1, 0], [x, y+1, 0],  # Bottom
                [x, y, height], [x+1, y, height], [x+1, y+1, height], [x, y+1, height]  # Top
            ]

            # Define the 6 faces of the cuboid
            faces = [
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
                [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
                [vertices[0], vertices[1], vertices[2], vertices[3]]   # Bottom
            ]

            # Create collection
            face_collection = Poly3DCollection(faces, alpha=0.8,
                                              facecolor=color,
                                              edgecolor='white',
                                              linewidth=0.5)
            ax1.add_collection3d(face_collection)

    ax1.set_xlim(0, grid_size)
    ax1.set_ylim(grid_size, 0)  # Inverted to flip face upside down
    ax1.set_zlim(0, 2.5)
    ax1.set_xlabel('X', color='white', fontsize=10)
    ax1.set_ylabel('Y', color='white', fontsize=10)
    ax1.set_zlabel('CaMKII Activity', color='white', fontsize=10)
    ax1.set_title('Face Pattern Formation', fontsize=14, fontweight='bold', color='white')
    ax1.tick_params(colors='white', labelsize=8)

    # Add text annotation for K_half plane
    ax1.text2D(0.05, 0.95, f'Red plane = K_half ({K_half})\nDecision boundary',
               transform=ax1.transAxes, color='#e74c3c', fontsize=9, fontweight='bold',
               verticalalignment='top')

    # Set fixed view angle: eyes at top, mouth at bottom
    # elev=30 looks down from above, azim=-60 shows from front-right
    ax1.view_init(elev=30, azim=-60)

    # ========== MIDDLE PANEL: Energy Landscape ==========
    x_range = np.linspace(-0.1, 1.2, 200)

    # Draw energy landscape
    y_energy = energy_landscape(x_range)
    y_energy_scaled = (y_energy - y_energy.min()) / (y_energy.max() - y_energy.min()) * 0.6
    ax2.fill_between(x_range, -0.1, y_energy_scaled, color='#16213e', alpha=0.8)
    ax2.plot(x_range, y_energy_scaled, color='#4a69bd', linewidth=2)

    # Draw K_half line
    ax2.axvline(K_half + 0.1, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8)
    ax2.text(K_half + 0.02, 0.85, 'K_half\n(Decision\nBoundary)',
             color='#e74c3c', fontsize=11, fontweight='bold',
             verticalalignment='top')

    # Label stable states
    ax2.text(0.05, 0.72, 'OFF\nState', color='#3498db', fontsize=12,
             fontweight='bold', ha='center')
    ax2.text(0.95, 0.72, 'ON\nState', color='#2ecc71', fontsize=12,
             fontweight='bold', ha='center')

    # Draw cells as balls on the landscape
    for i, x in enumerate(feat_states):
        y = energy_landscape(x)
        y_scaled = (y - y_energy.min()) / (y_energy.max() - y_energy.min()) * 0.6
        # Feature cells in green
        circle = Circle((x, y_scaled + 0.03), 0.025, color='#2ecc71',
                        ec='white', linewidth=1.5, zorder=10)
        ax2.add_patch(circle)

    for i, x in enumerate(bg_states):
        y = energy_landscape(x)
        y_scaled = (y - y_energy.min()) / (y_energy.max() - y_energy.min()) * 0.6
        # Background cells in blue
        circle = Circle((x, y_scaled + 0.03), 0.025, color='#3498db',
                        ec='white', linewidth=1.5, zorder=10)
        ax2.add_patch(circle)

    # Draw Ca²⁺ "hand" pushing cells (during active phases)
    if frame < phase3_end:
        # Arrow showing Ca force on feature cells
        if ca_f > 0.05:  # Lowered threshold so arrow appears early
            mean_feat = np.mean(feat_states)
            # Arrow positioned at y=0.65 (above the hill peak) so it's visible throughout motion
            ax2.annotate('', xy=(min(mean_feat + 0.15, 1.15), 0.65),
                        xytext=(mean_feat - 0.05, 0.65),
                        arrowprops=dict(arrowstyle='->', color='#f39c12', lw=3),
                        zorder=5)
            ax2.text(mean_feat + 0.05, 0.73, f'Ca²⁺\n({ca_f:.1f})',
                    color='#f39c12', fontsize=10, fontweight='bold', ha='center')

    ax2.set_xlim(-0.1, 1.2)
    ax2.set_ylim(-0.1, 1.0)
    ax2.set_xlabel('CaMKII Activity', fontsize=12, color='white')
    ax2.set_ylabel('Energy Landscape', fontsize=12, color='white')
    ax2.set_title('Bistability Landscape', fontsize=14, fontweight='bold', color='white')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('white')

    # Legend
    ax2.plot([], [], 'o', color='#2ecc71', markersize=10, label='Feature cells (high Vmem)')
    ax2.plot([], [], 'o', color='#3498db', markersize=10, label='Background cells (low Vmem)')
    ax2.legend(loc='upper left', fontsize=9, facecolor='#1a1a2e',
               edgecolor='white', labelcolor='white')

    # ========== RIGHT PANEL: Time Series ==========
    times = np.arange(frame + 1)

    # Plot Ca²⁺ levels
    ax3.fill_between(times, 0, all_ca_feature[:frame+1],
                     color='#f39c12', alpha=0.3, label='Ca²⁺ (feature)')
    ax3.plot(times, all_ca_feature[:frame+1], color='#f39c12', linewidth=2)

    ax3.fill_between(times, 0, all_ca_background[:frame+1],
                     color='#9b59b6', alpha=0.3, label='Ca²⁺ (background)')
    ax3.plot(times, all_ca_background[:frame+1], color='#9b59b6', linewidth=2)

    # Plot CaMKII states
    feat_mean = [np.mean(s) for s in all_feature_states[:frame+1]]
    feat_std = [np.std(s) for s in all_feature_states[:frame+1]]
    bg_mean = [np.mean(s) for s in all_background_states[:frame+1]]
    bg_std = [np.std(s) for s in all_background_states[:frame+1]]

    ax3.fill_between(times, np.array(feat_mean) - np.array(feat_std),
                     np.array(feat_mean) + np.array(feat_std),
                     color='#2ecc71', alpha=0.3)
    ax3.plot(times, feat_mean, color='#2ecc71', linewidth=2.5,
             label='CaMKII (feature)')

    ax3.fill_between(times, np.array(bg_mean) - np.array(bg_std),
                     np.array(bg_mean) + np.array(bg_std),
                     color='#3498db', alpha=0.3)
    ax3.plot(times, bg_mean, color='#3498db', linewidth=2.5,
             label='CaMKII (background)')

    # Draw K_half line
    ax3.axhline(K_half, color='#e74c3c', linestyle='--', linewidth=2, alpha=0.8)
    ax3.text(num_frames - 5, K_half + 0.03, 'K_half', color='#e74c3c',
             fontsize=11, fontweight='bold', ha='right')

    # Phase markers
    ax3.axvline(phase1_end, color='white', linestyle=':', alpha=0.5)
    ax3.axvline(phase2_end, color='white', linestyle=':', alpha=0.5)
    ax3.axvline(phase3_end, color='white', linestyle=':', alpha=0.5)

    ax3.set_xlim(0, num_frames)
    ax3.set_ylim(-0.05, 1.2)
    ax3.set_xlabel('Time', fontsize=12, color='white')
    ax3.set_ylabel('Activity Level', fontsize=12, color='white')
    ax3.set_title('Temporal Dynamics', fontsize=14, fontweight='bold', color='white')
    ax3.tick_params(colors='white')
    for spine in ax3.spines.values():
        spine.set_color('white')

    ax3.legend(loc='upper right', fontsize=9, facecolor='#1a1a2e',
               edgecolor='white', labelcolor='white')

    # Phase annotation box
    props = dict(boxstyle='round,pad=0.5', facecolor='#16213e',
                 edgecolor='#4a69bd', alpha=0.9)
    ax3.text(0.02, 0.98, f'{phase_text}\n\n{phase_desc}',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             color='white', bbox=props)

    # Main title
    fig.suptitle('Ca²⁺ "Drags and Drops" CaMKII into Bistable States → Face Pattern Emerges',
                 fontsize=16, fontweight='bold', color='white', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return []

# ============================================================
# Create animation
# ============================================================
print("Creating animation...")
anim = animation.FuncAnimation(fig, animate, frames=num_frames,
                                interval=50, blit=False)

if args.save_gif:
    output_file = f'{args.output}.gif'
    print(f"Saving animation as {output_file}...")
    anim.save(output_file, writer='pillow', fps=20, dpi=100)
    print(f"Saved: {output_file}")
elif args.save_mp4:
    output_file = f'{args.output}.mp4'
    print(f"Saving animation as {output_file}...")
    anim.save(output_file, writer='ffmpeg', fps=20, dpi=150)
    print(f"Saved: {output_file}")
else:
    print("Showing animation (close window to exit)...")
    print("Tip: Use --save_gif or --save_mp4 to save the animation")
    plt.show()

print("\nVisualization complete!")
print("\nKey concept illustrated:")
print("  1. Ca²⁺ rises in feature cells (high Vmem regions)")
print("  2. Ca²⁺ 'drags' feature cells past K_half into ON stable state")
print("  3. Background cells (low Ca²⁺) remain below K_half in OFF state")
print("  4. When Ca²⁺ decays, cells stay in their respective stable states")
print("  5. Pattern persists via competitive self-activation (bistable memory)")
