#!/usr/bin/env python3
"""
Interactive FacialGRN simulation - closer to the HTML experience
Allows step-by-step execution and parameter adjustment
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button, Slider
from geneRegulatoryNetwork import FacialGRN


class InteractiveFacialGRN:
    """Interactive visualization of FacialGRN similar to HTML interface"""

    def __init__(self):
        self.grn = FacialGRN(grid_size=40, device='cpu')

        # Setup figure and axes
        self.fig = plt.figure(figsize=(20, 12))
        self.setup_plots()
        self.setup_controls()

        # Initial draw
        self.update_all_plots()

    def setup_plots(self):
        """Setup all visualization axes"""
        gs = GridSpec(3, 4, figure=self.fig, hspace=0.35, wspace=0.3,
                     left=0.05, right=0.95, top=0.92, bottom=0.15)

        # Morphogen plots
        self.ax_shh = self.fig.add_subplot(gs[0, 0])
        self.ax_fgf8 = self.fig.add_subplot(gs[0, 1])
        self.ax_edn1 = self.fig.add_subplot(gs[0, 2])
        self.ax_combined = self.fig.add_subplot(gs[0, 3])

        # Gene expression plots
        self.ax_pax6 = self.fig.add_subplot(gs[1, 0])
        self.ax_lhx2 = self.fig.add_subplot(gs[1, 1])
        self.ax_alx = self.fig.add_subplot(gs[1, 2])
        self.ax_dlx = self.fig.add_subplot(gs[1, 3])

        # Features and profiles
        self.ax_features = self.fig.add_subplot(gs[2, 0:2])
        self.ax_profiles = self.fig.add_subplot(gs[2, 2:])

        # Initialize image objects
        self.images = {}
        self.setup_image_objects()

    def setup_image_objects(self):
        """Initialize all image objects"""
        state = self.grn.get_state()
        gs = self.grn.grid_size

        # Morphogens
        self.images['shh'] = self.ax_shh.imshow(state['morphogens']['shh'].cpu().numpy(),
                                                cmap='Reds', aspect='auto', origin='upper',
                                                vmin=0, vmax=1)
        self.ax_shh.set_title('Shh Gradient (Medial)', fontweight='bold')
        plt.colorbar(self.images['shh'], ax=self.ax_shh, fraction=0.046)

        self.images['fgf8'] = self.ax_fgf8.imshow(state['morphogens']['fgf8'].cpu().numpy(),
                                                  cmap='cool', aspect='auto', origin='upper',
                                                  vmin=0, vmax=1)
        self.ax_fgf8.set_title('Fgf8 Gradient (Lateral)', fontweight='bold')
        plt.colorbar(self.images['fgf8'], ax=self.ax_fgf8, fraction=0.046)

        self.images['edn1'] = self.ax_edn1.imshow(state['morphogens']['edn1'].cpu().numpy(),
                                                  cmap='YlOrBr', aspect='auto', origin='upper',
                                                  vmin=0, vmax=1)
        self.ax_edn1.set_title('Edn1 Gradient (Posterior)', fontweight='bold')
        plt.colorbar(self.images['edn1'], ax=self.ax_edn1, fraction=0.046)

        # Combined morphogens (RGB)
        rgb = self.create_rgb_image(state)
        self.images['combined'] = self.ax_combined.imshow(rgb, aspect='auto', origin='upper')
        self.ax_combined.set_title('Combined Morphogens', fontweight='bold')

        # Genes
        self.images['pax6'] = self.ax_pax6.imshow(state['genes']['pax6'].cpu().numpy(),
                                                  cmap='viridis', aspect='auto', origin='upper',
                                                  vmin=0, vmax=1)
        self.ax_pax6.set_title('Pax6 (Eye)', fontweight='bold')
        plt.colorbar(self.images['pax6'], ax=self.ax_pax6, fraction=0.046)

        self.images['lhx2'] = self.ax_lhx2.imshow(state['genes']['lhx2'].cpu().numpy(),
                                                  cmap='viridis', aspect='auto', origin='upper',
                                                  vmin=0, vmax=1)
        self.ax_lhx2.set_title('Lhx2 (Eye)', fontweight='bold')
        plt.colorbar(self.images['lhx2'], ax=self.ax_lhx2, fraction=0.046)

        self.images['alx'] = self.ax_alx.imshow(state['genes']['alx'].cpu().numpy(),
                                                cmap='viridis', aspect='auto', origin='upper',
                                                vmin=0, vmax=1)
        self.ax_alx.set_title('Alx (Nose)', fontweight='bold')
        plt.colorbar(self.images['alx'], ax=self.ax_alx, fraction=0.046)

        self.images['dlx'] = self.ax_dlx.imshow(state['genes']['dlx'].cpu().numpy(),
                                                cmap='viridis', aspect='auto', origin='upper',
                                                vmin=0, vmax=1)
        self.ax_dlx.set_title('Dlx (Jaw)', fontweight='bold')
        plt.colorbar(self.images['dlx'], ax=self.ax_dlx, fraction=0.046)

        # Features
        colors = ['#ecf0f1', '#9b59b6', '#e74c3c', '#f39c12']
        feature_cmap = ListedColormap(colors)
        self.images['features'] = self.ax_features.imshow(state['features'].cpu().numpy(),
                                                          cmap=feature_cmap, aspect='auto',
                                                          origin='upper', vmin=0, vmax=3)
        self.ax_features.set_title('Facial Features Pattern', fontweight='bold', fontsize=14)

        # Add legend for features
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ecf0f1', label='Undifferentiated'),
            Patch(facecolor='#9b59b6', label='Eye'),
            Patch(facecolor='#e74c3c', label='Nose'),
            Patch(facecolor='#f39c12', label='Jaw')
        ]
        self.ax_features.legend(handles=legend_elements, loc='upper right')

    def create_rgb_image(self, state):
        """Create RGB composite of morphogens"""
        gs = self.grn.grid_size
        shh = state['morphogens']['shh'].cpu().numpy()
        fgf8 = state['morphogens']['fgf8'].cpu().numpy()
        edn1 = state['morphogens']['edn1'].cpu().numpy()

        rgb = np.zeros((gs, gs, 3))
        rgb[:, :, 0] = shh + edn1
        rgb[:, :, 1] = fgf8 * 0.7 + edn1 * 0.6
        rgb[:, :, 2] = fgf8 * 0.8
        return np.clip(rgb, 0, 1)

    def update_profiles(self):
        """Update gene expression profiles"""
        self.ax_profiles.clear()

        state = self.grn.get_state()
        mid_x = self.grn.grid_size // 2

        profiles = {
            'Shh': (state['morphogens']['shh'][:, mid_x].cpu().numpy(), '#FF0000', '-', 3),
            'Fgf8': (state['morphogens']['fgf8'][:, mid_x].cpu().numpy(), '#00FFFF', '-', 3),
            'Edn1': (state['morphogens']['edn1'][:, mid_x].cpu().numpy(), '#FF00FF', '-', 3),
            'Pax6': (state['genes']['pax6'][:, mid_x].cpu().numpy(), '#00FF00', '--', 2),
            'Alx': (state['genes']['alx'][:, mid_x].cpu().numpy(), '#FFFF00', '--', 2),
            'Hand2': (state['genes']['hand2'][:, mid_x].cpu().numpy(), '#0000FF', '--', 2),
        }

        for name, (data, color, style, width) in profiles.items():
            self.ax_profiles.plot(data, label=name, color=color,
                                linestyle=style, linewidth=width)

        self.ax_profiles.set_xlabel('Anterior → Posterior', fontweight='bold')
        self.ax_profiles.set_ylabel('Expression Level', fontweight='bold')
        self.ax_profiles.set_title('Gene Expression Profiles (Midline)',
                                   fontweight='bold', fontsize=14)
        self.ax_profiles.legend(loc='upper right')
        self.ax_profiles.grid(True, alpha=0.3)
        self.ax_profiles.set_ylim([0, 1.1])

    def update_all_plots(self):
        """Update all visualization elements"""
        state = self.grn.get_state()

        # Update images
        self.images['shh'].set_array(state['morphogens']['shh'].cpu().numpy())
        self.images['fgf8'].set_array(state['morphogens']['fgf8'].cpu().numpy())
        self.images['edn1'].set_array(state['morphogens']['edn1'].cpu().numpy())
        self.images['combined'].set_array(self.create_rgb_image(state))
        self.images['pax6'].set_array(state['genes']['pax6'].cpu().numpy())
        self.images['lhx2'].set_array(state['genes']['lhx2'].cpu().numpy())
        self.images['alx'].set_array(state['genes']['alx'].cpu().numpy())
        self.images['dlx'].set_array(state['genes']['dlx'].cpu().numpy())
        self.images['features'].set_array(state['features'].cpu().numpy())

        # Update profiles
        self.update_profiles()

        # Update title
        self.fig.suptitle(f'Craniofacial Patterning Model - Time Step: {state["time"]}',
                         fontsize=16, fontweight='bold')

        self.fig.canvas.draw_idle()

    def setup_controls(self):
        """Setup interactive controls"""
        # Button axes
        ax_step = plt.axes([0.15, 0.05, 0.1, 0.04])
        ax_run = plt.axes([0.3, 0.05, 0.1, 0.04])
        ax_reset = plt.axes([0.45, 0.05, 0.1, 0.04])

        # Slider axes
        ax_shh_slider = plt.axes([0.65, 0.08, 0.25, 0.02])
        ax_fgf8_slider = plt.axes([0.65, 0.05, 0.25, 0.02])
        ax_edn1_slider = plt.axes([0.65, 0.02, 0.25, 0.02])

        # Create buttons
        self.btn_step = Button(ax_step, 'Step Forward')
        self.btn_run = Button(ax_run, 'Run 100 Steps')
        self.btn_reset = Button(ax_reset, 'Reset')

        # Create sliders
        self.slider_shh = Slider(ax_shh_slider, 'Shh', 0.0, 2.0, valinit=1.0)
        self.slider_fgf8 = Slider(ax_fgf8_slider, 'Fgf8', 0.0, 2.0, valinit=1.0)
        self.slider_edn1 = Slider(ax_edn1_slider, 'Edn1', 0.0, 2.0, valinit=1.0)

        # Connect callbacks
        self.btn_step.on_clicked(self.on_step)
        self.btn_run.on_clicked(self.on_run)
        self.btn_reset.on_clicked(self.on_reset)
        self.slider_shh.on_changed(self.on_slider_change)
        self.slider_fgf8.on_changed(self.on_slider_change)
        self.slider_edn1.on_changed(self.on_slider_change)

    def on_step(self, event):
        """Handle step button click"""
        self.grn.update_state()
        self.update_all_plots()

    def on_run(self, event):
        """Handle run button click"""
        print("Running 100 steps...")
        self.grn.simulate(num_steps=100)
        self.update_all_plots()
        print("Done!")

    def on_reset(self, event):
        """Handle reset button click"""
        print("Resetting simulation...")
        self.grn.reset()
        # Reset sliders
        self.slider_shh.set_val(1.0)
        self.slider_fgf8.set_val(1.0)
        self.slider_edn1.set_val(1.0)
        self.update_all_plots()
        print("Reset complete!")

    def on_slider_change(self, val):
        """Handle slider value changes"""
        self.grn.set_parameters(
            shhStrength=self.slider_shh.val,
            fgf8Strength=self.slider_fgf8.val,
            edn1Strength=self.slider_edn1.val
        )
        self.update_all_plots()

    def show(self):
        """Display the interactive plot"""
        plt.show()


def main():
    """Main function to run interactive visualization"""
    print("="*60)
    print("Interactive FacialGRN Visualization")
    print("="*60)
    print("\nControls:")
    print("  - Step Forward: Advance simulation by 1 step")
    print("  - Run 100 Steps: Run simulation for 100 steps")
    print("  - Reset: Reset to initial conditions")
    print("  - Sliders: Adjust morphogen strengths (resets simulation)")
    print("\nClose the window to exit.")
    print("="*60)

    app = InteractiveFacialGRN()
    app.show()


if __name__ == '__main__':
    main()
