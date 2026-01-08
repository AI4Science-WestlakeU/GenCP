import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import torch
import os
from typing import Optional, Tuple, List
import matplotlib.colors as mcolors
from datetime import datetime


class FluidFieldVisualizer:
    """Fluid field visualizer."""
    
    def __init__(self, args, grid_size = [64, 64], save_dir: str = "visualization_results", create_timestamp_folder: bool = True):
        """
        Initialize visualizer.
        
        Args:
            grid_size: Grid size (default 64x64)
            save_dir: Save directory
            create_timestamp_folder: Whether to create timestamp folder
        """
        self.grid_size = grid_size
        self.args = args
        
        if create_timestamp_folder:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.save_dir = os.path.join(save_dir, timestamp)
        else:
            self.save_dir = save_dir
            
        os.makedirs(self.save_dir, exist_ok=True)
    
    def visualize_field(self, 
                               u_pred: np.ndarray, 
                               u_true: np.ndarray = None,
                               title: str = "Velocity Field",
                               save_name: str = "velocity_field.png",
                               cmap: str = 'RdBu_r',
                               colorbar: bool = True,
                               vmin: Optional[float] = None,
                               vmax: Optional[float] = None,
                                use_interpolation: str = 'bilinear',
                                interpolation_factor: int = 4,
                                smooth_sigma: float = 0.8) -> None:
        
        # Compute global vmin and vmax to ensure reasonable colorbar range
        if vmin is None:
            vmin = min(u_pred.min(), u_true.min())
        if vmax is None:
            vmax = max(u_pred.max(), u_true.max())
        
        # Add offset if vmin and vmax are the same to avoid colorbar issues
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
        
        # Create three subplots: predicted, ground truth, error
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        extent = [0, self.grid_size[1]-1, 0, self.grid_size[0]-1]
        im1 = axes[0].imshow(
                u_pred,
                cmap=cmap, 
                alpha=0.8, 
                vmin=vmin, 
                vmax=vmax,
                extent=extent,
                origin='lower',
                interpolation=use_interpolation
            )
        if colorbar:
            cbar1 = plt.colorbar(im1, ax=axes[0])
        axes[0].set_title('Predicted', fontsize=12, fontweight='bold')
        axes[0].set_aspect('equal')
        
        im2 = axes[1].imshow(
                u_true,
                cmap=cmap, 
                alpha=0.8, 
                vmin=vmin, 
                vmax=vmax,
                extent=extent,
                origin='lower',
                interpolation=use_interpolation
            )
        if colorbar:
            cbar2 = plt.colorbar(im2, ax=axes[1])
        axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1].set_aspect('equal')
        
        # Compute error
        error = np.abs(u_pred - u_true)
        im3 = axes[2].imshow(
                error,
                cmap='Reds', 
                alpha=0.8, 
                vmin=error.min(), 
                vmax=error.max(),
                extent=extent,
                origin='lower',
                interpolation=use_interpolation
            )
        # axes[2].contour(X, Y, error, levels=20, colors='black', alpha=0.3, linewidths=0.5)
        if colorbar:
            cbar3 = plt.colorbar(im3, ax=axes[2])
            # cbar3.set_label('Error', fontsize=12)
        axes[2].set_title('Error', fontsize=12, fontweight='bold')
        axes[2].set_aspect('equal')
        # axes[2].grid(False, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        # plt.show()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Velocity contour image saved to: {save_path}")

    def visualize_time_series_gif(self, 
                                 u_pred: np.ndarray, 
                                 u_true: np.ndarray = None,
                                 title: str = "Time Series Animation",
                                 save_name: str = "time_series.gif",
                                 fps: int = 2,
                                 show_colorbar: bool = False,
                                 vmin: Optional[float] = None,
                                 vmax: Optional[float] = None) -> None:
        """
        Generate time series GIF animation.
        
        Args:
            u_pred: Predicted velocity field data (time_steps, grid_size, grid_size)
            u_true: Ground truth velocity field data (time_steps, grid_size, grid_size)
            title: Image title
            save_name: Save filename
            fps: Frame rate (frames per second)
            show_colorbar: Whether to show colorbar
            vmin: Minimum value, auto-computed from full time trajectory if None
            vmax: Maximum value, auto-computed from full time trajectory if None
        """
        import matplotlib.animation as animation

        # Compute vmin and vmax for full time trajectory to ensure reasonable colorbar range
        if vmin is None:
            vmin = min(u_pred.min(), u_true.min())
        if vmax is None:
            vmax = max(u_pred.max(), u_true.max())
        
        # Add offset if vmin and vmax are the same to avoid colorbar issues
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
        
        time_steps = u_pred.shape[0]
        
        # Compute global error min/max for consistent colorbar
        error_data = np.abs(u_pred - u_true)
        error_vmin, error_vmax = error_data.min(), error_data.max()
        
        # Set reasonable range if error range is too small
        if error_vmin == error_vmax:
            error_vmin = 0
            error_vmax = max(0.1, error_vmax + 0.1)
        
        # Create subplots, reserve space for colorbar
        if show_colorbar:
            fig = plt.figure(figsize=(20, 6))
            gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], 
                                 left=0.05, right=0.85, wspace=0.3)
            axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        extent = [0, self.grid_size[1]-1, 0, self.grid_size[0]-1]

        # Initialize image objects for subsequent updates
        im1 = axes[0].imshow(u_pred[0], cmap='RdBu_r', vmin=vmin, vmax=vmax, alpha=0.8)
        im2 = axes[1].imshow(u_true[0], cmap='RdBu_r', vmin=vmin, vmax=vmax, alpha=0.8)
        im3 = axes[2].imshow(error_data[0], cmap='Reds', vmin=error_vmin, vmax=error_vmax, alpha=0.8,
                                extent=extent,
                                origin='lower',
                                interpolation='bilinear')
        
        # Set initial titles and format
        axes[0].set_title('Predicted (t=0)', fontsize=12, fontweight='bold')
        axes[1].set_title('Ground Truth (t=0)', fontsize=12, fontweight='bold')
        axes[2].set_title('Error (t=0)', fontsize=12, fontweight='bold')
        
        for ax in axes:
            ax.set_aspect('equal')
        
        # Add colorbars (only once, based on full time trajectory range)
        if show_colorbar:
            cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8, pad=0.02)
            cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8, pad=0.02)
            cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8, pad=0.02)
            
            cbar1.set_label('Predicted Value', rotation=270, labelpad=15)
            cbar2.set_label('True Value', rotation=270, labelpad=15)
            cbar3.set_label('Absolute Error', rotation=270, labelpad=15)
        
        def animate(frame):
            # Update image data instead of clearing axes
            im1.set_array(u_pred[frame])
            im2.set_array(u_true[frame])
            
            # Compute error for current frame
            error_frame = np.abs(u_pred[frame] - u_true[frame])
            im3.set_array(error_frame)
            
            # Update titles
            axes[0].set_title(f'Predicted (t={frame})', fontsize=12, fontweight='bold')
            axes[1].set_title(f'Ground Truth (t={frame})', fontsize=12, fontweight='bold')
            axes[2].set_title(f'Error (t={frame})', fontsize=12, fontweight='bold')
            
            return [im1, im2, im3]
        
        ani = animation.FuncAnimation(fig, animate, frames=time_steps, 
                                     interval=1000//fps, blit=False, repeat=True)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.save_dir, save_name)
        ani.save(save_path, writer='pillow', fps=fps)
        plt.close()
        print(f"Time series GIF saved to: {save_path}")

    def visualize_sdf_field(self, 
                                   sdf_pred: np.ndarray, 
                                   sdf_true: np.ndarray = None,
                                   title: str = "SDF Contour Field",
                                   save_name: str = "sdf_contour_field.png",
                                   cmap: str = 'RdBu_r',
                                   colorbar: bool = True,
                                   vmin: Optional[float] = None,
                                   vmax: Optional[float] = None,
                                   index: Optional[int] = None,
                                   use_interpolation: str = 'bilinear', # bilinear  
                                   zero_tolerance: float = 1e-6) -> None:
        """
        Visualize SDF field
        
        Args:
            sdf_pred: Predicted SDF field (grid_size, grid_size)
            sdf_true: Ground truth SDF field (grid_size, grid_size)
            title: Image title
            save_name: Save filename
            cmap: Color map
            colorbar: Whether to show colorbar
            vmin: Colorbar minimum value, auto-computed if None
            vmax: Colorbar maximum value, auto-computed if None
            index: Index number, added to filename if provided
            zero_tolerance: Tolerance for SDF=0 (currently unused)
        """
        if vmin is None:
            vmin = min(sdf_pred.min(), sdf_true.min())
        if vmax is None:
            vmax = max(sdf_pred.max(), sdf_true.max())
        
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        extent = [0, self.grid_size[1]-1, 0, self.grid_size[0]-1]
        
        import matplotlib.colors as mcolors
        levels = np.linspace(vmin, vmax, 21)
        norm = mcolors.BoundaryNorm(levels, ncolors=cmap if isinstance(cmap, int) else plt.cm.get_cmap(cmap).N)
        
        im1 = axes[0].imshow(
            sdf_pred, 
            cmap=cmap, 
            alpha=0.8, 
            extent=extent,
            origin='lower',
            interpolation=use_interpolation
        )
        
        solid_mask_pred = (sdf_pred < self.args.sdf_threshold)

        if solid_mask_pred.any():
            axes[0].contour(
                solid_mask_pred.astype(float), 
                levels=[0.5], 
                colors='yellow', 
                linewidths=1,
                origin='lower'
            )
        
        if colorbar:
            cbar1 = plt.colorbar(im1, ax=axes[0])
        axes[0].set_title('Predicted SDF', fontsize=12, fontweight='bold')
        axes[0].set_aspect('equal')
        
        im2 = axes[1].imshow(
            sdf_true, 
            cmap=cmap, 
            alpha=0.8, 
            extent=extent,
            origin='lower',
            interpolation=use_interpolation
        )
        
        solid_mask_true = (sdf_true < self.args.sdf_threshold)
        
        if solid_mask_true.any():
            axes[1].contour(
                solid_mask_true.astype(float), 
                levels=[0.5], 
                colors='yellow', 
                linewidths=1,
                origin='lower'
            )
        
        if colorbar:
            cbar2 = plt.colorbar(im2, ax=axes[1])
        axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1].set_aspect('equal')
        
        error = np.abs(sdf_pred - sdf_true)
        im3 = axes[2].imshow(
            error, 
            cmap='Reds', 
            alpha=0.8,
            extent=extent,
            origin='lower',
            interpolation=use_interpolation
        )
        if colorbar:
            cbar3 = plt.colorbar(im3, ax=axes[2])
        axes[2].set_title('Absolute Error', fontsize=12, fontweight='bold')
        axes[2].set_aspect('equal')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if index is not None:
            name, ext = os.path.splitext(save_name)
            save_name = f"{name}_{index}{ext}"
        
        save_path = os.path.join(self.save_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"SDF contour field image saved to: {save_path}")

    def visualize_sdf_time_series_gif(self, 
                                     sdf_pred: np.ndarray, 
                                     sdf_true: np.ndarray = None,
                                     title: str = "SDF Time Series Animation",
                                     save_name: str = "sdf_time_series.gif",
                                     fps: int = 2,
                                     cmap: str = 'RdBu_r',
                                     vmin: Optional[float] = None,
                                     vmax: Optional[float] = None,
                                     index: Optional[int] = None,
                                     show_colorbar: bool = True) -> None:
        """
        Generate SDF field time series GIF animation
        
        Args:
            sdf_pred: Predicted SDF field data (time_steps, grid_size, grid_size)
            sdf_true: Ground truth SDF field data (time_steps, grid_size, grid_size)
            title: Image title
            save_name: Save filename
            fps: Frame rate (frames per second)
            cmap: Color map
            vmin: Colorbar minimum value, auto-computed if None
            vmax: Colorbar maximum value, auto-computed if None
            index: Index number, added to filename if provided
            show_colorbar: Whether to show colorbar
        """
        import matplotlib.animation as animation
        
        if sdf_true is None:
            sdf_true = np.zeros_like(sdf_pred)
        
        time_steps = sdf_pred.shape[0]
        
        if vmin is None:
            vmin = min(sdf_pred.min(), sdf_true.min())
        if vmax is None:
            vmax = max(sdf_pred.max(), sdf_true.max())
        
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
        
        error_data = np.abs(sdf_pred - sdf_true)
        error_vmin, error_vmax = error_data.min(), error_data.max()
        
        import matplotlib.colors as mcolors
        levels = np.linspace(vmin, vmax, 21)
        norm = mcolors.BoundaryNorm(levels, ncolors=cmap if isinstance(cmap, int) else plt.cm.get_cmap(cmap).N)
        
        error_levels = np.linspace(error_vmin, error_vmax, 21)
        error_norm = mcolors.BoundaryNorm(error_levels, ncolors=plt.cm.get_cmap('Reds').N)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        if show_colorbar:
            temp_im1 = axes[0].imshow(sdf_pred[0], cmap=cmap, 
                                       norm=norm, alpha=0.8, origin='lower')
            temp_im2 = axes[1].imshow(sdf_true[0], cmap=cmap,
                                       norm=norm, alpha=0.8, origin='lower')
            temp_im3 = axes[2].imshow(error_data[0], cmap='Reds',
                                       norm=error_norm, alpha=0.8, origin='lower')
            
            plt.colorbar(temp_im1, ax=axes[0], shrink=0.8)
            plt.colorbar(temp_im2, ax=axes[1], shrink=0.8)
            plt.colorbar(temp_im3, ax=axes[2], shrink=0.8)
            
            for ax in axes:
                ax.clear()
        
        def animate(frame):
            for ax in axes:
                ax.clear()
            
            im1 = axes[0].imshow(
                sdf_pred[frame], 
                cmap=cmap, 
                alpha=0.8, 
                norm=norm,
                origin='lower'
            )
            
            solid_mask_pred = sdf_pred[frame] < self.args.sdf_threshold

            if solid_mask_pred.any():
                axes[0].contour(
                    solid_mask_pred.astype(float), 
                    levels=[0.5], 
                    colors='yellow', 
                    linewidths=1,
                    origin='lower'
                )
            
            axes[0].set_title(f'Predicted SDF (t={frame})', fontsize=12, fontweight='bold')
            axes[0].set_aspect('equal')
            
            im2 = axes[1].imshow(
                sdf_true[frame], 
                cmap=cmap, 
                alpha=0.8, 
                norm=norm,
                origin='lower'
            )
            
            solid_mask_true = sdf_true[frame] < self.args.sdf_threshold
            
            if solid_mask_true.any():
                axes[1].contour(
                    solid_mask_true.astype(float), 
                    levels=[0.5], 
                    colors='yellow', 
                    linewidths=1,
                    origin='lower'
                )
            
            axes[1].set_title(f'Ground Truth SDF (t={frame})', fontsize=12, fontweight='bold')
            axes[1].set_aspect('equal')
            
            error = np.abs(sdf_pred[frame] - sdf_true[frame])
            im3 = axes[2].imshow(
                error, 
                cmap='Reds', 
                alpha=0.8,
                norm=error_norm,
                origin='lower'
            )
            
            axes[2].set_title(f'Absolute Error (t={frame})', fontsize=12, fontweight='bold')
            axes[2].set_aspect('equal')
            
            return [im1, im2, im3]
        
        ani = animation.FuncAnimation(fig, animate, frames=time_steps, 
                                     interval=1000//fps, blit=False, repeat=True)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if index is not None:
            name, ext = os.path.splitext(save_name)
            save_name = f"{name}_{index}{ext}"
        
        save_path = os.path.join(self.save_dir, save_name)
        ani.save(save_path, writer='pillow', fps=fps)
        plt.close()
        print(f"SDF time series GIF saved to: {save_path}")



def generate_test_fluid_data(grid_size: int = 64, time_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test fluid data
    
    Args:
        grid_size: Grid size
        time_steps: Number of time steps
    
    Returns:
        velocity_data: Velocity field data (time_steps, 2, grid_size, grid_size)
        pressure_data: Pressure field data (time_steps, grid_size, grid_size)
    """
    x = np.linspace(0, 2*np.pi, grid_size)
    y = np.linspace(0, 2*np.pi, grid_size)
    X, Y = np.meshgrid(x, y)
    
    velocity_data = np.zeros((time_steps, 2, grid_size, grid_size))
    pressure_data = np.zeros((time_steps, grid_size, grid_size))
    
    for t in range(time_steps):
        time_factor = 1.0 + 0.1 * t
        
        u = -np.sin(Y) * np.cos(X) * time_factor
        v = np.cos(Y) * np.sin(X) * time_factor
        
        noise_u = 0.1 * np.random.randn(grid_size, grid_size)
        noise_v = 0.1 * np.random.randn(grid_size, grid_size)
        
        velocity_data[t, 0] = u + noise_u
        velocity_data[t, 1] = v + noise_v
        
        pressure = -0.5 * (u**2 + v**2) + 0.1 * np.sin(X + Y) * time_factor
        pressure_data[t] = pressure
    
    return velocity_data, pressure_data


if __name__ == "__main__":
    print("Starting test data generation and visualization...")
    
    visualizer = FluidFieldVisualizer(grid_size=[64,64], save_dir="/path/to/visualization_results", create_timestamp_folder=False)
    
    velocity_data, pressure_data = generate_test_fluid_data(grid_size=64, time_steps=10)
    
    print(f"Generated data shape: velocity field {velocity_data.shape}, pressure field {pressure_data.shape}")

    print("Generating velocity field plots...")
    visualizer.visualize_field(
        u_pred=velocity_data[0, 0], 
        u_true=None, 
        title="Test Velocity Field (t=0)",
        save_name="test_velocity_field_x.png"
    )
    visualizer.visualize_field(
        u_pred=velocity_data[0, 1], 
        u_true=None, 
        title="Test Velocity Field (t=0)",
        save_name="test_velocity_field_y.png"
    )
    
    print("Generating pressure field plots...")
    visualizer.visualize_field(
        u_pred=pressure_data[0], 
        u_true=None, 
        title="Test Pressure Field (t=0)",
        save_name="test_pressure_field.png"
    )
    
    print("Generating time series GIF animation...")
    visualizer.visualize_time_series_gif(
        u_pred=velocity_data[:,0],
        u_true=None,
        title="Time Series Animation Test",
        save_name="time_series_animation.gif",
        fps=2
    )

    print(f"\nAll visualization results saved to: {visualizer.save_dir}")
    print("Visualization test completed!")