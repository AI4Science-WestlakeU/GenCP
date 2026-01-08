# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference script for NTcouple single-field models (neutron, solid, fluid).

Joint evolution paradigm inference implementation:
- [conditioning, target] evolve in joint space
- Only compute error on target portion
- Fully aligned with GenCP core idea from Double Cylinder
"""

import os
import sys
from pathlib import Path
from time import time
from typing import Dict
import argparse

import torch
import numpy as np
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Import custom modules
from data.ntcouple_dataset import NTcoupleDataset
from data.ntcouple_normalizer import NTcoupleNormalizer
from utils.visualize import FluidFieldVisualizer
from utils.utils import (
    parse_ode_args, 
    parse_sde_args, 
    parse_transport_args,
    add_args_from_config, 
    rel_l2_loss, 
    mse_loss
)

# Import models
from model.cno import CNO3d
from model.fno import FNO3d
from model.SiT_FNO import SiT_FNO
from model.SiT import SiT
from model.cno_surrogate import CNO3d as CNO3d_surrogate
from model.fno_surrogate import FNO3d as FNO3d_surrogate
from model.sit_fno_surrogate import SiT_FNO as SiT_FNO_surrogate

# Import torchcfm for flow matching
import torchcfm


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, field: str) -> Dict[str, float]:
    """
    Compute evaluation metrics for NTcouple predictions.
    
    Args:
        pred: Predicted tensor (B, C, T, H, W).
        target: Ground truth tensor (B, C, T, H, W).
        field: Field type ('neutron', 'solid', 'fluid').
    
    Returns:
        Dictionary of metrics.
    """
    metrics = {}
    
    # Overall MSE (mse_loss returns per-sample MSE, so we need to take mean)
    metrics['mse'] = mse_loss(pred, target).mean().item()
    
    # Per-channel metrics (important for multi-channel fields like fluid)
    if field == "fluid" and pred.shape[1] > 1:
        metrics['per_channel_mse'] = []
        metrics['per_channel_rel_l2'] = []
        channel_names = ['temperature', 'velocity_x', 'velocity_y', 'pressure']
        
        for c in range(pred.shape[1]):
            c_mse = mse_loss(pred[:, c:c+1], target[:, c:c+1]).mean().item()
            c_rel_l2 = rel_l2_loss(pred[:, c:c+1], target[:, c:c+1]).mean().item()
            metrics['per_channel_mse'].append(c_mse)
            metrics['per_channel_rel_l2'].append(c_rel_l2)
            
            ch_name = channel_names[c] if c < len(channel_names) else f"channel_{c}"
            metrics[f'mse_{ch_name}'] = c_mse
            metrics[f'rel_l2_{ch_name}'] = c_rel_l2
        
        # Use per-channel average as the main rel_l2 metric for fluid
        metrics['rel_l2'] = np.mean(metrics['per_channel_rel_l2'])
    else:
        # For single-channel fields (neutron, solid), use overall rel_l2
        metrics['rel_l2'] = rel_l2_loss(pred, target).mean().item()
    
    return metrics


def sample_cfm_ode_joint(
    cond: torch.Tensor,
    target_ref: torch.Tensor,
    model: torch.nn.Module,
    cfm: torchcfm.ConditionalFlowMatcher,
    args: argparse.Namespace,
    **model_kwargs
) -> torch.Tensor:
    """
    Sample from conditional flow matching ODE using joint evolution paradigm.
    
    Joint evolution paradigm inference (fully consistent with training):
    1. Add noise to entire joint state [cond, target] (consistent with training)
    2. In ODE loop, xt_joint obtained via CFM sampling (both cond and target are noisy)
    3. Update entire joint state
    4. Finally return only target portion
    
    Args:
        cond: Conditioning tensor (B, T, H, W, C_cond)
        target_ref: Target reference for shape (B, T, H, W, C_target)
        model: Flow matching model
        cfm: ConditionalFlowMatcher instance
        args: Arguments
        model_kwargs: Additional kwargs
    
    Returns:
        Predicted target tensor (B, T, H, W, C_target)
    """
    device = cond.device
    
    joint_ref = torch.cat([cond, target_ref], dim=-1)
    
    z = torch.randn_like(joint_ref, device=device)
    x = z.clone()
    
    # For neutron field: prepare BC for inpainting
    # BC is the second channel in cond (channel index 1)
    bc_clean = None
    bc_noise = None
    if args.field == "neutron":
        # Extract BC from cond: cond shape is (B, T, H, W, 2), BC is channel 1
        bc_clean = cond[..., 1:2]  # (B, T, H, W, 1) - clean BC (x1)
        # Generate noise for BC (consistent across all time steps)
        bc_noise = torch.randn_like(bc_clean, device=device)  # (x0)
        print(f"[BC Inpainting] Neutron field: BC inpainting enabled for leftmost column (W=0)")
    
    # For solid field: prepare left boundary temperature BC for inpainting
    left_bc_clean = None
    left_bc_noise = None
    if args.field == "solid":
        # Extract left boundary temperature from target_ref (clean GT)
        # target_ref shape is (B, T, H, W=8, C=1), extract leftmost column (W=0)
        left_bc_clean = target_ref[..., 0:1, :]  # (B, T, H, 1, 1) - clean left BC (x1)
        # Generate noise for left BC (consistent across all time steps)
        left_bc_noise = torch.randn_like(left_bc_clean, device=device)  # (x0)
        print(f"[BC Inpainting] Solid field: Left boundary temperature BC inpainting enabled for leftmost column (W=0)")
    
    num_steps = args.num_sampling_steps
    t0, t1 = 0.0, 1.0
    t_grid = torch.linspace(t0, t1, num_steps, device=device)
    dt = (t_grid[1] - t_grid[0]).item() if num_steps > 1 else 1.0
    
    for t in tqdm(t_grid[:-1], desc="Sampling progress", leave=False):
        tb = t.expand(x.shape[0]).to(device)
        
        tb_model = tb * 1000.0
        
        _, xt_joint, _ = cfm.sample_location_and_conditional_flow(
            z.float().to(device), 
            joint_ref.float().to(device), 
            tb
        )
        
        if args.field == "neutron":
            yt = xt_joint[..., :-1]
            x = torch.cat([yt, x[..., -1:]], dim=-1)
        elif args.field == "solid":
            yt = xt_joint[..., :-1]
            x = torch.cat([yt, x[..., -1:]], dim=-1)
        elif args.field == "fluid":
            yt = xt_joint[..., :-4]
            x = torch.cat([yt, x[..., -4:]], dim=-1)
        else:
            raise NotImplementedError(f"Field '{args.field}' not supported")
        
        C_cond = cond.shape[-1]
        C_target = target_ref.shape[-1]
        
        use_clean_cond = getattr(args, 'use_clean_cond', False)
        if use_clean_cond:
            x = torch.cat([cond, x[..., C_cond:]], dim=-1)
        
        use_clean_bc = getattr(args, 'use_clean_bc', True)
        if use_clean_bc and args.field == 'neutron' and C_cond == 2:
            x_cond = x[..., :C_cond]
            x_target = x[..., C_cond:]
            x_cond_clean_bc = torch.cat([x_cond[..., 0:1], cond[..., 1:2]], dim=-1)
            x = torch.cat([x_cond_clean_bc, x_target], dim=-1)
        
        use_clean_left_bc = getattr(args, 'use_clean_left_bc_for_solid', True)
        if use_clean_left_bc and args.field == 'solid' and C_cond == 3:
            # For solid: input has 3 channels: [neutron_field(ch0), fluid_pressure(ch1), left_boundary_temp(ch2)]
            # Keep left boundary temp BC (channel 2) clean during inference
            x_cond = x[..., :C_cond]
            x_target = x[..., C_cond:]
            # Replace left boundary temp BC channel with clean value
            x_cond_clean_left_bc = torch.cat([x_cond[..., 0:2], cond[..., 2:3]], dim=-1)
            x = torch.cat([x_cond_clean_left_bc, x_target], dim=-1)
        
        use_noise_concat = getattr(args, 'use_noise_concat', False)
        if use_noise_concat:
            x_cond = x[..., :C_cond]
            x_target_current = x[..., C_cond:]
            
            # === CRITICAL FIX for neutron field: BC should remain clean ===
            if args.field == 'neutron' and C_cond == 2:
                # Separate spatial_cond (channel 0) and BC (channel 1)
                x_spatial = x_cond[..., 0:1]  # noisy spatial_cond
                clean_spatial = cond[..., 0:1]  # clean spatial_cond
                clean_bc = cond[..., 1:2]  # clean bc
                
                # Compute noise only for spatial_cond
                noise_spatial = x_spatial - clean_spatial
                # BC noise should be zero (keep it clean)
                noise_bc = torch.zeros_like(clean_bc)
                
                # Reconstruct
                noise_cond = torch.cat([noise_spatial, noise_bc], dim=-1)
                clean_cond = torch.cat([clean_spatial, clean_bc], dim=-1)
            elif args.field == 'solid' and C_cond == 3:
                # Separate channels: neutron (ch0), fluid_pressure (ch1), left_boundary_temp (ch2)
                x_neu = x_cond[..., 0:1]  # noisy neutron
                x_fluid = x_cond[..., 1:2]  # noisy fluid_pressure
                x_left_bc = x_cond[..., 2:3]  # noisy left_boundary_temp
                
                clean_neu = cond[..., 0:1]  # clean neutron
                clean_fluid = cond[..., 1:2]  # clean fluid_pressure
                clean_left_bc = cond[..., 2:3]  # clean left_boundary_temp
                
                # Compute noise for neutron and fluid_pressure
                noise_neu = x_neu - clean_neu
                noise_fluid = x_fluid - clean_fluid
                # Left BC noise should be zero (keep it clean)
                noise_left_bc = torch.zeros_like(clean_left_bc)
                
                # Reconstruct
                noise_cond = torch.cat([noise_neu, noise_fluid, noise_left_bc], dim=-1)
                clean_cond = torch.cat([clean_neu, clean_fluid, clean_left_bc], dim=-1)
            else:
                clean_cond = cond
                noise_cond = x_cond - clean_cond
            
            x_model_input = torch.cat([noise_cond, clean_cond, x_target_current], dim=-1)
            dummy_x0 = x_model_input[:, :1, :, :, :]
            vt = model(x_model_input, tb_model, cond=model_kwargs.get("attrs"))
            x_target_updated = x_target_current + vt * dt
            x = torch.cat([x_cond, x_target_updated], dim=-1)
        else:
            dummy_x0 = x[:, :1, :, :, :]
            vt = model(x, tb_model, cond=model_kwargs.get("attrs"))
            
            if args.field == "neutron":
                x[..., -1:] = x[..., -1:] + vt * dt
            elif args.field == "solid":
                x[..., -1:] = x[..., -1:] + vt * dt
            elif args.field == "fluid":
                x[..., -4:] = x[..., -4:] + vt * dt
        
        if args.field == "neutron" and bc_clean is not None:
            # Time after this update step
            t_after = t + dt
            tb_after = t_after.expand(x.shape[0]).to(device)
            
            # Use CFM to sample BC at time t_after
            # This gives us: bc_t = sample_xt(bc_noise, bc_clean, t_after)
            _, bc_t, _ = cfm.sample_location_and_conditional_flow(
                bc_noise.float().to(device),  # x0: noise
                bc_clean.float().to(device),  # x1: clean BC
                tb_after                       # t: time
            )
            
            # Replace leftmost column (W=0) of neutron target with noisy BC
            # x shape: (B, T, H, W, C_total), target is last 1 channel
            # neutron field has W=20, we replace W=0
            x[..., 0:1, -1:] = bc_t[..., 0:1, :]  # (B, T, H, 1, 1)
        
        if args.field == "solid" and left_bc_clean is not None:
            # Time after this update step
            t_after = t + dt
            tb_after = t_after.expand(x.shape[0]).to(device)
            
            # Use CFM to sample left BC at time t_after
            # This gives us: left_bc_t = sample_xt(left_bc_noise, left_bc_clean, t_after)
            _, left_bc_t, _ = cfm.sample_location_and_conditional_flow(
                left_bc_noise.float().to(device),  # x0: noise
                left_bc_clean.float().to(device),   # x1: clean left BC
                tb_after                            # t: time
            )
            
            # Replace leftmost column (W=0) of solid target with noisy left BC
            # x shape: (B, T, H, W=8, C_total), target is last 1 channel
            # solid field has W=8, we replace W=0
            x[..., 0:1, -1:] = left_bc_t
    
    C_target = target_ref.shape[-1]
    return x[..., -C_target:]


def sample_surrogate_ntcouple(
    cond: torch.Tensor,
    target_ref: torch.Tensor,
    model: torch.nn.Module,
    args: argparse.Namespace,
    **model_kwargs
) -> torch.Tensor:
    """
    Sample from surrogate model for NTcouple.
    
    Note: Surrogate models only take conditioning input (cond), not the target.
    The model directly predicts the target from the condition.
    
    Args:
        cond: Conditioning tensor (B, T, H, W, C_cond)
        target_ref: Target reference for shape (B, T, H, W, C_target) - only used for shape reference
        model: Surrogate model
        args: Arguments
        model_kwargs: Additional kwargs
    
    Returns:
        Predicted target tensor (B, T, H, W, C_target)
    """
    device = cond.device
    
    # Surrogate model only takes condition as input (not concatenated with target)
    # Convert to (B, C, T, H, W) format for surrogate model
    model_input = cond.permute(0, 4, 1, 2, 3)  # (B, T, H, W, C_cond) -> (B, C_cond, T, H, W)
    
    # Forward pass through surrogate model
    with torch.no_grad():
        pred = model(model_input)  # (B, C_out, T, H, W)
    
    # Convert back to (B, T, H, W, C) format
    pred = pred.permute(0, 2, 3, 4, 1)  # (B, C, T, H, W) -> (B, T, H, W, C)
    
    return pred


def main(args):
    """Main inference function for NTcouple single-field models."""
    
    # Setup
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Validate field type
    if args.field not in ['neutron', 'solid', 'fluid']:
        raise ValueError(f"Invalid field '{args.field}'. Choose from: neutron, solid, fluid")
    
    print(f"\n{'='*60}")
    print(f"NTcouple Single-Field Inference (Joint Evolution)")
    print(f"{'='*60}")
    print(f"Field: {args.field}")
    print(f"Split: {args.split}")
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Load validation dataset
    print(f"Loading dataset from {args.dataset_path}...")
    val_dataset = NTcoupleDataset(
        field=args.field,
        split=args.split,
        n_samples=args.n_data_set,
        data_root=args.dataset_path,
        normalize=True  # Already normalized in dataset
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers
    )
    
    print(f"Loaded {len(val_dataset)} validation samples")
    print(f"Cond shape: {val_dataset.cond_shape}, Target shape: {val_dataset.target_shape}")
    
    # Initialize normalizer for postprocessing
    normalizer = NTcoupleNormalizer(field=args.field, device=device)
    
    # Load model
    print(f"\nInitializing {args.model_name} model...")
    use_surrogate = getattr(args, "use_surrogate", False)
    
    if use_surrogate:
        print("Using surrogate model")
        if args.model_name == 'CNO':
            model = CNO3d_surrogate(
                in_dim=args.in_dim,
                out_dim=args.out_dim,
                in_size=args.in_size,
                N_layers=args.depth,
                channel_multiplier=args.channel_multiplier
            ).to(device)
        elif args.model_name == 'SiT_FNO':
            model = SiT_FNO_surrogate(
                input_size=args.input_size,
                depth=args.depth,
                hidden_size=args.hidden_size,
                patch_size=args.patch_size,
                num_heads=args.num_heads,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                modes=args.modes
            ).to(device)
        else:
            raise ValueError(f"Unsupported surrogate model: {args.model_name}")
    else:
        if args.model_name == 'CNO':
            model = CNO3d(
                in_dim=args.in_dim,
                out_dim=args.out_dim,
                in_size=args.in_size,
                N_layers=args.depth,
                dataset_name=args.dataset_name,
                x0_is_use_noise=args.x0_is_use_noise,
                channel_multiplier=args.channel_multiplier,
                stage=args.field
            ).to(device)

        elif args.model_name == 'SiT_FNO':
            model = SiT_FNO(
                input_size=args.input_size,
                depth=args.depth,
                hidden_size=args.hidden_size,
                patch_size=args.patch_size,
                num_heads=args.num_heads,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                x0_is_use_noise=args.x0_is_use_noise,
                stage=args.field,
                dataset_name=args.dataset_name,
                modes=args.modes,
            ).to(device)

        else:
            raise ValueError(f"Unsupported model: {args.model_name}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("Loaded model_state_dict")
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        print("Loaded state_dict")
    else:
        raise KeyError("Checkpoint must contain 'model_state_dict' or 'state_dict'")
    
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded and set to eval mode")
    
    # Setup flow matching (only for non-surrogate models)
    if not use_surrogate:
        cfm = torchcfm.ConditionalFlowMatcher()
    
    # Create visualizer
    save_figs_path = args.save_figs_path if hasattr(args, 'save_figs_path') and args.save_figs_path else "./visualization_results"
    visualizer = FluidFieldVisualizer(
        args=args, 
        grid_size=args.input_size, 
        save_dir=save_figs_path, 
        create_timestamp_folder=True
    )
    print(f"Visualizer created, saving to: {visualizer.save_dir}")
    
    # Inference loop
    print(f"\n{'='*60}")
    print("Starting inference...")
    print(f"{'='*60}\n")
    
    all_metrics = []
    total_time = 0.0
    num_batches = 0
    vis_batch_idx = 0  # Track which batch to visualize
    
    for batch_idx, (cond, target, grid_x, grid_y, attrs) in enumerate(val_dataloader):
        start_time = time()
        
        # Dataset returns (B, C, T, H, W), convert to (B, T, H, W, C)
        cond = cond.to(device).float()      # (B, C_cond, T, H, W)
        target = target.to(device).float()  # (B, C_target, T, H, W)
        
        # Convert to (B, T, H, W, C) format
        cond = cond.permute(0, 2, 3, 4, 1).contiguous()      # (B, C, T, H, W) -> (B, T, H, W, C)
        target = target.permute(0, 2, 3, 4, 1).contiguous()  # (B, C, T, H, W) -> (B, T, H, W, C)
        
        # Model kwargs
        # Sample using appropriate method based on model type
        with torch.no_grad():
            if use_surrogate:
                samples = sample_surrogate_ntcouple(
                    cond,
                    target,
                    model,
                    args,
                    attrs=attrs  # Pass attrs for model
                )  # (B, T, H, W, C_target)
            else:
                samples = sample_cfm_ode_joint(
                    cond,
                    target,
                    model,
                    cfm,
                    args,
                    attrs=attrs  # Pass attrs for model
                )  # (B, T, H, W, C_target)
        
        # Apply Gaussian smoothing to denoise predictions (optional)
        manual_denoise = False
        if manual_denoise:
            from scipy.ndimage import gaussian_filter
            
            def denoise_flow_output(data, sigma=1.0):
                """Apply Gaussian smoothing to flow matching output"""
                if isinstance(data, torch.Tensor):
                    data_np = data.cpu().numpy()
                else:
                    data_np = data
                denoised = np.zeros_like(data_np)
                
                for b in range(data_np.shape[0]):
                    for t in range(data_np.shape[1]):
                        for c in range(data_np.shape[4]):
                            denoised[b, t, :, :, c] = gaussian_filter(
                                data_np[b, t, :, :, c], 
                                sigma=sigma
                            )
                
                return torch.from_numpy(denoised).to(data.device)
            
            if args.field == "neutron":
                samples_denoised = denoise_flow_output(samples, sigma=0.8)
                samples = samples_denoised
                if batch_idx == 0:
                    print(f"Applied Gaussian smoothing to {args.field} field (all channels, sigma=0.8)")
            elif args.field == "solid":
                samples_denoised = denoise_flow_output(samples, sigma=0.8)
                samples = samples_denoised
                if batch_idx == 0:
                    print(f"Applied Gaussian smoothing to {args.field} field (all channels, sigma=0.8)")
            elif args.field == "fluid":
                samples_denoised = denoise_flow_output(samples, sigma=0.8)
                samples[..., 1:3] = samples_denoised[..., 1:3]
                if batch_idx == 0:
                    print(f"Applied Gaussian smoothing to {args.field} field (velocity_x and velocity_y channels, sigma=0.8)")
        
        # Denormalize predictions and targets
        # Convert to (B, C, T, H, W) for normalizer
        samples_perm = samples.permute(0, 4, 1, 2, 3)  # (B, T, H, W, C) -> (B, C, T, H, W)
        target_perm = target.permute(0, 4, 1, 2, 3)
        
        samples_denorm = normalizer.renormalize(samples_perm, field=args.field)
        target_denorm = normalizer.renormalize(target_perm, field=args.field)
        
        # Compute metrics
        batch_metrics = compute_metrics(samples_denorm, target_denorm, args.field)
        all_metrics.append(batch_metrics)
        
        batch_time = time() - start_time
        total_time += batch_time
        num_batches += 1
        
        # Print batch results
        print(f"Batch {batch_idx+1}/{len(val_dataloader)}:")
        print(f"  MSE: {batch_metrics['mse']:.6e}")
        print(f"  Rel L2: {batch_metrics['rel_l2']:.6f}")
        
        if args.field == "fluid" and 'per_channel_mse' in batch_metrics:
            print(f"  Per-channel MSE: {[f'{x:.6e}' for x in batch_metrics['per_channel_mse']]}")
            print(f"  Per-channel Rel L2: {[f'{x:.6f}' for x in batch_metrics['per_channel_rel_l2']]}")
        
        print(f"  Time: {batch_time:.2f}s")
        print()
        
        # Visualize the first batch
        if vis_batch_idx == 0 and batch_idx == 0:
            print("  Generating visualizations...")
            # Convert to (B, T, H, W, C) for visualization
            samples_vis = samples_denorm.permute(0, 2, 3, 4, 1).cpu().numpy()  # (B, C, T, H, W) -> (B, T, H, W, C)
            target_vis = target_denorm.permute(0, 2, 3, 4, 1).cpu().numpy()
            
            # Use first sample in batch
            pred_sample = samples_vis[0]  # (T, H, W, C)
            true_sample = target_vis[0]   # (T, H, W, C)
            
            # Visualize based on field type
            if args.field == "neutron":
                # Neutron flux: single channel
                visualizer.visualize_field(
                    u_pred=pred_sample[-1, ..., 0],  # Last time step
                    u_true=true_sample[-1, ..., 0], 
                    title="Neutron Flux Field (t=15)",
                    save_name="neutron_field.png",
                )
                visualizer.visualize_time_series_gif(
                    u_pred=pred_sample[..., 0],  # (T, H, W)
                    u_true=true_sample[..., 0],
                    title="Time Series Animation - Neutron Flux",
                    save_name="time_series_neutron.gif",
                    show_colorbar=True
                )
            
            elif args.field == "solid":
                # Fuel temperature: single channel
                visualizer.visualize_field(
                    u_pred=pred_sample[-1, ..., 0],  # Last time step
                    u_true=true_sample[-1, ..., 0], 
                    title="Fuel Temperature Field (t=15)",
                    save_name="solid_field.png",
                )
                visualizer.visualize_time_series_gif(
                    u_pred=pred_sample[..., 0],  # (T, H, W)
                    u_true=true_sample[..., 0],
                    title="Time Series Animation - Fuel Temperature",
                    save_name="time_series_solid.gif",
                    show_colorbar=True
                )
            
            elif args.field == "fluid":
                # Fluid: 4 channels (temperature, vx, vy, pressure)
                channel_names = ["Temperature", "Velocity-X", "Velocity-Y", "Pressure"]
                save_names = ["fluid_temp", "fluid_vx", "fluid_vy", "fluid_p"]
                
                for c, (ch_name, save_name) in enumerate(zip(channel_names, save_names)):
                    visualizer.visualize_field(
                        u_pred=pred_sample[-1, ..., c],  # Last time step
                        u_true=true_sample[-1, ..., c], 
                        title=f"{ch_name} Field (t=15)",
                        save_name=f"{save_name}_field.png",
                    )
                    visualizer.visualize_time_series_gif(
                        u_pred=pred_sample[..., c],  # (T, H, W)
                        u_true=true_sample[..., c],
                        title=f"Time Series Animation - {ch_name}",
                        save_name=f"time_series_{save_name}.gif",
                        show_colorbar=True
                    )
            
            vis_batch_idx += 1
            print(f"  Visualizations saved to: {visualizer.save_dir}")
        
        # Limit number of batches for quick testing
        if args.max_eval_batches and batch_idx >= args.max_eval_batches - 1:
            print(f"Reached max_eval_batches={args.max_eval_batches}, stopping...")
            break
    
    # Aggregate metrics
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    
    avg_mse = np.mean([m['mse'] for m in all_metrics])
    avg_rel_l2 = np.mean([m['rel_l2'] for m in all_metrics])
    
    print(f"Average MSE: {avg_mse:.6e}")
    print(f"Average Rel L2: {avg_rel_l2:.6f}")
    
    if args.field == "fluid" and 'per_channel_mse' in all_metrics[0]:
        n_channels = len(all_metrics[0]['per_channel_mse'])
        channel_names = ['temperature', 'velocity_x', 'velocity_y', 'pressure']
        
        for c in range(n_channels):
            ch_name = channel_names[c] if c < len(channel_names) else f"channel_{c}"
            ch_mse = np.mean([m['per_channel_mse'][c] for m in all_metrics])
            ch_rel_l2 = np.mean([m['per_channel_rel_l2'][c] for m in all_metrics])
            print(f"\n{ch_name.upper()}:")
            print(f"  MSE: {ch_mse:.6e}")
            print(f"  Rel L2: {ch_rel_l2:.6f}")
    
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average time per batch: {total_time/num_batches:.2f}s")
    print(f"{'='*60}\n")
    
    # # Save results to file
    # if args.results_path:
    #     results_dir = Path(args.results_path) / "ntcouple" / args.field / args.split
    #     results_dir.mkdir(parents=True, exist_ok=True)
        
    #     results_file = results_dir / f"{args.model_name}_results.txt"
    #     with open(results_file, 'w') as f:
    #         f.write(f"NTcouple {args.field} Inference Results (Joint Evolution)\n")
    #         f.write(f"{'='*60}\n")
    #         f.write(f"Model: {args.model_name}\n")
    #         f.write(f"Checkpoint: {args.checkpoint_path}\n")
    #         f.write(f"Split: {args.split}\n")
    #         f.write(f"Num samples: {len(val_dataset)}\n")
    #         f.write(f"{'='*60}\n\n")
    #         f.write(f"Average MSE: {avg_mse:.6e}\n")
    #         f.write(f"Average Rel L2: {avg_rel_l2:.6f}\n")
            
    #         if args.field == "fluid" and 'per_channel_mse' in all_metrics[0]:
    #             f.write(f"\nPer-channel metrics:\n")
    #             for c in range(n_channels):
    #                 ch_name = channel_names[c] if c < len(channel_names) else f"channel_{c}"
    #                 ch_mse = np.mean([m['per_channel_mse'][c] for m in all_metrics])
    #                 ch_rel_l2 = np.mean([m['per_channel_rel_l2'][c] for m in all_metrics])
    #                 f.write(f"  {ch_name}: MSE={ch_mse:.6e}, Rel L2={ch_rel_l2:.6f}\n")
        
    #     print(f"Results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NTcouple single-field inference (joint evolution)")
    
    # Config file
    parser.add_argument("--config", type=str, required=True,
                       help="Path to config YAML file")
    
    # Inference-specific arguments
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument(
        "--split",
        type=str,
        default="decouple_val",
        choices=['decouple_train', 'decouple_val', 'couple_train', 'couple_val'],
        help="Dataset split for evaluation",
    )
    parser.add_argument("--max_eval_batches", type=int, default=None,
                       help="Max number of batches to evaluate (None = all)")
    parser.add_argument("--num-sampling-steps", type=int, default=25,
                       help="Number of ODE sampling steps")
    parser.add_argument("--save_figs_path", type=str, default="./visualization_results",
                       help="Path to save visualization figures")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Parse transport args (for flow matching)
    parse_transport_args(parser)
    
    # Load config from YAML
    args = add_args_from_config(parser)
    
    # Run inference
    main(args)

