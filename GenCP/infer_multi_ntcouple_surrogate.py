# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
NTcouple multi-field inference script.
Implements three-field (neutron, solid, fluid) interactive inference with different shapes.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from model.SiT import SiT
from model.cno import CNO3d
from model.fno import FNO3d
from model.SiT_FNO import SiT_FNO
from model.cno_surrogate import CNO3d as CNO3d_surrogate
from model.sit_fno_surrogate import SiT_FNO as SiT_FNO_surrogate
from infer_single_ntcouple import sample_surrogate_ntcouple
from utils.utils import parse_ode_args, parse_transport_args, add_args_from_config, mse_loss, rel_l2_loss
import argparse
from time import time
from typing import List, Sequence, Tuple
from tqdm import tqdm
import torchcfm

from data.ntcouple_dataset import NTcoupleDataset
from data.ntcouple_normalizer import NTcoupleNormalizer
from utils.visualize import FluidFieldVisualizer
import numpy as np
import os

# NTcouple field order (must match M2PDE convention)
FIELD_ORDER = ("neutron", "solid", "fluid")


def update_neutron_condition(
    estimates: Sequence[torch.Tensor],
    other_condition: Sequence[torch.Tensor],
    normalize_fn,
    renormalize_fn,
) -> torch.Tensor:
    """
    Construct the conditioning tensor for the neutron diffusion model.
    
    Args:
        estimates: Current denoised estimates [neutron, solid, fluid] in normalized space.
        other_condition: Sequence whose first element is the neutron boundary condition tensor.
        normalize_fn: Function to normalize physical values.
        renormalize_fn: Function to denormalize normalized values.
    
    Returns:
        Conditioning tensor with shape (B, C_cond, T, H, W) for neutron model.
        Note: neutron field has W=20 (8 fuel + 12 fluid channels)
    """
    bc = other_condition[0]  # Boundary condition: (B, 1, T, H, 1)
    solid = estimates[1]  # Solid field: (B, T, H, 8, 1)
    fluid = estimates[2][:, :, :, :, :1]  # (B, T, H, 12, 1) - temperature channel only
    
    # Convert solid and fluid to (B, C, T, H, W) format for concatenation
    # estimates are in (B, T, H, W, C) format
    solid = solid.permute(0, 4, 1, 2, 3)  # (B, T, H, 8, 1) -> (B, 1, T, H, 8)
    fluid = fluid.permute(0, 4, 1, 2, 3)  # (B, T, H, 12, 1) -> (B, 1, T, H, 12)
    
    # Concatenate solid and fluid spatially (along width dimension)
    stacked = torch.cat((solid, fluid), dim=-1)  # (B, 1, T, H, 20)
    
    # Expand boundary condition to match width
    bc = bc.repeat(1, 1, 1, 1, stacked.shape[-1])  # (B, 1, T, H, 20)
    
    # Concatenate along channel dimension: (B, 2, T, H, 20)
    # Channel 0: spatial_cond (fuel+fluid temperature)
    # Channel 1: bc_expanded (boundary condition)
    cond = torch.cat((stacked, bc), dim=1)
    
    # Convert to (B, T, H, W, C) format for GenCP models
    cond = cond.permute(0, 2, 3, 4, 1)  # (B, 2, T, H, 20) -> (B, T, H, 20, 2)
    
    return cond


def update_solid_condition(
    estimates: Sequence[torch.Tensor],
    other_condition: Sequence[torch.Tensor],
    normalize_fn,
    renormalize_fn,
) -> torch.Tensor:
    """
    Construct the conditioning tensor for the solid/fuel diffusion model.
    
    The solid model expects 3 input channels:
    1. Channel 0: neutron flux (first 8 axial channels)
    2. Channel 1: fluid inlet temperature (repeated across width)
    3. Channel 2: left boundary temperature from solid field (repeated across width)
    
    Returns:
        Conditioning tensor with shape (B, T, H, W, C=3) for solid model.
        Note: solid field has W=8 (fuel channels only)
    """
    del other_condition  # Only estimates are required
    # neutron field shape: (B, T, H, 20, 1), take first 8 width channels
    neu = estimates[0][:, :, :, :8, :]  # (B, T, H, 8, 1)
    
    # fluid field shape after permute: (B, T, H, 12, 4)
    fluid = estimates[2][:, :, :, :1, :1]  # (B, T, H, 1, 1) - inlet temperature (NOT pressure!)
    fluid = fluid.repeat(1, 1, 1, 8, 1)  # Expand to match solid width: (B, T, H, 8, 1)
    
    # Extract left boundary temperature from solid field (most left column)
    # solid field shape: (B, T, H, 8, 1)
    left_boundary_temp = estimates[1][:, :, :, :1, :]  # (B, T, H, 1, 1) - leftmost column
    left_boundary_temp = left_boundary_temp.repeat(1, 1, 1, 8, 1)  # Expand to match solid width: (B, T, H, 8, 1)
    
    # Concatenate along channel dimension: (B, T, H, 8, 3)
    # Channel 0: neutron flux
    # Channel 1: fluid inlet temperature
    # Channel 2: left boundary temperature
    cond = torch.cat((neu, fluid, left_boundary_temp), dim=-1)
    
    return cond


def update_fluid_condition(
    estimates: Sequence[torch.Tensor],
    other_condition: Sequence[torch.Tensor],
    normalize_fn,
    renormalize_fn,
) -> torch.Tensor:
    """
    Construct the conditioning tensor for the coolant diffusion model.
    
    The coolant model conditions on the heat flux derived from the current fuel estimate.
    
    Returns:
        Conditioning tensor with shape (B, T, H, W, C) for fluid model.
        Note: fluid field has W=12 (coolant channels only)
    """
    del other_condition  # Only the current estimate is required
    
    # Work directly in NORMALIZED space (estimates[1] is already normalized)
    # This matches training where flux is computed from normalized temperature
    fuel_norm = estimates[1]  # (B, T, H, 8, 1) - already in normalized space
    
    # Extract the two rightmost width channels (closest to fluid boundary)
    # These are used to construct the flux condition
    temperature_second_last = fuel_norm[:, :, :, -2:-1, :]  # (B, T, H, 1, 1)
    temperature_last = fuel_norm[:, :, :, -1:, :]  # (B, T, H, 1, 1)
    
    # Concatenate the two temperature channels
    # This represents the boundary condition information for fluid
    flux_cond = torch.cat((temperature_second_last, temperature_last), dim=-2)  # (B, T, H, 2, 1)
    
    # Repeat to match fluid width (12 channels)
    flux_expanded = flux_cond.repeat(1, 1, 1, int(estimates[2].shape[3]//2), 1)  # (B, T, H, 12, 1)
    
    return flux_expanded


def default_ntcouple_updates():
    """Return update callables for (neutron, solid, fluid) flow models."""
    return [
        update_neutron_condition,
        update_solid_condition,
        update_fluid_condition,
    ]


def create_ntcouple_surrogate_step_fn(models, pred_update_coeff=0.1, bc_batch=None):
    """
    Create surrogate step function for NTcouple three-field coupling.
    
    Args:
        models: Dict with keys 'neutron', 'solid', 'fluid', each containing a surrogate model
        pred_update_coeff: Update coefficient for iterative refinement
        bc_batch: Boundary condition tensor (B, 1, T, H, 1) for neutron field
    
    Returns:
        Function that takes (input_norm, neutron_pred_prev, solid_pred_prev, fluid_pred_prev)
        and returns (neutron_pred, solid_pred, fluid_pred, loss)
    """
    neutron_model = models['neutron']
    solid_model = models['solid']
    fluid_model = models['fluid']
    c = pred_update_coeff
    
    def single_step_surrogate(input_norm, neutron_pred_prev, solid_pred_prev, fluid_pred_prev):
        """
        Single step of surrogate inference for three fields.
        
        Note: Surrogate models only take conditioning input, not concatenated with target.
        
        Args:
            input_norm: Normalized input (not used directly, but kept for consistency)
            neutron_pred_prev: Previous neutron prediction (B, T, H, W, C)
            solid_pred_prev: Previous solid prediction (B, T, H, W, C)
            fluid_pred_prev: Previous fluid prediction (B, T, H, W, C)
        
        Returns:
            (neutron_pred, solid_pred, fluid_pred, loss)
        """
        # Build conditioning for each field using Jacobi iteration (all fields use prev values)
        # This ensures proper coupling: all fields are updated based on previous iteration state
        # Neutron: conditions on solid + fluid temperature + boundary condition
        cond_neutron = update_neutron_condition(
            [neutron_pred_prev, solid_pred_prev, fluid_pred_prev],
            [bc_batch] if bc_batch is not None else [], None, None
        )
        # Surrogate model only takes condition as input (not concatenated with target)
        model_input_neutron = cond_neutron.permute(0, 4, 1, 2, 3)  # (B, T, H, W, C) -> (B, C, T, H, W)
        neutron_pred = neutron_model(model_input_neutron)
        neutron_pred = neutron_pred.permute(0, 2, 3, 4, 1)  # (B, C, T, H, W) -> (B, T, H, W, C)
        
        # Solid: conditions on neutron + fluid inlet temperature (use prev values for proper coupling)
        cond_solid = update_solid_condition(
            [neutron_pred_prev, solid_pred_prev, fluid_pred_prev],
            [], None, None
        )
        # Surrogate model only takes condition as input
        model_input_solid = cond_solid.permute(0, 4, 1, 2, 3)  # (B, T, H, W, C) -> (B, C, T, H, W)
        solid_pred = solid_model(model_input_solid)
        solid_pred = solid_pred.permute(0, 2, 3, 4, 1)  # (B, C, T, H, W) -> (B, T, H, W, C)
        
        # Fluid: conditions on solid heat flux (use prev values for proper coupling)
        cond_fluid = update_fluid_condition(
            [neutron_pred_prev, solid_pred_prev, fluid_pred_prev],
            [], None, None
        )
        # Surrogate model only takes condition as input
        model_input_fluid = cond_fluid.permute(0, 4, 1, 2, 3)  # (B, T, H, W, C) -> (B, C, T, H, W)
        fluid_pred = fluid_model(model_input_fluid)
        fluid_pred = fluid_pred.permute(0, 2, 3, 4, 1)  # (B, C, T, H, W) -> (B, T, H, W, C)
        
        # Update with coefficient
        neutron_pred = c * neutron_pred + (1 - c) * neutron_pred_prev
        solid_pred = c * solid_pred + (1 - c) * solid_pred_prev
        fluid_pred = c * fluid_pred + (1 - c) * fluid_pred_prev
        
        # Compute loss
        loss1 = torch.norm(neutron_pred - neutron_pred_prev, dim=2, p=2).mean()
        loss2 = torch.norm(solid_pred - solid_pred_prev, dim=2, p=2).mean()
        loss3 = torch.norm(fluid_pred - fluid_pred_prev, dim=2, p=2).mean()
        loss = (loss1 + loss2 + loss3) / 3.0
        
        return neutron_pred, solid_pred, fluid_pred, loss
    
    return single_step_surrogate


def surrogate_inference_ntcouple(input_norm, target_norm, num_steps, device, args, attrs, step_fn):
    """
    Perform surrogate inference for NTcouple three-field coupling.
    
    Args:
        input_norm: Normalized input (not used directly)
        target_norm: Dict with keys 'neutron', 'solid', 'fluid', each (B, T, H, W, C)
        num_steps: Number of iteration steps
        device: Device
        args: Arguments
        attrs: Attributes
        step_fn: Step function from create_ntcouple_surrogate_step_fn
    
    Returns:
        List of predictions [neutron_pred, solid_pred, fluid_pred] in normalized space
    """
    print(f"Starting surrogate inference for {num_steps} steps...")
    start_time = time()
    
    # Initialize predictions
    neutron_pred_prev = torch.ones_like(target_norm['neutron']) * 0.5
    solid_pred_prev = torch.ones_like(target_norm['solid']) * 0.5
    fluid_pred_prev = torch.ones_like(target_norm['fluid']) * 0.5
    
    for step in range(num_steps):
        neutron_pred, solid_pred, fluid_pred, loss = step_fn(
            input_norm, neutron_pred_prev, solid_pred_prev, fluid_pred_prev
        )
        
        neutron_pred_prev = neutron_pred
        solid_pred_prev = solid_pred
        fluid_pred_prev = fluid_pred
        
        print(f"Step {step + 1}/{num_steps}, loss: {loss.item():.6f}")
    
    print(f"Surrogate inference took {time() - start_time:.2f} seconds.")
    return [neutron_pred, solid_pred, fluid_pred]


def compose_flow_ntcouple(
    model_list: List,
    shapes: List[Tuple],
    update_f: List,
    normalize_fn,
    renormalize_fn,
    timestep: int = 10,
    other_condition: List = [],
    device: str = "cuda",
    use_bc_inpainting: bool = True,
    args = None,
) -> List[torch.Tensor]:
    """
    Compose flow matching models for NTcouple three-field inference.
    
    Args:
        model_list: List of flow matching models [neutron_model, solid_model, fluid_model].
        shapes: List of target shapes for each field [(B, T, H, W_neu, C), (B, T, H, W_solid, C), (B, T, H, W_fluid, C)].
        update_f: List of update functions [update_neutron, update_solid, update_fluid].
        normalize_fn: Function to normalize physical values.
        renormalize_fn: Function to denormalize normalized values.
        timestep: Number of flow matching steps.
        other_condition: List of other conditions (e.g., boundary conditions).
        device: Device for computation.
        use_bc_inpainting: If True, use inpainting for neutron field's leftmost column with boundary condition.
    
    Returns:
        List of generated fields [neutron_pred, solid_pred, fluid_pred] in normalized space.
    """
    with torch.no_grad():
        n_compose = len(model_list)
        
        T = torch.linspace(0, 1, timestep + 1).to(device)
        
        # Initialize fields with random noise
        mult_p = []
        for s in shapes:
            mult_p.append(torch.randn(s, device=device))
        
        use_noise_concat = getattr(args, 'use_noise_concat', False) if args is not None else False
        if use_noise_concat:
            mult_p_clean_est = [None] * n_compose
            print("[INFO] Initializing clean state estimation for noise concatenation")
        else:
            mult_p_clean_est = None
        
        # For neutron field BC inpainting: prepare clean BC and noise
        bc_clean = None
        bc_noise = None
        if use_bc_inpainting and len(other_condition) > 0:
            # Extract clean BC: (B, 1, T, H, 1) -> (B, T, H, 1, 1)
            bc_clean = other_condition[0].permute(0, 2, 3, 4, 1)
            # Generate noise for BC (consistent across all time steps)
            bc_noise = torch.randn_like(bc_clean, device=device)
            print(f"[BC Inpainting] Neutron field: BC inpainting enabled for leftmost column (W=0)")
        
        # Single flow matching loop - use current mult_p as condition
        for t_idx in tqdm(range(0, timestep), desc="Flow matching steps"):
            for i in range(n_compose):
                # Get model and update function for this field
                model = model_list[i]
                update = update_f[i]
                
                # Construct conditioning from current state of other fields
                cond = update(
                    mult_p.copy(),
                    other_condition,
                    normalize_fn,
                    renormalize_fn,
                )
                
                # Single flow step
                t_curr = T[t_idx]
                t_next = T[t_idx + 1]
                dt = t_next - t_curr
                
                # Prepare model inputs
                # CNO model expects joint state [cond, target] concatenated along channel dimension
                x_target = mult_p[i]  # Current target state: (B, T, H, W, C_target)
                
                # Concatenate cond and target along channel dimension to form joint state
                x_joint = torch.cat([cond, x_target], dim=-1)  # (B, T, H, W, C_cond+C_target)
                
                # Construct dummy_x0 consistent with training: extract first timestep from x_joint
                # This matches train.py:606 and infer_single_ntcouple.py:156
                dummy_x0 = x_joint[:, :1, :, :, :]  # (B, 1, H, W, C_cond+C_target)
                
                # Model forward pass (CFM step)
                # Note: CNO models expect joint state [cond, target] as input
                if hasattr(model, 'forward'):
                    # For CNO/SiT models that expect (B, T, H, W, C) format
                    # Convert time to batch-sized tensor
                    t_batch = torch.full((x_joint.shape[0],), t_curr.item(), device=device, dtype=x_joint.dtype)
                    
                    C_cond = cond.shape[-1]
                    C_target = x_target.shape[-1]
                    
                    use_clean_cond = getattr(args, 'use_clean_cond', False) if args is not None else False
                    if use_clean_cond:
                        x_joint = torch.cat([cond, x_joint[..., C_cond:]], dim=-1)
                    
                    if use_noise_concat and mult_p_clean_est is not None:
                        if t_idx == 0 and i == 0:
                            print("[INFO] use_noise_concat=True: Starting cold-start procedure")
                        
                        x_cond_current = x_joint[..., :C_cond]
                        x_target_current = x_joint[..., C_cond:]
                        
                        if t_idx == 0:
                            if i == 0 and t_idx == 0:
                                print(f"[Cold Start] Step {t_idx}: Assuming clean_cond ≈ cond_noise (zero-noise assumption)")
                            
                            clean_cond_assumption = x_cond_current
                            noise_cond_zero = torch.zeros_like(x_cond_current)
                            
                            x_model_input = torch.cat([noise_cond_zero, clean_cond_assumption, x_target_current], dim=-1)
                            dummy_x0_input = x_model_input[:, :1, :, :, :]
                            vt_joint_pure = model(x_model_input, t_batch, dummy_x0_input, None)
                            
                            if vt_joint_pure.shape[-1] == C_cond + C_target:
                                vt_target = vt_joint_pure[..., -C_target:]
                            elif vt_joint_pure.shape[-1] == C_target:
                                vt_target = vt_joint_pure
                            else:
                                vt_target = torch.zeros_like(x_target_current)
                            
                            remaining_time = 1.0 - t_curr.item()
                            target_clean_estimate = x_target_current + vt_target * remaining_time
                            mult_p_clean_est[i] = target_clean_estimate
                            
                            if i == 0:
                                print(f"[Cold Start] Field {i}: Estimated clean state with zero-noise assumption")
                            
                            vt_joint = vt_joint_pure
                        
                        else:
                            clean_cond = update(
                                mult_p_clean_est.copy(),
                                other_condition,
                                normalize_fn,
                                renormalize_fn,
                            )
                            
                            noise_cond = x_cond_current - clean_cond
                            
                            x_model_input = torch.cat([noise_cond, clean_cond, x_target_current], dim=-1)
                            dummy_x0_input = x_model_input[:, :1, :, :, :]
                            vt_joint = model(x_model_input, t_batch, dummy_x0_input, None)
                            remaining_time = 1.0 - t_curr.item()
                            if vt_joint.shape[-1] == C_cond + C_target:
                                vt_target = vt_joint[..., -C_target:]
                            elif vt_joint.shape[-1] == C_target:
                                vt_target = vt_joint
                            else:
                                vt_target = torch.zeros_like(x_target_current)
                            
                            target_clean_estimate = x_target_current + vt_target * remaining_time
                            mult_p_clean_est[i] = target_clean_estimate
                    else:
                        # Model predicts velocity field for joint state
                        vt_joint = model(x_joint, t_batch, dummy_x0, None)  # cond is already in x_joint
                    
                    # CNO model outputs velocity for the entire joint state
                    # We need to extract only the target part
                    C_target = x_target.shape[-1]
                    C_cond = cond.shape[-1]
                    
                    # Model output should have same channels as input joint state
                    # Extract target part (last C_target channels)
                    if vt_joint.shape[-1] == C_cond + C_target:
                        # Model outputs joint state velocity, extract target part
                        vt = vt_joint[..., -C_target:]  # (B, T, H, W, C_target)
                    elif vt_joint.shape[-1] == C_target:
                        # Model outputs only target velocity
                        vt = vt_joint
                    else:
                        # Debug: print shapes for troubleshooting
                        if t_idx == 0 and i == 2:  # Only print for fluid field on first step
                                  f"vt_joint shape: {vt_joint.shape}, "
                                  f"C_cond: {C_cond}, C_target: {C_target}")
                        raise ValueError(
                            f"Field {i}: Unexpected model output channels: {vt_joint.shape[-1]}, "
                            f"expected {C_cond + C_target} (joint) or {C_target} (target only). "
                            f"x_joint shape: {x_joint.shape}, vt_joint shape: {vt_joint.shape}"
                        )
                    
                    # Update state: x_{t+dt} = x_t + v_t * dt
                    # Ensure vt has correct shape
                    assert vt.shape == x_target.shape, (
                        f"Field {i}: vt shape {vt.shape} != x_target shape {x_target.shape}"
                    )
                    mult_p[i] = x_target + vt * dt
                    
                    # Apply BC inpainting for neutron field (i=0) after state update
                    if i == 0 and use_bc_inpainting and bc_clean is not None:
                        # Use CFM to sample BC at time t_next
                        # Create CFM instance for BC sampling
                        cfm_bc = torchcfm.ConditionalFlowMatcher(sigma=0.0)
                        t_batch_next = torch.full((mult_p[i].shape[0],), t_next.item(), device=device, dtype=mult_p[i].dtype)
                        
                        # Sample BC at time t_next using CFM
                        # This gives us: bc_t = sample_xt(bc_noise, bc_clean, t_next)
                        _, bc_t, _ = cfm_bc.sample_location_and_conditional_flow(
                            bc_noise.float().to(device),  # x0: noise
                            bc_clean.float().to(device),  # x1: clean BC
                            t_batch_next                  # t: time
                        )
                        
                        # Replace leftmost column (W=0) of neutron field with sampled BC
                        # mult_p[i] shape: (B, T, H, W=20, C=1)
                        mult_p[i][:, :, :, 0:1, :] = bc_t[:, :, :, 0:1, :]
                        
                else:
                    raise ValueError(f"Model {i} does not have forward method")
        
        return mult_p


def load_model(args, device, model_type, dataset=None):
    """Load model for specified field type.
    
    Args:
        args: Arguments containing model configuration.
        device: Device to load model on.
        model_type: One of "ntcouple_neutron", "ntcouple_solid", "ntcouple_fluid".
        dataset: Optional dataset to infer input dimensions from.
    """
    use_surrogate = getattr(args, "use_surrogate", False)
    
    # For surrogate models, prioritize config file values over dataset values
    # to ensure consistency with training configuration
    if use_surrogate:
        # Get field-specific in_dim from config first
        if model_type == "ntcouple_neutron":
            # Neutron uses global in_dim or in_channels (no field-specific config)
            in_dim = getattr(args, "in_dim", getattr(args, "in_channels", 2))
        elif model_type == "ntcouple_solid":
            # Try in_dim_structure first, then in_channels_structure, then fallback
            in_dim = getattr(args, "in_dim_structure", 
                           getattr(args, "in_channels_structure", 
                                  getattr(args, "in_dim", getattr(args, "in_channels", 2))))
        elif model_type == "ntcouple_fluid":
            # Try in_dim_fluid first, then in_channels_fluid, then fallback
            in_dim = getattr(args, "in_dim_fluid",
                           getattr(args, "in_channels_fluid",
                                  getattr(args, "in_dim", getattr(args, "in_channels", 1))))
        else:
            in_dim = getattr(args, "in_dim", getattr(args, "in_channels", 2))
        
        # Get input_size from dataset if available, otherwise from args
        if dataset is not None:
            _, _, H, W = dataset.target.shape[1:]  # (C, T, H, W)
            input_size = [H, W]
            in_size = max(H, W)  # CNO uses max dimension
        else:
            input_size = args.input_size
            in_size = args.in_size
        
        print(f"  Using in_dim={in_dim} from config for {model_type} (surrogate model)")
    else:
        # For CFM models, use dataset values or args
        if dataset is not None:
            cond_channels = dataset.cond.shape[1]  # (C, T, H, W)
            target_channels = dataset.target.shape[1]
            if args.use_noise_concat:
                in_dim = cond_channels*2 + target_channels  # [cond, xt, E]
            else:
                in_dim = cond_channels + target_channels  # [cond, xt]
            
            _, _, H, W = dataset.target.shape[1:]  # (C, T, H, W)
            input_size = [H, W]
            in_size = max(H, W)  # CNO uses max dimension
        else:
            in_dim = args.in_dim
            input_size = args.input_size
            in_size = args.in_size
    
    if use_surrogate:
        if model_type == "ntcouple_fluid":
            if args.model_name == "CNO":
                # Use field-specific channel_multiplier if available
                channel_mult = getattr(args, "channel_multiplier_fluid", 
                                     getattr(args, "channel_multiplier", 16))
                print(f"  Using channel_multiplier={channel_mult} for {model_type} (CNO surrogate)")
                model = CNO3d_surrogate(in_dim=in_dim,
                            out_dim=args.out_dim_fluid,
                            in_size=in_size,
                            N_layers=args.depth_fluid,
                            channel_multiplier=channel_mult
                        ).to(device)
            elif args.model_name == "SiT_FNO":
                model = SiT_FNO_surrogate(input_size=input_size, depth=args.depth_fluid, hidden_size=args.hidden_size_fluid, patch_size=args.patch_size_fluid, 
                            num_heads=args.num_heads_fluid, in_channels=in_dim,
                            out_channels=args.out_channels_fluid, modes=args.modes_fluid).to(device)
            else:
                raise ValueError(f"Unsupported surrogate model: {args.model_name}")
            ckpt_path = args.ntcouple_fluid_checkpoint_path
        elif model_type == "ntcouple_solid":
            if args.model_name == "CNO":
                # Use field-specific channel_multiplier if available
                channel_mult = getattr(args, "channel_multiplier_structure", 
                                     getattr(args, "channel_multiplier", 16))
                print(f"  Using channel_multiplier={channel_mult} for {model_type} (CNO surrogate)")
                model = CNO3d_surrogate(in_dim=in_dim,
                            out_dim=args.out_dim_structure,
                            in_size=in_size,
                            N_layers=args.depth_structure,
                            channel_multiplier=channel_mult
                        ).to(device)
            elif args.model_name == "SiT_FNO":
                model = SiT_FNO_surrogate(input_size=input_size, depth=args.depth_structure, hidden_size=args.hidden_size_structure, patch_size=args.patch_size_structure, 
                            num_heads=args.num_heads_structure, in_channels=in_dim,
                            out_channels=args.out_channels_structure, modes=args.modes_structure).to(device)
            else:
                raise ValueError(f"Unsupported surrogate model: {args.model_name}")
            ckpt_path = args.ntcouple_solid_checkpoint_path
        elif model_type == "ntcouple_neutron":
            if args.model_name == "CNO":
                # Use field-specific channel_multiplier if available
                channel_mult = getattr(args, "channel_multiplier_neutron", 
                                     getattr(args, "channel_multiplier", 16))
                print(f"  Using channel_multiplier={channel_mult} for {model_type} (CNO surrogate)")
                model = CNO3d_surrogate(in_dim=in_dim,
                            out_dim=args.out_dim_neutron,
                            in_size=in_size,
                            N_layers=args.depth_neutron,
                            channel_multiplier=channel_mult
                        ).to(device)
            elif args.model_name == "SiT_FNO":
                model = SiT_FNO_surrogate(input_size=input_size, depth=args.depth_neutron, hidden_size=args.hidden_size_neutron, patch_size=args.patch_size_neutron, 
                            num_heads=args.num_heads_neutron, in_channels=in_dim,
                            out_channels=args.out_channels_neutron, modes=args.modes_neutron).to(device)
            else:
                raise ValueError(f"Unsupported surrogate model: {args.model_name}")
            ckpt_path = args.ntcouple_neutron_checkpoint_path
        else:
            raise ValueError(f"Model type {model_type} not supported. Expected one of: ntcouple_neutron, ntcouple_solid, ntcouple_fluid")
    
    elif getattr(args, "use_torchcfm", False):
        if model_type == "ntcouple_fluid":
            if args.model_name == "CNO":
                model = CNO3d(in_dim=in_dim, 
                            out_dim=args.out_dim_fluid, 
                            in_size=in_size, 
                            N_layers=args.depth_fluid,
                            dataset_name=args.dataset_name,
                            x0_is_use_noise=args.x0_is_use_noise,
                            stage=args.stage).to(device)
            elif args.model_name == "SiT_FNO":
                model = SiT_FNO(input_size=input_size, depth=args.depth_fluid, hidden_size=args.hidden_size_fluid, patch_size=args.patch_size_fluid, 
                            num_heads=args.num_heads_fluid, x0_is_use_noise=args.x0_is_use_noise, in_channels=in_dim,
                            out_channels=args.out_channels_fluid,
                            stage=model_type, modes=args.modes_fluid).to(device)
            ckpt_path = args.ntcouple_fluid_checkpoint_path
        elif model_type == "ntcouple_solid":
            if args.model_name == "CNO":
                model = CNO3d(in_dim=in_dim, 
                            out_dim=args.out_dim_structure, 
                            in_size=in_size, 
                            N_layers=args.depth_structure,
                            dataset_name=model_type,
                            x0_is_use_noise=args.x0_is_use_noise,
                            stage=args.stage).to(device)
            elif args.model_name == "SiT_FNO":
                model = SiT_FNO(input_size=input_size, depth=args.depth_structure, hidden_size=args.hidden_size_structure, patch_size=args.patch_size_structure, 
                            num_heads=args.num_heads_structure, x0_is_use_noise=args.x0_is_use_noise, in_channels=in_dim,
                            out_channels=args.out_channels_structure,
                            stage=model_type, modes=args.modes_structure).to(device)
            ckpt_path = args.ntcouple_solid_checkpoint_path
        elif model_type == "ntcouple_neutron":
            if args.model_name == "CNO":
                model = CNO3d(in_dim=in_dim, 
                            out_dim=args.out_dim_neutron, 
                            in_size=in_size, 
                            N_layers=args.depth_neutron,
                            dataset_name=args.dataset_name,
                            x0_is_use_noise=args.x0_is_use_noise,
                            stage=args.stage).to(device)
            elif args.model_name == "SiT_FNO":
                model = SiT_FNO(input_size=input_size, depth=args.depth_neutron, hidden_size=args.hidden_size_neutron, patch_size=args.patch_size_neutron, 
                            num_heads=args.num_heads_neutron, x0_is_use_noise=args.x0_is_use_noise, in_channels=in_dim,
                            out_channels=args.out_channels_neutron,
                            stage=model_type, modes=args.modes_neutron).to(device)
            ckpt_path = args.ntcouple_neutron_checkpoint_path
        else:
            raise ValueError(f"Model type {model_type} not supported. Expected one of: ntcouple_neutron, ntcouple_solid, ntcouple_fluid")

    print(f"Loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("Loading main model (model_state_dict)")
    else:
        raise ValueError(f"Checkpoint {ckpt_path} not supported")
        
    model.load_state_dict(state_dict)
    model.eval()

    return model


def main(args):
    # Record start time
    total_start_time = time()
    
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load datasets - only support ntcouple
    if args.dataset_name == 'ntcouple':
        # NTcouple: Load three separate datasets for three fields
        use_ntcouple = True
        datasets = {}
        dataloaders = {}
        normalizers = {}
        
        for field in FIELD_ORDER:
            print(f"Loading {field} dataset...")
            datasets[field] = NTcoupleDataset(
                field=field,
                split=getattr(args, 'eval_split', 'couple'),
                n_samples=getattr(args, 'n_samples', None),
                data_root=getattr(args, 'dataset_path', None),
                normalize=True
            )
            dataloaders[field] = torch.utils.data.DataLoader(
                datasets[field],
                batch_size=args.test_batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=args.num_workers
            )
            normalizers[field] = NTcoupleNormalizer(field=field, device=device)
        
        # Load three models (pass dataset to infer correct input dimensions)
        models = {}
        for field in FIELD_ORDER:
            print(f"Loading {field} model...")
            models[field] = load_model(args, device, f"ntcouple_{field}", dataset=datasets[field])
        
        # Get update functions
        updates = default_ntcouple_updates()
        
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}. Only 'ntcouple' is supported.")

    # NTcouple multi-field inference
    print("\n" + "="*60)
    print("NTcouple Multi-Field Inference")
    print("="*60)
    
    # Get data from all three fields
    conds = {}
    targets = {}
    for field in FIELD_ORDER:
        cond, target, _, _, _ = next(iter(dataloaders[field]))
        conds[field] = cond.to(device)  # (B, C, T, H, W)
        targets[field] = target.to(device)  # (B, C, T, H, W)
        print(f"{field}", conds[field].shape)
        print(f"{field}", targets[field].shape)
        # neutron torch.Size([4, 2, 16, 64, 20])
        # neutron torch.Size([4, 1, 16, 64, 20])
        # solid torch.Size([4, 2, 16, 64, 8])
        # solid torch.Size([4, 1, 16, 64, 8])
        # fluid torch.Size([4, 1, 16, 64, 12])
        # fluid torch.Size([4, 4, 16, 64, 12])
        
    # Convert to (B, T, H, W, C) format for GenCP
    for field in FIELD_ORDER:
        conds[field] = conds[field].permute(0, 2, 3, 4, 1)  # (B, C, T, H, W) -> (B, T, H, W, C)
        targets[field] = targets[field].permute(0, 2, 3, 4, 1)  # (B, C, T, H, W) -> (B, T, H, W, C)
        
    # Extract boundary condition from neutron cond
    # conds["neutron"] format: (B, T, H, W=20, C=2)
    # channel 0: spatial_cond (fuel+fluid temperature)
    # channel 1: bc_expanded (boundary condition) <- we need this!
    bc_batch = conds["neutron"][:, :, :, :1, 1:2]  # (B, T, H, 1, 1) - extract channel 1 (bc)
    bc_batch = bc_batch.permute(0, 4, 1, 2, 3)  # (B, 1, T, H, 1)

    # Get target shapes for each field
    shapes = [targets[field].shape for field in FIELD_ORDER]
    model_list = [models[field] for field in FIELD_ORDER]
    
    # Perform inference based on model type
    use_surrogate = getattr(args, "use_surrogate", False)
    
    if use_surrogate:
        # Perform iterative coupling inference for three fields
        print(f"Starting surrogate inference with {getattr(args, 'num_inference_steps', 100)} steps...")
        
        # Create surrogate step function with boundary condition
        pred_update_coeff = getattr(args, "pred_update_coeff", 0.1)
        step_fn = create_ntcouple_surrogate_step_fn(models, pred_update_coeff, bc_batch=bc_batch)
        
        # Prepare target dict for surrogate inference
        target_norm_dict = {
            'neutron': targets['neutron'],
            'solid': targets['solid'],
            'fluid': targets['fluid']
        }
        
        # Perform surrogate inference with iterative coupling
        generated = surrogate_inference_ntcouple(
            input_norm=None,  # Not used in surrogate mode
            target_norm=target_norm_dict,
            num_steps=getattr(args, 'num_inference_steps', 1000),
            device=device,
            args=args,
            attrs=None,
            step_fn=step_fn
        )
    else:
        # Perform composed flow matching inference
        print(f"Starting composed flow matching with {args.num_sampling_steps} steps...")
        
        def normalize_fn(x, field, component="target", context=None):
            """Normalize with proper context and component."""
            return normalizers[field].normalize(x, field=field, component=component, context=context)
        
        def renormalize_fn(x, field, component="target", context=None):
            """Renormalize with proper context and component."""
            return normalizers[field].renormalize(x, field=field, component=component, context=context)
        
        generated = compose_flow_ntcouple(
            model_list=model_list,
            shapes=shapes,
            update_f=updates,
            normalize_fn=normalize_fn,
            renormalize_fn=renormalize_fn,
            timestep=args.num_sampling_steps,
            other_condition=[bc_batch],
            device=device,
            use_bc_inpainting=True,  # Always use BC inpainting for neutron field
        )
    
    # Apply Gaussian smoothing to denoise predictions (optional)
    manual_denoise = False
    if manual_denoise:
        from scipy.ndimage import gaussian_filter

        def denoise_flow_output(data, sigma=1.0):
            """
            Apply Gaussian smoothing to flow matching output
            
            Args:
                data: Array of shape [B, T, H, W, C]
                sigma: Smoothing strength, larger values mean more smoothing
            """
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            denoised = np.zeros_like(data)
            
            for b in range(data.shape[0]):
                for t in range(data.shape[1]):
                    for c in range(data.shape[4]):
                        denoised[b, t, :, :, c] = gaussian_filter(
                            data[b, t, :, :, c], 
                            sigma=sigma
                        )
            
            return torch.from_numpy(denoised)
        
        generated_denoised = []
        for i, (field, pred_norm) in enumerate(zip(FIELD_ORDER, generated)):
            if field == "neutron":
                samples_denoised = denoise_flow_output(pred_norm, sigma=0.8)
                pred_norm = samples_denoised.to(device)
                print(f"Applied Gaussian smoothing to {field} field (all channels)")
            elif field == "solid":
                samples_denoised = denoise_flow_output(pred_norm, sigma=0.8)
                pred_norm = samples_denoised.to(device)
                print(f"Applied Gaussian smoothing to {field} field (all channels)")
            elif field == "fluid":
                samples_denoised = denoise_flow_output(pred_norm, sigma=0.8)
                pred_norm[..., 1:3] = samples_denoised[..., 1:3].to(device)
                print(f"Applied Gaussian smoothing to {field} field (velocity_x and velocity_y channels)")
            generated_denoised.append(pred_norm)
        
        generated = generated_denoised
    
    # Convert predictions back to physical space and evaluate
    predictions_phys = {}
    for field, pred_norm in zip(FIELD_ORDER, generated):
        # Convert from (B, T, H, W, C) to (B, C, T, H, W) for normalizer
        pred_norm_converted = pred_norm.permute(0, 4, 1, 2, 3)  # (B, T, H, W, C) -> (B, C, T, H, W)
        target_converted = targets[field].permute(0, 4, 1, 2, 3)  # (B, T, H, W, C) -> (B, C, T, H, W)
        # Renormalize using unified normalization parameters
        pred_phys = normalizers[field].renormalize(pred_norm_converted.cpu(), field=field)
        target_phys = normalizers[field].renormalize(target_converted.cpu(), field=field)
        predictions_phys[field] = pred_phys
        targets[field] = target_phys

    # Print metrics 
    for field in FIELD_ORDER:
        pred = predictions_phys[field]
        target = targets[field]
        mse = mse_loss(pred, target).mean().item()
        
        # For fluid field, use per-channel average rel_l2 (more fair across different physical quantities)
        if field == "fluid" and pred.shape[1] > 1:
            channel_rel_l2 = []
            channel_names = ['temperature', 'velocity_x', 'velocity_y', 'pressure']
            for c in range(pred.shape[1]):
                c_rel_l2 = rel_l2_loss(pred[:, c:c+1], target[:, c:c+1]).mean().item()
                channel_rel_l2.append(c_rel_l2)
                ch_name = channel_names[c] if c < len(channel_names) else f"channel_{c}"
                print(f"  {field}.{ch_name} - Rel L2: {c_rel_l2:.6f}")
            rel_err = np.mean(channel_rel_l2)
            print(f"{field} - MSE: {mse:.6e}, Rel L2 (per-channel avg): {rel_err:.6f}")
        else:
            rel_err = rel_l2_loss(pred, target).mean().item()
            print(f"{field} - MSE: {mse:.6e}, Rel L2: {rel_err:.6f}")
    
    # Visualization
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    # Convert to numpy for visualization (use first batch)
    batch_idx = 0
    save_figs_path = getattr(args, 'save_figs_path', "./visualization_results/ntcouple_multi")
    
    for field in FIELD_ORDER:
        # predictions_phys and targets are in (B, C, T, H, W) format
        # Convert to (T, H, W, C) for visualization
        pred_vis = predictions_phys[field][batch_idx].permute(1, 2, 3, 0).cpu().numpy()  # (C, T, H, W) -> (T, H, W, C)
        target_vis = targets[field][batch_idx].permute(1, 2, 3, 0).cpu().numpy()  # (C, T, H, W) -> (T, H, W, C)
        
        # Get field-specific grid size [H, W]
        H, W = pred_vis.shape[1], pred_vis.shape[2]
        
        # Create visualizer for this field
        field_save_dir = os.path.join(save_figs_path, field)
        visualizer = FluidFieldVisualizer(
            args=args,
            grid_size=[H, W],
            save_dir=field_save_dir,
            create_timestamp_folder=True
        )
        print(f"  Visualizing {field} field (grid_size: [{H}, {W}])...")
        
        if field == "neutron":
            # Neutron flux: single channel (T, H, W, 1)
            visualizer.visualize_field(
                u_pred=pred_vis[-1, ..., 0],  # Last time step, first channel
                u_true=target_vis[-1, ..., 0],
                title="Neutron Flux Field (t=15)",
                save_name="neutron_field.png",
                cmap='viridis'
            )
            visualizer.visualize_time_series_gif(
                u_pred=pred_vis[..., 0],  # (T, H, W)
                u_true=target_vis[..., 0],
                title="Time Series Animation - Neutron Flux",
                save_name="time_series_neutron.gif",
                fps=2,
                show_colorbar=True
            )
        
        elif field == "solid":
            # Fuel temperature: single channel (T, H, W, 1)
            visualizer.visualize_field(
                u_pred=pred_vis[-1, ..., 0],  # Last time step, first channel
                u_true=target_vis[-1, ..., 0],
                title="Fuel Temperature Field (t=15)",
                save_name="solid_field.png",
                cmap='hot'
            )
            visualizer.visualize_time_series_gif(
                u_pred=pred_vis[..., 0],  # (T, H, W)
                u_true=target_vis[..., 0],
                title="Time Series Animation - Fuel Temperature",
                save_name="time_series_solid.gif",
                fps=2,
                show_colorbar=True
            )
        
        elif field == "fluid":
            # Fluid: 4 channels (temperature, vx, vy, pressure) (T, H, W, 4)
            channel_names = ["Temperature", "Velocity-X", "Velocity-Y", "Pressure"]
            save_names = ["fluid_temp", "fluid_vx", "fluid_vy", "fluid_p"]
            cmaps = ['hot', 'RdBu_r', 'RdBu_r', 'viridis']
            
            for c, (ch_name, save_name, cmap) in enumerate(zip(channel_names, save_names, cmaps)):
                visualizer.visualize_field(
                    u_pred=pred_vis[-1, ..., c],  # Last time step, channel c
                    u_true=target_vis[-1, ..., c],
                    title=f"{ch_name} Field (t=15)",
                    save_name=f"{save_name}_field.png",
                    cmap=cmap
                )
                visualizer.visualize_time_series_gif(
                    u_pred=pred_vis[..., c],  # (T, H, W)
                    u_true=target_vis[..., c],
                    title=f"Time Series Animation - {ch_name}",
                    save_name=f"time_series_{save_name}.gif",
                    fps=2,
                    show_colorbar=True
                )
        
        print(f"    Visualizations saved to: {visualizer.save_dir}")
    
    print(f"\nAll visualizations saved to: {save_figs_path}")
    
    # Calculate and print total runtime
    total_end_time = time()
    total_runtime = total_end_time - total_start_time
    hours = int(total_runtime // 3600)
    minutes = int((total_runtime % 3600) // 60)
    seconds = total_runtime % 60
    
    print("\n" + "="*60)
    print("Runtime Statistics")
    print("="*60)
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Total runtime: {hours:02d}:{minutes:02d}:{seconds:05.2f} (HH:MM:SS.ss)")
    print("NTcouple inference completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    mode = "ODE" # ["ODE", "SDE"]

    parser.add_argument("--config", type=str, default="configs/sample_ntcouple.yaml")
    parser.add_argument("--num-sampling-steps", type=int, default=10,
                       help="Number of flow matching steps")
    parser.add_argument("--seed", type=int, default=0)
    
    # NTcouple specific arguments
    parser.add_argument("--ntcouple-neutron-checkpoint-path", type=str, default=None,
                       help="Path to neutron model checkpoint for NTcouple")
    parser.add_argument("--ntcouple-solid-checkpoint-path", type=str, default=None,
                       help="Path to solid model checkpoint for NTcouple")
    parser.add_argument("--ntcouple-fluid-checkpoint-path", type=str, default=None,
                       help="Path to fluid model checkpoint for NTcouple")
    parser.add_argument(
        "--eval-split",
        type=str,
        default="couple_val",
        choices=['decouple_train', 'decouple_val', 'couple_train', 'couple_val'],
        help="Evaluation split for NTcouple (decouple or couple variants only)",
    )
    parser.add_argument("--n-samples", type=int, default=None,
                       help="Number of samples to evaluate (None for all)")
    parser.add_argument("--max-eval-batches", type=int, default=1,
                       help="Maximum number of batches to evaluate")
    parser.add_argument("--save-figs-path", type=str, default="./visualization_results/ntcouple_multi",
                       help="Path to save visualization figures")

    parse_transport_args(parser)
    parser.add_argument("--use-torchcfm", action="store_true", default=False, 
                       help="Use torchcfm for inference")
    parser.add_argument("--use-surrogate", action="store_true", default=False,
                       help="Use surrogate models for inference")
    parser.add_argument("--num-inference-steps", type=int, default=100,
                       help="Number of inference steps for surrogate models")
    parser.add_argument("--pred-update-coeff", type=float, default=0.1,
                       help="Update coefficient for surrogate iterative refinement")

    mode = "ODE"  # Only ODE mode supported for ntcouple
    parse_ode_args(parser)
    
    args = add_args_from_config(parser)
    main(args)