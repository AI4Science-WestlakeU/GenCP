# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from pre-trained SiT models for FSI (Fluid-Structure Interaction).
Implements dual-field interactive autoregressive inference.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torchcfm
from model.SiT import SiT
from model.cno import CNO3d
from model.fno import FNO3d
from model.SiT_FNO import SiT_FNO
from model.cno_surrogate import CNO3d as CNO3d_surrogate
from model.sit_fno_surrogate import SiT_FNO as SiT_FNO_surrogate
from utils.utils import parse_ode_args, parse_sde_args, parse_transport_args, find_model, add_args_from_config, rel_l2_loss
import argparse
from time import time

from utils.visualize import FluidFieldVisualizer

from data.turek_hron_dataset import TurekHronDataset
from data.double_cylinder_dataset import DoubleCylinderDataset

def create_couple_step_fn_fsi(model_fluid, model_structure, device, use_torchcfm=True, mode="ODE", flag = "jacobi", **sde_kwargs):
    
    """Create single-step sampling function supporting ODE and SDE modes"""
    
    print(f"flag: {flag}")
    if mode == "ODE":
        def single_step_cfm_ode(x, t, dt, z_init, **model_kwargs):
            """CFM ODE single-step advance function"""
            if isinstance(t, (int, float)):
                tb = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
            else:
                tb = t.to(x.device)
            
            cfm = torchcfm.ConditionalFlowMatcher()

            x0 = model_kwargs.get("x0")
            gt = model_kwargs.get("gt")
            cond = model_kwargs.get("cond")

            x00 = x0[:, -1:]
            x000 = x00.repeat(1, 12, 1, 1, 1)

            _, zt, _ = cfm.sample_location_and_conditional_flow(z_init.float().to(device), x000.float().to(device), tb.to(device))
            
            x_fluid_cond_struct0 = x.clone()
            x_fluid_cond_struct0[..., -1:] = zt[..., -1:]
            x_struct_cond_fluid0 = x.clone()
            x_struct_cond_fluid0[..., :-1] = zt[..., :-1]

            if flag == "jacobi":
                vt_fluid = model_fluid(x, tb, x0, model_kwargs.get("cond"))
                vt_structure = model_structure(x, tb, x0, model_kwargs.get("cond"))
                vt = torch.cat([vt_fluid, vt_structure], dim=-1)
                return x + vt * dt
            elif flag == "lie":
                vt_fluid = model_fluid(x, tb, x0, model_kwargs.get("cond"))
                x_fluid = x + vt_fluid* dt
                x_fluid[...,-1:] = x[...,-1:]
                vt_structure = model_structure(x_fluid, tb, x0, model_kwargs.get("cond"))
                x_structure = x_fluid + vt_structure * dt
                x_structure[..., :-1] = x_fluid[..., :-1]
                x = x_structure
                return x
            elif flag == "strang":
                vt_fluid = model_fluid(x, tb, x0, model_kwargs.get("cond"))
                x_fluid = x + vt_fluid* dt * 0.5
                x_fluid[...,-1:] = x[...,-1:]

                vt_structure = model_structure(x_fluid, tb, x0, model_kwargs.get("cond"))
                x_structure = x_fluid + vt_structure * dt
                x_structure[..., :-1] = x_fluid[..., :-1]

                vt_fluid_2 = model_fluid(x_structure, tb, x0, model_kwargs.get("cond"))
                x = x_structure + vt_fluid_2 * dt * 0.5
                x[..., -1:] = x_structure[..., -1:]

                return x

        return single_step_cfm_ode
    
def create_couple_step_fn_surrogate(model_fluid, model_structure, pred_update_coeff, dataset_name, input_step, output_step):
    assert output_step > input_step, "output_step must be greater than input_step"
    assert output_step % input_step == 0, "output_step must be divisible by input_step"
    
    def update_f_fluid(input, structure_pred_prev):
        input = input.repeat(1, output_step // input_step, 1, 1, 1)
        z_prev = torch.cat([input, structure_pred_prev], dim=-1)
        z_prev = z_prev.permute(0, 4, 1, 2, 3)
        fluid_pred = model_fluid(z_prev)
        fluid_pred = fluid_pred.permute(0, 2, 3, 4, 1)
        return fluid_pred
    
    def update_f_structure(input, fluid_pred_prev):
        input = input.repeat(1, output_step // input_step, 1, 1, 1)
        z_prev = torch.cat([input, fluid_pred_prev], dim=-1)
        z_prev = z_prev.permute(0, 4, 1, 2, 3)
        structure_pred = model_structure(z_prev)
        structure_pred = structure_pred.permute(0, 2, 3, 4, 1)
        return structure_pred
    
    c = pred_update_coeff
    
    def single_step_surrogate(input, fluid_pred_prev, structure_pred_prev):
        fluid_pred = update_f_fluid(input, structure_pred_prev)
        structure_pred = update_f_structure(input, fluid_pred_prev)
        
        fluid_pred = c * fluid_pred + (1 - c) * fluid_pred_prev
        structure_pred = c * structure_pred + (1 - c) * structure_pred_prev
        
        loss1 = torch.norm(fluid_pred - fluid_pred_prev, dim=2, p=2)
        loss2 = torch.norm(structure_pred - structure_pred_prev, dim=2, p=2)
        loss = (loss1 + loss2).mean()

        return fluid_pred, structure_pred, loss


    return single_step_surrogate

def load_model(args, device, model_type="fluid"):
    """Load model and create sampler for specified model type"""

    if getattr(args, "use_torchcfm", False):
        if model_type == "fluid":
            if args.model_name == "SiT":    
                model = SiT(input_size=args.input_size, depth=args.depth_fluid, hidden_size=args.hidden_size_fluid, patch_size=args.patch_size_fluid, 
                            num_heads=args.num_heads_fluid, x0_is_use_noise=args.x0_is_use_noise, in_channels=args.in_channels,
                            stage=model_type).to(device)
            elif args.model_name == "CNO":
                model = CNO3d(in_dim=args.in_dim, 
                            out_dim=args.out_dim_fluid, 
                            in_size=args.in_size, 
                            N_layers=args.depth_fluid,
                            dataset_name=args.dataset_name,
                            x0_is_use_noise=args.x0_is_use_noise,
                            stage=args.stage).to(device)
            elif args.model_name == "FNO":
                model = FNO3d(modes1=args.modes_t_fluid, 
                            modes2=args.modes_x_fluid, 
                            modes3=args.modes_y_fluid, 
                            n_layers=args.depth_fluid, 
                            width=args.width_structure, 
                            shape_in=(args.input_step, args.input_size[0], args.input_size[1], args.in_channels), 
                            shape_out=(args.output_step, args.input_size[0], args.input_size[1], args.out_channels_fluid),
                            dataset_name=model_type,
                            x0_is_use_noise=args.x0_is_use_noise,
                            stage=args.stage).to(device)
            elif args.model_name == "SiT_FNO":
                model = SiT_FNO(input_size=args.input_size, depth=args.depth_fluid, hidden_size=args.hidden_size_fluid, patch_size=args.patch_size_fluid, 
                            num_heads=args.num_heads_fluid, x0_is_use_noise=args.x0_is_use_noise, in_channels=args.in_channels,
                            out_channels=args.out_channels_fluid,
                            stage=model_type, modes=args.modes_fluid).to(device)
            ckpt_path = args.fluid_checkpoint_path
        elif model_type == "structure":  # structure
            if args.model_name == "SiT":
                model = SiT(input_size=args.input_size, depth=args.depth_structure, hidden_size=args.hidden_size_structure, patch_size=args.patch_size_structure, 
                                num_heads=args.num_heads_structure, x0_is_use_noise=args.x0_is_use_noise, in_channels=args.in_channels,
                                stage=model_type).to(device)
            elif args.model_name == "CNO":
                model = CNO3d(in_dim=args.in_dim, 
                            out_dim=args.out_dim_structure, 
                            in_size=args.in_size, 
                            N_layers=args.depth_structure,
                            dataset_name=model_type,
                            x0_is_use_noise=args.x0_is_use_noise,
                            stage=args.stage).to(device)
            elif args.model_name == "FNO":
                model = FNO3d(modes1=args.modes_t_structure, 
                            modes2=args.modes_x_structure, 
                            modes3=args.modes_y_structure, 
                            n_layers=args.depth_structure, 
                            width=args.width_structure, 
                            shape_in=(args.input_step, args.input_size[0], args.input_size[1], args.in_channels), 
                            shape_out=(args.output_step, args.input_size[0], args.input_size[1], args.out_channels_structure),
                            dataset_name=model_type,
                            x0_is_use_noise=args.x0_is_use_noise,
                            stage=args.stage).to(device)
            elif args.model_name == "SiT_FNO":
                model = SiT_FNO(input_size=args.input_size, depth=args.depth_structure, hidden_size=args.hidden_size_structure, patch_size=args.patch_size_structure, 
                            num_heads=args.num_heads_structure, x0_is_use_noise=args.x0_is_use_noise, in_channels=args.in_channels,
                            out_channels=args.out_channels_structure,
                            stage=model_type, modes=args.modes_structure).to(device)
            ckpt_path = args.structure_checkpoint_path
        else:
            raise ValueError(f"Model type {model_type} not supported")
        
    elif getattr(args, "use_surrogate", False):
            if model_type == "fluid":
                if args.model_name == "CNO":
                    model = CNO3d_surrogate(in_dim=args.in_dim_fluid,
                        out_dim=args.out_dim_fluid,
                        in_size=args.in_size_fluid,
                        N_layers=args.depth_fluid,
                        channel_multiplier=args.channel_multiplier_fluid,
                    ).to(device)
                elif args.model_name == "SiT_FNO":
                    model = SiT_FNO_surrogate(input_size=args.input_size, depth=args.depth_fluid, hidden_size=args.hidden_size_fluid, patch_size=args.patch_size, 
                                num_heads=args.num_heads_fluid, in_channels=args.in_channels_fluid,
                                out_channels=args.out_channels_fluid, modes=args.modes_fluid).to(device)
                else:
                    raise ValueError(f"Model type {args.model_name} not supported")
                ckpt_path = args.fluid_checkpoint_path
            

            elif model_type == "structure":
                if args.model_name == "CNO":
                    model = CNO3d_surrogate(in_dim=args.in_dim_structure,
                        out_dim=args.out_dim_structure,
                        in_size=args.in_size_structure,
                        N_layers=args.depth_structure,
                        channel_multiplier=args.channel_multiplier_structure,
                    ).to(device)
                elif args.model_name == "SiT_FNO":
                    model = SiT_FNO_surrogate(input_size=args.input_size, depth=args.depth_structure, hidden_size=args.hidden_size_structure, patch_size=args.patch_size, 
                                num_heads=args.num_heads_structure, in_channels=args.in_channels_structure,
                                out_channels=args.out_channels_structure, modes=args.modes_structure).to(device)
                else:
                    raise ValueError(f"Model type {args.model_name} not supported")
                ckpt_path = args.structure_checkpoint_path
            
            else:
                raise ValueError(f"Model type {model_type} not supported")

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


def coupled_flow_sampling_algorithm_cp(couple_step_fn, input_norm, target_norm, num_flow_steps, device, args, attrs, physical_step=0):
    """
    CFM-based multi-physics coupled sampling implementation (coupled data training version)
    """
    print(f"Starting algorithm-based CFM coupled sampling with {num_flow_steps} steps...")
    

    z_flow = torch.randn_like(target_norm, device=device)
    z_init = z_flow.clone()
    
    dt = 1.0 / num_flow_steps
    
    for s in range(num_flow_steps):
        cfm_time = s * dt
     
        model_kwargs = dict(x0=input_norm, gt=target_norm, cond=attrs)
        z_flow = couple_step_fn(z_flow, cfm_time, dt, z_init, **model_kwargs)
        
        if (s + 1) % 1 == 0:
            print(f"  Algorithm flow step {s}/{num_flow_steps} completed")
    
    final_result = z_flow

    return final_result

def autoregressive_inference(input_norm, target_norm, num_steps, device, args, attrs, couple_step_fn):
    """Perform autoregressive inference with dual-field interaction (all data already normalized)"""
    
    initial_state = input_norm.clone()
    
    predictions_norm = []
    
    print(f"Starting autoregressive inference for {num_steps} steps...")
    start_time = time()
    
    use_coupled_sampling = getattr(args, 'use_coupled_sampling', False)
    print(f"use_coupled_sampling (1+1)×10...: {use_coupled_sampling}")
    
    for step in range(num_steps):
        print(f"Step {step + 1}/{num_steps}")

        if use_coupled_sampling and getattr(args, "use_torchcfm", False):
            next_state = coupled_flow_sampling_algorithm_cp(couple_step_fn=couple_step_fn, 
                            input_norm=initial_state, target_norm=target_norm, num_flow_steps=args.num_sampling_steps, 
                            device=device, args=args, attrs=attrs, physical_step=step)
            
        # Store this step's prediction (squeeze out the time dimension)
        predictions_norm.append(next_state)  # [batch, height, width, 4]

        print(f"Step {step + 1} completed")
    
    predictions_norm = predictions_norm[-1]
    print(f"Autoregressive inference took {time() - start_time:.2f} seconds.")
    return predictions_norm


def surrogate_inference(input_norm, target_norm, num_steps, device, args, attrs, couple_step_fn):
    """Perform surrogate inference with dual-field interaction (all data already normalized)"""
    
    predictions_norm = []
    print(f"Starting surrogate inference for {num_steps} steps...")
    start_time = time()
    
    pred_init = torch.ones_like(target_norm) * 0.5
    fluid_pred_prev = pred_init[..., :-1]
    structure_pred_prev = pred_init[..., -1:]
    
    for step in range(num_steps):
        fluid_pred, structure_pred, loss = couple_step_fn(input_norm, fluid_pred_prev, structure_pred_prev)
        result = torch.cat([fluid_pred.clone(), structure_pred.clone()], dim=-1)
        predictions_norm.append(result)
        fluid_pred_prev = fluid_pred
        structure_pred_prev = structure_pred
        
        print(f"Step {step + 1}/{num_steps}, loss: {loss.item():.6f}")

    predictions_norm = predictions_norm[-1]
    print(f"Surrogate inference took {time() - start_time:.2f} seconds.")
    return predictions_norm

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load datasets
    if args.dataset_name == 'turek_hron_data':
        train_dataset = TurekHronDataset(dataset_path=args.dataset_path, length=args.length, input_size=args.input_step, output_size=args.output_step, stride=args.stride, mode='train', num_delta_t=args.num_delta_t,
                                                stage=args.stage, dt=args.dt)
        val_dataset = TurekHronDataset(dataset_path=args.dataset_path, length=args.length, input_size=args.input_step, output_size=args.output_step, stride=args.stride, mode='val', num_delta_t=args.num_delta_t,
                                                stage='couple', dt=args.dt)
    if args.dataset_name == 'double_cylinder_data':
        train_dataset = DoubleCylinderDataset(dataset_path=args.dataset_path, length=args.length, input_size=args.input_step, output_size=args.output_step, stride=args.stride, mode='train', num_delta_t=args.num_delta_t,
                                                stage=args.stage, dt=args.dt)
        val_dataset = DoubleCylinderDataset(dataset_path=args.dataset_path, length=args.length, input_size=args.input_step, output_size=args.output_step, stride=args.stride, mode='val', num_delta_t=args.num_delta_t,
                                                stage='couple', dt=args.dt)
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, 
                                            shuffle=True, pin_memory=True, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size,
                                            shuffle=False, pin_memory=True, num_workers=args.num_workers)

    # Setup data normalizer 
    from data.data_normalizer import RangeNormalizer
    data_normalizer = RangeNormalizer(train_dataset, device=device, mode='train',
                                        batch_size=args.train_batch_size)

    # Load both models and samplers
    print("Loading fluid model...")
    fluid_model = load_model(args, device, "fluid")
    print("Loading structure model...")
    structure_model = load_model(args, device, "structure")

    print("Loading coupling model...")
    if getattr(args, "use_torchcfm", False):
        couple_step_fn = create_couple_step_fn_fsi(fluid_model, structure_model, device, flag=args.flag)
    
    elif getattr(args, "use_surrogate", False):
        couple_step_fn = create_couple_step_fn_surrogate(fluid_model, structure_model, args.pred_update_coeff, args.dataset_name, args.input_step, args.output_step)
    
    else:
        raise ValueError("Not supported paradigm")

    # Get initial conditions from validation dataset
    rel_l2_fluid_u = 0.0
    rel_l2_fluid_v = 0.0
    rel_l2_fluid_p = 0.0
    rel_l2_structure = 0.0

    num = 0
    
    start_time = time()
    for input, target, grid_x, grid_y, attrs in val_dataloader:

        input = input.to(device)
        target = target.to(device)

        attrs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in attrs.items()}
        
        # Normalize entire batch at once
        input_norm, target_norm = data_normalizer.preprocess(input, target)

        # Perform autoregressive inference for entire batch
        start_time = time()
        
        if getattr(args, "use_torchcfm", False):
            predictions_norm = autoregressive_inference(
                input_norm, target_norm, args.num_inference_steps, device, args, attrs,
                couple_step_fn
            )
        
        elif getattr(args, "use_surrogate", False):
            predictions_norm = surrogate_inference(
                input_norm, target_norm, args.num_inference_steps, device, args, attrs,
                couple_step_fn
            )
        
        else:
            raise ValueError("Not supported paradigm")
        
        print(f"Autoregressive inference took {time() - start_time:.2f} seconds.")

        # Get final predictions from the AR trajectory
        _, final_prediction = data_normalizer.postprocess(predictions_norm, predictions_norm)

        manual_denoise = True
        if manual_denoise:
            from scipy.ndimage import gaussian_filter
            import numpy as np

            def denoise_flow_output(data, sigma=1.0):
                """Apply Gaussian smoothing to flow matching output"""
                data = data.cpu()
                denoised = np.zeros_like(data)
                
                for b in range(data.shape[0]):
                    for t in range(data.shape[1]):
                        for c in range(data.shape[4]):
                            denoised[b, t, :, :, c] = gaussian_filter(
                                data[b, t, :, :, c], 
                                sigma=sigma
                            )
                
                return denoised

            samples_denoised = denoise_flow_output(final_prediction, sigma=0.8)
            final_prediction[..., -1:] = torch.tensor(samples_denoised[..., -1:]).to(device)

        # Visualize results for first sample
        visualizer = FluidFieldVisualizer(args=args, grid_size=args.input_size, save_dir=args.save_figs_path, create_timestamp_folder=True)
        
        if visualizer is not None:
            sdf_mask = torch.where(target[..., -1:] > args.sdf_threshold, torch.tensor(1.0), torch.tensor(0.0))
       
            final_fluid_u = final_prediction[..., 0:1]  # Shape: [batch, time, height, width, 1]
            final_fluid_v = final_prediction[..., 1:2]  # Shape: [batch, time, height, width, 1]
            final_fluid_p = final_prediction[..., 2:3]  # Shape: [batch, time, height, width, 1]
            final_structure = final_prediction[..., -1:]  # Shape: [batch, time, height, width, 1]

            rel_l2_fluid_u += rel_l2_loss(final_fluid_u*sdf_mask, target[..., 0:1]*sdf_mask).mean().item()
            rel_l2_fluid_v += rel_l2_loss(final_fluid_v*sdf_mask, target[..., 1:2]*sdf_mask).mean().item()
            rel_l2_fluid_p += rel_l2_loss(final_fluid_p*sdf_mask, target[..., 2:3]*sdf_mask).mean().item()
            rel_l2_structure += rel_l2_loss(final_structure*sdf_mask, target[..., 3:4]*sdf_mask).mean().item()

            num += 1

            print(f"rel_l2_fluid_u: {rel_l2_fluid_u/(num):.6f}")
            print(f"rel_l2_fluid_v: {rel_l2_fluid_v/(num):.6f}")
            print(f"rel_l2_fluid_p: {rel_l2_fluid_p/(num):.6f}")
            print(f"rel_l2_structure: {rel_l2_structure/(num):.6f}")

            final_prediction = final_prediction.cpu().numpy()
            target = target.cpu().numpy()

            sdf_mask = np.where(target[0,-1,...,-1] > args.sdf_threshold, 1.0, 0.0)# .cpu().numpy()
            sdf_video_mask = np.where(target[0,...,-1] > args.sdf_threshold, 1.0, 0.0)# .cpu().numpy()

            if True:
                # Visualize velocity field (first channel of fluid)
                visualizer.visualize_field(
                    u_pred=final_prediction[0, -1, ..., 0] * sdf_mask,  # Velocity x
                    u_true=target[0, -1, ..., 0] * sdf_mask,  # Ground truth velocity x
                    title="Velocity-X Field",
                    save_name="final_velocity_field_u.png"
                )
                
                visualizer.visualize_field(
                    u_pred=final_prediction[0, -1, ..., 1] * sdf_mask,  # Velocity x
                    u_true=target[0, -1, ..., 1] * sdf_mask,  # Ground truth velocity x
                    title="Velocity-Y Field",
                    save_name="final_velocity_field_v.png"
                )
                
                visualizer.visualize_field(
                    u_pred=final_prediction[0, -1, ..., 2] * sdf_mask,  # Velocity x
                    u_true=target[0, -1, ..., 2] * sdf_mask,  # Ground truth velocity x
                    title="Pressure Field",
                    save_name="final_velocity_field_p.png"
                )
                
                # Visualize structure field
                visualizer.visualize_sdf_field(
                    sdf_pred=final_prediction[0, -1, ..., 3],  # SDF
                    sdf_true=target[0, -1, ..., 3],  # Ground truth SDF
                    title="SDF Field",
                    save_name="final_structure_field.png"
                )

                visualizer.visualize_time_series_gif(
                    u_pred=final_prediction[0, ..., 0] * sdf_video_mask,  # Velocity x [b, t, h, w, c]
                    u_true=target[0, ..., 0] * sdf_video_mask,  # Ground truth velocity x
                    title="Time Series Animation - Velocity X",
                    save_name="time_series_animation_u.gif",
                    fps=2,
                    show_colorbar=True
                )

                visualizer.visualize_time_series_gif(
                    u_pred=final_prediction[0, ..., 1] * sdf_video_mask,  # Velocity x [b, t, h, w, c]
                    u_true=target[0, ..., 1] * sdf_video_mask,  # Ground truth velocity x
                    title="Time Series Animation - Velocity V",
                    save_name="time_series_animation_v.gif",
                    fps=2,
                    show_colorbar=True
                )

                visualizer.visualize_time_series_gif(
                    u_pred=final_prediction[0, ..., 2] * sdf_video_mask,  # Velocity x [b, t, h, w, c]
                    u_true=target[0, ..., 2] * sdf_video_mask,  # Ground truth velocity x
                    title="Time Series Animation - Velocity P",
                    save_name="time_series_animation_p.gif",
                    fps=2,
                    show_colorbar=True
                )

                visualizer.visualize_sdf_time_series_gif(
                    sdf_pred=final_prediction[0, ..., 3],  # Structure field [t, h, w]
                    sdf_true=target[0, ..., 3],  # Ground truth structure field
                    title="Time Series Animation - Structure (SDF)",
                    save_name="time_series_animation_structure.gif",
                    fps=2
                )

        print("Inference completed. Results saved to:", args.save_figs_path)
        
        print(f"model_name: {args.model_name}")

        if num >= 5:
            break
        
    end_time = time()
    print(f"Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    mode = "ODE" # ["ODE", "SDE"]

    parser.add_argument("--config", type=str, default="configs/sample_fsi.yaml")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=64)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    
    # New arguments for dual-field inference
    parser.add_argument("--fluid-checkpoint-path", type=str, default="checkpoints/fluid_model.pth",
                       help="Path to fluid model checkpoint")
    parser.add_argument("--structure-checkpoint-path", type=str, default="checkpoints/structure_model.pth",
                       help="Path to structure model checkpoint")
    parser.add_argument("--num-inference-steps", type=int, default=10,
                       help="Number of autoregressive inference steps")
    parser.add_argument("--mode", type=str, default="ODE", choices=["ODE", "SDE"])
    
    # Arguments for coupled sampling
    parser.add_argument("--use-coupled-sampling", action="store_true", default=False,
                       help="Use alternating coupled flow sampling (1+1)×N instead of weak coupling")
    parser.add_argument("--use-algorithm-coupling", action="store_true", default=False,
                       help="Use the algorithm-based coupling method from the provided pseudocode")
    parser.add_argument("--use-fixed-coupling", action="store_true", default=True,
                       help="Use the fixed coupling method with stable targets")
    parser.add_argument("--debug-flow", action="store_true", default=False,
                       help="Enable debug mode to save visualization for each flow step")

    parse_transport_args(parser)
    parser.add_argument("--use-torchcfm", action="store_true", default=True, help="Use torchcfm inference branch (does not create transport/Sampler)")

    if mode == "ODE":
        parse_ode_args(parser)
    elif mode == "SDE":
        parse_sde_args(parser)
    
    args = add_args_from_config(parser)
    main(args)