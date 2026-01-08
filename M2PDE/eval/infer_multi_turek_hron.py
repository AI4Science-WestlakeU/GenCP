# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from pre-trained SiT models for FSI (Fluid-Structure Interaction).
Implements dual-field interactive autoregressive inference.
"""
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
from time import time
import os
import yaml

import numpy as np

from filepath import ABSOLUTE_PATH
from data.turek_hron import TurekHronDataset, RangeNormalizer
from models.unet3d import Unet3D
from models.sit_fno import SiT_FNO
from models.cno import CNO3d
from functools import partial
from paradigms.diffusion import GaussianDiffusion
from utils import get_parameter_net, FluidFieldVisualizer
from collections import OrderedDict

from tqdm import tqdm

DEFAULT_EVAL_SAMPLES = 50

def load_model_ema(model_path):
    state_dict = torch.load(model_path)["ema"]
    prefix = "online_model.model."
    online_model_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            online_model_dict[k[len(prefix):]] = v
    return online_model_dict

def load_model(model_path):
    state_dict = torch.load(model_path)["model"]
    prefix = "model."
    model_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(prefix):
            model_dict[k[len(prefix):]] = v
    return model_dict

def rel_l2_loss(pred, target):
    b = pred.size(0)
    pred = pred.reshape(b, -1)
    target = target.reshape(b, -1)
    return (torch.norm(pred - target, dim=1) / torch.norm(target, dim=1))

def prepare_data(stage, input, target):

    input = input.repeat(1, target.shape[1] // input.shape[1], 1, 1, 1) # (b, 3, h, w, 4) -> (b, 12, h, w, 4)
    if stage == "fluid":
        cond = target[..., -1:] # (b, 12, h, w, 1)
        cond = torch.cat([input, cond], dim=-1) # (b, 12, h, w, 5)
        data = target[..., :-1] # (b, 12, h, w, 3)
    elif stage == "structure":
        cond = target[..., :-1] # (b, 12, h, w, 3)
        cond = torch.cat([input, cond], dim=-1) # (b, 12, h, w, 7)
        data = target[..., -1:] # (b, 12, h, w, 1)
    elif stage == "couple":
        pass
    else:
        raise ValueError(f"Invalid stage: {stage}")
    
    return cond, data


def create_couple_step_fn(model_fluid, model_structure, input_step, output_step):
    assert output_step > input_step, "output_step must be greater than input_step"
    assert output_step % input_step == 0, "output_step must be divisible by input_step"

    def update_fluid(alpha, t, fluid_pred, structure_estimate, structure_estimate_before, input_data):
        """Update function for fluid field following compose logic"""
        # Use estimated structure field as condition
        input = input_data.repeat(1, output_step // input_step, 1, 1, 1) # (b, 5, h, w, 4) -> (b, 10, h, w, 4)
        structure_cond = structure_estimate  # (b, 10, h, w, 1)

        # Create condition by concatenating input and estimated structure
        cond = torch.cat([input, structure_cond], dim=-1) # (b, 10, h, w, 5)
        cond = cond.permute(0, 4, 1, 2, 3) # (b, t, h, w, c) -> (b, c, t, h, w)

        # Apply alpha blending if not first iteration
        if alpha < 1.0:
            cond_before = torch.cat([input, structure_estimate_before], dim=-1).permute(0, 4, 1, 2, 3)
            cond = alpha * cond + (1 - alpha) * cond_before
            
        fluid_pred = fluid_pred.permute(0, 4, 1, 2, 3) # (b, c, t, h, w) -> (b, t, h, w, c)

        fluid_pred, fluid_estimate = model_fluid.p_sample(fluid_pred.clone(), t, cond)
        
        fluid_pred = fluid_pred.permute(0, 2, 3, 4, 1) # (b, t, h, w, c) -> (b, c, t, h, w)
        fluid_estimate = fluid_estimate.permute(0, 2, 3, 4, 1) # (b, c, t, h, w) -> (b, t, h, w, c)

        return fluid_pred, fluid_estimate

    def update_structure(alpha, t, structure_pred, fluid_estimate, fluid_estimate_before, input_data):
        """Update function for structure field following compose logic"""
        # Use estimated fluid field as condition
        input = input_data.repeat(1, output_step // input_step, 1, 1, 1) # (b, 5, h, w, 4) -> (b, 10, h, w, 4)
        fluid_cond = fluid_estimate  # (b, 10, h, w, 3)

        # Create condition by concatenating input and estimated fluid
        cond = torch.cat([input, fluid_cond], dim=-1) # (b, 10, h, w, 7)
        cond = cond.permute(0, 4, 1, 2, 3) # (b, t, h, w, c) -> (b, c, t, h, w)

        # Apply alpha blending if not first iteration
        if alpha < 1.0:
            cond_before = torch.cat([input, fluid_estimate_before], dim=-1).permute(0, 4, 1, 2, 3)
            cond = alpha * cond + (1 - alpha) * cond_before

        structure_pred = structure_pred.permute(0, 4, 1, 2, 3) # (b, c, t, h, w) -> (b, t, h, w, c)

        structure_pred, structure_estimate = model_structure.p_sample(structure_pred.clone(), t, cond)

        structure_pred = structure_pred.permute(0, 2, 3, 4, 1) # (b, t, h, w, c) -> (b, c, t, h, w)
        structure_estimate = structure_estimate.permute(0, 2, 3, 4, 1) # (b, c, t, h, w) -> (b, t, h, w, c)

        return structure_pred, structure_estimate

    return update_fluid, update_structure


def parse_model_kwargs(exapmle_cond, exapmle_data, model_name, model_ckpt_path, stage):
    model_config_yaml_path = os.path.join(os.path.dirname(model_ckpt_path), "config.yaml")
    with open(model_config_yaml_path, "r") as file:
        model_config = yaml.load(file, Loader=yaml.FullLoader)
    
    suffix = f"_{stage}"

    if model_name == "Unet":
        model_kwargs = {
            f"dim{suffix}": model_config["dim"],
            f"out_dim{suffix}": exapmle_data.shape[1],
            f"cond_channels{suffix}": exapmle_cond.shape[1],
        }
    elif model_name == "SiT_FNO":
        model_kwargs = {
            f"in_channels{suffix}": exapmle_cond.shape[1] + exapmle_data.shape[1],
            f"out_channels{suffix}": exapmle_data.shape[1],
            f"input_size{suffix}": model_config["input_size"],
            f"depth{suffix}": model_config["depth"],
            f"hidden_size{suffix}": model_config["hidden_size"],
            f"patch_size{suffix}": model_config["patch_size"],
            f"num_heads{suffix}": model_config["num_heads"],
            f"modes{suffix}": model_config["modes"],
        }
    elif model_name == "CNO":
        model_kwargs = {
            f"in_dim{suffix}": exapmle_cond.shape[1] + exapmle_data.shape[1],
            f"out_dim{suffix}": exapmle_data.shape[1],
            f"in_size{suffix}": model_config["input_size"],
            f"n_layers{suffix}": model_config["n_layers"],
            f"channel_multiplier{suffix}": model_config["channel_multiplier"],
        }
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    model_kwargs[f"seq_length{suffix}"] = tuple(exapmle_data.shape[1:])
    model_kwargs[f"diffusion_step{suffix}"] = model_config["diffusion_step"]
    
    return model_kwargs

def load_model(args, model_kwargs, ckpt_path, model_type="fluid"):
    """Load model and create sampler for specified model type"""
    if model_type == "fluid":
        if args.model_name == "Unet": 
            model = Unet3D(
                dim=model_kwargs["dim_fluid"],
                out_dim=model_kwargs["out_dim_fluid"],
                cond_channels=model_kwargs["cond_channels_fluid"],
                dim_mults=(1, 2, 4),
                use_sparse_linear_attn=False,
                attn_dim_head=16,
                expects_x=True,
            )
        elif args.model_name == "SiT_FNO":
            model = SiT_FNO(
                input_size=model_kwargs["input_size_fluid"], 
                depth=model_kwargs["depth_fluid"], 
                hidden_size=model_kwargs["hidden_size_fluid"],
                patch_size=model_kwargs["patch_size_fluid"], 
                num_heads=model_kwargs["num_heads_fluid"],
                in_channels=model_kwargs["in_channels_fluid"],
                out_channels=model_kwargs["out_channels_fluid"], 
                modes=model_kwargs["modes_fluid"])
        elif args.model_name == "CNO":
            model = CNO3d(
                in_dim=model_kwargs["in_dim_fluid"],
                out_dim=model_kwargs["out_dim_fluid"],
                in_size=max(model_kwargs["in_size_fluid"]),
                N_layers=model_kwargs["n_layers_fluid"],
                channel_multiplier=model_kwargs["channel_multiplier_fluid"])
        else:
            raise ValueError(f"Invalid model type: {args.model_name}")
    elif model_type == "structure":
        if args.model_name == "Unet":
            model = Unet3D(
                dim=model_kwargs["dim_structure"],
                out_dim=model_kwargs["out_dim_structure"],
                cond_channels=model_kwargs["cond_channels_structure"],
                dim_mults=(1, 2, 4),
                use_sparse_linear_attn=False,
                attn_dim_head=16,
                expects_x=True,
            )
        elif args.model_name == "SiT_FNO":
            model = SiT_FNO(
                in_channels=model_kwargs["in_channels_structure"],
                out_channels=model_kwargs["out_channels_structure"],
                input_size=model_kwargs["input_size_structure"], 
                depth=model_kwargs["depth_structure"], 
                hidden_size=model_kwargs["hidden_size_structure"],
                patch_size=model_kwargs["patch_size_structure"], 
                num_heads=model_kwargs["num_heads_structure"],
                modes=model_kwargs["modes_structure"])
        elif args.model_name == "CNO":
            model = CNO3d(
                in_dim=model_kwargs["in_dim_structure"],
                out_dim=model_kwargs["out_dim_structure"],
                in_size=max(model_kwargs["in_size_structure"]),
                N_layers=model_kwargs["n_layers_structure"],
                channel_multiplier=model_kwargs["channel_multiplier_structure"])
        else:
            raise ValueError(f"Invalid model type: {args.model_name}")
    else:
        raise ValueError(f"Invalid model type: {args.model_name}")
        

    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    state_dict = load_model_ema(ckpt_path)
    # state_dict = load_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    
    model = GaussianDiffusion(
        model,
        seq_length=model_kwargs[f"seq_length_{model_type}"],
        timesteps=model_kwargs[f"diffusion_step_{model_type}"], 
        auto_normalize=False, 
        sampling_timesteps=args.ddim_sampling_timesteps 
        )

    get_parameter_net(model)

    return model

def inference(input_norm, target_norm, num_steps, device, args, update_fluid_fn, update_structure_fn, num_iter=2):
    """Perform inference with dual-field interaction following compose logic (all data already normalized)"""

    print(f"Starting compose inference for {num_steps} steps with {num_iter} iterations...")
    start_time = time()

    # Get target shape for initialization
    batch_size, time_steps, height, width, channels = target_norm.shape
    
    fluid_field_shape = (batch_size, time_steps, height, width, 3)
    structure_field_shape = (batch_size, time_steps, height, width, 1)
    
    # fluid has 3 channels (u, v, p), structure has 1 sdf channel
    fluid_estimate = torch.randn(fluid_field_shape, device=device)
    structure_estimate = torch.randn(structure_field_shape, device=device)

    for k in range(num_iter):
        print(f"Iteration {k + 1}/{num_iter}")
        fluid_estimate_before = fluid_estimate.clone()
        structure_estimate_before = structure_estimate.clone()

        # Re-initialize predictions for this iteration
        fluid_estimate = torch.randn(fluid_field_shape, device=device)
        structure_estimate = torch.randn(structure_field_shape, device=device)
        fluid_pred = torch.randn(fluid_field_shape, device=device)
        structure_pred = torch.randn(structure_field_shape, device=device)

        # Time step loop (following compose logic, reversed)
        for t in tqdm(reversed(range(num_steps)), desc=f"Sampling timestep for iter {k+1}", total=num_steps):
            # Calculate alpha for controlling update strength
            alpha = 1 - t / (num_steps - 1) if k > 0 else 1.0

            # Update fluid field
            fluid_pred, fluid_estimate = update_fluid_fn(
                alpha,
                t,
                fluid_pred,
                structure_estimate,
                structure_estimate_before,
                input_norm,
            )
            
            # Update structure field
            structure_pred, structure_estimate = update_structure_fn(
                alpha,
                t,
                structure_pred,
                fluid_estimate,
                fluid_estimate_before,
                input_norm,
            )
            
    # Combine final predictions
    predictions_norm = torch.cat([fluid_pred, structure_pred], dim=-1)
    print(f"Compose inference took {time() - start_time:.2f} seconds.")
    return predictions_norm

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load datasets
    # TODO: what is the stage for train set here?
    fluid_trainset = TurekHronDataset(dataset_path=ABSOLUTE_PATH, length=args.length, input_size=args.input_step, output_size=args.output_step, stride=args.stride, mode='train', num_delta_t=args.num_delta_t,
                                            stage='fluid', dt=args.dt)
    structure_trainset = TurekHronDataset(dataset_path=ABSOLUTE_PATH, length=args.length, input_size=args.input_step, output_size=args.output_step, stride=args.stride, mode='train', num_delta_t=args.num_delta_t,
                                            stage='structure', dt=args.dt)
    couple_trainset = TurekHronDataset(dataset_path=ABSOLUTE_PATH, length=args.length, input_size=args.input_step, output_size=args.output_step, stride=args.stride, mode='train', num_delta_t=args.num_delta_t,
                                            stage='couple', dt=args.dt)
    
    fluid_data_prepare_fn = partial(prepare_data, 'fluid')
    structure_data_prepare_fn = partial(prepare_data, 'structure')
    
    val_dataset = TurekHronDataset(dataset_path=ABSOLUTE_PATH, length=args.length, input_size=args.input_step, output_size=args.output_step, stride=args.stride, mode='val', num_delta_t=args.num_delta_t,
                                            stage='couple', dt=args.dt)
    
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size,
                                            shuffle=False, pin_memory=True, num_workers=args.num_workers)

    data_normalizer = RangeNormalizer(couple_trainset,
                                        batch_size=args.test_batch_size)

    fluid_example_data = fluid_trainset[0]
    fluid_cond, fluid_data = fluid_data_prepare_fn(fluid_example_data[0].unsqueeze(0), fluid_example_data[1].unsqueeze(0))
    structure_example_data = structure_trainset[0]
    structure_cond, structure_data = structure_data_prepare_fn(structure_example_data[0].unsqueeze(0), structure_example_data[1].unsqueeze(0))

    fluid_cond = fluid_cond.permute(0, 4, 1, 2, 3)
    fluid_data = fluid_data.permute(0, 4, 1, 2, 3)
    structure_cond = structure_cond.permute(0, 4, 1, 2, 3)
    structure_data = structure_data.permute(0, 4, 1, 2, 3)

    fluid_model_kwargs = parse_model_kwargs(fluid_cond, fluid_data, args.model_name, args.fluid_checkpoint_path, 'fluid')
    structure_model_kwargs = parse_model_kwargs(structure_cond, structure_data, args.model_name, args.structure_checkpoint_path, 'structure')

    fluid_model = load_model(args, fluid_model_kwargs, args.fluid_checkpoint_path, "fluid")
    structure_model = load_model(args, structure_model_kwargs, args.structure_checkpoint_path, "structure")

    fluid_model = fluid_model.to(device)
    structure_model = structure_model.to(device)

    print("Loading coupling model...")
    update_fluid_fn, update_structure_fn = create_couple_step_fn(fluid_model, structure_model, args.input_step, args.output_step)
    
    rel_l2_fluid_u = 0
    rel_l2_fluid_v = 0
    rel_l2_fluid_p = 0
    rel_l2_structure = 0
    num = 0
    
    # Get initial conditions from validation dataset
    processed_samples = 0
    sample_limit = args.eval_samples

    for input, target, _, _, _ in val_dataloader:
        batch_size = input.size(0)
        if sample_limit is not None and processed_samples >= sample_limit:
            break
        if sample_limit is not None:
            remaining = sample_limit - processed_samples
            if remaining <= 0:
                break
            if batch_size > remaining:
                input = input[:remaining]
                target = target[:remaining]
                batch_size = remaining

        # Normalize entire batch at once
        input_norm, target_norm = data_normalizer.preprocess(input, target)
        
        input = input.to(device)
        target = target.to(device)

        input_norm = input_norm.to(device)
        target_norm = target_norm.to(device)

        # Perform compose inference for entire batch
        start_time = time()

        assert fluid_model.num_timesteps == structure_model.num_timesteps
        num_steps = fluid_model.num_timesteps

        predictions_norm = inference(
            input_norm, target_norm, num_steps, device, args,
            update_fluid_fn, update_structure_fn, num_iter=args.num_iter
        )
        
        print(f"Autoregressive inference took {time() - start_time:.2f} seconds.")

        # Get final predictions from the AR trajectory
        _, final_prediction = data_normalizer.postprocess(input_norm.cpu(), predictions_norm.cpu())

        final_prediction = final_prediction.to(device)

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
        else:
            final_prediction = final_prediction


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
        processed_samples += batch_size

        print(f"rel_l2_fluid_u: {rel_l2_fluid_u/(num):.6f}")
        print(f"rel_l2_fluid_v: {rel_l2_fluid_v/(num):.6f}")
        print(f"rel_l2_fluid_p: {rel_l2_fluid_p/(num):.6f}")
        print(f"rel_l2_structure: {rel_l2_structure/(num):.6f}")

        final_prediction = final_prediction.cpu().numpy()
        target = target.cpu().numpy()

        sdf_mask = np.where(target[0,-1,...,-1] > args.sdf_threshold, 1.0, 0.0)# .cpu().numpy()
        sdf_video_mask = np.where(target[0,...,-1] > args.sdf_threshold, 1.0, 0.0)# .cpu().numpy()

        if num == 1:
            visualizer = FluidFieldVisualizer(args=args, grid_size=args.input_size, save_dir=args.save_figs_path, create_timestamp_folder=True)
        
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

            # visualizer.visualize_time_series_gif(
            #     u_pred=final_prediction[0, ..., 0] * sdf_video_mask,  # Velocity x [b, t, h, w, c]
            #     u_true=target[0, ..., 0] * sdf_video_mask,  # Ground truth velocity x
            #     title="Time Series Animation - Velocity X",
            #     save_name="time_series_animation_u.gif",
            #     fps=2,
            #     show_colorbar=True
            # )

            # visualizer.visualize_time_series_gif(
            #     u_pred=final_prediction[0, ..., 1] * sdf_video_mask,  # Velocity x [b, t, h, w, c]
            #     u_true=target[0, ..., 1] * sdf_video_mask,  # Ground truth velocity x
            #     title="Time Series Animation - Velocity V",
            #     save_name="time_series_animation_v.gif",
            #     fps=2,
            #     show_colorbar=True
            # )

            # visualizer.visualize_time_series_gif(
            #     u_pred=final_prediction[0, ..., 2] * sdf_video_mask,  # Velocity x [b, t, h, w, c]
            #     u_true=target[0, ..., 2] * sdf_video_mask,  # Ground truth velocity x
            #     title="Time Series Animation - Velocity P",
            #     save_name="time_series_animation_p.gif",
            #     fps=2,
            #     show_colorbar=True
            # )

            # visualizer.visualize_sdf_time_series_gif(
            #     sdf_pred=final_prediction[0, ..., 3],  # Structure field [t, h, w]
            #     sdf_true=target[0, ..., 3],  # Ground truth structure field
            #     title="Time Series Animation - Structure (SDF)",
            #     save_name="time_series_animation_structure.gif",
            #     fps=2
            # )

        print("Inference completed. Results saved to:", args.save_figs_path)
        
        if (sample_limit is not None and processed_samples >= sample_limit) or num >= args.max_test_batch_num:
            break
    
def parse_tuple(s):
    """Parse a string like '108,88' into a tuple (108, 88)"""
    return tuple(int(x.strip()) for x in s.split(','))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # New arguments for dual-field inference
    parser.add_argument("--fluid_checkpoint_path", type=str, default="checkpoints/fluid_model.pth",
                       help="Path to fluid model checkpoint")
    parser.add_argument("--structure_checkpoint_path", type=str, default="checkpoints/structure_model.pth",
                       help="Path to structure model checkpoint")
    
    parser.add_argument("--model_name", type=str, default="Unet")
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--max_test_batch_num", default=5, type=int, help="max test batch num")
    parser.add_argument(
        "--eval-samples",
        type=lambda v: None if v in {"", None, "none", "None", 0, "0"} else max(1, int(v)),
        default=DEFAULT_EVAL_SAMPLES,
        help="Evaluate at most this many samples (use 0/none for full dataset).",
    )

    parser.add_argument("--length", default=999, type=int, help="length")
    parser.add_argument("--input_step", default=3, type=int, help="input step")
    parser.add_argument("--output_step", default=12, type=int, help="output step")
    parser.add_argument("--stride", default=1, type=int, help="stride")
    parser.add_argument("--num_delta_t", default=0, type=int, help="num delta t")
    parser.add_argument("--dt", default=5, type=int, help="dt")

    parser.add_argument("--input_size", default="108,88", type=parse_tuple, help="input size as tuple (height,width)")
    parser.add_argument("--save_figs_path", default="./visualization_results", type=str, help="save figs path")
    
    parser.add_argument("--num_iter", type=int, default=2)
    parser.add_argument("--sdf_threshold", type=float, default=0.02)
    parser.add_argument("--ddim_sampling_timesteps", type=int, default=250)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    main(args)