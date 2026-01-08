# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
from colorsys import yiq_to_rgb
from numpy import False_
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from utils.utils import parse_ode_args, parse_sde_args, parse_transport_args, find_model, add_args_from_config, rel_l2_loss, mse_loss
import argparse
import sys
from time import time
from tqdm import tqdm

from data.turek_hron_dataset import TurekHronDataset
from data.double_cylinder_dataset import DoubleCylinderDataset

from utils.visualize import FluidFieldVisualizer
import pdb
import torchcfm
from model.cno_surrogate import CNO3d as CNO3d_surrogate
from model.fno_surrogate import FNO3d as FNO3d_surrogate
from model.fno import FNO3d
from model.cno import CNO3d
from model.SiT_FNO import SiT_FNO
from model.sit_fno_surrogate import SiT_FNO as SiT_FNO_surrogate

# NOTE: simple workaround for preparing input for surrogate model without mask
def prepare_data_for_surrogate(stage, input, target):
    
    input = input.repeat(1, target.shape[1] // input.shape[1], 1, 1, 1) # (b, 5, h, w, 4) -> (b, 10, h, w, 4)
    if stage == "fluid":
        cond = target[..., -1:] # (b, 10, h, w, 1)
        input = torch.cat([input, cond], dim=-1) # (b, 10, h, w, 5)
        target = target[..., :-1] # (b, 10, h, w, 3)
    elif stage == "structure":
        cond = target[..., :-1] # (b, 10, h, w, 3)
        input = torch.cat([input, cond], dim=-1) # (b, 10, h, w, 7)
        target = target[..., -1:] # (b, 10, h, w, 1)
    elif stage == "couple" or stage == "joint":
        pass
    else:
        raise ValueError(f"Invalid stage: {stage}")
    
    return input, target


def main(mode, args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from model.SiT import SiT

    # Load datasets
    if args.dataset_name == 'turek_hron_data':
        train_dataset = TurekHronDataset(dataset_path=args.dataset_path, length=args.length, input_size=args.input_step, output_size=args.output_step, stride=args.stride, mode='train', num_delta_t=args.num_delta_t
                                                 , stage=args.stage, dt=args.dt)
        val_dataset = TurekHronDataset(dataset_path=args.dataset_path, length=args.length, input_size=args.input_step, output_size=args.output_step, stride=args.stride, mode='val', num_delta_t=args.num_delta_t
                                               , stage=args.stage, dt=args.dt)
    if args.dataset_name == 'double_cylinder_data':
        train_dataset = DoubleCylinderDataset(dataset_path=args.dataset_path, length=args.length, input_size=args.input_step, output_size=args.output_step, stride=args.stride, mode='train', num_delta_t=args.num_delta_t
                                                 , stage=args.stage, dt=args.dt)
        val_dataset = DoubleCylinderDataset(dataset_path=args.dataset_path, length=args.length, input_size=args.input_step, output_size=args.output_step, stride=args.stride, mode='val', num_delta_t=args.num_delta_t
                                               , stage=args.stage, dt=args.dt)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, 
                                            shuffle=True, pin_memory=True, num_workers=args.num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size,
                                            shuffle=False, pin_memory=True, num_workers=args.num_workers)

    from data.data_normalizer import RangeNormalizer
    data_normalizer = RangeNormalizer(train_dataset, device=device, mode='train',
                                    batch_size=args.train_batch_size)

    if getattr(args, "use_torchcfm", False):
        if args.model_name == "SiT":
            model = SiT(input_size=args.input_size, depth=args.depth, hidden_size=args.hidden_size, patch_size=args.patch_size, num_heads=args.num_heads, in_channels=args.in_channels, x0_is_use_noise=args.x0_is_use_noise, stage=args.stage, dataset_name=args.dataset_name).to(device)

        elif args.model_name == 'SiT_FNO':
            model = SiT_FNO(input_size=args.input_size, 
                            depth=args.depth, 
                            hidden_size=args.hidden_size, 
                            patch_size=args.patch_size, 
                            num_heads=args.num_heads, 
                            in_channels=args.in_channels, 
                            out_channels=args.out_channels, 
                            x0_is_use_noise=args.x0_is_use_noise, 
                            stage=args.stage, 
                            dataset_name=args.dataset_name,
                            modes=args.modes).to(device)
            
        elif args.model_name == 'FNO':
            model = FNO3d(modes1=args.modes_t, 
                          modes2=args.modes_x, 
                          modes3=args.modes_y, 
                          n_layers=args.depth, 
                          width=args.width, 
                          shape_in=(args.input_step, args.input_size[0], args.input_size[1], args.in_channels), 
                          shape_out=(args.output_step, args.input_size[0], args.input_size[1], args.out_channels),
                          dataset_name=args.dataset_name,
                          x0_is_use_noise=args.x0_is_use_noise,
                          stage=args.stage).to(device)
            
        elif args.model_name == 'CNO':
            model = CNO3d(in_dim=args.in_dim, 
                        out_dim=args.out_dim, 
                        in_size=args.in_size, 
                        N_layers=args.depth,
                        dataset_name=args.dataset_name,
                        x0_is_use_noise=args.x0_is_use_noise,
                        stage=args.stage).to(device)
            
        else:
            raise ValueError(f"Model {args.model_name} not supported")
        
    elif getattr(args, "use_surrogate", False):
        
        if args.model_name == 'CNO':
            model = CNO3d_surrogate(in_dim=args.in_dim,
                out_dim=args.out_dim,
                in_size=args.in_size,
                N_layers=args.depth,
                channel_multiplier=args.channel_multiplier,
            )
        elif args.model_name == 'FNO':
            model = FNO3d_surrogate(modes1=args.modes_t, 
                                modes2=args.modes_x, 
                                modes3=args.modes_y, 
                                n_layers=args.n_layers, 
                                width=args.width, 
                                shape_in=args.input_shape, 
                                shape_out=args.output_shape)
        elif args.model_name == 'SiT_FNO':
            model = SiT_FNO_surrogate(input_size=args.input_size, 
                    depth=args.depth, 
                    hidden_size=args.hidden_size, 
                    patch_size=args.patch_size, 
                    num_heads=args.num_heads, 
                    in_channels=args.in_channels, 
                    out_channels=args.out_channels,
                    modes=args.modes)
        else:
            raise ValueError(f"Model {args.model_name} not supported")
        
    
    ckpt_path = args.checkpoint_path 
    print(f"Loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print("Loading main model (model_state_dict)")
    else:
        raise ValueError(f"Checkpoint {ckpt_path} not supported")
    
    model.load_state_dict(state_dict)
    model.eval()
    if getattr(args, "use_torchcfm", False):
        cfm = torchcfm.ConditionalFlowMatcher()
        
        def sample_cfm_ode(x_init, target_norm, model, **model_kwargs):

            x = x_init.clone()
            
            num_steps = args.num_sampling_steps
            
            t0, t1 = (0.0, 1.0)
            t_grid = torch.linspace(t0, t1, num_steps, device=x.device)

            dt = (t_grid[1] - t_grid[0]).item() if num_steps > 1 else (1.0 if not args.reverse else -1.0)

            xs = [x]
            for t in tqdm(t_grid[:-1], desc="Sampling progress"):

                tb = t.expand(x.shape[0]).to(x)
                _, xt, _ = cfm.sample_location_and_conditional_flow(x_init, target_norm, tb)

                print(args.stage)

                if args.stage == "fluid":
                    yt = xt[..., -1:]
                    x = torch.cat([x[..., :-1], yt], dim=-1)
                elif args.stage == "structure":
                    yt = xt[..., :-1]
                    x = torch.cat([yt, x[..., -1:]], dim=-1)
                elif args.stage == "joint":  
                    yt = xt[..., :]
                    x = x
                                
                vt = model(x, tb, model_kwargs.get("x0"), model_kwargs.get("cond"))

                channel = getattr(args, 'out_channels', 4)
                if channel == 3 or channel == 1:
                    if args.stage == "fluid":
                        x[..., :-1] = x[..., :-1] + vt * dt
                    elif args.stage == "structure":
                        x[..., -1:] = x[..., -1:] + vt * dt
                    elif args.stage == "joint":
                        x = x + vt * dt
                elif channel == 4:
                    if args.stage == "fluid":
                        x[..., :-1] = x[..., :-1] + vt[..., :-1] * dt
                    elif args.stage == "structure":
                        x[..., -1:] = x[..., -1:] + vt[..., -1:] * dt
                    elif args.stage == "joint":
                        x = x + vt * dt

                xs.append(x)

            return xs[-1]

        sample_fn = sample_cfm_ode
    
    elif getattr(args, "use_surrogate", False):

        def sample_surrogate(z, target_norm, model, **model_kwargs):
            if args.model_name in ["UNet3d", "CNO", "SiT_FNO"]:
                z = z.permute(0, 4, 1, 2, 3)
                pred = model(z)
                pred = pred.permute(0, 2, 3, 4, 1)
            elif args.model_name == "FNO":
                pred = model(z)
            else: 
                raise NotImplementedError("Model not supported")

            return pred

        sample_fn = sample_surrogate

    rel_l2_mean_u = 0.0
    rel_l2_mean_v = 0.0
    rel_l2_mean_p = 0.0
    rel_l2_mean_sdf = 0.0

    rel_l2_mean_u = 0
    rel_l2_mean_v = 0
    rel_l2_mean_p = 0
    rel_l2_mean_sdf = 0
    num = 0

    start_time = time()
    for input, target, grid_x, grid_y, attrs in val_dataloader:
        input_norm, target_norm = data_normalizer.preprocess(input, target)
        start_time = time()

        if getattr(args, "use_torchcfm", False):
            if args.x0_is_use_noise:
                z = torch.randn_like(target, device=device)
            else:
                z = input_norm.clone().to(device).float()
                
        elif getattr(args, "use_surrogate", False):
            z = input_norm.clone().to(device).float()
            z, _ = prepare_data_for_surrogate(args.stage, z, target_norm)
            
        else:
            raise NotImplementedError("Not supported paradigm")

        target_norm = target_norm.to(device).float()
        input_norm = input_norm.to(device).float()

        model_kwargs = dict(x0=input_norm, cond=attrs)
        samples = sample_fn(z, target_norm, model.to(device), **model_kwargs)

        if getattr(args, "use_surrogate", False):
            if args.stage == "fluid":
                pred_ = torch.zeros_like(target_norm)
                pred_[..., :-1] = samples
                samples = pred_
            elif args.stage == "structure":
                pred_ = torch.zeros_like(target_norm)
                pred_[..., -1:] = samples
                samples = pred_ 

        _, samples_denorm = data_normalizer.postprocess(input_norm, samples)

        target = target.to(device).float()


        manual_denoise = True
        if manual_denoise:
            from scipy.ndimage import gaussian_filter
            import numpy as np
            def denoise_flow_output(data, sigma=1.0):
                """Apply Gaussian smoothing to flow matching output."""
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

            samples_denoised = denoise_flow_output(samples_denorm, sigma=0.8)
            samples_denorm[..., -1:] = torch.tensor(samples_denoised[..., -1:]).to(device)

        if True:
            sdf_mask = torch.where(target[..., -1:] > args.sdf_threshold, torch.tensor(1.0), torch.tensor(0.0))

            rel_l2_per_sample_u = rel_l2_loss(samples_denorm[..., 0:1] * sdf_mask, target[..., 0:1] * sdf_mask)
            rel_l2_per_sample_v = rel_l2_loss(samples_denorm[..., 1:2] * sdf_mask, target[..., 1:2] * sdf_mask)
            rel_l2_per_sample_p = rel_l2_loss(samples_denorm[..., 2:3] * sdf_mask, target[..., 2:3] * sdf_mask)
            rel_l2_per_sample_sdf = rel_l2_loss(samples_denorm[..., 3:4] * sdf_mask, target[..., 3:4] * sdf_mask)

            rel_l2_mean_u += rel_l2_per_sample_u.mean().item()
            rel_l2_mean_v += rel_l2_per_sample_v.mean().item()
            rel_l2_mean_p += rel_l2_per_sample_p.mean().item()
            rel_l2_mean_sdf += rel_l2_per_sample_sdf.mean().item()

            num += 1

            print(f"Validation metrics (denorm): relL2_mean_u={rel_l2_mean_u/num:.6f}")
            print(f"Validation metrics (denorm): relL2_mean_v={rel_l2_mean_v/num:.6f}")
            print(f"Validation metrics (denorm): relL2_mean_p={rel_l2_mean_p/num:.6f}")
            print(f"Validation metrics (denorm): relL2_mean_sdf={rel_l2_mean_sdf/num:.6f}")

        print(f"Sampling took {time() - start_time:.2f} seconds.")

        samples_denorm = samples_denorm.cpu().numpy()
        target = target.cpu().numpy()
        
        visualizer = FluidFieldVisualizer(args=args, grid_size=args.input_size, save_dir=args.save_figs_path, create_timestamp_folder=True)

        if True:
            sdf_mask = np.where(target[0,-1,...,-1] > args.sdf_threshold, 1.0, 0.0)
            sdf_video_mask = np.where(target[0,...,-1] > args.sdf_threshold, 1.0, 0.0)

            if args.stage == "fluid":
                visualizer.visualize_field(
                    u_pred=samples_denorm[0,-1,...,0] * sdf_mask, 
                    u_true=target[0,-1,...,0] * sdf_mask, 
                    title="Velocity-X Field",
                    save_name="u_field.png",
                )
                visualizer.visualize_field(
                    u_pred=samples_denorm[0,-1,...,1] * sdf_mask, 
                    u_true=target[0,-1,...,1] * sdf_mask, 
                    title="Velocity-Y Field",
                    save_name="v_field.png"
                )
                visualizer.visualize_field(
                    u_pred=samples_denorm[0,-1,...,2] * sdf_mask, 
                    u_true=target[0,-1,...,2] * sdf_mask, 
                    title="Pressure Field",
                    save_name="p_field.png"
                )
                visualizer.visualize_time_series_gif(
                    u_pred=samples_denorm[0, ..., 0] * sdf_video_mask,
                    u_true=target[0, ..., 0] * sdf_video_mask,
                    title="Time Series Animation - Velocity X",
                    save_name="time_series_animation_u.gif",
                    show_colorbar=True
                )
                visualizer.visualize_time_series_gif(
                    u_pred=samples_denorm[0, ..., 1] * sdf_video_mask,
                    u_true=target[0, ..., 1] * sdf_video_mask,
                    title="Time Series Animation - Velocity Y",
                    save_name="time_series_animation_v.gif",
                    show_colorbar=True
                )
                visualizer.visualize_time_series_gif(
                    u_pred=samples_denorm[0, ..., 2] * sdf_video_mask,
                    u_true=target[0, ..., 2] * sdf_video_mask,
                    title="Time Series Animation - Pressure",
                    save_name="time_series_animation_p.gif",
                    show_colorbar=True
                )
                
            elif args.stage == "structure":
                visualizer.visualize_sdf_field(
                    sdf_pred=samples_denorm[0,-1,...,3],
                    sdf_true=target[0,-1,...,3], 
                    title="SDF Field",
                    save_name="sdf_field.png"
                )
                visualizer.visualize_sdf_time_series_gif(
                    sdf_pred=samples_denorm[0, ..., 3],
                    sdf_true=target[0, ..., 3],
                    title="Time Series Animation - SDF",
                    save_name="time_series_animation.gif",
                    show_colorbar=True
                )
                
            elif args.stage == "joint":
                visualizer.visualize_field(
                    u_pred=samples_denorm[0,-1,...,0] * sdf_mask, 
                    u_true=target[0,-1,...,0] * sdf_mask, 
                    title="Velocity-X Field",
                    save_name="u_field.png",
                )
                visualizer.visualize_field(
                    u_pred=samples_denorm[0,-1,...,1] * sdf_mask, 
                    u_true=target[0,-1,...,1] * sdf_mask, 
                    title="Velocity-Y Field",
                    save_name="v_field.png"
                )
                visualizer.visualize_field(
                    u_pred=samples_denorm[0,-1,...,2] * sdf_mask, 
                    u_true=target[0,-1,...,2] * sdf_mask, 
                    title="Pressure Field",
                    save_name="p_field.png"
                )
                visualizer.visualize_time_series_gif(
                    u_pred=samples_denorm[0, ..., 0] * sdf_video_mask,
                    u_true=target[0, ..., 0] * sdf_video_mask,
                    title="Time Series Animation - Velocity X",
                    save_name="time_series_animation_u.gif",
                    show_colorbar=True
                )
                visualizer.visualize_time_series_gif(
                    u_pred=samples_denorm[0, ..., 1] * sdf_video_mask,
                    u_true=target[0, ..., 1] * sdf_video_mask,
                    title="Time Series Animation - Velocity Y",
                    save_name="time_series_animation_v.gif",
                    show_colorbar=True
                )
                visualizer.visualize_time_series_gif(
                    u_pred=samples_denorm[0, ..., 2] * sdf_video_mask,
                    u_true=target[0, ..., 2] * sdf_video_mask,
                    title="Time Series Animation - Pressure",
                    save_name="time_series_animation_p.gif",
                    show_colorbar=True
                )
                visualizer.visualize_sdf_field(
                    sdf_pred=samples_denorm[0,-1,...,3],
                    sdf_true=target[0,-1,...,3], 
                    title="SDF Field",
                    save_name="sdf_field.png"
                )
                visualizer.visualize_sdf_time_series_gif(
                    sdf_pred=samples_denorm[0, ..., 3],
                    sdf_true=target[0, ..., 3],
                    title="Time Series Animation - SDF",
                    save_name="time_series_animation.gif",
                    show_colorbar=True
                )
            
        if num >= 5:
            break
    
    end_time = time()
    print(f"Total time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    mode = "ODE" # ["ODE", "SDE"]

    parser.add_argument("--config", type=str, default="configs/fluidzero_data.yaml") 
    # parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=64)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    
    parse_transport_args(parser)
    parser.add_argument("--use-torchcfm", action="store_true", default=True, help="Use torchcfm inference branch (does not create transport/Sampler)")
    if mode == "ODE":
        parse_ode_args(parser)
    elif mode == "SDE":
        parse_sde_args(parser)
    
    args = add_args_from_config(parser)
    main(mode, args)