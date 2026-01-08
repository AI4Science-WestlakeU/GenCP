# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
from time import time

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

def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load datasets
    train_dataset = TurekHronDataset(dataset_path=ABSOLUTE_PATH, length=args.length, input_size=args.input_step, output_size=args.output_step, stride=args.stride, mode='train', num_delta_t=args.num_delta_t
                                            , stage=args.stage, dt=args.dt)
    val_dataset = TurekHronDataset(dataset_path=ABSOLUTE_PATH, length=args.length, input_size=args.input_step, output_size=args.output_step, stride=args.stride, mode='val', num_delta_t=args.num_delta_t
                                            , stage=args.stage, dt=args.dt)

    # Setup data normalizer
    data_normalizer = RangeNormalizer(val_dataset, batch_size=args.test_batch_size)
    data_prepare_fn = partial(prepare_data, args.stage)
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.test_batch_size, 
        shuffle=False, 
        pin_memory=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        prefetch_factor=16,
        )
    
    example_data = val_dataset[0]
    cond, data = data_prepare_fn(example_data[0].unsqueeze(0), example_data[1].unsqueeze(0))
    
    cond = cond.permute(0, 4, 1, 2, 3)
    data = data.permute(0, 4, 1, 2, 3)
    
    if args.model_name == "Unet":
        # cond_channels from prepared tensor
            model = Unet3D(
                dim=args.dim,
                out_dim=data.shape[1],
                cond_channels=cond.shape[1],
                dim_mults=(1, 2, 4),
                use_sparse_linear_attn=False,
                attn_dim_head=16,
                expects_x=True,
            )
    elif args.model_name == "SiT_FNO":
        model = SiT_FNO(input_size=args.input_size, 
                depth=args.depth, 
                hidden_size=args.hidden_size, 
                patch_size=args.patch_size, 
                num_heads=args.num_heads, 
                in_channels=data.shape[1] + cond.shape[1],
                out_channels=data.shape[1],
                modes=args.modes)
    elif args.model_name == "CNO":
        model = CNO3d(
            in_dim=data.shape[1] + cond.shape[1],
            out_dim=data.shape[1],
            in_size=max(args.input_size),
            N_layers=args.n_layers,
            channel_multiplier=args.channel_multiplier)
    else:
        raise ValueError(f"Invalid model type: {args.model_name}")

    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.checkpoint_path 
    state_dict = load_model_ema(ckpt_path)
    # state_dict = load_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    
    model = GaussianDiffusion(
        model,
        seq_length=tuple(data.shape[1:]), 
        timesteps=args.diffusion_step, 
        auto_normalize=False, 
        sampling_timesteps=args.ddim_sampling_timesteps 
        ).to(device)

    get_parameter_net(model)

    rel_l2_mean_u = 0
    rel_l2_mean_v = 0
    rel_l2_mean_p = 0
    rel_l2_mean_sdf = 0
    num = 0
    
    for input, target, _, _, _ in tqdm(val_dataloader, desc="Sampling"):
        start_time = time()
        
        input_norm, target_norm = data_normalizer.preprocess(input, target)
        cond, _ = data_prepare_fn(input_norm, target_norm)
        
        target_norm = target_norm.to(device).float()
        input_norm = input_norm.to(device).float()
        cond = cond.to(device).float()

        # Sample images:
        batchsize = target_norm.shape[0]
        
        cond = cond.permute(0, 4, 1, 2, 3)

        samples = model.sample(batchsize, cond)
        
        samples = samples.permute(0, 2, 3, 4, 1)

        if args.stage == "fluid":
            pred_ = target_norm.clone()
            pred_[..., :-1] = samples
            samples = pred_
        elif args.stage == "structure":
            pred_ = target_norm.clone()
            pred_[..., -1:] = samples
            samples = pred_ 
        
        _, samples_denorm = data_normalizer.postprocess(input_norm.cpu(), samples.cpu())

        samples_denorm = samples_denorm.to(device).float()
        target = target.to(device).float()


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

            samples_denoised = denoise_flow_output(samples_denorm, sigma=0.8)
            samples_denorm[..., -1:] = torch.tensor(samples_denoised[..., -1:]).to(device)
        else:
            samples_denorm = samples_denorm
        

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
        
        if num == 1:
            visualizer = FluidFieldVisualizer(args=args, grid_size=args.input_size, save_dir=args.save_figs_path, create_timestamp_folder=True)
            sdf_mask = np.where(target[0,-1,...,-1] > args.sdf_threshold, 1.0, 0.0)# .cpu().numpy()
            sdf_video_mask = np.where(target[0,...,-1] > args.sdf_threshold, 1.0, 0.0)# .cpu().numpy()

            # sdf_mask = np.ones_like(sdf_mask)# .cpu().numpy()
            # sdf_video_mask = np.ones_like(sdf_video_mask)# .cpu().numpy()

            if args.stage == "fluid":
                visualizer.visualize_field(
                    u_pred=samples_denorm[0,-1,...,0] * sdf_mask, 
                    u_true=target[0,-1,...,0] * sdf_mask, 
                    title="U Field",
                    save_name="u_field.png",
                )
                visualizer.visualize_field(
                    u_pred=samples_denorm[0,-1,...,1] * sdf_mask, 
                    u_true=target[0,-1,...,1] * sdf_mask, 
                    title="V Field",
                    save_name="v_field.png"
                )
                visualizer.visualize_field(
                    u_pred=samples_denorm[0,-1,...,2] * sdf_mask, 
                    u_true=target[0,-1,...,2] * sdf_mask, 
                    title="P Field",
                    save_name="p_field.png"
                )
                visualizer.visualize_time_series_gif(
                    u_pred=samples_denorm[0, ..., 0] * sdf_video_mask,  # Velocity x [b, t, h, w, c]
                    u_true=target[0, ..., 0] * sdf_video_mask,  # Ground truth velocity x
                    title="Time Series Animation - Velocity X",
                    save_name="time_series_animation_u.gif",
                    show_colorbar=True
                )
                visualizer.visualize_time_series_gif(
                    u_pred=samples_denorm[0, ..., 1] * sdf_video_mask,  # Velocity x [b, t, h, w, c]
                    u_true=target[0, ..., 1] * sdf_video_mask,  # Ground truth velocity x
                    title="Time Series Animation - Velocity X",
                    save_name="time_series_animation_v.gif",
                    show_colorbar=True
                )
                visualizer.visualize_time_series_gif(
                    u_pred=samples_denorm[0, ..., 2] * sdf_video_mask,  # Velocity x [b, t, h, w, c]
                    u_true=target[0, ..., 2] * sdf_video_mask,  # Ground truth velocity x
                    title="Time Series Animation - Velocity X",
                    save_name="time_series_animation_p.gif",
                    show_colorbar=True
                )
                # visualizer.visualize_sdf_field(
                #     sdf_pred=samples_denorm[0,-1,...,3], # * sdf_mask, 
                #     sdf_true=target[0,-1,...,3], 
                #     title="SDF Field",
                #     save_name="sdf_field.png"
                # )
                # visualizer.visualize_sdf_time_series_gif(
                #     sdf_pred=samples_denorm[0, ..., 3].cpu().numpy(), # * sdf_video_mask,  # SDF data [t, h, w]
                #     sdf_true=target[0, ..., 3].cpu().numpy(),  # Ground truth SDF
                #     title="Time Series Animation - SDF",
                #     save_name="time_series_animation.gif",
                #     show_colorbar=True
                # )
                
            elif args.stage == "structure":
                visualizer.visualize_sdf_field(
                    sdf_pred=samples_denorm[0,-1,...,3], # * sdf_mask, 
                    sdf_true=target[0,-1,...,3], 
                    title="SDF Field",
                    save_name="sdf_field.png"
                )
                visualizer.visualize_sdf_time_series_gif(
                    sdf_pred=samples_denorm[0, ..., 3], # * sdf_video_mask,  # SDF data [t, h, w]
                    sdf_true=target[0, ..., 3],  # Ground truth SDF
                    title="Time Series Animation - SDF",
                    save_name="time_series_animation.gif",
                    show_colorbar=True
                )
        
        if num >= args.max_test_batch_num:
            break
        
def parse_tuple(s):
    """Parse a string like '108,88' into a tuple (108, 88)"""
    return tuple(int(x.strip()) for x in s.split(','))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--stage", type=str, default="fluid")
    parser.add_argument("--model_name", type=str, default="Unet")
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--max_test_batch_num", default=5, type=int, help="max test batch num")

    parser.add_argument("--length", default=999, type=int, help="length")
    parser.add_argument("--input_step", default=3, type=int, help="input step")
    parser.add_argument("--output_step", default=12, type=int, help="output step")
    parser.add_argument("--stride", default=1, type=int, help="stride")
    parser.add_argument("--num_delta_t", default=0, type=int, help="num delta t")
    parser.add_argument("--dt", default=5, type=int, help="dt")

    parser.add_argument("--input_size", default="108,88", type=parse_tuple, help="input size as tuple (height,width)")
    parser.add_argument("--save_figs_path", default="./visualization_results", type=str, help="save figs path")
    
    parser.add_argument("--sdf_threshold", type=float, default=0.02)
    parser.add_argument("--ddim_sampling_timesteps", type=int, default=250)
    parser.add_argument("--diffusion_step", type=int, default=250)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    
    # model unet
    parser.add_argument("--dim", type=int, default=8)
    
    # model sit_fno
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--patch_size", type=parse_tuple, default="2,2")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--modes", type=int, default=4)
    
    # model cno
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--channel_multiplier", type=int, default=32)
    
    args = parser.parse_args()
    
    
    main(args)