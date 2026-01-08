import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset

from paradigms.diffusion import GaussianDiffusion
from paradigms.fm_wrapper import FMWrapper
from paradigms.fm_couple import FMCoupleWrapper
from train import Trainer
from models.cno import CNO3d
from models.sit_fno import SiT_FNO
from models.unet3d import Unet3D
from utils import create_res, get_parameter_net, save_config_from_args, set_seed
from data.ntcouple import load_nt_dataset_emb

DEFAULT_PATCH_SIZE: Tuple[int, int] = (2, 2)


def _parse_int_tuple(raw: str) -> Tuple[int, ...]:
    """Convert a comma-separated string into a tuple of integers."""
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected at least one integer.")
    return tuple(int(part) for part in parts)


def _resolve_size(
    override: Optional[Tuple[int, ...]], fallback: Tuple[int, int]
) -> Tuple[int, int]:
    """Return a 2D (height, width) tuple, using the override when provided."""
    if override is None:
        return fallback
    if len(override) == 1:
        return override[0], override[0]
    if len(override) >= 2:
        return override[0], override[1]
    raise ValueError("Override size must contain at least one integer.")


def _positive_int_or_none(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    value = value.strip()
    if value.lower() in {"", "none"}:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def build_model(args: argparse.Namespace, data: torch.Tensor, cond: torch.Tensor) -> nn.Module:
    """Instantiate the requested NTcouple diffusion backbone."""
    out_channels = data.shape[1]
    cond_channels = cond.shape[1]
    seq_shape = tuple(data.shape[1:])

    if args.model_type == "Unet":
        base_model: nn.Module = Unet3D(
            dim=args.dim,
            out_dim=out_channels,
            cond_channels=cond_channels,
            dim_mults=(1, 2, 4),
            use_sparse_linear_attn=False,
            attn_dim_head=16,
            expects_x=True,
        )
    elif args.model_type == "SiT_FNO":
        height, width = data.shape[-2], data.shape[-1]
        input_size = _resolve_size(args.input_size, (height, width))
        patch_size = _resolve_size(args.patch_size, DEFAULT_PATCH_SIZE)
        num_frames = args.num_frames or data.shape[2]
        base_model = SiT_FNO(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=out_channels + cond_channels,
            out_channels=out_channels,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            num_frames=num_frames,
            modes=args.modes,
        )
    elif args.model_type == "CNO":
        cno_size = args.cno_input_size or max(data.shape[2], data.shape[3], data.shape[4])
        base_model = CNO3d(
            in_dim=out_channels + cond_channels,
            out_dim=out_channels,
            in_size=cno_size,
            N_layers=args.n_layers,
            channel_multiplier=args.channel_multiplier,
        )
    else:
        raise ValueError(f"Unsupported model_type {args.model_type}")

    if args.paradigm == "fm":
        return FMWrapper(
            base_model,
            seq_shape=seq_shape,
            time_sampling=args.fm_time_sampling,
            logit_mu=args.fm_logit_mu,
            logit_sigma=args.fm_logit_sigma,
            sampling_steps=args.fm_sampling_steps,
            solver_step_size=args.fm_step_size,
        )

    if args.paradigm == "fm_couple":
        return FMCoupleWrapper(
            base_model,
            seq_shape=seq_shape,
            time_sampling=args.fm_time_sampling,
            logit_mu=args.fm_logit_mu,
            logit_sigma=args.fm_logit_sigma,
            sampling_steps=args.fm_sampling_steps,
            solver_step_size=args.fm_step_size,
        )

    return GaussianDiffusion(
        base_model,
        seq_length=seq_shape,
        timesteps=args.diffusion_step,
        sampling_timesteps=args.sample_steps,
        auto_normalize=False,
    )

def create_training_functions():
    def func_train(model: nn.Module, batch):
        data, cond = batch
        return model(data, cond)

    def func_val(model: nn.Module, batch, loss_fn=F.mse_loss):
        device = next(model.parameters()).device
        data, cond = batch
        data = data.to(device)
        cond = cond.to(device)
        batchsize = data.shape[0]
        outputs = model.sample(batchsize, cond)
        return loss_fn(data, outputs)

    return func_train, func_val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NTcouple baselines.")
    parser.add_argument("--exp-id", default="ntcouple", type=str, help="Experiment folder id.")
    parser.add_argument("--stage", default="neutron", choices=["neutron", "solid", "fluid"], help="Physics field.")
    parser.add_argument(
        "--paradigm",
        default="diffusion",
        choices=["diffusion", "fm", "fm_couple"],
        help="Training paradigm.",
    )
    parser.add_argument(
        "--dataset",
        default="decouple_train",
        type=str,
        choices=["decouple_train", "decouple_val", "couple_train", "couple_val"],
        help="Dataset split (decouple_train/decouple_val/couple_train/couple_val/).",
    )
    parser.add_argument("--data-root", default=None, type=str, help="Optional dataset root override.")
    parser.add_argument("--n-dataset", default=None, type=lambda v: None if v in {"", "none", "None"} else int(v))
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--batchsize", default=32, type=int, help="Training batch size.")
    parser.add_argument("--val-batchsize", default=None, type=int, help="Optional validation batch size.")
    parser.add_argument(
        "--val-samples",
        type=_positive_int_or_none,
        default=50,
        help="Limit validation to the first N samples (use 0 or 'none' for all).",
    )
    parser.add_argument("--num-steps", default=100000, type=int, help="Training iterations.")
    parser.add_argument("--diffusion-step", default=250, type=int, help="Number of diffusion steps.")
    parser.add_argument("--sample-steps", default=None, type=lambda v: None if v in {"", "none", "None"} else int(v))
    parser.add_argument("--checkpoint", default=5000, type=int, help="Checkpoint period.")
    parser.add_argument("--overall-results-path", default="results", type=str, help="Root results directory.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--dim", default=8, type=int, help="UNet base dimension.")
    parser.add_argument("--gradient-accumulate-every", default=2, type=int, help="Gradient accumulation steps.")
    parser.add_argument("--mixed-precision-type", default="no", type=str, help="Accelerate mixed precision type.")
    parser.add_argument("--num-workers", default=4, type=int, help="Dataloader workers.")
    parser.add_argument("--prefetch-factor", default=2, type=int, help="Dataloader prefetch factor.")
    parser.add_argument("--gap", default=32, type=int, help="Validation holdout size.")
    parser.add_argument("--model-type", default="Unet", choices=["Unet", "SiT_FNO", "CNO"], help="Backbone architecture to use for the diffusion model.")
    # SiT_FNO specific arguments
    parser.add_argument("--input-size", default=None, type=_parse_int_tuple, help="Override (height,width) for SiT_FNO spatial embedding.")
    parser.add_argument("--patch-size", default=None, type=_parse_int_tuple, help="Override (height,width) patch size for SiT_FNO.")
    parser.add_argument("--num-frames", default=None, type=int, help="Override temporal length for SiT_FNO.")
    parser.add_argument("--hidden-size", default=128, type=int, help="SiT_FNO hidden dimension.")
    parser.add_argument("--depth", default=6, type=int, help="SiT_FNO transformer depth.")
    parser.add_argument("--num-heads", default=4, type=int, help="SiT_FNO attention heads.")
    parser.add_argument("--modes", default=4, type=int, help="Fourier modes for the SiT_FNO compact FNO head.")
    # CNO specific arguments
    parser.add_argument("--n-layers", default=2, type=int, help="CNO hierarchical depth (number of U/D blocks).")
    parser.add_argument("--channel-multiplier", default=16, type=int, help="CNO channel multiplier controlling width.")
    parser.add_argument("--cno-input-size", default=None, type=int, help="Override cubic input size for CNO (defaults to max(frames, height, width)).")
    # Flow matching specific arguments
    parser.add_argument(
        "--fm-time-sampling",
        default="logit_normal",
        choices=["logit_normal", "uniform"],
        help="Time sampling strategy for flow matching.",
    )
    parser.add_argument("--fm-logit-mu", default=-0.8, type=float, help="Mean used for logit-normal time sampling.")
    parser.add_argument("--fm-logit-sigma", default=0.8, type=float, help="Std used for logit-normal time sampling.")
    parser.add_argument("--fm-sampling-steps", default=20, type=int, help="Number of ODE steps during FM sampling.")
    parser.add_argument("--fm-step-size", default=0.1, type=float, help="Fixed solver step size for FM sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cond, data = load_nt_dataset_emb(
        field=args.stage,
        dataset=args.dataset,
        n_data_set=args.n_dataset,
        data_root=Path(args.data_root) if args.data_root else None,
    )
    if args.gap <= 0 or args.gap >= cond.shape[0]:
        raise ValueError(f"gap must be between 1 and dataset size-1, got {args.gap} for dataset size {cond.shape[0]}")

    interval = -args.gap
    train_dataset = TensorDataset(data[:interval], cond[:interval])
    val_data = data[interval:]
    val_cond = cond[interval:]
    if args.val_samples:
        limit = min(len(val_data), args.val_samples)
        val_data = val_data[:limit]
        val_cond = val_cond[:limit]
    val_dataset = TensorDataset(val_data, val_cond)

    model = build_model(args, data, cond)
    get_parameter_net(model)

    train_fn, val_fn = create_training_functions()

    if args.val_batchsize is None:
        val_batchsize = args.batchsize * 10
    else:
        val_batchsize = args.val_batchsize

    results_path = create_res(args.overall_results_path, folder_name=args.exp_id)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_tag = f"{args.dataset}_{args.stage}_{args.n_dataset or 'full'}"
    results_folder = os.path.join(results_path, dataset_tag, time_stamp)
    os.makedirs(results_folder, exist_ok=True)
    save_config_from_args(args, results_folder)

    trainer = Trainer(
        model=model,
        data_train=train_dataset,
        train_function=train_fn,
        val_function=val_fn,
        data_val=val_dataset,
        train_lr=args.lr,
        train_num_steps=args.num_steps,
        train_batch_size=args.batchsize,
        val_batch_size=val_batchsize,
        save_every=args.checkpoint,
        results_folder=results_folder,
        mixed_precision_type=args.mixed_precision_type,
        gradient_accumulate_every=args.gradient_accumulate_every,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    trainer.train()


if __name__ == "__main__":
    main()

