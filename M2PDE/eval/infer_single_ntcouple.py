import argparse
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from data.ntcouple import load_nt_dataset_emb, load_nt_dataset_emb_eval, renormalize
from models.cno import CNO3d
from models.sit_fno import SiT_FNO
from eval.viz_utils import (
    parse_frame_indices,
    resolve_viz_dir,
    save_frame_grid,
    select_frame_indices,
    stage_channel_labels,
)
from models.unet3d import Unet3D
from paradigms.diffusion import GaussianDiffusion
from paradigms.fm_wrapper import FMWrapper
from utils import relative_error

DEFAULT_PATCH_SIZE: Tuple[int, int] = (2, 2)
VALID_NTCOUPLE_SPLITS = ("decouple_train", "decouple_val", "couple_train", "couple_val")
DEFAULT_EVAL_SAMPLES = 50


def _optional_positive_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    value = value.strip()
    if value.lower() in {"", "none"}:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _parse_int_tuple(raw: str) -> Tuple[int, ...]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("Expected at least one integer.")
    return tuple(int(part) for part in parts)


def _resolve_size(override: Optional[Tuple[int, ...]], fallback: Tuple[int, int]) -> Tuple[int, int]:
    if override is None:
        return fallback
    if len(override) == 1:
        return override[0], override[0]
    if len(override) >= 2:
        return override[0], override[1]
    raise ValueError("Override size must contain at least one integer.")

def _load_dataset(field: str, dataset: str, n_samples: Optional[int], data_root: Optional[Path]):
    dataset = dataset.lower()
    if dataset in {"couple_train", "couple_val"}:
        return load_nt_dataset_emb_eval(field=field, n_data_set=n_samples, data_root=data_root, dataset=dataset)
    return load_nt_dataset_emb(field=field, dataset=dataset, n_data_set=n_samples, data_root=data_root)


def _load_ema_weights(checkpoint_path: Path):
    state = torch.load(checkpoint_path, map_location="cpu")
    if "ema" not in state:
        raise ValueError(f"EMA weights not found in checkpoint {checkpoint_path}")
    ema_state = state["ema"]
    prefix = "online_model.model."
    trimmed = {k[len(prefix):]: v for k, v in ema_state.items() if k.startswith(prefix)}
    if not trimmed:
        # fallback to plain model weights
        trimmed = {k.partition(".")[2]: v for k, v in ema_state.items() if "." in k}
    return trimmed


def _load_model_weights(checkpoint_path: Path):
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in state:
        raise ValueError(f"Model weights not found in checkpoint {checkpoint_path}")
    model_state = state["model"]
    prefix = "model."
    trimmed = {k[len(prefix):]: v for k, v in model_state.items() if k.startswith(prefix)}
    if not trimmed:
        trimmed = model_state
    return trimmed


def _build_backbone_from_config(
    config: Dict,
    cond: torch.Tensor,
    data: torch.Tensor,
    fallback_args: argparse.Namespace
) -> torch.nn.Module:
    model_type = config.get("model_type", fallback_args.model_type)
    out_channels = data.shape[1]
    cond_channels = cond.shape[1]

    if model_type == "Unet":
        dim = config.get("dim", fallback_args.dim)
        return Unet3D(
            dim=dim,
            out_dim=out_channels,
            cond_channels=cond_channels,
            dim_mults=(1, 2, 4),
            use_sparse_linear_attn=False,
            attn_dim_head=16,
            expects_x=True,
        )

    if model_type == "CNO":
        cno_size = config.get("cno_input_size")
        if cno_size is None:
             cno_size = fallback_args.cno_input_size
        if cno_size is None:
             cno_size = max(data.shape[2], data.shape[3], data.shape[4])
             
        return CNO3d(
            in_dim=out_channels + cond_channels,
            out_dim=out_channels,
            in_size=cno_size,
            N_layers=config.get("n_layers", fallback_args.n_layers),
            channel_multiplier=config.get("channel_multiplier", fallback_args.channel_multiplier),
        )

    if model_type == "SiT_FNO":
        height, width = data.shape[-2], data.shape[-1]
        
        input_size_raw = config.get("input_size")
        if input_size_raw is None:
            input_size = _resolve_size(fallback_args.input_size, (height, width))
        elif isinstance(input_size_raw, str):
            input_size = _parse_int_tuple(input_size_raw)
            input_size = _resolve_size(input_size, (height, width))
        elif isinstance(input_size_raw, int):
             input_size = (input_size_raw, input_size_raw)
        else:
             input_size = tuple(input_size_raw)

        patch_size_raw = config.get("patch_size")
        if patch_size_raw is None:
             patch_size = _resolve_size(fallback_args.patch_size, DEFAULT_PATCH_SIZE)
        elif isinstance(patch_size_raw, str):
             patch_size = _resolve_size(_parse_int_tuple(patch_size_raw), DEFAULT_PATCH_SIZE)
        elif isinstance(patch_size_raw, int):
             patch_size = (patch_size_raw, patch_size_raw)
        else:
             patch_size = tuple(patch_size_raw)

        num_frames = config.get("num_frames")
        if num_frames is None:
            num_frames = fallback_args.num_frames or data.shape[2]
            
        return SiT_FNO(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=out_channels + cond_channels,
            out_channels=out_channels,
            hidden_size=config.get("hidden_size", fallback_args.hidden_size),
            depth=config.get("depth", fallback_args.depth),
            num_heads=config.get("num_heads", fallback_args.num_heads),
            num_frames=num_frames,
            modes=config.get("modes", fallback_args.modes),
        )

    raise ValueError(f"Unsupported model_type {model_type}")


def _build_generative_model(
    args: argparse.Namespace,
    checkpoint: Path,
    cond: torch.Tensor,
    data: torch.Tensor,
    diffusion_step: int,
    sample_steps: Optional[int],
    device: torch.device,
):
    config_path = checkpoint.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    else:
        print(f"Warning: config.yaml not found at {config_path}, using args/defaults.")
        config = {}

    paradigm = config.get("paradigm", args.paradigm)
    backbone = _build_backbone_from_config(config, cond, data, args)
    seq_shape = tuple(data.shape[1:])

    if paradigm == "fm":
        fm_kwargs = dict(
            seq_shape=seq_shape,
            time_sampling=config.get("fm_time_sampling", "logit_normal"),
            logit_mu=config.get("fm_logit_mu", -0.8),
            logit_sigma=config.get("fm_logit_sigma", 0.8),
            sampling_steps=config.get("fm_sampling_steps", args.fm_sampling_steps),
            solver_step_size=config.get("fm_step_size", args.fm_step_size),
        )
        # Load weights into backbone first
        backbone.load_state_dict(_load_ema_weights(checkpoint))
        
        wrapper = FMWrapper(backbone, **fm_kwargs).to(device)
        wrapper.eval()
        return wrapper, "fm"

    backbone.load_state_dict(_load_ema_weights(checkpoint))
    diff_steps = config.get("diffusion_step", diffusion_step)
    sample_t = config.get("sample_steps", sample_steps)
    diffusion = GaussianDiffusion(
        backbone,
        seq_length=seq_shape,
        timesteps=diff_steps,
        sampling_timesteps=sample_t,
        auto_normalize=False,
    ).to(device)
    diffusion.eval()
    return diffusion, "diffusion"


def evaluate_model(
    model: nn.Module,
    cond: torch.Tensor,
    data: torch.Tensor,
    stage: str,
    device: torch.device,
    batch_size: int,
    paradigm: str,
) -> dict:
    mse_total = 0.0
    rel_total = 0.0
    count = 0
    num_channels = data.shape[1]
    channel_mse = [0.0 for _ in range(num_channels)]
    channel_rel = [0.0 for _ in range(num_channels)]

    with torch.no_grad():
        for start in tqdm(range(0, cond.shape[0], batch_size), desc="Sampling"):
            end = min(start + batch_size, cond.shape[0])
            cond_batch = cond[start:end].to(device)
            data_batch = data[start:end]
            preds = model.sample(cond_batch.shape[0], cond_batch).cpu()

            preds_phys = renormalize(preds, stage)
            data_phys = renormalize(data_batch, stage)

            mse_total += F.mse_loss(preds_phys, data_phys).item() * cond_batch.shape[0]
            rel_total += relative_error(data_phys, preds_phys) * cond_batch.shape[0]
            for ch in range(num_channels):
                pred_ch = preds_phys[:, ch : ch + 1]
                data_ch = data_phys[:, ch : ch + 1]
                channel_mse[ch] += F.mse_loss(pred_ch, data_ch).item() * cond_batch.shape[0]
                channel_rel[ch] += relative_error(data_ch, pred_ch) * cond_batch.shape[0]
            count += cond_batch.shape[0]

    labels = stage_channel_labels(stage, num_channels)
    channel_metrics = {
        label: {
            "mse": channel_mse[idx] / count if count else 0.0,
            "relative_error": channel_rel[idx] / count if count else 0.0,
        }
        for idx, label in enumerate(labels)
    }

    return {
        "mode": paradigm,
        "stage": stage,
        "mse": mse_total / count,
        "relative_error": rel_total / count,
        "samples": count,
        "channels": channel_metrics,
    }


def _resolve_output_path(output_arg: Optional[str], checkpoint: Path, stage: str) -> Path:
    if output_arg:
        output_path = Path(output_arg)
    else:
        output_path = checkpoint.parent / f"{checkpoint.stem}_{stage}_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _generate_visualizations(
    model: nn.Module,
    cond: torch.Tensor,
    data: torch.Tensor,
    stage: str,
    sample_index: int,
    frame_indices: list[int],
    viz_dir: Path,
    device: torch.device,
) -> Optional[Dict[str, object]]:
    if not frame_indices or cond.shape[0] == 0:
        return None

    sample_index = max(0, min(sample_index, data.shape[0] - 1))
    cond_sample = cond[sample_index : sample_index + 1].to(device)
    data_sample = data[sample_index : sample_index + 1]

    with torch.no_grad():
        pred_norm = model.sample(cond_sample.shape[0], cond_sample).cpu()

    pred_phys = renormalize(pred_norm, stage)[0]
    true_phys = renormalize(data_sample, stage)[0]

    saved_paths = save_frame_grid(true_phys, pred_phys, stage, frame_indices, sample_index, viz_dir)
    if not saved_paths:
        return None

    return {
        "sample_index": sample_index,
        "frames": frame_indices,
        "directory": str(viz_dir),
        "images": [str(path) for path in saved_paths],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer NTcouple diffusion or surrogate checkpoints.")
    parser.add_argument("--stage", choices=["neutron", "solid", "fluid"], required=True, help="Physics field.")
    parser.add_argument("--paradigm", default="diffusion", choices=["diffusion", "fm"], help="Model paradigm to load.")
    parser.add_argument(
        "--dataset",
        default="couple_val",
        choices=VALID_NTCOUPLE_SPLITS,
        help="Dataset split to evaluate on.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint (.pt) file.")
    parser.add_argument("--data-root", default=None, help="Optional dataset root override.")
    parser.add_argument(
        "--n-samples",
        default=DEFAULT_EVAL_SAMPLES,
        type=_optional_positive_int,
        help="Max number of samples to evaluate (use 0/none for full dataset).",
    )
    parser.add_argument("--batch-size", default=16, type=int, help="Evaluation batch size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--diffusion-step", default=250, type=int, help="Diffusion steps (fallback).")
    parser.add_argument("--sample-steps", default=None, type=lambda v: None if v in {"", "none", "None"} else int(v))
    parser.add_argument("--dim", default=8, type=int, help="UNet base dimension (fallback).")
    parser.add_argument("--output", default=None, help="Optional JSON file to store metrics.")
    parser.add_argument("--model-type", default="Unet", choices=["Unet", "SiT_FNO", "CNO"], help="Backbone architecture (fallback).")
    parser.add_argument("--n-layers", default=2, type=int, help="CNO hierarchical depth (fallback).")
    parser.add_argument("--channel-multiplier", default=16, type=int, help="CNO channel multiplier (fallback).")
    parser.add_argument("--cno-input-size", default=None, type=int, help="Override cubic input size for CNO.")
    parser.add_argument("--input-size", default=None, type=_parse_int_tuple, help="Override (height,width) for SiT_FNO spatial embedding.")
    parser.add_argument("--patch-size", default=None, type=_parse_int_tuple, help="Override (height,width) patch size for SiT_FNO.")
    parser.add_argument("--num-frames", default=None, type=int, help="Override temporal length for SiT_FNO.")
    parser.add_argument("--hidden-size", default=128, type=int, help="SiT_FNO hidden dimension.")
    parser.add_argument("--depth", default=6, type=int, help="SiT_FNO transformer depth.")
    parser.add_argument("--num-heads", default=4, type=int, help="SiT_FNO attention heads.")
    parser.add_argument("--modes", default=4, type=int, help="Fourier modes for SiT_FNO compact head.")
    parser.add_argument("--fm-sampling-steps", default=20, type=int, help="Flow Matching ODE steps during sampling.")
    parser.add_argument("--fm-step-size", default=0.1, type=float, help="Flow Matching solver step size.")
    parser.add_argument("--viz-frames", default=None, help="Comma-separated frame indices to visualize (default 0-4).")
    parser.add_argument("--viz-sample", default=0, type=int, help="Dataset sample index to visualize.")
    parser.add_argument("--viz-samples", default=None, type=int, help="Limit inference to the first N samples for fast visualization.")
    parser.add_argument("--viz-dir", default=None, help="Directory to store visualization PNGs (defaults near JSON).")
    parser.add_argument("--disable-viz", action="store_true", help="Skip visualization export.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    device = torch.device(args.device)

    data_root = Path(args.data_root) if args.data_root else None
    cond, data = _load_dataset(args.stage, args.dataset, args.n_samples, data_root)
    if args.viz_samples is not None:
        if args.viz_samples <= 0:
            raise ValueError("--viz-samples must be positive when provided.")
        max_count = min(args.viz_samples, cond.shape[0])
        cond = cond[:max_count]
        data = data[:max_count]

    checkpoint = Path(args.checkpoint)
    model, paradigm = _build_generative_model(
        args=args,
        checkpoint=checkpoint,
        cond=cond,
        data=data,
        diffusion_step=args.diffusion_step,
        sample_steps=args.sample_steps,
        device=device,
    )
    metrics = evaluate_model(
        model=model,
        cond=cond,
        data=data,
        stage=args.stage,
        device=device,
        batch_size=args.batch_size,
        paradigm=paradigm,
    )

    requested_frames = parse_frame_indices(args.viz_frames)
    frame_indices = select_frame_indices(total_frames=data.shape[2], requested=requested_frames)
    viz_info = None
    output_path = _resolve_output_path(args.output, checkpoint, args.stage)
    if not args.disable_viz:
        viz_dir = resolve_viz_dir(output_path, args.viz_dir)
        viz_info = _generate_visualizations(
            model=model,
            cond=cond,
            data=data,
            stage=args.stage,
            sample_index=args.viz_sample,
            frame_indices=frame_indices,
            viz_dir=viz_dir,
            device=device,
        )
    if viz_info:
        metrics["visualizations"] = viz_info

    print(json.dumps(metrics, indent=2))
    output_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
