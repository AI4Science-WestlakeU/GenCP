import argparse
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml

import torch
import torch.nn.functional as F

from data.ntcouple import (
    FIELD_ORDER,
    default_ntcouple_updates,
    normalize,
    load_nt_dataset_emb,
    load_nt_dataset_emb_eval,
    renormalize,
)
from eval.compose import compose_diffusion
from eval.viz_utils import (
    parse_frame_indices,
    resolve_viz_dir,
    save_frame_grid,
    select_frame_indices,
    stage_channel_labels,
)
from models.cno import CNO3d
from models.sit_fno import SiT_FNO
from models.unet3d import Unet3D
from paradigms.diffusion import GaussianDiffusion
from utils import relative_error

DEFAULT_PATCH_SIZE: Tuple[int, int] = (2, 2)
VALID_NTCOUPLE_SPLITS = ("decouple_train", "decouple_val", "couple_train", "couple_val")


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
        trimmed = {k.partition(".")[2]: v for k, v in ema_state.items() if "." in k}
    return trimmed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer composed NTcouple diffusion checkpoints.")
    parser.add_argument(
        "--eval-split",
        default="couple_val",
        choices=VALID_NTCOUPLE_SPLITS,
        help="Evaluation split.",
    )
    parser.add_argument("--data-root", default=None, help="Optional dataset root override.")
    parser.add_argument("--n-samples", default=None, type=lambda v: None if v in {"", "none", "None"} else int(v))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Note: diffusion-step, n-layers etc. might be overridden by config if present
    parser.add_argument("--diffusion-step", default=250, type=int, help="Diffusion timesteps (fallback if not in config).")
    parser.add_argument("--sample-steps", default=None, type=lambda v: None if v in {"", "none", "None"} else int(v))
    
    # Legacy args, might be ignored if config is used
    parser.add_argument("--dim-neutron", default=8, type=int, help="UNet base dim for neutron.")
    parser.add_argument("--dim-solid", default=8, type=int, help="UNet base dim for solid.")
    parser.add_argument("--dim-fluid", default=16, type=int, help="UNet base dim for fluid.")
    
    parser.add_argument("--checkpoint-neutron", required=True, help="Neutron checkpoint path.")
    parser.add_argument("--checkpoint-solid", required=True, help="Solid checkpoint path.")
    parser.add_argument("--checkpoint-fluid", required=True, help="Fluid checkpoint path.")
    parser.add_argument("--outer-iters", default=2, type=int, help="Outer iterations for composition.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--viz-frames", default=None, help="Comma-separated frame indices to visualize (default 0-4).")
    parser.add_argument("--viz-sample", default=0, type=int, help="Dataset sample index used for visualization.")
    parser.add_argument("--viz-samples", default=None, type=int, help="Limit inference to the first N samples before visualization.")
    parser.add_argument("--viz-dir", default=None, help="Directory to store visualization PNGs (defaults near JSON).")
    parser.add_argument("--disable-viz", action="store_true", help="Skip visualization export.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
    
    # These args act as overrides or fallbacks now
    parser.add_argument("--model-type", default="Unet", choices=["Unet", "SiT_FNO", "CNO"], help="Backbone architecture (fallback).")
    parser.add_argument("--n-layers", default=2, type=int, help="CNO hierarchical depth (fallback).")
    parser.add_argument("--channel-multiplier", default=16, type=int, help="CNO channel multiplier (fallback).")
    parser.add_argument("--cno-input-size", default=None, type=int, help="Override cubic input size for CNO.")
    parser.add_argument("--batch-size", default=None, type=int, help="Batch size for composed inference (defaults to full dataset).")
    parser.add_argument("--input-size", default=None, type=_parse_int_tuple, help="Override (height,width) for SiT_FNO spatial embedding.")
    parser.add_argument("--patch-size", default=None, type=_parse_int_tuple, help="Override (height,width) patch size for SiT_FNO.")
    parser.add_argument("--num-frames", default=None, type=int, help="Override temporal length for SiT_FNO.")
    parser.add_argument("--hidden-size", default=128, type=int, help="SiT_FNO hidden dimension.")
    parser.add_argument("--depth", default=6, type=int, help="SiT_FNO transformer depth.")
    parser.add_argument("--num-heads", default=4, type=int, help="SiT_FNO attention heads.")
    parser.add_argument("--modes", default=4, type=int, help="Fourier modes for SiT_FNO compact head.")
    return parser.parse_args()


def _resolve_output_path(output_arg: Optional[str], checkpoint_fluid: Path, eval_split: str) -> Path:
    if output_arg:
        path = Path(output_arg)
    else:
        path = checkpoint_fluid.parent / f"{eval_split}_coupled_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _generate_visualizations(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    requested_frames: Optional[list[int]],
    sample_index: int,
    viz_dir: Path,
) -> Dict[str, Dict[str, object]]:
    viz_metadata: Dict[str, Dict[str, object]] = {}
    for field, pred_tensor in preds.items():
        target_tensor = targets[field]
        if target_tensor.shape[0] == 0:
            continue
        frame_indices = select_frame_indices(total_frames=target_tensor.shape[2], requested=requested_frames)
        if not frame_indices:
            continue

        sample_idx = max(0, min(sample_index, target_tensor.shape[0] - 1))
        field_dir = viz_dir / field
        saved_paths = save_frame_grid(
            true_sample=target_tensor[sample_idx],
            pred_sample=pred_tensor[sample_idx],
            stage=field,
            frame_indices=frame_indices,
            sample_idx=sample_idx,
            output_dir=field_dir,
        )
        if saved_paths:
            viz_metadata[field] = {
                "frames": frame_indices,
                "sample_index": sample_idx,
                "directory": str(field_dir),
                "images": [str(path) for path in saved_paths],
            }
    return viz_metadata


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
        dim = config.get("dim", getattr(fallback_args, "dim", 8)) # Fallback or config
        # The args had dim_neutron, dim_solid etc. but config just has dim.
        # We assume config is correct for the loaded model.
        
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
             # Try fallback args
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


def _summarize_field_metrics(pred_phys: torch.Tensor, target_phys: torch.Tensor, stage: str) -> Dict[str, Dict[str, float]]:
    mse = F.mse_loss(pred_phys, target_phys).item()
    rel = relative_error(target_phys, pred_phys)
    labels = stage_channel_labels(stage, pred_phys.shape[1])
    channel_metrics = {}
    for idx, label in enumerate(labels):
        pred_ch = pred_phys[:, idx : idx + 1]
        target_ch = target_phys[:, idx : idx + 1]
        channel_metrics[label] = {
            "mse": F.mse_loss(pred_ch, target_ch).item(),
            "relative_error": relative_error(target_ch, pred_ch),
        }
    return {"mse": mse, "relative_error": rel, "channels": channel_metrics}


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

    conds: Dict[str, torch.Tensor] = {}
    targets: Dict[str, torch.Tensor] = {}
    for field in FIELD_ORDER:
        cond, data = _load_dataset(field, args.eval_split, args.n_samples, data_root)
        if args.viz_samples is not None:
            if args.viz_samples <= 0:
                raise ValueError("--viz-samples must be positive when provided.")
            limit = min(args.viz_samples, cond.shape[0])
            cond = cond[:limit]
            data = data[:limit]
        conds[field] = cond
        targets[field] = data

    # Checkpoints map
    checkpoints = {
        "neutron": Path(args.checkpoint_neutron),
        "solid": Path(args.checkpoint_solid),
        "fluid": Path(args.checkpoint_fluid),
    }

    diffusions = {}
    for field in FIELD_ORDER:
        cond = conds[field]
        data = targets[field]
        ckpt_path = checkpoints[field]
        
        # Load config
        config_path = ckpt_path.parent / "config.yaml"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            print(f"Warning: config.yaml not found for {field} at {config_path}, using args/defaults.")
            config = {}

        # Determine diffusion steps from config or args
        diff_steps = config.get("diffusion_step", args.diffusion_step)
        
        model = _build_backbone_from_config(config, cond, data, args)
        
        diffusion = GaussianDiffusion(
            model,
            seq_length=tuple(data.shape[1:]),
            timesteps=diff_steps,
            sampling_timesteps=args.sample_steps,
            auto_normalize=False,
        ).to(device)
        
        diffusion.model.load_state_dict(_load_ema_weights(ckpt_path))
        diffusion.eval()
        diffusions[field] = diffusion

    updates = default_ntcouple_updates()
    total_samples = conds["neutron"].shape[0] # Assuming all have same len
    shapes = [targets[field].shape for field in FIELD_ORDER]
    metrics = {}
    batch_size = args.batch_size or total_samples
    batch_size = max(1, min(batch_size, total_samples))
    field_labels = {field: stage_channel_labels(field, targets[field].shape[1]) for field in FIELD_ORDER}
    aggregates = {
        field: {
            "mse": 0.0,
            "relative_error": 0.0,
            "channels": {label: {"mse": 0.0, "relative_error": 0.0} for label in field_labels[field]},
        }
        for field in FIELD_ORDER
    }
    viz_sample = max(0, min(args.viz_sample, total_samples - 1))
    viz_predictions: Dict[str, torch.Tensor] = {}
    viz_targets: Dict[str, torch.Tensor] = {}

    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)
        current_batch = end - start
        batch_conds = {field: conds[field][start:end] for field in FIELD_ORDER}
        batch_targets = {field: targets[field][start:end] for field in FIELD_ORDER}
        bc_channel = batch_conds["neutron"][:, -1:]
        bc_batch = bc_channel[..., :1]

        with torch.no_grad():
            generated = compose_diffusion(
                [diffusions[field] for field in FIELD_ORDER],
                [batch_targets[field].shape for field in FIELD_ORDER],
                updates,
                normalize,
                renormalize,
                other_condition=[bc_batch.to(device)],
                num_iter=args.outer_iters,
                device=device,
            )

        for field, pred_norm in zip(FIELD_ORDER, generated):
            pred_phys = renormalize(pred_norm.cpu(), field)
            target_phys = renormalize(batch_targets[field], field)
            field_metrics = _summarize_field_metrics(pred_phys, target_phys, field)
            aggregates[field]["mse"] += field_metrics["mse"] * current_batch
            aggregates[field]["relative_error"] += field_metrics["relative_error"] * current_batch
            for label, stats in field_metrics["channels"].items():
                aggregates[field]["channels"][label]["mse"] += stats["mse"] * current_batch
                aggregates[field]["channels"][label]["relative_error"] += stats["relative_error"] * current_batch

            if start <= viz_sample < end:
                sample_idx = viz_sample - start
                viz_predictions[field] = pred_phys[sample_idx : sample_idx + 1]
                viz_targets[field] = target_phys[sample_idx : sample_idx + 1]

    for field in FIELD_ORDER:
        field_total = aggregates[field]
        metrics[field] = {
            "mse": field_total["mse"] / total_samples,
            "relative_error": field_total["relative_error"] / total_samples,
            "channels": {
                label: {
                    "mse": stats["mse"] / total_samples,
                    "relative_error": stats["relative_error"] / total_samples,
                }
                for label, stats in field_total["channels"].items()
            },
        }

    requested_frames = parse_frame_indices(args.viz_frames)
    output_path = _resolve_output_path(args.output, checkpoints["fluid"], args.eval_split)
    if not args.disable_viz:
        viz_dir = resolve_viz_dir(output_path, args.viz_dir)
        if viz_predictions and viz_targets:
            viz_info = _generate_visualizations(
                preds={k: v for k, v in viz_predictions.items()},
                targets={k: v for k, v in viz_targets.items()},
                requested_frames=requested_frames,
                sample_index=0,
                viz_dir=viz_dir,
            )
        else:
            viz_info = {}
    else:
        viz_info = {}
    if viz_info:
        metrics["visualizations"] = viz_info

    print(json.dumps(metrics, indent=2))
    output_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
