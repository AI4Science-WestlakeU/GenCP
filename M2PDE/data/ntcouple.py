import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Literal

import torch
import numpy as np
import torch.nn.functional as F

try:
    from ..models.unet3d import Unet3D
    from ..paradigms.diffusion import GaussianDiffusion
    from ..utils import relative_error
    from ..eval.compose import compose_diffusion
except ImportError:  # Script execution fallback
    from models.unet3d import Unet3D  # type: ignore
    from paradigms.diffusion import GaussianDiffusion  # type: ignore
    from utils import relative_error  # type: ignore
    from eval.compose import compose_diffusion  # type: ignore

FieldType = Literal["neutron", "solid", "fluid"]
SplitType = Literal[
    "iter1_train",
    "iter1_val",
    "iter2_train",
    "iter2_val",
    "iter3_train",
    "iter3_val",
    "iter4_train",
    "iter4_val",
    "couple_train",
    "couple_val",
]


def resolve_ntcouple_root(data_root: Optional[Path] = None) -> Path:
    """
    Resolve NTcouple dataset root using the same priority as GenCP loader.

    Priority:
        1. Explicit ``data_root`` argument.
        2. Environment variable ``NTCOUPLE_DATA_ROOT``.
        3. Common dataset locations inside the repo or user home.
    """
    if data_root is not None:
        path = Path(data_root).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"Specified data_root does not exist: {data_root}")

    env_root = os.getenv("NTCOUPLE_DATA_ROOT")
    if env_root:
        path = Path(env_root).expanduser()
        if path.exists():
            return path

    common_paths = [
        Path("/path/to/dataset/NTcouple/"),
        Path.home() / "data/ntcouple/",
        Path(__file__).resolve().parent.parent.parent / "data/ntcouple/",
    ]

    for path in common_paths:
        if path.exists():
            return path

    searched = ", ".join(str(p) for p in common_paths)
    raise FileNotFoundError(
        "NTcouple dataset not found. Please either:\n"
        "  1. Set NTCOUPLE_DATA_ROOT=/path/to/ntcouple\n"
        "  2. Place the data in one of the common locations\n"
        "  3. Pass data_root explicitly\n"
        f"Searched locations: {searched}"
    )

def _ensure_5d(x: torch.Tensor, name: str, allow_fallback: bool = False) -> torch.Tensor:
    """
    Make sure tensors follow the NTcouple convention (batch, channels, frames, height, width).

    Args:
        x: Input tensor.
        name: Friendly identifier used in error messages.
        allow_fallback: If True, automatically unsqueeze/reshape to reach 5D.

    Returns:
        Tensor guaranteed to be 5-dimensional.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")
    if x.dim() == 5:
        return x
    if not allow_fallback:
        raise ValueError(f"{name} must be a 5D tensor, got {x.dim()}D tensor")
    else:
        # fallback: expand dims to 5D by prepending singleton channel dims
        while x.dim() < 5:
            x = x.unsqueeze(0)
        # re-arrange to (b,c,f,h,w) if we over-padded
        if x.dim() > 5:
            # collapse leading dims into batch
            leading = x.shape[: x.dim() - 5]
            b = 1
            for d in leading:
                b *= d
            x = x.reshape((b,) + x.shape[-5:])
    return x


def _match_spatiotemporal(
    x: torch.Tensor, target_shape: Tuple[int, int, int], allow_fallback: bool = False
) -> torch.Tensor:
    """
    Broadcast or interpolate tensors to match a target ``(frames, height, width)`` shape.

    Args:
        x: Tensor shaped (batch, channels, frames, height, width).
        target_shape: Desired ``(frames, height, width)`` triple.
        allow_fallback: If True, resize with nearest-neighbour interpolation when spatial sizes differ.

    Returns:
        Tensor with the requested spatiotemporal dimensions.
    """
    # expects x shape (b,c,f,h,w); target is (f,h,w)
    if x.shape[2:] == target_shape:
        return x
    if not allow_fallback:
        raise ValueError(f"x shape {x.shape} does not match target shape {target_shape}")
    else:
        b, c, f, h, w = x.shape
        tf, th, tw = target_shape
        size = (tf, th, tw)
        # 5D interpolate with nearest to avoid introducing new values
        return F.interpolate(x, size=size, mode="nearest")


def make_cond_tensor(
    cond_list: List[torch.Tensor], data: torch.Tensor, pad_channels_to: Optional[int] = None
) -> torch.Tensor:
    """
    Stack and align conditioning tensors so they match a target data tensor.

    Args:
        cond_list: Sequence of tensors that will be concatenated along channel dimension.
        data: Reference tensor whose ``(frames, height, width)`` dimensions must be matched.
        pad_channels_to: Optional channel count to pad with ones for compatibility with legacy checkpoints.

    Returns:
        Concatenated conditioning tensor with shape ``(batch, channels, frames, height, width)``.
    """
    data = _ensure_5d(data, "data")
    target_fhw = data.shape[2:]
    adjusted: List[torch.Tensor] = []
    for idx, cond in enumerate(cond_list):
        ci = _ensure_5d(cond, f"cond[{idx}]")
        ci = _match_spatiotemporal(ci, target_fhw)
        adjusted.append(ci)

    cond_tensor = torch.cat(adjusted, dim=1) if len(adjusted) > 0 else None
    if cond_tensor is None:
        raise ValueError("cond_list is empty; at least one conditioning tensor is required")

    if pad_channels_to is not None and cond_tensor.shape[1] < pad_channels_to:
        pad_c = pad_channels_to - cond_tensor.shape[1]
        ones_pad = torch.ones(
            cond_tensor.shape[0], pad_c, cond_tensor.shape[2], cond_tensor.shape[3], cond_tensor.shape[4], device=cond_tensor.device, dtype=cond_tensor.dtype
        )
        cond_tensor = torch.cat([cond_tensor, ones_pad], dim=1)
    return cond_tensor


def shape_report(name: str, x: torch.Tensor) -> str:
    """Return a readable description of a tensor's 5D shape."""
    x = _ensure_5d(x, name)
    b, c, f, h, w = x.shape
    return f"{name}: b={b}, c={c}, f={f}, h={h}, w={w}"


def report_shapes(field: str, data: torch.Tensor, cond_list: List[torch.Tensor]) -> str:
    """Compose a one-line summary of data/condition tensor shapes for logging."""
    lines = [f"Field={field}", shape_report("data", data)]
    for i, c in enumerate(cond_list):
        lines.append(shape_report(f"cond[{i}]", c))
    return " | ".join(lines)


def normalize(x: torch.Tensor, field: str) -> torch.Tensor:
    """
    Normalize NTcouple values into [-1, 1] using fixed physical bounds.

    Args:
        x: Tensor shaped (batch, channels, frames, height, width).
        field: One of ``{"solid", "fluid", "neutron", "flux"}``.

    Returns:
        Tensor of the same shape with values in [-1, 1].
    """
    solid_bound = [100.0, 1500.0]
    flux_bound = [-2000.0, 4000.0]
    fluid_bound = [
        [100.0, 1200.0],
        [-50.0, 250.0],
        [-0.006, 0.006],
        [0.0, 0.6],
    ]
    neutron_bound = [0.0, 3.258]

    x = x.clone().to(dtype=torch.float32)

    if field == "solid":
        return (x - solid_bound[0]) / (solid_bound[1] - solid_bound[0]) * 2.0 - 1.0

    if field == "fluid":
        if x.shape[1] > len(fluid_bound):
            raise ValueError(f"Fluid tensor has {x.shape[1]} channels, expected ≤ {len(fluid_bound)}.")
        for chan in range(x.shape[1]):
            lower, upper = fluid_bound[chan]
            x[:, chan] = (x[:, chan] - lower) / (upper - lower) * 2.0 - 1.0
        return x

    if field == "neutron":
        return (torch.log(x + 1.0) - neutron_bound[0]) / (neutron_bound[1] - neutron_bound[0]) * 2.0 - 1.0

    if field == "flux":
        return (x - flux_bound[0]) / (flux_bound[1] - flux_bound[0]) * 2.0 - 1.0

    raise ValueError(f"Unsupported field '{field}'")


def renormalize(x: torch.Tensor, field: str) -> torch.Tensor:
    """
    Convert normalized [-1, 1] values back to their physical ranges.

    Args:
        x: Tensor shaped (batch, channels, frames, height, width).
        field: One of ``{"solid", "fluid", "neutron", "flux"}``.

    Returns:
        Tensor of the same shape with values in the original units.
    """
    solid_bound = [100.0, 1500.0]
    flux_bound = [-2000.0, 4000.0]
    fluid_bound = [
        [100.0, 1200.0],
        [-50.0, 250.0],
        [-0.006, 0.006],
        [0.0, 0.6],
    ]
    neutron_bound = [0.0, 3.258]

    x = x.clone().to(dtype=torch.float32)

    if field == "solid":
        return (x + 1.0) * 0.5 * (solid_bound[1] - solid_bound[0]) + solid_bound[0]

    if field == "fluid":
        if x.shape[1] > len(fluid_bound):
            raise ValueError(f"Fluid tensor has {x.shape[1]} channels, expected ≤ {len(fluid_bound)}.")
        for chan in range(x.shape[1]):
            lower, upper = fluid_bound[chan]
            x[:, chan] = (x[:, chan] + 1.0) * 0.5 * (upper - lower) + lower
        return x

    if field == "neutron":
        return torch.exp((x + 1.0) * 0.5 * (neutron_bound[1] - neutron_bound[0]) + neutron_bound[0]) - 1.0

    if field == "flux":
        return (x + 1.0) * 0.5 * (flux_bound[1] - flux_bound[0]) + flux_bound[0]

    raise ValueError(f"Unsupported field '{field}'")



def _resolve_root(data_root: Optional[Path]) -> Path:
    """Resolve the dataset root using GenCP-compatible fallback hierarchy."""
    return resolve_ntcouple_root(data_root)


def _load_tensor(file_path: Path, limit: Optional[int]) -> torch.Tensor:
    array = np.load(str(file_path))
    if limit is not None:
        array = array[:limit]
    return torch.from_numpy(array).float()


_DECOUPLED_SPLITS = {
    "iter1_train",
    "iter1_val",
    "iter2_train",
    "iter2_val",
    "iter3_train",
    "iter3_val",
    "iter4_train",
    "iter4_val",
}
_COUPLED_SPLITS = {"couple_train", "couple_val"}


def load_ntcouple_dataset(
    field: FieldType,
    split: SplitType = "iter2_train",
    n_samples: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load NTcouple conditioning tensors and target fields for a given split.

    Args:
        field: ``"neutron"``, ``"solid"``, or ``"fluid"``.
        split: Dataset split name (iter1–iter4 decoupled or ``couple_train``/``couple_val``).
        n_samples: Optional slice on the batch dimension for quick experiments.
        data_root: Override the dataset root (defaults to ``resolve_ntcouple_root`` search order).

    Returns:
        A tuple ``(cond, data)`` of torch tensors already normalized to [-1, 1].
    """
    field = field.lower()
    split = split.lower()
    if split not in _DECOUPLED_SPLITS | _COUPLED_SPLITS:
        raise ValueError(f"Unsupported split '{split}'.")

    root = _resolve_root(data_root)
    folder = root / split
    if not folder.exists():
        raise FileNotFoundError(
            f"NTcouple split '{split}' not found under '{folder}'. "
            "Set NTCOUPLE_DATA_ROOT or download the dataset."
        )

    if field == "neutron":
        if split in _DECOUPLED_SPLITS:
            bc = normalize(_load_tensor(folder / "bc_neu.npy", n_samples), "neutron")
            fuel = normalize(_load_tensor(folder / "fuel_neu.npy", n_samples), "solid")
            fluid = normalize(_load_tensor(folder / "fluid_neu.npy", n_samples), "fluid")
        else:
            bc = normalize(_load_tensor(folder / "bc.npy", n_samples), "neutron")
            fuel = normalize(_load_tensor(folder / "fuel.npy", n_samples), "solid")
            fluid_raw = _load_tensor(folder / "fluid.npy", n_samples)
            fluid = normalize(fluid_raw[:, :1], "fluid")

        data = normalize(_load_tensor(folder / "neu.npy", n_samples), "neutron")
        stacked_cond = torch.cat((fuel, fluid), dim=-1)
        bc = bc.repeat(1, 1, 1, 1, stacked_cond.shape[-1])
        cond = torch.cat((stacked_cond, bc), dim=1)
        return cond, data

    if field == "solid":
        fuel_raw = _load_tensor(folder / "fuel.npy", n_samples)
        left_boundary = fuel_raw[:, :, :, :, 0:1]

        if split in _DECOUPLED_SPLITS:
            neu = normalize(_load_tensor(folder / "neu_fuel.npy", n_samples), "neutron")
            fluid = normalize(_load_tensor(folder / "fluid_fuel.npy", n_samples), "fluid")
        else:
            neu_raw = _load_tensor(folder / "neu.npy", n_samples)
            neu = normalize(neu_raw[:, :, :, :, :8], "neutron")
            fluid_raw = _load_tensor(folder / "fluid.npy", n_samples)
            fluid = normalize(fluid_raw[:, :1, :, :, :1], "fluid")

        data = normalize(fuel_raw, "solid")
        left_boundary = normalize(left_boundary, "solid")
        fluid_expanded = fluid.repeat(1, 1, 1, 1, neu.shape[-1])
        left_boundary_expanded = left_boundary.repeat(1, 1, 1, 1, neu.shape[-1])
        cond = torch.cat((neu, fluid_expanded, left_boundary_expanded), dim=1)
        return cond, data

    if field == "fluid":
        data = normalize(_load_tensor(folder / "fluid.npy", n_samples), "fluid")

        if split in _DECOUPLED_SPLITS:
            fuel_boundary = normalize(_load_tensor(folder / "delta_fuel_fluid.npy", n_samples), "solid")
        else:
            fuel_temp = _load_tensor(folder / "fuel.npy", n_samples)
            temperature_prev = fuel_temp[:, :1, :, :, -2:-1]
            temperature_curr = fuel_temp[:, :1, :, :, -1:]
            boundary = torch.cat((temperature_prev, temperature_curr), dim=-1)
            fuel_boundary = normalize(boundary, "solid")

        if fuel_boundary.shape[2] != data.shape[2] or fuel_boundary.shape[3] != data.shape[3]:
            raise ValueError(
                f"Fuel boundary spatiotemporal shape {fuel_boundary.shape[2:]} "
                f"does not match fluid target {data.shape[2:]}"
            )

        boundary_width = fuel_boundary.shape[-1]
        fluid_width = data.shape[-1]
        if boundary_width == 0 or fluid_width % boundary_width != 0:
            raise ValueError(
                f"Fuel boundary width {boundary_width} does not divide fluid width {fluid_width}"
            )

        repeat_factor = fluid_width // boundary_width
        if repeat_factor > 1:
            fuel_boundary = fuel_boundary.repeat(1, 1, 1, 1, repeat_factor)
        return fuel_boundary, data

    raise ValueError(f"Unsupported field '{field}'.")


def load_nt_dataset_emb(
    field: FieldType = "neutron",
    dataset: SplitType = "iter2_train",
    n_data_set: Optional[int] = None,
    data_root: Optional[Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backwards-compatible entrypoint mirroring the original API.

    Args:
        field: Physics field to train (``neutron``, ``solid``, ``fluid``).
        dataset: Dataset split name.
        n_data_set: Optional limit on the number of samples.
        data_root: Optional dataset root override.
    """
    return load_ntcouple_dataset(field=field, split=dataset, n_samples=n_data_set, data_root=data_root)


def load_nt_dataset_emb_eval(
    field: FieldType = "neutron",
    n_data_set: Optional[int] = 50,
    data_root: Optional[Path] = None,
    dataset: SplitType = "couple_val",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Legacy helper used in evaluation notebooks (defaults to the coupled split).

    Args:
        field: Physics field to evaluate (``neutron``, ``solid``, ``fluid``).
        n_data_set: Optional limit on the number of samples. Defaults to 50 so that
            validation/evaluation runs stay lightweight; pass ``None`` (for example,
            ``--n-samples none`` on the CLI) to use the full split.
        data_root: Optional dataset root override.
    """
    return load_ntcouple_dataset(field=field, split=dataset, n_samples=n_data_set, data_root=data_root)

def inspect_shapes(field: str, data, cond_list):
    """Utility for quick CLI inspection of tensor shapes."""
    try:
        rep = report_shapes(field, data, cond_list)
    except Exception as e:
        rep = f"shape inspection failed: {e}"
    print(rep)
    return rep


# ---------------------------------------------------------------------------
# Composition helpers (moved from eval/ntcouple_utils.py)
# ---------------------------------------------------------------------------

FIELD_ORDER = ("neutron", "solid", "fluid")


def _blend_fields(alpha: float, estimates: Sequence[torch.Tensor], previous: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    """Blend current and previous denoised estimates with the given relaxation factor."""
    return [alpha * est + (1.0 - alpha) * prev for est, prev in zip(estimates, previous)]


def _thermal_conductivity(temperature: torch.Tensor) -> torch.Tensor:
    """Thermal conductivity model used by the original notebook implementation."""
    return (
        17.5 * (1 - 0.223) / (1 + 0.161)
        + 1.54e-2 * (1 + 0.0061) / (1 + 0.161) * temperature
        + 9.38e-6 * torch.pow(temperature, 2)
    )


def update_neutron_condition(
    alpha: float,
    estimates: Sequence[torch.Tensor],
    previous: Sequence[torch.Tensor],
    other_condition: Sequence[torch.Tensor],
    normalize_fn=normalize,
    renormalize_fn=renormalize,
) -> torch.Tensor:
    """
    Construct the conditioning tensor for the neutron diffusion model.

    Args:
        alpha: Relaxation factor in [0, 1].
        estimates: Current denoised estimates [neutron, solid, fluid].
        previous: Estimates from the previous outer iteration (same layout as ``estimates``).
        other_condition: Sequence whose first element is the neutron boundary condition tensor.
    """
    blended = _blend_fields(alpha, estimates, previous)
    bc = other_condition[0]
    solid = blended[1]
    fluid = blended[2][:, :1]  # only coolant inlet channel is needed
    stacked = torch.cat((solid, fluid), dim=-1)
    bc = bc.repeat(1, 1, 1, 1, stacked.shape[-1])
    return torch.cat((stacked, bc), dim=1)


def update_solid_condition(
    alpha: float,
    estimates: Sequence[torch.Tensor],
    previous: Sequence[torch.Tensor],
    other_condition: Sequence[torch.Tensor],
    normalize_fn=normalize,
    renormalize_fn=renormalize,
) -> torch.Tensor:
    """
    Construct the conditioning tensor for the solid/fuel diffusion model.

    Mirrors GenCP dataset construction:
        - neutron field contributes the first eight axial channels
        - coolant inlet pressure is repeated across the width
        - left boundary fuel temperature is broadcast as the third conditioning channel
    """
    del other_condition  # only estimates are required
    blended = _blend_fields(alpha, estimates, previous)
    neu = blended[0][..., :8]
    fluid = blended[2][:, :1, :, :, :1]
    fluid_expanded = fluid.repeat(1, 1, 1, 1, neu.shape[-1])

    left_boundary = blended[1][..., :1]  # fuel temperature at left-most column (already normalized)
    left_boundary_expanded = left_boundary.repeat(1, 1, 1, 1, neu.shape[-1])

    return torch.cat((neu, fluid_expanded, left_boundary_expanded), dim=1)


def update_fluid_condition(
    alpha: float,
    estimates: Sequence[torch.Tensor],
    previous: Sequence[torch.Tensor],
    other_condition: Sequence[torch.Tensor],
    normalize_fn=normalize,
    renormalize_fn=renormalize,
) -> torch.Tensor:
    """
    Construct the conditioning tensor for the coolant diffusion model.

    The coolant model conditions on the normalized fuel boundary temperatures (two right-most columns),
    repeated to match the coolant spatial width, mimicking dataset construction.
    """
    del alpha, previous, other_condition  # only the current estimate is required

    fuel_phys = renormalize_fn(estimates[1], "solid")
    temperature_prev = fuel_phys[..., -2:-1]
    temperature_curr = fuel_phys[..., -1:]
    boundary = torch.cat((temperature_prev, temperature_curr), dim=-1)

    boundary_norm = normalize_fn(boundary, "solid")
    fluid_width = estimates[2].shape[-1]
    boundary_width = boundary_norm.shape[-1]
    if boundary_width == 0 or fluid_width % boundary_width != 0:
        raise ValueError(
            f"Fuel boundary width {boundary_width} incompatible with fluid width {fluid_width}"
        )
    repeat_factor = fluid_width // boundary_width
    return boundary_norm.repeat(1, 1, 1, 1, repeat_factor)


def default_ntcouple_updates():
    """Return update callables for (neutron, solid, fluid) diffusion models."""
    return [
        update_neutron_condition,
        update_solid_condition,
        update_fluid_condition,
    ]


# ---------------------------------------------------------------------------
# Evaluation entry point (moved from eval/ntcouple_runner.py)
# ---------------------------------------------------------------------------

def _parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NTcouple diffusion models across different regimes.")
    parser.add_argument(
        "--mode",
        choices=("decoupled", "decoupled_to_coupled", "coupled"),
        required=True,
        help="Evaluation regime to run.",
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        default=None,
        help="Dataset split for evaluation (defaults to iter2_val for decoupled, couple_val otherwise).",
    )
    parser.add_argument("--dataset-root", type=str, default=None, help="Optional NTcouple dataset root override.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples to evaluate.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for sampling.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--timesteps", type=int, default=250, help="Total diffusion timesteps.")
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=None,
        help="Optional number of steps for DDIM sampling (defaults to full timesteps).",
    )
    parser.add_argument("--dim-neutron", type=int, default=8, help="UNet latent dimension for neutron model.")
    parser.add_argument("--dim-solid", type=int, default=8, help="UNet latent dimension for solid model.")
    parser.add_argument("--dim-fluid", type=int, default=16, help="UNet latent dimension for fluid model.")
    parser.add_argument("--checkpoint-neutron", type=str, default=None, help="Path to neutron checkpoint (.pt).")
    parser.add_argument("--checkpoint-solid", type=str, default=None, help="Path to solid checkpoint (.pt).")
    parser.add_argument("--checkpoint-fluid", type=str, default=None, help="Path to fluid checkpoint (.pt).")
    parser.add_argument(
        "--outer-iters",
        type=int,
        default=2,
        help="Number of outer iterations for composed sampling (only used when compose is enabled).",
    )
    parser.add_argument(
        "--compose",
        action="store_true",
        help="If set, run coupled composition evaluation in addition to per-field sampling.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional JSON path to store metrics.")
    return parser.parse_args()


def _default_eval_split(mode: str) -> str:
    return "iter2_val" if mode == "decoupled" else "couple_val"


def _load_checkpoint(unet: Unet3D, checkpoint_path: Optional[str], device: torch.device) -> None:
    if not checkpoint_path:
        return
    state = torch.load(checkpoint_path, map_location=device)
    if "ema" in state:
        ema_state = state["ema"]
        prefix = "online_model.model."
        cleaned = {k[len(prefix) :]: v for k, v in ema_state.items() if k.startswith(prefix)}
        unet.load_state_dict(cleaned, strict=False)
    elif "model_state_dict" in state:
        unet.load_state_dict(state["model_state_dict"], strict=False)
    elif "model" in state:
        unet.load_state_dict(state["model"], strict=False)
    else:
        raise ValueError(f"Unsupported checkpoint structure at {checkpoint_path}")


def _build_diffusion(
    field: str,
    cond: torch.Tensor,
    data: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Optional[str],
) -> GaussianDiffusion:
    dim_map = {
        "neutron": args.dim_neutron,
        "solid": args.dim_solid,
        "fluid": args.dim_fluid,
    }
    unet = Unet3D(
        dim=dim_map[field],
        out_dim=data.shape[1],
        cond_channels=cond.shape[1],
        dim_mults=(1, 2, 4),
        use_sparse_linear_attn=False,
        attn_dim_head=16,
        expects_x=True,
    ).to(device)
    _load_checkpoint(unet, checkpoint_path, device)

    diffusion = GaussianDiffusion(
        unet,
        seq_length=tuple(data.shape[1:]),
        timesteps=args.timesteps,
        sampling_timesteps=args.sample_steps,
        auto_normalize=False,
    ).to(device)
    return diffusion


def _evaluate_field(
    field: str,
    diffusion: GaussianDiffusion,
    cond: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    preds: List[torch.Tensor] = []
    diffusion.eval()
    with torch.no_grad():
        for start in range(0, cond.shape[0], batch_size):
            end = min(start + batch_size, cond.shape[0])
            cond_batch = cond[start:end].to(device)
            pred_batch = diffusion.sample(cond_batch.shape[0], cond_batch).cpu()
            preds.append(pred_batch)
    pred = torch.cat(preds, dim=0)
    pred_phys = renormalize(pred, field)
    target_phys = renormalize(target.cpu(), field)
    mse = F.mse_loss(pred_phys, target_phys).item()
    rel = relative_error(target_phys, pred_phys).item()
    return {"mse": mse, "relative_error": rel}


def _evaluate_composition(
    diffusions: Dict[str, GaussianDiffusion],
    targets: Dict[str, torch.Tensor],
    conds: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    updates = default_ntcouple_updates()
    model_list = [diffusions[field] for field in FIELD_ORDER]
    shapes = [targets[field].shape for field in FIELD_ORDER]
    bc_channel = conds["neutron"][:, -1:].to(device)
    bc = bc_channel[..., :1]
    generated = compose_diffusion(
        model_list,
        shapes,
        updates,
        normalize,
        renormalize,
        other_condition=[bc],
        num_iter=args.outer_iters,
        device=device,
    )

    metrics: Dict[str, Dict[str, float]] = {}
    for field, pred_norm in zip(FIELD_ORDER, generated):
        pred_phys = renormalize(pred_norm.cpu(), field)
        target_phys = renormalize(targets[field].cpu(), field)
        metrics[field] = {
            "mse": F.mse_loss(pred_phys, target_phys).item(),
            "relative_error": relative_error(target_phys, pred_phys).item(),
        }
    return metrics


def run_evaluation_cli() -> None:
    """Entry point for command line evaluation."""
    args = _parse_eval_args()
    eval_split = args.eval_split or _default_eval_split(args.mode)
    device = torch.device(args.device)
    root = Path(args.dataset_root) if args.dataset_root else None

    conds: Dict[str, torch.Tensor] = {}
    targets: Dict[str, torch.Tensor] = {}
    for field in FIELD_ORDER:
        cond, data = load_ntcouple_dataset(field, split=eval_split, n_samples=args.limit, data_root=root)
        conds[field] = cond
        targets[field] = data

    checkpoints = {
        "neutron": args.checkpoint_neutron,
        "solid": args.checkpoint_solid,
        "fluid": args.checkpoint_fluid,
    }

    diffusions = {
        field: _build_diffusion(field, conds[field], targets[field], args, device, checkpoints[field])
        for field in FIELD_ORDER
    }

    metrics: Dict[str, Dict[str, float]] = {}
    for field in FIELD_ORDER:
        metrics[field] = _evaluate_field(field, diffusions[field], conds[field], targets[field], args.batch_size, device)

    if args.compose or args.mode in {"decoupled_to_coupled", "coupled"}:
        metrics["composed"] = _evaluate_composition(diffusions, targets, conds, args, device)

    print(json.dumps(metrics, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run_evaluation_cli()


