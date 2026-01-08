#!/usr/bin/env python3
"""
Unified visualization tool for NTcouple models across GenCP, M2PDE, and surrogate suites.

Features:
- Runs single-field inference on decouple_val and couple_val splits.
- Runs coupled inference (three-field composition) on couple_val.
- Supports GenCP (flow matching), M2PDE (diffusion/flow), and surrogate checkpoints.
- Generates per-channel comparison figures (prediction, ground truth, absolute error).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Constants and Suite Definitions
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
GENCP_ROOT = ROOT / "GenCP"
M2PDE_ROOT = ROOT / "M2PDE"

FIELD_ORDER = ["neutron", "solid", "fluid"]
CHANNEL_NAMES = {
    "neutron": ["flux"],
    "solid": ["temperature"],
    "fluid": ["temperature", "velocity_x", "velocity_y", "pressure"],
}

DECOUPLED_SPLITS = {
    "decouple_train",
    "decouple_val",
}
COUPLED_SPLITS = {"couple_train", "couple_val"}
MAX_TOTAL_SAMPLES = 10
SURROGATE_COUPLED_STEPS = 100
SURROGATE_UPDATE_COEFF = 0.1


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    runner: str  # "gencp" or "m2pde"
    paradigm: str  # "diffusion", "fm", or "surrogate"
    model_type: str  # "CNO" or "SiT_FNO"
    checkpoints: Dict[str, str]


SUITES: Dict[str, SuiteConfig] = {
    "GenCP-FNO": SuiteConfig(
        name="GenCP-FNO",
        runner="gencp",
        paradigm="fm",
        model_type="SiT_FNO",
        checkpoints={
            "neutron": "/path/to/results/ntcouple/neutron_SiT_FNO/ckpt/best_main_model.pth",
            "solid": "/path/to/results/ntcouple/solid_SiT_FNO/ckpt/best_main_model.pth",
            "fluid": "/path/to/results/ntcouple/fluid_SiT_FNO/ckpt/best_main_model.pth",
        },
    ),
    "GenCP-CNO": SuiteConfig(
        name="GenCP-CNO",
        runner="gencp",
        paradigm="fm",
        model_type="CNO",
        checkpoints={
            "neutron": "/path/to/results/ntcouple/neutron_CNO/ckpt/best_main_model.pth",
            "solid": "/path/to/results/ntcouple/solid_CNO/ckpt/best_main_model.pth",
            "fluid": "/path/to/results/ntcouple/fluid_CNO/ckpt/best_main_model.pth",
        },
    ),
    "M2PDE-FNO": SuiteConfig(
        name="M2PDE-FNO",
        runner="m2pde",
        paradigm="diffusion",
        model_type="SiT_FNO",
        checkpoints={
            "neutron": "/path/to/results/ntcouple_m2pde_neutron_cno/checkpoint/model.pt",
            "solid": "/path/to/results/ntcouple_m2pde_solid_cno/checkpoint/model.pt",
            "fluid": "/path/to/results/ntcouple_m2pde_fluid_cno/checkpoint/model.pt",
        },
    ),
    "M2PDE-CNO": SuiteConfig(
        name="M2PDE-CNO",
        runner="m2pde",
        paradigm="diffusion",
        model_type="CNO",
        checkpoints={
            "neutron": "/path/to/results/ntcouple_m2pde_neutron_cno/checkpoint/model.pt",
            "solid": "/path/to/results/ntcouple_m2pde_solid_cno/checkpoint/model.pt",
            "fluid": "/path/to/results/ntcouple_m2pde_fluid_cno/checkpoint/model.pt",
        },
    ),
    "Surrogate-CNO": SuiteConfig(
        name="Surrogate-CNO",
        runner="gencp",
        paradigm="surrogate",
        model_type="CNO",
        checkpoints={
            "neutron": "/path/to/results/ckpt_surrogate/neutron_CNO_surrogate/checkpoint/best_main_model.pth",
            "solid": "/path/to/results/ckpt_surrogate/solid_CNO_surrogate/checkpoint/best_main_model.pth",
            "fluid": "/path/to/results/ckpt_surrogate/fluid_CNO_surrogate/checkpoint/best_main_model.pth",
        },
    ),
    "Surrogate-FNO": SuiteConfig(
        name="Surrogate-FNO",
        runner="gencp",
        paradigm="surrogate",
        model_type="SiT_FNO",
        checkpoints={
            "neutron": "/path/to/results/ckpt_surrogate/neutron_SiT_FNO_surrogate/checkpoint/best_main_model.pth",
            "solid": "/path/to/results/ckpt_surrogate/solid_SiT_FNO_surrogate/checkpoint/best_main_model.pth",
            "fluid": "/path/to/results/ckpt_surrogate/fluid_SiT_FNO_surrogate/checkpoint/best_main_model.pth",
        },
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class PathContext:
    """Temporarily prepend a path to sys.path while clearing conflicting modules."""

    def __init__(self, path: Path):
        self.path = str(path)
        self.added = False
        self._removed: List[str] = []

    def __enter__(self):
        if self.path not in sys.path:
            sys.path.insert(0, self.path)
            self.added = True
        # Drop known conflicting modules so each project can re-import its own
        for name in list(sys.modules):
            if name.startswith(("data", "model", "models", "utils", "paradigms", "eval")):
                self._removed.append(name)
                del sys.modules[name]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.added and self.path in sys.path:
            sys.path.remove(self.path)
        # do not restore removed modules; subsequent contexts will re-import


def read_yaml_config(ckpt_path: Path) -> Dict:
    cfg_path = ckpt_path.parent / "config.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def move_attrs(attrs: Dict, device: torch.device) -> Dict:
    def to_device(val):
        if isinstance(val, torch.Tensor):
            return val.to(device)
        if isinstance(val, dict):
            return {k: to_device(v) for k, v in val.items()}
        return val

    return {k: to_device(v) for k, v in attrs.items()}


def _resolve_nt_root(data_root: Optional[str]) -> Path:
    with PathContext(GENCP_ROOT):
        from data.ntcouple_dataset import resolve_ntcouple_root

        return resolve_ntcouple_root(None if data_root is None else Path(data_root))


def _load_npy_slice(split_dir: Path, filename: str, indices: List[int]) -> torch.Tensor:
    path = split_dir / filename
    arr = np.load(path, mmap_mode="r")
    subset = np.take(arr, indices, axis=0)
    return torch.from_numpy(subset).float()


def _load_sparse_field(field: str, split: str, indices: List[int], data_root: Optional[str], normalizers: Dict[str, object]):
    root = _resolve_nt_root(data_root)
    split_dir = root / split
    is_decoupled = split in DECOUPLED_SPLITS

    if field == "neutron":
        if is_decoupled:
            bc = _load_npy_slice(split_dir, "bc_neu.npy", indices)
            fuel = _load_npy_slice(split_dir, "fuel_neu.npy", indices)
            fluid = _load_npy_slice(split_dir, "fluid_neu.npy", indices)
        else:
            bc = _load_npy_slice(split_dir, "bc.npy", indices)
            fuel = _load_npy_slice(split_dir, "fuel.npy", indices)
            fluid = _load_npy_slice(split_dir, "fluid.npy", indices)[:, :1]
        target = _load_npy_slice(split_dir, "neu.npy", indices)

        bc = normalizers["neutron"].normalize(bc, field="neutron")
        fuel = normalizers["solid"].normalize(fuel, field="solid")
        fluid = normalizers["fluid"].normalize(fluid, field="fluid")
        target = normalizers["neutron"].normalize(target, field="neutron")

        stacked = torch.cat((fuel, fluid), dim=-1)
        bc = bc.repeat(1, 1, 1, 1, stacked.shape[-1])
        cond = torch.cat((stacked, bc), dim=1)
        return cond, target

    if field == "solid":
        fuel_raw = _load_npy_slice(split_dir, "fuel.npy", indices)
        target = normalizers["solid"].normalize(fuel_raw.clone(), field="solid")
        left_boundary = normalizers["solid"].normalize(fuel_raw[..., :1], field="solid")

        if is_decoupled:
            neu = normalizers["neutron"].normalize(_load_npy_slice(split_dir, "neu_fuel.npy", indices), field="neutron")
            fluid = normalizers["fluid"].normalize(_load_npy_slice(split_dir, "fluid_fuel.npy", indices), field="fluid")
        else:
            neu_raw = _load_npy_slice(split_dir, "neu.npy", indices)[:, :, :, :, :8]
            neu = normalizers["neutron"].normalize(neu_raw, field="neutron")
            fluid_raw = _load_npy_slice(split_dir, "fluid.npy", indices)[:, :1, :, :, :1]
            fluid = normalizers["fluid"].normalize(fluid_raw, field="fluid")

        fluid_expanded = fluid.repeat(1, 1, 1, 1, neu.shape[-1])
        left_bc_expanded = left_boundary.repeat(1, 1, 1, 1, neu.shape[-1])
        cond = torch.cat((neu, fluid_expanded, left_bc_expanded), dim=1)
        return cond, target

    if field == "fluid":
        target = normalizers["fluid"].normalize(_load_npy_slice(split_dir, "fluid.npy", indices), field="fluid")
        if is_decoupled:
            boundary = normalizers["solid"].normalize(_load_npy_slice(split_dir, "delta_fuel_fluid.npy", indices), field="solid")
        else:
            fuel_temp = _load_npy_slice(split_dir, "fuel.npy", indices)
            prev_col = fuel_temp[:, :, :, :, -2:-1]
            last_col = fuel_temp[:, :, :, :, -1:]
            boundary_raw = torch.cat((prev_col, last_col), dim=-1)
            boundary = normalizers["solid"].normalize(boundary_raw, field="solid")
        repeat_factor = target.shape[-1] // boundary.shape[-1]
        if repeat_factor > 1:
            boundary = boundary.repeat(1, 1, 1, 1, repeat_factor)
        cond = boundary
        return cond, target

    raise ValueError(f"Unsupported field {field}")


def load_sparse_sample(field: str, split: str, idx: int, data_root: Optional[str], normalizers_cpu: Dict[str, object]):
    cond, target = _load_sparse_field(field, split, [idx], data_root, normalizers_cpu)
    cond = cond.clone()
    target = target.clone()
    grid_x = torch.arange(target.shape[-2], dtype=torch.float32)
    grid_y = torch.arange(target.shape[-1], dtype=torch.float32)
    attrs = {
        "field": field,
        "split": split,
        "idx": torch.tensor([idx], dtype=torch.long),
        "grid_x": grid_x,
        "grid_y": grid_y,
    }
    return cond, target, grid_x, grid_y, attrs

# ---------------------------------------------------------------------------
# GenCP / Surrogate Loading & Inference
# ---------------------------------------------------------------------------

def build_gencp_model(cfg: SuiteConfig, field: str, device: torch.device):
    ckpt = Path(cfg.checkpoints[field])
    if not ckpt.exists():
        return None
    model_cfg = read_yaml_config(ckpt)

    with PathContext(GENCP_ROOT):
        from model.cno import CNO3d
        from model.SiT_FNO import SiT_FNO
        from model.cno_surrogate import CNO3d as CNO3dSur
        from model.sit_fno_surrogate import SiT_FNO as SiT_FNOSur

        model_name = model_cfg.get("model_name", cfg.model_type)
        use_surrogate = cfg.paradigm == "surrogate"

        if model_name == "CNO":
            ModelCls = CNO3dSur if use_surrogate else CNO3d
            common_kwargs = dict(
                in_dim=model_cfg.get("in_dim", model_cfg.get("in_channels", 4)),
                out_dim=model_cfg.get("out_dim", model_cfg.get("out_channels", 1)),
                in_size=model_cfg.get("in_size", model_cfg.get("input_size", [64, 64])[0]),
                N_layers=model_cfg.get("depth", 4),
                channel_multiplier=model_cfg.get("channel_multiplier", 16),
            )
            if use_surrogate:
                model = ModelCls(**common_kwargs).to(device)
            else:
                model = ModelCls(
                    **common_kwargs,
                    dataset_name=model_cfg.get("dataset_name", "ntcouple"),
                    x0_is_use_noise=model_cfg.get("x0_is_use_noise", True),
                    stage=field,
                ).to(device)
        else:
            ModelCls = SiT_FNOSur if use_surrogate else SiT_FNO
            input_size = tuple(model_cfg.get("input_size", [64, 64]))
            patch_size = tuple(model_cfg.get("patch_size", [2, 2]))
            common_kwargs = dict(
                input_size=input_size,
                patch_size=patch_size,
                in_channels=model_cfg.get("in_channels", 4),
                out_channels=model_cfg.get("out_channels", 1 if field != "fluid" else 4),
                hidden_size=model_cfg.get("hidden_size", 128),
                depth=model_cfg.get("depth", 6),
                num_heads=model_cfg.get("num_heads", 4),
                modes=model_cfg.get("modes", 4),
            )
            if use_surrogate:
                model = ModelCls(**common_kwargs).to(device)
            else:
                model = ModelCls(
                    **common_kwargs,
                    stage=field,
                    dataset_name=model_cfg.get("dataset_name", "ntcouple"),
                    x0_is_use_noise=model_cfg.get("x0_is_use_noise", True),
                ).to(device)

        state = torch.load(ckpt, map_location=device)
        key = "model_state_dict" if "model_state_dict" in state else "state_dict"
        model.load_state_dict(state[key], strict=False)
        model.eval()
        meta = {
            "num_sampling_steps": model_cfg.get("num_sampling_steps", 25),
            "use_clean_left_bc_for_solid": model_cfg.get("use_clean_left_bc_for_solid", True),
            "use_noise_concat": model_cfg.get("use_noise_concat", False),
            "use_clean_cond": model_cfg.get("use_clean_cond", False),
        }
        return {"model": model, "meta": meta, "is_surrogate": use_surrogate}


def run_gencp_single(field: str, runner: Dict, cond: torch.Tensor, target: torch.Tensor, attrs: Dict, device):
    with PathContext(GENCP_ROOT):
        import torchcfm
        from infer_single_ntcouple import sample_cfm_ode_joint, sample_surrogate_ntcouple

        args = argparse.Namespace(
            field=field,
            num_sampling_steps=runner["meta"]["num_sampling_steps"],
            use_clean_left_bc_for_solid=runner["meta"].get("use_clean_left_bc_for_solid", True),
            use_noise_concat=runner["meta"].get("use_noise_concat", False),
            use_clean_cond=runner["meta"].get("use_clean_cond", False),
        )
        cond_bt = cond.permute(0, 2, 3, 4, 1).contiguous()
        target_bt = target.permute(0, 2, 3, 4, 1).contiguous()

        model = runner["model"]
        with torch.no_grad():
            if runner.get("is_surrogate"):
                preds = sample_surrogate_ntcouple(cond_bt, target_bt, model, args, attrs=attrs)
            else:
                cfm = torchcfm.ConditionalFlowMatcher()
                preds = sample_cfm_ode_joint(cond_bt, target_bt, model, cfm, args, attrs=attrs)
        return preds.permute(0, 4, 1, 2, 3).contiguous()


def run_gencp_coupled(runners: Dict[str, Dict], conds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], attrs: Dict[str, Dict], device):
    with PathContext(GENCP_ROOT):
        from infer_multi_ntcouple import compose_flow_ntcouple, default_ntcouple_updates
        from data.ntcouple_normalizer import NTcoupleNormalizer

        normalizers = {f: NTcoupleNormalizer(field=f, device=device) for f in FIELD_ORDER}

        def normalize_fn(x, field):
            return normalizers[field].normalize(x, field=field)

        def renormalize_fn(x, field):
            return normalizers[field].renormalize(x, field=field)

        model_list = [runners[f]["model"] for f in FIELD_ORDER]
        shapes = [targets[f].shape for f in FIELD_ORDER]

        cond_bt = {f: conds[f].permute(0, 2, 3, 4, 1).contiguous() for f in FIELD_ORDER}
        target_bt = {f: targets[f].permute(0, 2, 3, 4, 1).contiguous() for f in FIELD_ORDER}

        bc_batch = cond_bt["neutron"][:, :, :, :1, 1:2]
        left_bc = target_bt["solid"][:, :, :, 0:1, :]

        preds = compose_flow_ntcouple(
            model_list=model_list,
            shapes=[target_bt[f].shape for f in FIELD_ORDER],
            update_f=default_ntcouple_updates(),
            normalize_fn=normalize_fn,
            renormalize_fn=renormalize_fn,
            timestep=runners["neutron"]["meta"]["num_sampling_steps"],
            other_condition=[bc_batch, left_bc],
            device=device,
            use_bc_inpainting=True,
            gt_targets=None,
        )

        result = {}
        for field, pred in zip(FIELD_ORDER, preds):
            result[field] = pred.permute(0, 4, 1, 2, 3).contiguous()
        return result


def run_surrogate_coupled(runners: Dict[str, Dict], conds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], device: torch.device):
    """
    Run coupled inference for surrogate suites using the dedicated multi-field surrogate flow.
    """
    with PathContext(GENCP_ROOT):
        from infer_multi_ntcouple_surrogate import (
            create_ntcouple_surrogate_step_fn,
            surrogate_inference_ntcouple,
        )

        models = {field: runners[field]["model"] for field in FIELD_ORDER}

    cond_bt = conds["neutron"].permute(0, 2, 3, 4, 1).contiguous()
    bc_batch = cond_bt[:, :, :, :1, 1:2].permute(0, 4, 1, 2, 3).contiguous()
    targets_bt = {
        field: targets[field].permute(0, 2, 3, 4, 1).contiguous() for field in FIELD_ORDER
    }

    step_fn = create_ntcouple_surrogate_step_fn(
        models,
        pred_update_coeff=SURROGATE_UPDATE_COEFF,
        bc_batch=bc_batch,
    )
    with torch.no_grad():
        preds = surrogate_inference_ntcouple(
            input_norm=None,
            target_norm=targets_bt,
            num_steps=SURROGATE_COUPLED_STEPS,
            device=device,
            args=None,
            attrs=None,
            step_fn=step_fn,
        )

    return {
        field: pred.permute(0, 4, 1, 2, 3).contiguous()
        for field, pred in zip(FIELD_ORDER, preds)
    }


# ---------------------------------------------------------------------------
# M2PDE Loading & Inference
# ---------------------------------------------------------------------------

_M2PDE_REF_CACHE: Dict[Tuple[str, str, Optional[str]], Tuple[torch.Tensor, torch.Tensor]] = {}


def get_m2pde_reference(field: str, dataset: str, data_root: Optional[str], device: torch.device):
    key = (field, dataset, data_root)
    if key in _M2PDE_REF_CACHE:
        return _M2PDE_REF_CACHE[key]
    with PathContext(M2PDE_ROOT):
        from data.ntcouple import load_nt_dataset_emb

        cond, data = load_nt_dataset_emb(
            field=field,
            dataset=dataset,
            n_data_set=1,
            data_root=None if data_root is None else Path(data_root),
        )
        cond = cond.to(device)
        data = data.to(device)
        _M2PDE_REF_CACHE[key] = (cond, data)
        return cond, data


def build_m2pde_model(cfg: SuiteConfig, field: str, device, data_root: Optional[str]):
    ckpt = Path(cfg.checkpoints[field])
    if not ckpt.exists():
        return None
    config = read_yaml_config(ckpt)
    dataset_name = config.get("dataset", "decouple_val")
    model_type = config.get("model_type", cfg.model_type)
    cond, target = get_m2pde_reference(field, dataset_name, data_root, device)
    expected_in_dim = None
    if model_type == "CNO":
        state = torch.load(ckpt, map_location="cpu")
        ema = state.get("ema", {})
        weight_key = next(
            (k for k in ema if k.endswith("lift.inter_CNOBlock.convolution.weight")),
            None,
        )
        if weight_key:
            expected_in_dim = ema[weight_key].shape[1]
    cond_trimmed = cond
    target_channels = target.shape[1]
    if expected_in_dim is not None:
        desired_cond = expected_in_dim - target_channels
        if desired_cond > 0 and cond.shape[1] != desired_cond:
            cond_trimmed = cond[:, :desired_cond, ...]

    with PathContext(M2PDE_ROOT):
        from eval.infer_single_ntcouple import _build_generative_model

        args = argparse.Namespace(
            paradigm="diffusion",
            model_type=model_type,
            dim=8,
            n_layers=2,
            channel_multiplier=16,
            diffusion_step=250,
            sample_steps=None,
            fm_sampling_steps=20,
            fm_step_size=0.1,
        )
        model, paradigm = _build_generative_model(
            args=args,
            checkpoint=ckpt,
            cond=cond_trimmed,
            data=target,
            diffusion_step=250,
            sample_steps=None,
            device=device,
        )
        if hasattr(model, "model") and expected_in_dim is not None:
            desired_cond = expected_in_dim - target_channels
            if desired_cond > 0:
                model.model = _CondTrimWrapper(model.model, desired_cond)
        model.eval()
        return {
            "model": model,
            "paradigm": paradigm,
            "expected_in_dim": expected_in_dim,
            "target_channels": target_channels,
        }


def run_m2pde_single(runner: Dict, cond: torch.Tensor):
    with torch.no_grad():
        model = runner["model"]
        model_device = next(model.parameters()).device if isinstance(model, torch.nn.Module) else cond.device
        cond_adj = _align_m2pde_cond(cond, runner).to(model_device)
        return model.sample(cond_adj.shape[0], cond_adj)


def _align_m2pde_cond(cond: torch.Tensor, runner: Dict) -> torch.Tensor:
    expected = runner.get("expected_in_dim")
    target_ch = runner.get("target_channels")
    if expected is None or target_ch is None:
        return cond
    desired = expected - target_ch
    if desired <= 0 or desired == cond.shape[1]:
        return cond
    if desired < cond.shape[1]:
        return cond[:, :desired, ...]
    raise ValueError(
        f"Runner expects {desired} conditioning channels but received {cond.shape[1]}"
    )


class _CondTrimWrapper(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, desired_channels: int):
        super().__init__()
        self.base = base
        self.desired_channels = desired_channels

    def forward(self, x, t, cond, *args, **kwargs):
        if self.desired_channels > 0 and cond.shape[1] != self.desired_channels:
            cond = cond[:, : self.desired_channels, ...]
        return self.base(x, t, cond, *args, **kwargs)

    def __getattr__(self, name):
        if name in {"base", "desired_channels"}:
            return super().__getattr__(name)
        return getattr(self.base, name)


def run_m2pde_coupled(runners: Dict[str, Dict], conds: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], device):
    with PathContext(M2PDE_ROOT):
        from eval.compose import compose_diffusion, compose_flow
        from data.ntcouple import default_ntcouple_updates, normalize, renormalize

        aligned_conds = {
            field: _align_m2pde_cond(conds[field], runners[field]) for field in FIELD_ORDER
        }
        model_list = [runners[f]["model"] for f in FIELD_ORDER]
        shapes = [targets[f].shape for f in FIELD_ORDER]
        bc_channel = aligned_conds["neutron"][:, -1:]
        bc = bc_channel[..., :1].to(device)
        paradigm = runners["neutron"]["paradigm"]

        if paradigm == "diffusion":
            preds = compose_diffusion(
                model_list,
                shapes,
                default_ntcouple_updates(),
                normalize,
                renormalize,
                other_condition=[bc],
                num_iter=2,
                device=device,
            )
        else:
            preds = compose_flow(
                model_list,
                shapes,
                default_ntcouple_updates(),
                normalize,
                renormalize,
                other_condition=[bc],
                timestep=20,
                num_iter=2,
                device=device,
            )
        return {f: p for f, p in zip(FIELD_ORDER, preds)}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_comparisons(output_dir: Path, label: str, field: str, sample_idx: int, gt: np.ndarray, preds: Dict[str, np.ndarray]):
    names = CHANNEL_NAMES[field]
    t_idx = -1
    for ch_idx, ch_name in enumerate(names):
        fig, axes = plt.subplots(len(preds), 3, figsize=(8, 3 * len(preds)))
        if len(preds) == 1:
            axes = np.expand_dims(axes, 0)
        gt_img = gt[t_idx, :, :, ch_idx]
        for row, (suite, pred) in enumerate(sorted(preds.items())):
            suite_label = suite.replace("FNO", "FNO*")
            pred_img = pred[t_idx, :, :, ch_idx]
            err_img = np.abs(pred_img - gt_img)
            ax_pred, ax_gt, ax_err = axes[row]
            im0 = ax_pred.imshow(pred_img, origin="lower", cmap="viridis")
            ax_pred.set_title(f"{suite_label} Pred")
            plt.colorbar(im0, ax=ax_pred, fraction=0.046, pad=0.02)
            im1 = ax_gt.imshow(gt_img, origin="lower", cmap="viridis")
            ax_gt.set_title("Ground Truth")
            plt.colorbar(im1, ax=ax_gt, fraction=0.046, pad=0.02)
            im2 = ax_err.imshow(err_img, origin="lower", cmap="magma")
            mse = float(np.mean((pred_img - gt_img) ** 2))
            ax_err.set_title(f"Abs Error (MSE {mse:.2e})")
            plt.colorbar(im2, ax=ax_err, fraction=0.046, pad=0.02)
        plt.tight_layout()
        fname = output_dir / f"{field}_sample{sample_idx}_{label}_{ch_name}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)


def get_normalizer(device):
    with PathContext(GENCP_ROOT):
        from data.ntcouple_normalizer import NTcoupleNormalizer

        return {f: NTcoupleNormalizer(field=f, device=device) for f in FIELD_ORDER}


def tensors_to_phys(tensor: torch.Tensor, field: str, normalizer, to_numpy=True):
    renorm = normalizer[field].renormalize(tensor, field=field)
    if to_numpy:
        return renorm.detach().permute(0, 2, 3, 4, 1).cpu().numpy()
    return renorm


def main():
    parser = argparse.ArgumentParser(description="NTcouple visualization utility")
    parser.add_argument("--output-dir", default="visualization_results/ntcouple", type=str)
    parser.add_argument("--indices", nargs="+", type=int, default=[10, 11, 12])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--suites", nargs="+", default=list(SUITES.keys()), help="Subset of suite names to include")
    parser.add_argument("--data-root", type=str, default=None, help="Optional data root override for NTcoupleDataset")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_indices = sorted(set(args.indices))
    if len(unique_indices) > MAX_TOTAL_SAMPLES:
        raise ValueError(
            f"Requested {len(unique_indices)} unique samples; maximum allowed is {MAX_TOTAL_SAMPLES}."
        )

    loader_normalizers = get_normalizer(torch.device("cpu"))
    normalizers = get_normalizer(device)

    sample_cache: Dict[Tuple[str, str, int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]] = {}

    def fetch_sample(split: str, field: str, idx: int):
        key = (split, field, idx)
        if key not in sample_cache:
            sample_cache[key] = load_sparse_sample(field, split, idx, args.data_root, loader_normalizers)
        return sample_cache[key]

    splits = ["decouple_val", "couple_val"]
    sample_store: Dict[str, Dict[str, Dict[int, Tuple[torch.Tensor, torch.Tensor, Dict]]]] = {
        split: {field: {} for field in FIELD_ORDER} for split in splits
    }
    gt_phys: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {
        split: {field: {} for field in FIELD_ORDER} for split in splits
    }
    results_single: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]] = {
        split: {idx: {field: {} for field in FIELD_ORDER} for idx in unique_indices} for split in splits
    }
    results_coupled: Dict[str, Dict[int, Dict[str, Dict[str, np.ndarray]]]] = {
        split: {idx: {field: {} for field in FIELD_ORDER} for idx in unique_indices} for split in splits
    }

    for split in splits:
        for idx in unique_indices:
            for field in FIELD_ORDER:
                cond, target, _, _, attrs = fetch_sample(split, field, idx)
                sample_store[split][field][idx] = (cond.clone(), target.clone(), attrs.copy())
                gt_phys[split][field][idx] = tensors_to_phys(target, field, normalizers)[0]

    def load_suite_field_model(cfg: SuiteConfig, field: str):
        ckpt = cfg.checkpoints.get(field)
        if not ckpt:
            return None
        if cfg.runner == "gencp":
            return build_gencp_model(cfg, field, device)
        return build_m2pde_model(cfg, field, device, args.data_root)

    for suite_name in args.suites:
        if suite_name not in SUITES:
            print(f"[WARN] Suite {suite_name} not recognized, skipping.")
            continue
        cfg = SUITES[suite_name]

        for field in FIELD_ORDER:
            runner = load_suite_field_model(cfg, field)
            if not runner:
                continue
            for split in splits:
                for idx in unique_indices:
                    if idx not in sample_store[split][field]:
                        continue
                    cond_cpu, target_cpu, attrs = sample_store[split][field][idx]
                    cond = cond_cpu.to(device)
                    target = target_cpu.to(device)
                    attrs_dev = move_attrs(attrs, device)
                    if cfg.runner == "gencp":
                        preds = run_gencp_single(field, runner, cond, target, attrs_dev, device)
                    else:
                        preds = run_m2pde_single(runner, cond)
                    results_single[split][idx][field][suite_name] = tensors_to_phys(preds, field, normalizers)[0]
            runner["model"].to("cpu")
            torch.cuda.empty_cache()

        if "couple_val" in splits and cfg.runner in {"gencp", "m2pde"}:
            if cfg.paradigm == "surrogate":
                runners = {}
                for field in FIELD_ORDER:
                    runner = load_suite_field_model(cfg, field)
                    if runner:
                        runners[field] = runner
                if len(runners) == len(FIELD_ORDER):
                    split = "couple_val"
                    for idx in unique_indices:
                        conds = {}
                        targets = {}
                        for field in FIELD_ORDER:
                            cond_cpu, target_cpu, _ = sample_store[split][field][idx]
                            conds[field] = cond_cpu.to(device)
                            targets[field] = target_cpu.to(device)
                        preds = run_surrogate_coupled(runners, conds, targets, device)
                        for field in FIELD_ORDER:
                            results_coupled[split][idx][field][suite_name] = tensors_to_phys(preds[field], field, normalizers)[0]
                for runner in runners.values():
                    runner["model"].to("cpu")
                torch.cuda.empty_cache()
            elif cfg.paradigm != "surrogate":
                runners = {}
                for field in FIELD_ORDER:
                    runner = load_suite_field_model(cfg, field)
                    if runner:
                        runners[field] = runner
                if len(runners) == len(FIELD_ORDER):
                    split = "couple_val"
                    for idx in unique_indices:
                        conds = {}
                        targets = {}
                        attrs_map = {}
                        for field in FIELD_ORDER:
                            cond_cpu, target_cpu, attrs = sample_store[split][field][idx]
                            conds[field] = cond_cpu.to(device)
                            targets[field] = target_cpu.to(device)
                            attrs_map[field] = move_attrs(attrs, device)
                        if cfg.runner == "gencp":
                            preds = run_gencp_coupled(runners, conds, targets, attrs_map, device)
                        else:
                            preds = run_m2pde_coupled(runners, conds, targets, device)
                        for field in FIELD_ORDER:
                            results_coupled[split][idx][field][suite_name] = tensors_to_phys(preds[field], field, normalizers)[0]
                for runner in runners.values():
                    runner["model"].to("cpu")
                torch.cuda.empty_cache()

    for split in splits:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx in unique_indices:
            for field in FIELD_ORDER:
                pred_single = results_single[split][idx][field]
                if pred_single:
                    plot_comparisons(split_dir, "single", field, idx, gt_phys[split][field][idx], pred_single)
                pred_coupled = results_coupled[split][idx][field]
                if pred_coupled:
                    plot_comparisons(split_dir, "coupled", field, idx, gt_phys[split][field][idx], pred_coupled)

    print(f"Visualization complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

