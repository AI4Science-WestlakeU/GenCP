"""
NTcouple dataset loader for GenCP framework.

This module provides PyTorch Dataset wrapper for nuclear-thermal coupling data,
adapted from M2PDE implementation to work with GenCP's training pipeline.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Literal

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

from .ntcouple_normalizer import NTcoupleNormalizer, compute_heat_flux_from_spatial_gradient

FieldType = Literal["neutron", "solid", "fluid"]
SplitType = Literal["iter1_train", "iter1_val", "iter2_train", "iter2_val", "iter3_train", "iter3_val", "iter4_train", "iter4_val", "couple_train", "couple_val"]


def resolve_ntcouple_root(data_root: Optional[Path] = None) -> Path:
    """
    Resolve NTcouple dataset root with fallback hierarchy.
    
    Priority:
        1. Explicit data_root argument
        2. Environment variable NTCOUPLE_DATA_ROOT
        3. Common dataset locations
    
    Args:
        data_root: Optional explicit dataset root path.
    
    Returns:
        Resolved Path object pointing to dataset root.
    
    Raises:
        FileNotFoundError: If dataset cannot be found in any standard location.
    """
    # Priority 1: Explicit argument
    if data_root is not None:
        path = Path(data_root).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"Specified data_root does not exist: {data_root}")
    
    # Priority 2: Environment variable
    env_root = os.getenv('NTCOUPLE_DATA_ROOT')
    if env_root:
        path = Path(env_root).expanduser()
        if path.exists():
            return path
    
    # Priority 3: Common locations
    common_paths = [
        Path('/path/to/gencp_dataset/NTcouple/'),
        Path.home() / 'data/ntcouple/',
        Path(__file__).parent.parent.parent / 'data/ntcouple/',
    ]
    
    for path in common_paths:
        if path.exists():
            return path
    
    # Not found - provide helpful error message
    raise FileNotFoundError(
        "NTcouple dataset not found. Please either:\n"
        "  1. Set environment variable: export NTCOUPLE_DATA_ROOT=/path/to/ntcouple/\n"
        "  2. Place data in one of:\n" +
        "\n".join(f"     - {p}" for p in common_paths) +
        f"\n  3. Pass data_root argument explicitly\n\n"
        f"Searched locations: {', '.join(str(p) for p in common_paths)}"
    )


def _ensure_5d(x: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """
    Ensure tensor has 5D shape (batch, channels, frames, height, width).
    
    Args:
        x: Input tensor.
        name: Tensor name for error messages.
    
    Returns:
        5D tensor with shape (B, C, T, H, W).
    
    Raises:
        ValueError: If tensor cannot be reshaped to 5D.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(x)}")
    
    if x.dim() == 5:
        return x
    
    # Attempt to reshape by prepending singleton dimensions
    while x.dim() < 5:
        x = x.unsqueeze(0)
    
    if x.dim() > 5:
        raise ValueError(f"{name} has {x.dim()} dimensions, cannot safely convert to 5D")
    
    return x


def _match_spatiotemporal(
    x: torch.Tensor, 
    target_shape: Tuple[int, int, int], 
    mode: str = "nearest"
) -> torch.Tensor:
    """
    Resize tensor to match target (frames, height, width) shape.
    
    Args:
        x: Input tensor with shape (B, C, T, H, W).
        target_shape: Target (T, H, W) dimensions.
        mode: Interpolation mode ('nearest', 'trilinear').
    
    Returns:
        Resized tensor with target spatial-temporal dimensions.
    """
    if x.shape[2:] == target_shape:
        return x
    
    # Use nearest interpolation to avoid introducing non-physical values
    return F.interpolate(x, size=target_shape, mode=mode)


def _load_npy_tensor(file_path: Path, limit: Optional[int] = None) -> torch.Tensor:
    """Load numpy array from file and convert to PyTorch tensor."""
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    array = np.load(str(file_path))
    if limit is not None and limit > 0:
        array = array[:limit]
    return torch.from_numpy(array).float()


class NTcoupleDataset(Dataset):
    """
    PyTorch Dataset for NTcouple nuclear-thermal coupling data.
    
    Loads conditioning and target data for one of three physics fields (neutron,
    solid fuel, fluid coolant) from specified split (decoupled iter1/iter2 or
    coupled couple/couple_val).
    
    The dataset pre-loads all data into memory for fast access during training.
    Data is automatically normalized to [-1, 1] range using field-specific bounds.
    
    Usage:
        >>> dataset = NTcoupleDataset(field='neutron', split='iter1')
        >>> cond, target = dataset[0]
        >>> print(cond.shape, target.shape)  # (3, 10, 10, 8), (1, 10, 10, 8)
    
    Args:
        field: Physics field - 'neutron', 'solid', or 'fluid'.
        split: Dataset split - 'iter1'/'iter2' (decoupled) or 'couple'/'couple_val' (coupled).
        n_samples: Optional limit on number of samples (for quick experiments).
        data_root: Optional dataset root path (defaults to auto-detection).
        normalize: If True, apply normalization to [-1, 1] (default True).
    """
    
    _DECOUPLED_SPLITS = {"iter1_train", "iter1_val", "iter2_train", "iter2_val", "iter3_train", "iter3_val", "iter4_train", "iter4_val"}
    _COUPLED_SPLITS = {"couple_val", "couple_train"}
    
    def __init__(
        self,
        field: FieldType = "neutron",
        split: SplitType = "iter1",
        n_samples: Optional[int] = None,
        data_root: Optional[Path] = None,
        normalize: bool = True
    ):
        """
        Initialize NTcouple dataset.
        
        Args:
            field: Physical field to load - one of {'neutron', 'solid', 'fluid'}.
            split: Dataset split - one of {'iter1', 'iter2'} for decoupled training,
                   or {'couple', 'couple_val'} for coupled evaluation.
            n_samples: Optional limit on number of samples to load.
            data_root: Optional explicit path to dataset root.
            normalize: If True, normalize data to [-1, 1] (required for proper concatenation).
        
        Note:
            Unlike DoubleCylinder, NTcouple MUST normalize inside dataset because different
            physical fields need to be normalized separately before concatenation.
        """
        super().__init__()
        
        self.field = field.lower()
        self.split = split.lower()
        self.n_samples = n_samples
        self.normalize_data = normalize
        
        # Validate inputs
        if self.field not in ["neutron", "solid", "fluid"]:
            raise ValueError(f"Unsupported field '{field}'. Choose from: neutron, solid, fluid")
        
        if self.split not in self._DECOUPLED_SPLITS | self._COUPLED_SPLITS:
            raise ValueError(
                f"Unsupported split '{split}'. Choose from: "
                f"{', '.join(self._DECOUPLED_SPLITS | self._COUPLED_SPLITS)}"
            )
        
        # Resolve dataset root
        self.data_root = resolve_ntcouple_root(data_root)
        self.split_dir = self.data_root / self.split
        
        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.split_dir}\n"
            )
        
        # Load data (will be cached in memory)
        self.cond, self.target = self._load_field_data()
        
        # Validate shapes
        if self.cond.shape[0] != self.target.shape[0]:
            raise ValueError(
                f"Conditioning and target batch sizes mismatch: "
                f"{self.cond.shape[0]} vs {self.target.shape[0]}"
            )
    
    def _load_field_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load conditioning and target tensors for the specified field and split.
        
        Returns:
            Tuple of (conditioning, target) tensors, normalized if self.normalize_data=True.
        
        Note:
            Unlike DoubleCylinder, normalization MUST happen here because different physical
            fields need separate normalization before concatenation.
        """
        is_decoupled = self.split in self._DECOUPLED_SPLITS
        normalizer = NTcoupleNormalizer()
        
        if self.field == "neutron":
            # Neutron field: joint evolution [fuel_temp, fluid_temp, bc, neutron]
            # Physics: predict neutron field using solid temp + fluid temp + boundary conditions
            if is_decoupled:
                bc = _load_npy_tensor(self.split_dir / "bc_neu.npy", self.n_samples)
                fuel = _load_npy_tensor(self.split_dir / "fuel_neu.npy", self.n_samples)
                fluid = _load_npy_tensor(self.split_dir / "fluid_neu.npy", self.n_samples)
            else:
                bc = _load_npy_tensor(self.split_dir / "bc.npy", self.n_samples)
                fuel = _load_npy_tensor(self.split_dir / "fuel.npy", self.n_samples)
                fluid_raw = _load_npy_tensor(self.split_dir / "fluid.npy", self.n_samples)
                fluid = fluid_raw[:, :1]  # Only inlet temperature channel
            
            target = _load_npy_tensor(self.split_dir / "neu.npy", self.n_samples)
            
            # Normalize BEFORE concatenation (critical for NTcouple!)
            if self.normalize_data:
                bc = normalizer.normalize(bc, field="neutron")
                fuel = normalizer.normalize(fuel, field="solid")
                fluid = normalizer.normalize(fluid, field="fluid")
                target = normalizer.normalize(target, field="neutron")
            
            # Core concatenation logic (following GenCP joint evolution paradigm):
            # 1. Spatial concatenation: fuel(W=8) + fluid(W=12) → spatial_cond(W=20)
            spatial_cond = torch.cat((fuel, fluid), dim=-1)
            
            # 2. Expand boundary condition: bc(W=1) → bc_expanded(W=20)
            bc_expanded = bc.repeat(1, 1, 1, 1, spatial_cond.shape[-1])
            
            # 3. Channel concatenation: form conditioning (B, C=2, T, H, W=20)
            #    - channel 0: spatial_cond (fuel+fluid temperature field)
            #    - channel 1: bc_expanded (boundary condition)
            cond = torch.cat((spatial_cond, bc_expanded), dim=1)
            
            return cond, target
        
        elif self.field == "solid":
            # Solid field: joint evolution [neutron_flux, fluid_pressure, left_boundary_temp, fuel_temp]
            # Physics: predict solid temperature using neutron flux + fluid pressure + left boundary temp
            if is_decoupled:
                neu = _load_npy_tensor(self.split_dir / "neu_fuel.npy", self.n_samples)
                fluid = _load_npy_tensor(self.split_dir / "fluid_fuel.npy", self.n_samples)
            else:
                neu_raw = _load_npy_tensor(self.split_dir / "neu.npy", self.n_samples)
                neu = neu_raw[:, :, :, :, :8]  # Only first 8 axial channels
                fluid_raw = _load_npy_tensor(self.split_dir / "fluid.npy", self.n_samples)
                fluid = fluid_raw[:, :1, :, :, :1]  # Only pressure at inlet
            
            target = _load_npy_tensor(self.split_dir / "fuel.npy", self.n_samples)
            
            # Extract leftmost column as boundary condition (before normalization)
            left_boundary_temp = target[:, :, :, :, 0:1]
            
            # Normalize BEFORE concatenation
            if self.normalize_data:
                neu = normalizer.normalize(neu, field="neutron")
                fluid = normalizer.normalize(fluid, field="fluid")
                target = normalizer.normalize(target, field="solid")
                left_boundary_temp = normalizer.normalize(left_boundary_temp, field="solid")
            
            # Core concatenation logic:
            # 1. Expand fluid pressure to fuel width
            fluid_expanded = fluid.repeat(1, 1, 1, 1, neu.shape[-1])
            
            # 2. Expand left boundary temperature to fuel width
            left_boundary_expanded = left_boundary_temp.repeat(1, 1, 1, 1, neu.shape[-1])
            
            # 3. Channel concatenation: form conditioning (B, C=3, T, H, W=8)
            #    - channel 0: neu (neutron flux)
            #    - channel 1: fluid_expanded (fluid pressure)
            #    - channel 2: left_boundary_expanded (left boundary temperature)
            cond = torch.cat((neu, fluid_expanded, left_boundary_expanded), dim=1)
            
            return cond, target
        
        elif self.field == "fluid":
            # Fluid field: joint evolution [boundary_temperature, fluid_field]
            # Physics: predict fluid field (temperature+velocity+pressure) using fuel rod boundary temperature
            target = _load_npy_tensor(self.split_dir / "fluid.npy", self.n_samples)
            
            if is_decoupled:
                # iter1: directly load precomputed boundary condition
                fuel_boundary = _load_npy_tensor(self.split_dir / "delta_fuel_fluid.npy", self.n_samples)
            else:
                # couple: extract boundary condition from solid temperature
                # Key: match inference logic - normalize entire solid temp field first, then extract boundary
                fuel_temp = _load_npy_tensor(self.split_dir / "fuel.npy", self.n_samples)

                # Extract rightmost two columns from normalized temperature (match inference logic)
                temperature_second_last = fuel_temp[:, :, :, :, -2:-1]
                temperature_last = fuel_temp[:, :, :, :, -1:]
                
                # Concatenate along spatial width dimension to form boundary condition with width=2
                fuel_boundary = torch.cat((temperature_second_last, temperature_last), dim=-1)
            
            # Normalize
            if self.normalize_data:
                fuel_boundary = normalizer.normalize(fuel_boundary, field="solid")
                target = normalizer.normalize(target, field="fluid")
            
            # Core concatenation logic:
            fuel_boundary = fuel_boundary.repeat(1, 1, 1, 1, target.shape[-1] // 2)
            
            # Conditioning is the boundary temperature field (B, 1, T, H, W)
            cond = fuel_boundary
            
            return cond, target
              
        else:
            raise ValueError(f"Unsupported field: {self.field}")
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.target.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index.
        
        Returns:
            Tuple of (conditioning, target, grid_x, grid_y, attrs) for compatibility with GenCP training loop.
            - conditioning: (C, T, H, W) conditioning tensor
            - target: (C, T, H, W) target tensor
            - grid_x: (H,) dummy grid (not used for NTcouple)
            - grid_y: (W,) dummy grid (not used for NTcouple)
            - attrs: dict of metadata
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        cond = self.cond[idx]
        target = self.target[idx]
        
        # Create dummy grids for compatibility (NTcouple doesn't use explicit grids)
        _, _, H, W = target.shape  # (C, T, H, W)
        grid_x = torch.arange(H, dtype=torch.float32)
        grid_y = torch.arange(W, dtype=torch.float32)
        
        # Create metadata dict
        attrs = {
            'field': self.field,
            'split': self.split,
            'idx': torch.tensor([idx], dtype=torch.long)
        }
        
        return cond, target, grid_x, grid_y, attrs
    
    def get_batch(self, indices: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get multiple samples as a batch.
        
        Args:
            indices: List of sample indices.
        
        Returns:
            Tuple of (conditioning, target) batches with shape (B, C, T, H, W).
        """
        cond_batch = self.cond[indices]
        target_batch = self.target[indices]
        return cond_batch, target_batch
    
    @property
    def is_decoupled(self) -> bool:
        """Check if dataset is from decoupled split."""
        return self.split in self._DECOUPLED_SPLITS
    
    @property
    def is_coupled(self) -> bool:
        """Check if dataset is from coupled split."""
        return self.split in self._COUPLED_SPLITS
    
    @property
    def cond_shape(self) -> torch.Size:
        """Shape of conditioning tensor (without batch dimension)."""
        return self.cond.shape[1:]
    
    @property
    def target_shape(self) -> torch.Size:
        """Shape of target tensor (without batch dimension)."""
        return self.target.shape[1:]
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary containing dataset metadata and statistics.
        """
        return {
            'field': self.field,
            'split': self.split,
            'n_samples': len(self),
            'cond_shape': tuple(self.cond_shape),
            'target_shape': tuple(self.target_shape),
            'data_root': str(self.data_root),
            'is_decoupled': self.is_decoupled,
            'is_coupled': self.is_coupled,
            'normalized': self.normalize_data,
            'cond_channels': self.cond.shape[1],
            'target_channels': self.target.shape[1],
        }
    
    def __repr__(self) -> str:
        """String representation of dataset."""
        stats = self.get_stats()
        return (
            f"NTcoupleDataset(\n"
            f"  field='{stats['field']}',\n"
            f"  split='{stats['split']}',\n"
            f"  n_samples={stats['n_samples']},\n"
            f"  cond_shape={stats['cond_shape']},\n"
            f"  target_shape={stats['target_shape']},\n"
            f"  normalized={stats['normalized']}\n"
            f")"
        )


if __name__ == '__main__':
    """Test script for dataset loading."""
    import sys
    
    print("=" * 60)
    print("NTcouple Dataset Test")
    print("=" * 60)
    
    # Test all fields and splits
    for field in ['neutron', 'solid', 'fluid']:
        for split in ['iter1', 'iter2']:
            try:
                print(f"\n[Testing {field} - {split}]")
                dataset = NTcoupleDataset(
                    field=field,
                    split=split,
                    n_samples=5  # Small sample for testing
                )
                
                print(f"✓ Dataset loaded successfully")
                print(f"  Samples: {len(dataset)}")
                print(f"  Cond shape: {dataset.cond_shape}")
                print(f"  Target shape: {dataset.target_shape}")
                
                # Test __getitem__
                cond, target = dataset[0]
                print(f"  Sample[0] cond: {cond.shape}, target: {target.shape}")
                
                # Check normalization range
                print(f"  Cond range: [{cond.min():.3f}, {cond.max():.3f}]")
                print(f"  Target range: [{target.min():.3f}, {target.max():.3f}]")
                
            except Exception as e:
                print(f"✗ Failed: {e}")
                if '--verbose' in sys.argv:
                    import traceback
                    traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

