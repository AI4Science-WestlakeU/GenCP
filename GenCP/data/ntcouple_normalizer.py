"""
NTcouple dataset normalization utilities.

This module provides fixed-bound normalization for nuclear-thermal coupling data,
ported from M2PDE framework to maintain consistency with baseline experiments.
"""

import torch
from typing import Literal

FieldType = Literal["solid", "fluid", "neutron", "flux"]


class NTcoupleNormalizer:
    """
    Fixed-bound normalizer for NTcouple physical fields.
    
    Normalizes values to [-1, 1] range using pre-defined physical bounds
    for each field type. This ensures consistency with M2PDE baseline.
    
    Supported fields:
        - solid: Fuel rod temperature [100, 1500] K
        - fluid: Coolant [temperature, velocity_x, velocity_y, pressure]
        - neutron: Neutron flux (log-transformed) [0, 3.258]
        - flux: Heat flux [-2000, 4000] W/m²
    
    Usage in training loop (similar to RangeNormalizer for DoubleCylinder):
        normalizer = NTcoupleNormalizer(field='neutron')
        input_norm, target_norm = normalizer.preprocess(input, target)
    """
    
    # Physical bounds for each field type
    BOUNDS = {
        'solid': [100.0, 1500.0],
        'flux': [-2000.0, 4000.0],
        'fluid': [
            [100.0, 1200.0],   # channel 0: temperature (K)
            [-50.0, 250.0],    # channel 1: velocity_x (m/s)
            [-0.006, 0.006],   # channel 2: velocity_y (m/s)
            [0.0, 0.6]         # channel 3: pressure (MPa)
        ],
        'neutron': [0.0, 3.258]  # log-transformed neutron flux
    }
    
    def __init__(self, field: FieldType = 'neutron', device: str = 'cuda'):
        """
        Initialize normalizer for specific field.
        
        Args:
            field: Physical field type - one of {'solid', 'fluid', 'neutron', 'flux'}.
            device: Device for tensor operations ('cuda' or 'cpu').
        """
        if field not in self.BOUNDS:
            raise ValueError(f"Unsupported field '{field}'. Choose from {list(self.BOUNDS.keys())}")
        self.field = field
        self.device = device
    
    @staticmethod
    def normalize(x: torch.Tensor, field: FieldType) -> torch.Tensor:
        """
        Normalize physical values to [-1, 1] range.
        
        Args:
            x: Input tensor with shape (batch, channels, frames, height, width).
               Expected to be in physical units.
            field: Field type - one of {'solid', 'fluid', 'neutron', 'flux'}.
        
        Returns:
            Normalized tensor in [-1, 1] range with same shape as input.
        
        Note:
            - Neutron field applies log(x+1) transformation before normalization
            - Fluid field normalizes each channel independently
            - Creates a copy of input to avoid in-place modification
        """
        if field not in NTcoupleNormalizer.BOUNDS:
            raise ValueError(f"Unsupported field '{field}'. Choose from {list(NTcoupleNormalizer.BOUNDS.keys())}")
        
        x = x.clone().to(dtype=torch.float32)
        
        if field == "solid":
            lower, upper = NTcoupleNormalizer.BOUNDS['solid']
            return (x - lower) / (upper - lower) * 2.0 - 1.0
        
        if field == "fluid":
            bounds = NTcoupleNormalizer.BOUNDS['fluid']
            if x.shape[1] > len(bounds):
                raise ValueError(
                    f"Fluid tensor has {x.shape[1]} channels, expected ≤ {len(bounds)}. "
                    f"Expected channels: [temperature, velocity_x, velocity_y, pressure]"
                )
            # Normalize each channel independently
            for chan in range(x.shape[1]):
                lower, upper = bounds[chan]
                x[:, chan] = (x[:, chan] - lower) / (upper - lower) * 2.0 - 1.0
            return x
        
        if field == "neutron":
            # Log-transform before normalization to handle large dynamic range
            lower, upper = NTcoupleNormalizer.BOUNDS['neutron']
            x_log = torch.log(x + 1.0)
            return (x_log - lower) / (upper - lower) * 2.0 - 1.0
        
        if field == "flux":
            lower, upper = NTcoupleNormalizer.BOUNDS['flux']
            return (x - lower) / (upper - lower) * 2.0 - 1.0
        
        raise ValueError(f"Unsupported field '{field}'.")
    
    @staticmethod
    def renormalize(x: torch.Tensor, field: FieldType) -> torch.Tensor:
        """
        Convert normalized [-1, 1] values back to physical units.
        
        Args:
            x: Normalized tensor with shape (batch, channels, frames, height, width).
               Expected to be in [-1, 1] range.
            field: Field type - one of {'solid', 'fluid', 'neutron', 'flux'}.
        
        Returns:
            Denormalized tensor in physical units with same shape as input.
        
        Note:
            - Neutron field applies exp(x)-1 after denormalization to invert log transform
            - Fluid field denormalizes each channel independently
            - Creates a copy of input to avoid in-place modification
        """
        if field not in NTcoupleNormalizer.BOUNDS:
            raise ValueError(f"Unsupported field '{field}'. Choose from {list(NTcoupleNormalizer.BOUNDS.keys())}")
        
        x = x.clone().to(dtype=torch.float32)
        
        if field == "solid":
            lower, upper = NTcoupleNormalizer.BOUNDS['solid']
            return (x + 1.0) * 0.5 * (upper - lower) + lower
        
        if field == "fluid":
            bounds = NTcoupleNormalizer.BOUNDS['fluid']
            if x.shape[1] > len(bounds):
                raise ValueError(
                    f"Fluid tensor has {x.shape[1]} channels, expected ≤ {len(bounds)}. "
                    f"Expected channels: [temperature, velocity_x, velocity_y, pressure]"
                )
            # Denormalize each channel independently
            for chan in range(x.shape[1]):
                lower, upper = bounds[chan]
                x[:, chan] = (x[:, chan] + 1.0) * 0.5 * (upper - lower) + lower
            return x
        
        if field == "neutron":
            # Invert log-transform: exp(denormalized) - 1
            lower, upper = NTcoupleNormalizer.BOUNDS['neutron']
            x_denorm = (x + 1.0) * 0.5 * (upper - lower) + lower
            return torch.exp(x_denorm) - 1.0
        
        if field == "flux":
            lower, upper = NTcoupleNormalizer.BOUNDS['flux']
            return (x + 1.0) * 0.5 * (upper - lower) + lower
        
        raise ValueError(f"Unsupported field '{field}'.")
    
    @staticmethod
    def get_bounds(field: FieldType):
        """
        Get physical bounds for a given field.
        
        Args:
            field: Field type.
        
        Returns:
            List of bounds [min, max] or list of per-channel bounds for fluid.
        """
        if field not in NTcoupleNormalizer.BOUNDS:
            raise ValueError(f"Unsupported field '{field}'.")
        return NTcoupleNormalizer.BOUNDS[field]
    
    def preprocess(self, x: torch.Tensor, y: torch.Tensor):
        """
        Preprocess input and target tensors for training (normalize to [-1, 1]).
        
        **Note**: For NTcouple, data is already normalized in the dataset.
        This method is mainly for compatibility but will normalize if called.
        
        Args:
            x: Input conditioning tensor (may be raw or already normalized).
            y: Target tensor (may be raw or already normalized).
        
        Returns:
            Tuple of (normalized_input, normalized_target).
        """
        # Move to device if needed
        if self.device:
            x = x.to(self.device)
            y = y.to(self.device)
        
        # Normalize using field-specific logic
        x_norm = self.normalize(x, self.field)
        y_norm = self.normalize(y, self.field)
        
        return x_norm, y_norm
    
    def postprocess(self, x: torch.Tensor, y: torch.Tensor):
        """
        Postprocess normalized tensors back to physical units.
        
        Inverse of preprocess() for evaluation/visualization.
        
        Args:
            x: Normalized conditioning tensor in [-1, 1].
            y: Normalized target tensor in [-1, 1].
        
        Returns:
            Tuple of (denormalized_input, denormalized_target) in physical units.
        """
        # Move to device if needed
        if self.device:
            x = x.to(self.device)
            y = y.to(self.device)
        
        x_denorm = self.renormalize(x, self.field)
        y_denorm = self.renormalize(y, self.field)
        
        return x_denorm, y_denorm


def thermal_conductivity(temperature: torch.Tensor) -> torch.Tensor:
    """
    Compute thermal conductivity for nuclear fuel as a function of temperature.
    
    Uses empirical correlation from nuclear engineering literature:
    k(T) = k_0 * (1 - ε) / (1 + β) + k_1 * (1 + δ) / (1 + β) * T + k_2 * T²
    
    where:
        k_0 = 17.5, k_1 = 1.54e-2, k_2 = 9.38e-6
        ε = 0.223 (porosity correction)
        δ = 0.0061 (temperature coefficient)
        β = 0.161 (crack correction)
    
    Args:
        temperature: Fuel temperature tensor in Kelvin, any shape.
    
    Returns:
        Thermal conductivity in W/(m·K) with same shape as input.
    
    Note:
        This correlation is used in fluid condition update to compute heat flux
        from fuel to coolant: q = k(T) * (T_prev - T_current).
    """
    k0_term = 17.5 * (1 - 0.223) / (1 + 0.161)
    k1_term = 1.54e-2 * (1 + 0.0061) / (1 + 0.161)
    k2_term = 9.38e-6
    
    return k0_term + k1_term * temperature + k2_term * torch.pow(temperature, 2)


def compute_heat_flux_from_spatial_gradient(temperature_prev: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    """
    Compute heat flux from spatial temperature gradient.
    
    This function computes heat flux at the fuel-fluid interface using:
        flux = (T_prev - T_current) * thermal_conductivity(T_current)
    
    where T_prev is the temperature at the previous spatial point (more interior),
    and T_current is the temperature at the current spatial point (at interface).
    
    Args:
        temperature_prev: Temperature at previous spatial point (B, C, T, H, W) in Kelvin.
        temperature: Temperature at current spatial point (B, C, T, H, W) in Kelvin.
    
    Returns:
        Heat flux tensor in W/m², shape (B, C, T, H, W).
    
    Example:
        >>> temp_prev = torch.randn(4, 1, 16, 64, 1) * 100 + 700  # ~[600, 800] K
        >>> temp_curr = torch.randn(4, 1, 16, 64, 1) * 100 + 700
        >>> flux = compute_heat_flux_from_spatial_gradient(temp_prev, temp_curr)
        >>> flux.shape
        torch.Size([4, 1, 16, 64, 1])
    """
    if temperature_prev.shape != temperature.shape:
        raise ValueError(f"Shape mismatch: prev {temperature_prev.shape} vs curr {temperature.shape}")
    
    # Compute thermal conductivity at current temperature
    k = thermal_conductivity(temperature)  # (B, C, T, H, W)
    
    # Compute heat flux
    flux = (temperature_prev - temperature) * k
    
    return flux



# Legacy aliases for backward compatibility with M2PDE interface
normalize = NTcoupleNormalizer.normalize
renormalize = NTcoupleNormalizer.renormalize