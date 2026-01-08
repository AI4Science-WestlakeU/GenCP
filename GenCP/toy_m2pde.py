import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, beta, gamma, entropy
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from torch.utils.data import DataLoader, TensorDataset
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
try:
    from matplotlib.animation import FuncAnimation, PillowWriter
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("⚠️ Pillow not available, GIF generation will be skipped")

import os
import logging
from datetime import datetime

# Set default font for better compatibility
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# Set random seed for reproducibility
torch.manual_seed(42)

# Set matplotlib style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define beautiful color schemes
COLORS = {
    'primary': '#2E86AB',      # Deep blue
    'secondary': '#A23B72',    # Deep pink
    'accent': '#F18F01',       # Orange
    'success': '#C73E1D',      # Red
    'info': '#7209B7',         # Purple
    'warning': '#F77F00',      # Orange-red
    'light': '#F8F9FA',        # Light gray
    'dark': '#212529'          # Dark gray
}

# Create custom colormaps
def create_beautiful_colormap():
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    return LinearSegmentedColormap.from_list('beautiful', colors)

def setup_logging(experiment_name="m2pde_experiment", log_dir="/path/to/plot_toy"):
    """
    Setup logging system
    
    Args:
        experiment_name: Experiment name
        log_dir: Log save directory
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/similarity_metrics_{experiment_name}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized for experiment: {experiment_name}")
    logger.info(f"Log file: {log_filename}")
    
    return logger, log_filename


class AnalyticalJointDistribution:
    """
    Construct an analytical joint distribution p(x,y), ensuring x and y are not independent
    Uses Bivariate Normal distribution + nonlinear transformations to construct complex dependencies
    Supports multiple distribution patterns from simple to complex
    """
    def __init__(self, mu_x=0.0, sigma_x=1.0, mu_y=0.0, sigma_y=1.0, rho=0.7, 
                 complexity_level='simple', n_components=3):
        """
        Construct analytical joint distribution
        Args:
            mu_x, mu_y: Means of marginal distributions
            sigma_x, sigma_y: Standard deviations of marginal distributions
            rho: Correlation coefficient, controls dependency strength between x and y
            complexity_level: 'simple', 'medium', 'complex', 'extreme'
            n_components: Number of mixture components (only used in complex and extreme modes)
        """
        self.mu_x = mu_x
        self.sigma_x = sigma_x
        self.mu_y = mu_y
        self.sigma_y = sigma_y
        self.rho = rho
        self.complexity_level = complexity_level
        self.n_components = n_components
        
        if complexity_level == 'simple':
            self._setup_simple_distribution()
        else:
            self._setup_complex_distribution()
        
        print(f"Constructing analytical joint distribution:")
        print(f"  Complexity level: {complexity_level}")
        print(f"  Mean vector: μ = [{mu_x:.2f}, {mu_y:.2f}]")
        print(f"  Base parameters: σ_x={sigma_x:.2f}, σ_y={sigma_y:.2f}, ρ={rho:.2f}")
        if complexity_level != 'simple':
            print(f"  Number of components: {n_components}")
    
    def _setup_simple_distribution(self):
        """Setup simple bivariate normal distribution"""
        self.sigma_xy = self.rho * self.sigma_x * self.sigma_y
        self.cov_matrix = torch.tensor([[self.sigma_x**2, self.sigma_xy],
                                       [self.sigma_xy, self.sigma_y**2]], dtype=torch.float32)
        self.mean_vector = torch.tensor([self.mu_x, self.mu_y], dtype=torch.float32)
        
        self.det_cov = torch.det(self.cov_matrix)
        self.inv_cov = torch.inverse(self.cov_matrix)
        
        self.components = None
    
    def _setup_complex_distribution(self):
        """Setup complex mixture distribution"""
        self.components = self._create_mixture_components()
        self.sigma_xy = self.rho * self.sigma_x * self.sigma_y
        self.cov_matrix = torch.tensor([[self.sigma_x**2, self.sigma_xy],
                                       [self.sigma_xy, self.sigma_y**2]], dtype=torch.float32)
        self.mean_vector = torch.tensor([self.mu_x, self.mu_y], dtype=torch.float32)
        self.det_cov = torch.det(self.cov_matrix)
        self.inv_cov = torch.inverse(self.cov_matrix)
    
    def _create_mixture_components(self):
        """Create mixture distribution components"""
        components = []
        
        if self.complexity_level == 'medium':
            components.extend([
                {
                    'weight': 0.6,
                    'mean': torch.tensor([self.mu_x, self.mu_y], dtype=torch.float32),
                    'cov': torch.tensor([[self.sigma_x**2, 0.8*self.sigma_x*self.sigma_y],
                                       [0.8*self.sigma_x*self.sigma_y, self.sigma_y**2]], dtype=torch.float32)
                },
                {
                    'weight': 0.4,
                    'mean': torch.tensor([self.mu_x + 1.5, self.mu_y - 1.0], dtype=torch.float32),
                    'cov': torch.tensor([[self.sigma_x**2*0.5, -0.6*self.sigma_x*self.sigma_y*0.7],
                                       [-0.6*self.sigma_x*self.sigma_y*0.7, self.sigma_y**2*0.8]], dtype=torch.float32)
                }
            ])
        
        elif self.complexity_level == 'complex':
            for i in range(self.n_components):
                angle = 2 * torch.pi * i / self.n_components
                offset_x = 2.0 * torch.cos(torch.tensor(angle)).item()
                offset_y = 2.0 * torch.sin(torch.tensor(angle)).item()
                
                rho_i = self.rho * (0.5 + 0.5 * torch.cos(torch.tensor(angle)).item())
                scale_x = 0.5 + 0.5 * torch.sin(torch.tensor(angle + torch.pi/4)).item()
                scale_y = 0.5 + 0.5 * torch.cos(torch.tensor(angle + torch.pi/4)).item()
                
                components.append({
                    'weight': 1.0 / self.n_components,
                    'mean': torch.tensor([self.mu_x + offset_x, self.mu_y + offset_y], dtype=torch.float32),
                    'cov': torch.tensor([[self.sigma_x**2 * scale_x**2, rho_i*self.sigma_x*self.sigma_y*scale_x*scale_y],
                                       [rho_i*self.sigma_x*self.sigma_y*scale_x*scale_y, self.sigma_y**2 * scale_y**2]], dtype=torch.float32)
                })
        
        elif self.complexity_level == 'extreme':
            for i in range(self.n_components):
                angle = 4 * torch.pi * i / self.n_components
                radius = 1.5 + 0.5 * torch.sin(torch.tensor(angle)).item()
                offset_x = radius * torch.cos(torch.tensor(angle)).item()
                offset_y = radius * torch.sin(torch.tensor(angle)).item()
                
                rho_i = self.rho * (0.3 + 0.7 * torch.sin(torch.tensor(angle + torch.pi/3)).item())
                scale_x = 0.3 + 0.7 * torch.abs(torch.sin(torch.tensor(angle))).item()
                scale_y = 0.3 + 0.7 * torch.abs(torch.cos(torch.tensor(angle))).item()
                
                rotation_angle = angle * 0.5
                cos_r = torch.cos(torch.tensor(rotation_angle)).item()
                sin_r = torch.sin(torch.tensor(rotation_angle)).item()
                rotation_matrix = torch.tensor([[cos_r, -sin_r], [sin_r, cos_r]], dtype=torch.float32)
                
                base_cov = torch.tensor([[self.sigma_x**2 * scale_x**2, rho_i*self.sigma_x*self.sigma_y*scale_x*scale_y],
                                       [rho_i*self.sigma_x*self.sigma_y*scale_x*scale_y, self.sigma_y**2 * scale_y**2]], dtype=torch.float32)
                
                rotated_cov = rotation_matrix @ base_cov @ rotation_matrix.T
                
                components.append({
                    'weight': 1.0 / self.n_components,
                    'mean': torch.tensor([self.mu_x + offset_x, self.mu_y + offset_y], dtype=torch.float32),
                    'cov': rotated_cov
                })
        
        return components
    
    def sample_joint(self, n_samples=10000):
        """Sample from joint distribution"""
        if self.complexity_level == 'simple':
            mvn = torch.distributions.MultivariateNormal(self.mean_vector, self.cov_matrix)
            samples = mvn.sample((n_samples,))
        else:
            all_samples = []
            
            for component in self.components:
                n_comp_samples = int(n_samples * component['weight'])
                if n_comp_samples > 0:
                    mvn = torch.distributions.MultivariateNormal(component['mean'], component['cov'])
                    comp_samples = mvn.sample((n_comp_samples,))
                    all_samples.append(comp_samples)
            
            if all_samples:
                samples = torch.cat(all_samples, dim=0)
                
                if self.complexity_level in ['complex', 'extreme']:
                    noise_scale = 0.1 if self.complexity_level == 'complex' else 0.2
                    noise = torch.randn_like(samples) * noise_scale
                    samples = samples + noise
                
                if self.complexity_level == 'extreme':
                    x_mod = samples[:, 0] + 0.3 * torch.sin(2 * torch.pi * samples[:, 1])
                    y_mod = samples[:, 1] + 0.3 * torch.cos(2 * torch.pi * samples[:, 0])
                    samples = torch.stack([x_mod, y_mod], dim=1)
                
                perm = torch.randperm(len(samples))
                samples = samples[perm]
            else:
                mvn = torch.distributions.MultivariateNormal(self.mean_vector, self.cov_matrix)
                samples = mvn.sample((n_samples,))
        
        return samples
    
    def pdf_joint(self, x, y):
        """Compute joint probability density p(x,y)"""
        if self.complexity_level == 'simple':
            xy = torch.stack([x, y], dim=-1)
            centered = xy - self.mean_vector
            
            mahal_sq = torch.sum(centered @ self.inv_cov * centered, dim=-1)
            
            normalizer = 1.0 / (2 * torch.pi * torch.sqrt(torch.tensor(self.det_cov)))
            pdf = normalizer * torch.exp(-0.5 * mahal_sq)
            return pdf
        else:
            xy = torch.stack([x, y], dim=-1)
            total_pdf = torch.zeros(xy.shape[:-1], dtype=torch.float32)
            
            for component in self.components:
                centered = xy - component['mean']
                
                try:
                    inv_cov = torch.inverse(component['cov'])
                    det_cov = torch.det(component['cov'])
                except:
                    inv_cov = torch.pinverse(component['cov'])
                    det_cov = torch.tensor(1e-6, dtype=torch.float32)
                
                mahal_sq = torch.sum(centered @ inv_cov * centered, dim=-1)
                
                normalizer = 1.0 / (2 * torch.pi * torch.sqrt(torch.tensor(det_cov)))
                comp_pdf = normalizer * torch.exp(-0.5 * mahal_sq)
                
                total_pdf += component['weight'] * comp_pdf
            
            return total_pdf
    
    def pdf_marginal_x(self, x):
        """Compute marginal probability density p(x)"""
        return (1.0 / (torch.sqrt(torch.tensor(2 * torch.pi)) * self.sigma_x)) * \
               torch.exp(-0.5 * ((x - self.mu_x) / self.sigma_x)**2)
    
    def pdf_marginal_y(self, y):
        """Compute marginal probability density p(y)"""
        return (1.0 / (torch.sqrt(torch.tensor(2 * torch.pi)) * self.sigma_y)) * \
               torch.exp(-0.5 * ((y - self.mu_y) / self.sigma_y)**2)
    
    def pdf_conditional_x_given_y(self, x, y):
        """Compute conditional probability density p(x|y)"""
        mu_cond = self.mu_x + self.rho * (self.sigma_x / self.sigma_y) * (y - self.mu_y)
        sigma_cond = self.sigma_x * torch.sqrt(torch.tensor(1 - self.rho**2))
        
        return (1.0 / (torch.sqrt(torch.tensor(2 * torch.pi)) * sigma_cond)) * \
               torch.exp(-0.5 * ((x - mu_cond) / sigma_cond)**2)
    
    def pdf_conditional_y_given_x(self, y, x):
        """Compute conditional probability density p(y|x)"""
        mu_cond = self.mu_y + self.rho * (self.sigma_y / self.sigma_x) * (x - self.mu_x)
        sigma_cond = self.sigma_y * torch.sqrt(torch.tensor(1 - self.rho**2))
        
        return (1.0 / (torch.sqrt(torch.tensor(2 * torch.pi)) * sigma_cond)) * \
               torch.exp(-0.5 * ((y - mu_cond) / sigma_cond)**2)
    
    def sample_conditional_x_given_y(self, y_values, n_samples_per_y=1):
        """Sample conditional distribution p(x|y) given y values"""
        x_samples = []
        for y in y_values:
            mu_cond = self.mu_x + self.rho * (self.sigma_x / self.sigma_y) * (y - self.mu_y)
            sigma_cond = self.sigma_x * torch.sqrt(torch.tensor(1 - self.rho**2))
            x_cond = torch.normal(mu_cond, sigma_cond, size=(n_samples_per_y,))
            x_samples.append(x_cond)
        return torch.cat(x_samples)
    
    def sample_conditional_y_given_x(self, x_values, n_samples_per_x=1):
        """Sample conditional distribution p(y|x) given x values"""
        y_samples = []
        for x in x_values:
            mu_cond = self.mu_y + self.rho * (self.sigma_y / self.sigma_x) * (x - self.mu_x)
            sigma_cond = self.sigma_y * torch.sqrt(torch.tensor(1 - self.rho**2))
            y_cond = torch.normal(mu_cond, sigma_cond, size=(n_samples_per_x,))
            y_samples.append(y_cond)
        return torch.cat(y_samples)

class MixedMarginalDistribution:
    """
    Construct mixed marginal distributions p'(x) and p'(y) by mixing other distributions
    while keeping conditional probabilities p(x|y) and p(y|x) unchanged
    """
    def __init__(self, analytical_joint, mix_ratio=0.3):
        """
        Args:
            analytical_joint: Original analytical joint distribution
            mix_ratio: Mixing ratio, 0 means pure original distribution, 1 means pure mixed distribution
        """
        self.analytical_joint = analytical_joint
        self.mix_ratio = mix_ratio
        
        print(f"Constructing mixed marginal distributions:")
        print(f"  Mix ratio: {mix_ratio:.2f}")
        print(f"  Original distribution retention ratio: {1-mix_ratio:.2f}")
    
    def sample_mixed_marginal_x(self, n_samples=10000):
        """Sample mixed marginal distribution p'(x)"""
        n_original = int(n_samples * (1 - self.mix_ratio))
        n_mixed = n_samples - n_original
        
        x_original = torch.normal(self.analytical_joint.mu_x, self.analytical_joint.sigma_x, 
                                 size=(n_original,))
        
        x_mixed = torch.distributions.Uniform(-3, 3).sample((n_mixed,))
        
        all_x = torch.cat([x_original, x_mixed])
        perm = torch.randperm(len(all_x))
        return all_x[perm]
    
    def sample_mixed_marginal_y(self, n_samples=10000):
        """Sample mixed marginal distribution p'(y)"""
        n_original = int(n_samples * (1 - self.mix_ratio))
        n_mixed = n_samples - n_original
        
        y_original = torch.normal(self.analytical_joint.mu_y, self.analytical_joint.sigma_y, 
                                 size=(n_original,))
        
        y_mixed = torch.distributions.Beta(2, 5).sample((n_mixed,)) * 4 - 2
        
        all_y = torch.cat([y_original, y_mixed])
        perm = torch.randperm(len(all_y))
        return all_y[perm]
    
    def construct_mixed_joint_x(self, n_samples=10000):
        """
        Construct mixed joint distribution p'(x,y)_x from p'(x) and p(x|y)
        Uses p'(x,y)_x = p'(x) * p(y|x), implemented via sampling
        """
        n_original = int(n_samples * (1 - self.mix_ratio))
        n_mixed = n_samples - n_original

        original_pairs = self.analytical_joint.sample_joint(n_original) if n_original > 0 else torch.empty(0, 2)
        if n_mixed > 0:
            x_new = torch.distributions.Uniform(-3, 3).sample((n_mixed,))
            y_new = self._sample_conditional_y_given_x(x_new)
            mixed_pairs = torch.stack([x_new, y_new], dim=1)
        else:
            mixed_pairs = torch.empty(0, 2)

        if original_pairs.numel() == 0:
            all_pairs = mixed_pairs
        elif mixed_pairs.numel() == 0:
            all_pairs = original_pairs
        else:
            all_pairs = torch.cat([original_pairs, mixed_pairs], dim=0)
        if all_pairs.shape[0] > 0:
            perm = torch.randperm(all_pairs.shape[0])
            all_pairs = all_pairs[perm]
        return all_pairs
    
    def construct_mixed_joint_y(self, n_samples=10000):
        """
        Construct mixed joint distribution p'(x,y)_y from p'(y) and p(x|y)
        Uses p'(x,y)_y = p'(y) * p(x|y)
        """
        n_original = int(n_samples * (1 - self.mix_ratio))
        n_mixed = n_samples - n_original

        original_pairs = self.analytical_joint.sample_joint(n_original) if n_original > 0 else torch.empty(0, 2)

        if n_mixed > 0:
            y_new = torch.distributions.Beta(2, 5).sample((n_mixed,)) * 4 - 2
            x_new = self._sample_conditional_x_given_y(y_new)
            mixed_pairs = torch.stack([x_new, y_new], dim=1)
        else:
            mixed_pairs = torch.empty(0, 2)

        if original_pairs.numel() == 0:
            all_pairs = mixed_pairs
        elif mixed_pairs.numel() == 0:
            all_pairs = original_pairs
        else:
            all_pairs = torch.cat([original_pairs, mixed_pairs], dim=0)
        if all_pairs.shape[0] > 0:
            perm = torch.randperm(all_pairs.shape[0])
            all_pairs = all_pairs[perm]
        return all_pairs

    def _sample_conditional_y_given_x(self, x_values):
        """Sample p(y|x) given x values; supports simple Gaussian and mixture Gaussian"""
        if not hasattr(self.analytical_joint, 'components') or self.analytical_joint.components is None:
            mu_cond = (self.analytical_joint.mu_y + 
                      self.analytical_joint.rho * (self.analytical_joint.sigma_y / self.analytical_joint.sigma_x) * 
                      (x_values - self.analytical_joint.mu_x))
            sigma_cond = self.analytical_joint.sigma_y * torch.sqrt(torch.tensor(1 - self.analytical_joint.rho**2))
            return torch.normal(mu_cond, sigma_cond)
        
        y_samples = []
        eps = 1e-6
        for xv in x_values:
            weights = []
            mu_conds = []
            sigma_conds = []
            for comp in self.analytical_joint.components:
                mean = comp['mean']
                cov = comp['cov']
                weight = comp['weight']
                mu_xk, mu_yk = mean[0].item(), mean[1].item()
                sx2 = cov[0, 0].item()
                sxy = cov[0, 1].item()
                sy2 = cov[1, 1].item()
                px = (1.0 / (torch.sqrt(torch.tensor(2 * torch.pi * sx2))) *
                      torch.exp(-0.5 * ((xv - mu_xk) ** 2) / sx2))
                weights.append(weight * px)
                mu_c = mu_yk + (sxy / (sx2 + eps)) * (xv.item() - mu_xk)
                var_c = max(sy2 - (sxy * sxy) / (sx2 + eps), eps)
                mu_conds.append(mu_c)
                sigma_conds.append(torch.sqrt(torch.tensor(var_c)))
            weights = torch.stack(weights)
            weights = weights / (weights.sum() + eps)
            comp_idx = torch.multinomial(weights, num_samples=1).item()
            y_samples.append(torch.normal(torch.tensor(mu_conds[comp_idx]), sigma_conds[comp_idx], size=(1,)))
        return torch.cat(y_samples)

    def _sample_conditional_x_given_y(self, y_values):
        """Sample p(x|y) given y values; supports simple Gaussian and mixture Gaussian"""
        if not hasattr(self.analytical_joint, 'components') or self.analytical_joint.components is None:
            mu_cond = (self.analytical_joint.mu_x + 
                      self.analytical_joint.rho * (self.analytical_joint.sigma_x / self.analytical_joint.sigma_y) * 
                      (y_values - self.analytical_joint.mu_y))
            sigma_cond = self.analytical_joint.sigma_x * torch.sqrt(torch.tensor(1 - self.analytical_joint.rho**2))
            return torch.normal(mu_cond, sigma_cond)
        
        x_samples = []
        eps = 1e-6
        for yv in y_values:
            weights = []
            mu_conds = []
            sigma_conds = []
            for comp in self.analytical_joint.components:
                mean = comp['mean']
                cov = comp['cov']
                weight = comp['weight']
                mu_xk, mu_yk = mean[0].item(), mean[1].item()
                sx2 = cov[0, 0].item()
                sxy = cov[0, 1].item()
                sy2 = cov[1, 1].item()
                py = (1.0 / (torch.sqrt(torch.tensor(2 * torch.pi * sy2))) *
                      torch.exp(-0.5 * ((yv - mu_yk) ** 2) / sy2))
                weights.append(weight * py)
                mu_c = mu_xk + (sxy / (sy2 + eps)) * (yv.item() - mu_yk)
                var_c = max(sx2 - (sxy * sxy) / (sy2 + eps), eps)
                mu_conds.append(mu_c)
                sigma_conds.append(torch.sqrt(torch.tensor(var_c)))
            weights = torch.stack(weights)
            weights = weights / (weights.sum() + eps)
            comp_idx = torch.multinomial(weights, num_samples=1).item()
            x_samples.append(torch.normal(torch.tensor(mu_conds[comp_idx]), sigma_conds[comp_idx], size=(1,)))
        return torch.cat(x_samples)

class FlowMatchingNetVtx(nn.Module):
    def __init__(self, input_dim=1, cond_dim=1, hidden_dim=256):
        super(FlowMatchingNetVtx, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + cond_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, xt, yt_cond, t):
        # xt, yt_cond shape: (batch, 1)
        if xt.dim() == 1:
            xt = xt.unsqueeze(-1)
        if yt_cond.dim() == 1:
            yt_cond = yt_cond.unsqueeze(-1)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_expanded = t.unsqueeze(-1).expand(xt.shape[0], -1)
        input_all = torch.cat([xt, yt_cond, t_expanded], dim=-1)
        return self.network(input_all).squeeze(-1)

class FlowMatchingNetVty(nn.Module):
    def __init__(self, input_dim=1, cond_dim=1, hidden_dim=256):
        super(FlowMatchingNetVty, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim + cond_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, yt, xt_cond, t):
        if yt.dim() == 1:
            yt = yt.unsqueeze(-1)
        if xt_cond.dim() == 1:
            xt_cond = xt_cond.unsqueeze(-1)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t_expanded = t.unsqueeze(-1).expand(yt.shape[0], -1)
        input_all = torch.cat([yt, xt_cond, t_expanded], dim=-1)
        return self.network(input_all).squeeze(-1)

def get_flow_matching_loss_vtx(model_vtx, x0, x1, y1, device):
    batch_size = x0.shape[0]
    xt0 = x0[:, 0].unsqueeze(-1)
    xt1 = x1[:, 0].unsqueeze(-1)
    yt1 = y1[:, 1].unsqueeze(-1)
    t = torch.rand(batch_size, device=device)
    xt = (1 - t).unsqueeze(-1) * xt0 + t.unsqueeze(-1) * xt1
    v_true_x = (xt1 - xt0).squeeze(-1)
    v_pred_x = model_vtx(xt, yt1, t)
    loss = nn.MSELoss()(v_pred_x, v_true_x)
    return loss

def get_flow_matching_loss_vty(model_vty, x0, x1, x1_cond, device):
    batch_size = x0.shape[0]
    yt0 = x0[:, 1].unsqueeze(-1)
    yt1 = x1[:, 1].unsqueeze(-1)
    xt1 = x1_cond[:, 0].unsqueeze(-1)
    t = torch.rand(batch_size, device=device)
    yt = (1 - t).unsqueeze(-1) * yt0 + t.unsqueeze(-1) * yt1
    v_true_y = (yt1 - yt0).squeeze(-1)
    v_pred_y = model_vty(yt, xt1, t)
    loss = nn.MSELoss()(v_pred_y, v_true_y)
    return loss

def train_flow_matching_conditional(model_vtx, model_vty, target_data, num_epochs=1000, batch_size=256, lr=1e-3, device='cpu'):
    model_vtx = model_vtx.to(device)
    model_vty = model_vty.to(device)
    target_data = target_data.to(device).float()
    optimizer_vtx = optim.Adam(model_vtx.parameters(), lr=lr)
    optimizer_vty = optim.Adam(model_vty.parameters(), lr=lr)
    dataset = TensorDataset(target_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    losses_vtx = []
    losses_vty = []
    for epoch in range(num_epochs):
        epoch_loss_vtx = 0
        epoch_loss_vty = 0
        num_batches = 0
        for batch in dataloader:
            x1 = batch[0]
            x0 = torch.randn_like(x1)
            optimizer_vtx.zero_grad()
            loss_vtx = get_flow_matching_loss_vtx(model_vtx, x0, x1, x1, device)
            loss_vtx.backward()
            optimizer_vtx.step()
            optimizer_vty.zero_grad()
            loss_vty = get_flow_matching_loss_vty(model_vty, x0, x1, x1, device)
            loss_vty.backward()
            optimizer_vty.step()
            epoch_loss_vtx += loss_vtx.item()
            epoch_loss_vty += loss_vty.item()
            num_batches += 1
        avg_loss_vtx = epoch_loss_vtx / num_batches
        avg_loss_vty = epoch_loss_vty / num_batches
        losses_vtx.append(avg_loss_vtx)
        losses_vty.append(avg_loss_vty)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss Vtx: {avg_loss_vtx:.6f}, Loss Vty: {avg_loss_vty:.6f}")
    return losses_vtx, losses_vty

def train_flow_matching_conditional_two_datasets(model_vtx, model_vty, dataset_x, dataset_y, 
                                                num_epochs=1000, batch_size=256, lr=1e-3, device='cpu'):
    """
    Train vtx and vty models separately using two datasets
    dataset_x: Dataset for training vtx p'(x,y)_x
    dataset_y: Dataset for training vty p'(x,y)_y
    """
    model_vtx = model_vtx.to(device)
    model_vty = model_vty.to(device)
    dataset_x = dataset_x.to(device).float()
    dataset_y = dataset_y.to(device).float()
    
    optimizer_vtx = optim.Adam(model_vtx.parameters(), lr=lr)
    optimizer_vty = optim.Adam(model_vty.parameters(), lr=lr)
    
    dataset_x_loader = TensorDataset(dataset_x)
    dataloader_x = DataLoader(dataset_x_loader, batch_size=batch_size, shuffle=True)
    
    dataset_y_loader = TensorDataset(dataset_y)
    dataloader_y = DataLoader(dataset_y_loader, batch_size=batch_size, shuffle=True)
    
    losses_vtx = []
    losses_vty = []
    
    for epoch in range(num_epochs):
        epoch_loss_vtx = 0
        epoch_loss_vty = 0
        num_batches_x = 0
        num_batches_y = 0
        
        for batch in dataloader_x:
            x1 = batch[0]
            x0 = torch.randn_like(x1)
            optimizer_vtx.zero_grad()
            loss_vtx = get_flow_matching_loss_vtx(model_vtx, x0, x1, x1, device)
            loss_vtx.backward()
            optimizer_vtx.step()
            epoch_loss_vtx += loss_vtx.item()
            num_batches_x += 1
        
        for batch in dataloader_y:
            x1 = batch[0]
            x0 = torch.randn_like(x1)
            optimizer_vty.zero_grad()
            loss_vty = get_flow_matching_loss_vty(model_vty, x0, x1, x1, device)
            loss_vty.backward()
            optimizer_vty.step()
            epoch_loss_vty += loss_vty.item()
            num_batches_y += 1
        
        avg_loss_vtx = epoch_loss_vtx / num_batches_x if num_batches_x > 0 else 0
        avg_loss_vty = epoch_loss_vty / num_batches_y if num_batches_y > 0 else 0
        
        losses_vtx.append(avg_loss_vtx)
        losses_vty.append(avg_loss_vty)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss Vtx: {avg_loss_vtx:.6f}, Loss Vty: {avg_loss_vty:.6f}")
    
    return losses_vtx, losses_vty

def sample_conditional_flow_matching(model_vtx, model_vty, mix_ratio=0.0, num_samples=1000, 
                                   num_steps=100, num_external_loops=3, device='cpu', 
                                   visualize=False, target_name="M2PDE Conditional 2D Distribution"):
    """
    Implement mutual iteration process for x and y fields according to M2PDE algorithm
    
    Args:
        model_vtx: x field model, input (xt, y_cond, t) -> vtx
        model_vty: y field model, input (yt, x_cond, t) -> vty
        num_external_loops: Number of external loops K
        num_steps: Number of internal denoising steps S
    """
    model_vtx = model_vtx.to(device)
    model_vty = model_vty.to(device)
    model_vtx.eval()
    model_vty.eval()
    
    z_x_est = torch.randn(num_samples, device=device)
    z_y_est = torch.randn(num_samples, device=device)
    
    if visualize:
        intermediate_states = []
        initial_state = torch.stack([z_x_est, z_y_est], dim=1)
        intermediate_states.append(initial_state.cpu().numpy())
    
    with torch.no_grad():
        for k in range(1, num_external_loops + 1):
            print(f"🔄 External loop k={k}/{num_external_loops}")
            
            z_x_est_prev = z_x_est.clone()
            z_y_est_prev = z_y_est.clone()
            
            z_x_est = torch.randn(num_samples, device=device)
            z_y_est = torch.randn(num_samples, device=device)
            
            z_x = torch.randn(num_samples, device=device)
            z_y = torch.randn(num_samples, device=device)
            
            dt = 1.0 / num_steps
            for step in range(num_steps, 0, -1):
                t = torch.full((num_samples,), step * dt, device=device)
                
                if k > 1:
                    lambda_weight = 1.0 - step / num_steps
                else:
                    lambda_weight = 1.0
                
                weighted_y_cond = lambda_weight * z_y_est + (1 - lambda_weight) * z_y_est_prev
                
                v_x = model_vtx(z_x.unsqueeze(-1), weighted_y_cond.unsqueeze(-1), t)
                
                z_x = z_x + dt * v_x
                
                z_x_est = z_x + v_x * (1.0 - t + dt)
                
                weighted_x_cond = lambda_weight * z_x_est + (1 - lambda_weight) * z_x_est_prev
                
                v_y = model_vty(z_y.unsqueeze(-1), weighted_x_cond.unsqueeze(-1), t)
                
                z_y = z_y + dt * v_y
                
                z_y_est = z_y + v_y * (1.0 - t + dt)
                
                if visualize and k == num_external_loops and step % (num_steps // 10) == 0:
                    current_state = torch.stack([z_x, z_y], dim=1)
                    intermediate_states.append(current_state.cpu().numpy())
    final_result = torch.stack([z_x, z_y], dim=1)
    
    if visualize:
        intermediate_states.append(final_result.cpu().numpy())
        visualize_2d_sampling_process(intermediate_states, target_name, num_steps)
        if PILLOW_AVAILABLE:
            create_2d_sampling_gif(intermediate_states, target_name, num_steps, mix_ratio)
        else:
            print("💡 Tip: Install Pillow to generate GIF animations")
    
    return final_result


def visualize_2d_sampling_process(intermediate_states, target_name, num_steps):
    """
    Visualize the dynamic 2D sampling process from Gaussian to target distribution
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor(COLORS['light'])
    
    # Plot 1: 2D scatter evolution
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(intermediate_states)))
    
    for i, (state, color) in enumerate(zip(intermediate_states, colors)):
        alpha = 0.3 + 0.7 * (i / (len(intermediate_states) - 1))
        ax1.scatter(state[:, 0], state[:, 1], alpha=alpha, s=1, color=color, 
                   label=f't={i/(len(intermediate_states)-1):.1f}' if i % 3 == 0 else "")
    
    ax1.set_title(f'2D Evolution of (Gaussian → Target)', 
                  fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax1.set_xlabel('X (Beta)', fontsize=12, color=COLORS['dark'])
    ax1.set_ylabel('Y (Mixture)', fontsize=12, color=COLORS['dark'])
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor(COLORS['light'])
    
    # Plot 2: Marginal X evolution
    ax2 = axes[0, 1]
    for i, (state, color) in enumerate(zip(intermediate_states, colors)):
        alpha = 0.3 + 0.7 * (i / (len(intermediate_states) - 1))
        ax2.hist(state[:, 0], bins=30, alpha=alpha, color=color, density=True)
    
    ax2.set_title('X (Beta) Marginal Evolution', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax2.set_xlabel('X Value', fontsize=12, color=COLORS['dark'])
    ax2.set_ylabel('Density', fontsize=12, color=COLORS['dark'])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor(COLORS['light'])
    
    # Plot 3: Marginal Y evolution
    ax3 = axes[1, 0]
    for i, (state, color) in enumerate(zip(intermediate_states, colors)):
        alpha = 0.3 + 0.7 * (i / (len(intermediate_states) - 1))
        ax3.hist(state[:, 1], bins=30, alpha=alpha, color=color, density=True)
    
    ax3.set_title('Y (Mixture) Marginal Evolution', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax3.set_xlabel('Y Value', fontsize=12, color=COLORS['dark'])
    ax3.set_ylabel('Density', fontsize=12, color=COLORS['dark'])
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_facecolor(COLORS['light'])
    
    # Plot 4: Statistics evolution
    ax4 = axes[1, 1]
    x_means = [np.mean(state[:, 0]) for state in intermediate_states]
    y_means = [np.mean(state[:, 1]) for state in intermediate_states]
    x_stds = [np.std(state[:, 0]) for state in intermediate_states]
    y_stds = [np.std(state[:, 1]) for state in intermediate_states]
    times = np.linspace(0, 1, len(intermediate_states))
    
    ax4.plot(times, x_means, 'o-', color=COLORS['primary'], linewidth=2, label='X Mean')
    ax4.plot(times, y_means, 's-', color=COLORS['secondary'], linewidth=2, label='Y Mean')
    ax4.plot(times, x_stds, '^-', color=COLORS['accent'], linewidth=2, label='X Std')
    ax4.plot(times, y_stds, 'v-', color=COLORS['info'], linewidth=2, label='Y Std')
    
    ax4.set_title('Statistics Evolution', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax4.set_xlabel('Time (t)', fontsize=12, color=COLORS['dark'])
    ax4.set_ylabel('Value', fontsize=12, color=COLORS['dark'])
    ax4.legend()
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_facecolor(COLORS['light'])
    
    # Add overall title
    # fig.suptitle(f'Dynamic 2D Flow Matching Sampling: {target_name}', 
    #              fontsize=16, fontweight='bold', color=COLORS['dark'], y=0.95)
    
    plt.tight_layout()
    plt.show()
    
    # Print evolution statistics
    print(f"\n🔄 {target_name} 2D Sampling Evolution:")
    print("=" * 60)
    print(f"X (Beta) - Initial: μ={x_means[0]:.4f}, σ={x_stds[0]:.4f}")
    print(f"X (Beta) - Final:   μ={x_means[-1]:.4f}, σ={x_stds[-1]:.4f}")
    print(f"Y (Mixture) - Initial: μ={y_means[0]:.4f}, σ={y_stds[0]:.4f}")
    print(f"Y (Mixture) - Final:   μ={y_means[-1]:.4f}, σ={y_stds[-1]:.4f}")

def create_2d_sampling_gif(intermediate_states, target_name, num_steps, mix_ratio, gif_filename=None):
    """
    Create an animated GIF showing the 2D sampling process
    """
    if not PILLOW_AVAILABLE:
        print("⚠️ Cannot create GIF: Pillow not available")
        return None
    
    m2pde_dir = "/path/to/plot_toy"
    os.makedirs(m2pde_dir, exist_ok=True)
        
    if gif_filename is None:
        gif_filename = f"{m2pde_dir}/flow_matching_2d_{target_name.lower().replace(' ', '_')}_sampling_{mix_ratio}.gif"
    
    all_x = np.concatenate([state[:, 0] for state in intermediate_states])
    all_y = np.concatenate([state[:, 1] for state in intermediate_states])
    
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    
    x_padding = (x_max - x_min) * 0.2
    y_padding = (y_max - y_min) * 0.2
    x_lim = [x_min - x_padding, x_max + x_padding]
    y_lim = [y_min - y_padding, y_max + y_padding]
    
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(COLORS['light'])
    
    # Set up the plots with dynamic limits
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.set_xlabel('X (Beta)', fontsize=12, color=COLORS['dark'])
    ax1.set_ylabel('Y (Mixture)', fontsize=12, color=COLORS['dark'])
    ax1.set_title(f'2D {target_name} Evolution', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor(COLORS['light'])
    
    ax2.set_xlim(x_lim)
    ax2.set_ylim(0, 1.5)
    ax2.set_xlabel('Value', fontsize=12, color=COLORS['dark'])
    ax2.set_ylabel('Density', fontsize=12, color=COLORS['dark'])
    ax2.set_title('Marginal Distributions', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor(COLORS['light'])
    
    def animate(frame):
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
        # Re-setup axes with dynamic limits
        ax1.set_xlim(x_lim)
        ax1.set_ylim(y_lim)
        ax1.set_xlabel('X (Beta)', fontsize=12, color=COLORS['dark'])
        ax1.set_ylabel('Y (Mixture)', fontsize=12, color=COLORS['dark'])
        ax1.set_title(f'2D {target_name} Evolution', fontsize=14, fontweight='bold', color=COLORS['dark'])
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_facecolor(COLORS['light'])
        
        ax2.set_xlim(x_lim)
        ax2.set_ylim(0, 1.5)
        ax2.set_xlabel('Value', fontsize=12, color=COLORS['dark'])
        ax2.set_ylabel('Density', fontsize=12, color=COLORS['dark'])
        ax2.set_title('Marginal Distributions', fontsize=14, fontweight='bold', color=COLORS['dark'])
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_facecolor(COLORS['light'])
        
        # Current state
        current_state = intermediate_states[frame]
        current_time = frame / (len(intermediate_states) - 1)
        
        # Plot 2D scatter
        ax1.scatter(current_state[:, 0], current_state[:, 1], alpha=0.6, s=1, color=COLORS['primary'])
        
        # Plot marginal distributions
        hist_x, bins_x = np.histogram(current_state[:, 0], bins=30, density=True)
        hist_y, bins_y = np.histogram(current_state[:, 1], bins=30, density=True)
        bin_centers_x = (bins_x[:-1] + bins_x[1:]) / 2
        bin_centers_y = (bins_y[:-1] + bins_y[1:]) / 2
        
        ax2.plot(bin_centers_x, hist_x, color=COLORS['primary'], linewidth=2, label='X (Beta)')
        ax2.plot(bin_centers_y, hist_y, color=COLORS['secondary'], linewidth=2, label='Y (Mixture)')
        ax2.legend()
        
        # Add time indicator
        ax1.text(0.02, 0.98, f'Time: t = {current_time:.2f}', transform=ax1.transAxes, 
                verticalalignment='top', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add statistics
        mean_x = np.mean(current_state[:, 0])
        mean_y = np.mean(current_state[:, 1])
        ax1.text(0.02, 0.85, f'X Mean: {mean_x:.3f}\nY Mean: {mean_y:.3f}', transform=ax1.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        return []
    
    # Create animation
    print(f"🎬 Creating 2D GIF animation for {target_name}...")
    anim = FuncAnimation(fig, animate, frames=len(intermediate_states), 
                        interval=200, blit=False, repeat=True)
    
    # Save as GIF
    writer = PillowWriter(fps=5)
    anim.save(gif_filename, writer=writer)
    plt.close(fig)
    
    print(f"✅ 2D GIF saved as: {gif_filename}")
    return gif_filename

def save_distribution_plots(original_samples, generated_samples, target_name, mix_ratio):
    """Save original and generated data distribution plots to m2pde folder"""
    m2pde_dir = "/path/to/plot_toy"
    os.makedirs(m2pde_dir, exist_ok=True)
    
    if isinstance(original_samples, torch.Tensor):
        original_samples = original_samples.cpu().numpy()
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.cpu().numpy()
    
    all_x = np.concatenate([original_samples[:, 0]])
    all_y = np.concatenate([original_samples[:, 1]])
    
    x_min, x_max = np.min(all_x), np.max(all_x)
    y_min, y_max = np.min(all_y), np.max(all_y)
    
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    x_lim = [x_min - x_padding, x_max + x_padding]
    y_lim = [y_min - y_padding, y_max + y_padding]
    
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    fig1.patch.set_facecolor(COLORS['light'])
    
    ax1.scatter(original_samples[:, 0], original_samples[:, 1], 
               alpha=0.6, s=1, color=COLORS['primary'])
    ax1.set_xlabel('X', fontsize=12, color=COLORS['dark'])
    ax1.set_ylabel('Y', fontsize=12, color=COLORS['dark'])
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor(COLORS['light'])
    
    plt.tight_layout()
    filename1 = f"{m2pde_dir}/original_distribution_{target_name.lower().replace(' ', '_')}_mix_{mix_ratio}.png"
    plt.savefig(filename1, dpi=300, bbox_inches='tight', facecolor=COLORS['light'])
    plt.close(fig1)
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    fig2.patch.set_facecolor(COLORS['light'])
    
    ax2.scatter(generated_samples[:, 0], generated_samples[:, 1], 
               alpha=0.6, s=1, color=COLORS['secondary'])
    # ax2.set_title(f'Generated Distribution\n({target_name})', 
    #               fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax2.set_xlabel('X', fontsize=12, color=COLORS['dark'])
    ax2.set_ylabel('Y', fontsize=12, color=COLORS['dark'])
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor(COLORS['light'])
    
    plt.tight_layout()
    filename2 = f"{m2pde_dir}/generated_distribution_{target_name.lower().replace(' ', '_')}_mix_{mix_ratio}.png"
    plt.savefig(filename2, dpi=300, bbox_inches='tight', facecolor=COLORS['light'])
    plt.close(fig2)
    
    print(f"✅ Original distribution plot saved as: {filename1}")
    print(f"✅ Generated distribution plot saved as: {filename2}")
    
    return [filename1, filename2]

def visualize_2d_flow_matching_results(target_data, generated_data, target_name, losses, initial_data=None):
    """
    Visualize 2D Flow Matching results with comparison to initial distribution
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor(COLORS['light'])
    
    # Calculate unified axis limits for all 2D scatter plots
    # all_x_data = [target_data[:, 0].cpu().numpy(), generated_data[:, 0].cpu().numpy()]
    # all_y_data = [target_data[:, 1].cpu().numpy(), generated_data[:, 1].cpu().numpy()]
    
    all_x_data = [target_data[:, 0].cpu().numpy()]
    all_y_data = [target_data[:, 1].cpu().numpy()]
    
    if initial_data is not None:
        all_x_data.append(initial_data[:, 0].cpu().numpy())
        all_y_data.append(initial_data[:, 1].cpu().numpy())
    
    x_min, x_max = np.min([np.min(x) for x in all_x_data]), np.max([np.max(x) for x in all_x_data])
    y_min, y_max = np.min([np.min(y) for y in all_y_data]), np.max([np.max(y) for y in all_y_data])
    
    # Add some padding
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    x_lim = [x_min - x_padding, x_max + x_padding]
    y_lim = [y_min - y_padding, y_max + y_padding]
    
    # Calculate unified axis limits for marginal histograms
    x_marginal_min, x_marginal_max = x_min, x_max
    y_marginal_min, y_marginal_max = y_min, y_max
    
    # Plot 1: Target 2D distribution
    ax1 = axes[0, 0]
    ax1.scatter(target_data[:, 0].cpu().numpy(), target_data[:, 1].cpu().numpy(), 
               alpha=0.6, s=1, color=COLORS['primary'])
    ax1.set_title(f'Target \n(Complex 2D Joint Distribution)', 
                  fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax1.set_xlabel('X', fontsize=12, color=COLORS['dark'])
    ax1.set_ylabel('Y', fontsize=12, color=COLORS['dark'])
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor(COLORS['light'])
    
    # Plot 2: Generated 2D distribution
    ax2 = axes[0, 1]
    ax2.scatter(generated_data[:, 0].cpu().numpy(), generated_data[:, 1].cpu().numpy(), 
               alpha=0.6, s=1, color=COLORS['secondary'])
    ax2.set_title(f'Generated \n(Flow Matching Result)', 
                  fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax2.set_xlabel('X', fontsize=12, color=COLORS['dark'])
    ax2.set_ylabel('Y', fontsize=12, color=COLORS['dark'])
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor(COLORS['light'])
    
    # Plot 3: Initial vs Generated comparison (if initial_data provided)
    ax3 = axes[0, 2]
    if initial_data is not None:
        ax3.scatter(initial_data[:, 0].cpu().numpy(), initial_data[:, 1].cpu().numpy(), 
                   alpha=0.4, s=1, color=COLORS['info'], label='Initial')
        ax3.scatter(generated_data[:, 0].cpu().numpy(), generated_data[:, 1].cpu().numpy(), 
                   alpha=0.6, s=1, color=COLORS['secondary'], label='Generated')
        ax3.set_title('Initial vs Generated\nDistribution Comparison', 
                      fontsize=14, fontweight='bold', color=COLORS['dark'])
        ax3.set_xlim(x_lim)
        ax3.set_ylim(y_lim)
        ax3.legend()
    else:
        ax3.plot(losses, color=COLORS['accent'], linewidth=2)
        ax3.set_title('Training Loss', fontsize=14, fontweight='bold', color=COLORS['dark'])
        ax3.set_xlabel('Epoch', fontsize=12, color=COLORS['dark'])
        ax3.set_ylabel('Loss', fontsize=12, color=COLORS['dark'])
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_facecolor(COLORS['light'])
    
    # Plot 4: X marginal comparison
    ax4 = axes[1, 0]
    ax4.hist(target_data[:, 0].cpu().numpy(), bins=50, alpha=0.7, label='Target', 
             color=COLORS['primary'], density=True, edgecolor='white')
    ax4.hist(generated_data[:, 0].cpu().numpy(), bins=50, alpha=0.7, label='Generated', 
             color=COLORS['secondary'], density=True, edgecolor='white')
    if initial_data is not None:
        ax4.hist(initial_data[:, 0].cpu().numpy(), bins=50, alpha=0.5, label='Initial', 
                 color=COLORS['info'], density=True, edgecolor='white')
    ax4.set_title('X Marginal Distribution', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax4.set_xlabel('X Value', fontsize=12, color=COLORS['dark'])
    ax4.set_ylabel('Density', fontsize=12, color=COLORS['dark'])
    ax4.set_xlim(x_marginal_min, x_marginal_max)
    ax4.legend()
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_facecolor(COLORS['light'])
    
    # Plot 5: Y marginal comparison
    ax5 = axes[1, 1]
    ax5.hist(target_data[:, 1].cpu().numpy(), bins=50, alpha=0.7, label='Target', 
             color=COLORS['primary'], density=True, edgecolor='white')
    ax5.hist(generated_data[:, 1].cpu().numpy(), bins=50, alpha=0.7, label='Generated', 
             color=COLORS['secondary'], density=True, edgecolor='white')
    if initial_data is not None:
        ax5.hist(initial_data[:, 1].cpu().numpy(), bins=50, alpha=0.5, label='Initial', 
                 color=COLORS['info'], density=True, edgecolor='white')
    ax5.set_title('Y Marginal Distribution', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax5.set_xlabel('Y Value', fontsize=12, color=COLORS['dark'])
    ax5.set_ylabel('Density', fontsize=12, color=COLORS['dark'])
    ax5.set_xlim(y_marginal_min, y_marginal_max)
    ax5.legend()
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_facecolor(COLORS['light'])
    
    # Plot 6: Statistics comparison
    ax6 = axes[1, 2]
    stats_target = [target_data[:, 0].mean().item(), target_data[:, 0].std().item(), 
                   target_data[:, 1].mean().item(), target_data[:, 1].std().item()]
    stats_generated = [generated_data[:, 0].mean().item(), generated_data[:, 0].std().item(), 
                      generated_data[:, 1].mean().item(), generated_data[:, 1].std().item()]
    
    x_pos = np.arange(len(['X Mean', 'X Std', 'Y Mean', 'Y Std']))
    width = 0.35
    
    ax6.bar(x_pos - width/2, stats_target, width, label='Target', 
            color=COLORS['primary'], alpha=0.8)
    ax6.bar(x_pos + width/2, stats_generated, width, label='Generated', 
            color=COLORS['secondary'], alpha=0.8)
    
    if initial_data is not None:
        stats_initial = [initial_data[:, 0].mean().item(), initial_data[:, 0].std().item(), 
                        initial_data[:, 1].mean().item(), initial_data[:, 1].std().item()]
        ax6.bar(x_pos, stats_initial, width*0.7, label='Initial', 
                color=COLORS['info'], alpha=0.6)
    
    ax6.set_title('Statistics Comparison', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax6.set_xlabel('Statistics', fontsize=12, color=COLORS['dark'])
    ax6.set_ylabel('Value', fontsize=12, color=COLORS['dark'])
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(['X Mean', 'X Std', 'Y Mean', 'Y Std'])
    ax6.legend()
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.set_facecolor(COLORS['light'])
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n🎯 {target_name} 2D Flow Matching Results:")
    print("=" * 60)
    print(f"Target X - Mean: {target_data[:, 0].mean().item():.4f}, Std: {target_data[:, 0].std().item():.4f}")
    print(f"Generated X - Mean: {generated_data[:, 0].mean().item():.4f}, Std: {generated_data[:, 0].std().item():.4f}")
    print(f"Target Y - Mean: {target_data[:, 1].mean().item():.4f}, Std: {target_data[:, 1].std().item():.4f}")
    print(f"Generated Y - Mean: {generated_data[:, 1].mean().item():.4f}, Std: {generated_data[:, 1].std().item():.4f}")
    if initial_data is not None:
        print(f"Initial X - Mean: {initial_data[:, 0].mean().item():.4f}, Std: {initial_data[:, 0].std().item():.4f}")
        print(f"Initial Y - Mean: {initial_data[:, 1].mean().item():.4f}, Std: {initial_data[:, 1].std().item():.4f}")
    if isinstance(losses, list) and len(losses) > 0:
        print(f"Final Training Loss: {losses[-1]:.6f}")
    
    # Calculate correlation coefficients
    target_corr = torch.corrcoef(torch.stack([target_data[:, 0], target_data[:, 1]]))[0, 1].item()
    generated_corr = torch.corrcoef(torch.stack([generated_data[:, 0], generated_data[:, 1]]))[0, 1].item()
    print(f"Target Correlation: {target_corr:.4f}")
    print(f"Generated Correlation: {generated_corr:.4f}")
    if initial_data is not None:
        initial_corr = torch.corrcoef(torch.stack([initial_data[:, 0], initial_data[:, 1]]))[0, 1].item()
        print(f"Initial Correlation: {initial_corr:.4f}")

def visualize_analytical_distributions(analytical_joint, mixed_marginal, original_joint_samples, 
                                     mixed_joint_x_samples, mixed_joint_y_samples):
    """
    Visualize various distributions constructed from analytical distributions
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.patch.set_facecolor(COLORS['light'])
    all_x_data = [original_joint_samples[:, 0].numpy(), 
                  mixed_joint_x_samples[:, 0].numpy(), 
                  mixed_joint_y_samples[:, 0].numpy()]
    all_y_data = [original_joint_samples[:, 1].numpy(), 
                  mixed_joint_x_samples[:, 1].numpy(), 
                  mixed_joint_y_samples[:, 1].numpy()]
    
    x_min, x_max = np.min([np.min(x) for x in all_x_data]), np.max([np.max(x) for x in all_x_data])
    y_min, y_max = np.min([np.min(y) for y in all_y_data]), np.max([np.max(y) for y in all_y_data])
    
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    x_lim = [x_min - x_padding, x_max + x_padding]
    y_lim = [y_min - y_padding, y_max + y_padding]
    
    ax1 = axes[0, 0]
    ax1.scatter(original_joint_samples[:, 0].numpy(), original_joint_samples[:, 1].numpy(), 
               alpha=0.6, s=1, color=COLORS['primary'])
    ax1.set_title('Original Joint Distribution p(x,y)', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax1.set_xlabel('X', fontsize=12, color=COLORS['dark'])
    ax1.set_ylabel('Y', fontsize=12, color=COLORS['dark'])
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor(COLORS['light'])
    
    # Plot 2: X marginal from constructed mixed joint_x
    ax2 = axes[0, 1]
    ax2.hist(original_joint_samples[:, 0].numpy(), bins=50, alpha=0.7, label='Original p(x)', 
             color=COLORS['primary'], density=True, edgecolor='white')
    ax2.hist(mixed_joint_x_samples[:, 0].numpy(), bins=50, alpha=0.7, label="Mixed p'(x) (from joint_x)", 
             color=COLORS['secondary'], density=True, edgecolor='white')
    ax2.set_title('X Marginal Distribution Comparison', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax2.set_xlabel('X Value', fontsize=12, color=COLORS['dark'])
    ax2.set_ylabel('Density', fontsize=12, color=COLORS['dark'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor(COLORS['light'])
    
    # Plot 3: Y marginal from constructed mixed joint_y
    ax3 = axes[0, 2]
    ax3.hist(original_joint_samples[:, 1].numpy(), bins=50, alpha=0.7, label='Original p(y)', 
             color=COLORS['primary'], density=True, edgecolor='white')
    ax3.hist(mixed_joint_y_samples[:, 1].numpy(), bins=50, alpha=0.7, label="Mixed p'(y) (from joint_y)", 
             color=COLORS['accent'], density=True, edgecolor='white')
    ax3.set_title('Y Marginal Distribution Comparison', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax3.set_xlabel('Y Value', fontsize=12, color=COLORS['dark'])
    ax3.set_ylabel('Density', fontsize=12, color=COLORS['dark'])
    ax3.legend()
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_facecolor(COLORS['light'])
    
    ax4 = axes[1, 0]
    ax4.scatter(mixed_joint_x_samples[:, 0].numpy(), mixed_joint_x_samples[:, 1].numpy(), 
               alpha=0.6, s=1, color=COLORS['secondary'])
    ax4.set_title("Mixed Joint Distribution p'(x,y)_x\n(Based on p'(x) and p(y|x))", 
                  fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax4.set_xlabel('X', fontsize=12, color=COLORS['dark'])
    ax4.set_ylabel('Y', fontsize=12, color=COLORS['dark'])
    ax4.set_xlim(x_lim)
    ax4.set_ylim(y_lim)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_facecolor(COLORS['light'])
    
    ax5 = axes[1, 1]
    ax5.scatter(mixed_joint_y_samples[:, 0].numpy(), mixed_joint_y_samples[:, 1].numpy(), 
               alpha=0.6, s=1, color=COLORS['accent'])
    ax5.set_title("Mixed Joint Distribution p'(x,y)_y\n(Based on p'(y) and p(x|y))", 
                  fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax5.set_xlabel('X', fontsize=12, color=COLORS['dark'])
    ax5.set_ylabel('Y', fontsize=12, color=COLORS['dark'])
    ax5.set_xlim(x_lim)
    ax5.set_ylim(y_lim)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.set_facecolor(COLORS['light'])
    
    ax6 = axes[1, 2]
    ax6.scatter(original_joint_samples[:, 0].numpy(), original_joint_samples[:, 1].numpy(), 
               alpha=0.3, s=0.5, color=COLORS['primary'], label='Original p(x,y)')
    ax6.scatter(mixed_joint_x_samples[:, 0].numpy(), mixed_joint_x_samples[:, 1].numpy(), 
               alpha=0.3, s=0.5, color=COLORS['secondary'], label="p'(x,y)_x")
    ax6.scatter(mixed_joint_y_samples[:, 0].numpy(), mixed_joint_y_samples[:, 1].numpy(), 
               alpha=0.3, s=0.5, color=COLORS['accent'], label="p'(x,y)_y")
    ax6.set_title('Three Joint Distributions Comparison', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax6.set_xlabel('X', fontsize=12, color=COLORS['dark'])
    ax6.set_ylabel('Y', fontsize=12, color=COLORS['dark'])
    ax6.set_xlim(x_lim)
    ax6.set_ylim(y_lim)
    ax6.legend()
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.set_facecolor(COLORS['light'])
    
    ax7 = axes[2, 0]
    ax7.hist(original_joint_samples[:, 0].numpy(), bins=50, alpha=0.7, label='Original p(x)', 
             color=COLORS['primary'], density=True, edgecolor='white')
    ax7.hist(mixed_joint_x_samples[:, 0].numpy(), bins=50, alpha=0.7, label="X from p'(x,y)_x", 
             color=COLORS['secondary'], density=True, edgecolor='white')
    ax7.hist(mixed_joint_y_samples[:, 0].numpy(), bins=50, alpha=0.7, label="X from p'(x,y)_y", 
             color=COLORS['accent'], density=True, edgecolor='white')
    ax7.set_title('X Marginal Distribution Detailed Comparison', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax7.set_xlabel('X Value', fontsize=12, color=COLORS['dark'])
    ax7.set_ylabel('Density', fontsize=12, color=COLORS['dark'])
    ax7.legend()
    ax7.grid(True, alpha=0.3, linestyle='--')
    ax7.set_facecolor(COLORS['light'])
    
    ax8 = axes[2, 1]
    ax8.hist(original_joint_samples[:, 1].numpy(), bins=50, alpha=0.7, label='Original p(y)', 
             color=COLORS['primary'], density=True, edgecolor='white')
    ax8.hist(mixed_joint_x_samples[:, 1].numpy(), bins=50, alpha=0.7, label="Y from p'(x,y)_x", 
             color=COLORS['secondary'], density=True, edgecolor='white')
    ax8.hist(mixed_joint_y_samples[:, 1].numpy(), bins=50, alpha=0.7, label="Y from p'(x,y)_y", 
             color=COLORS['accent'], density=True, edgecolor='white')
    ax8.set_title('Y Marginal Distribution Detailed Comparison', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax8.set_xlabel('Y Value', fontsize=12, color=COLORS['dark'])
    ax8.set_ylabel('Density', fontsize=12, color=COLORS['dark'])
    ax8.legend()
    ax8.grid(True, alpha=0.3, linestyle='--')
    ax8.set_facecolor(COLORS['light'])
    
    ax9 = axes[2, 2]
    stats_original = [
        original_joint_samples[:, 0].mean().item(), original_joint_samples[:, 0].std().item(),
        original_joint_samples[:, 1].mean().item(), original_joint_samples[:, 1].std().item()
    ]
    stats_mixed_x = [
        mixed_joint_x_samples[:, 0].mean().item(), mixed_joint_x_samples[:, 0].std().item(),
        mixed_joint_x_samples[:, 1].mean().item(), mixed_joint_x_samples[:, 1].std().item()
    ]
    stats_mixed_y = [
        mixed_joint_y_samples[:, 0].mean().item(), mixed_joint_y_samples[:, 0].std().item(),
        mixed_joint_y_samples[:, 1].mean().item(), mixed_joint_y_samples[:, 1].std().item()
    ]
    
    x_pos = np.arange(len(['X Mean', 'X Std', 'Y Mean', 'Y Std']))
    width = 0.25
    
    ax9.bar(x_pos - width, stats_original, width, label='Original p(x,y)', 
            color=COLORS['primary'], alpha=0.8)
    ax9.bar(x_pos, stats_mixed_x, width, label="p'(x,y)_x", 
            color=COLORS['secondary'], alpha=0.8)
    ax9.bar(x_pos + width, stats_mixed_y, width, label="p'(x,y)_y", 
            color=COLORS['accent'], alpha=0.8)
    
    ax9.set_title('Statistics Comparison', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax9.set_xlabel('Statistics', fontsize=12, color=COLORS['dark'])
    ax9.set_ylabel('Value', fontsize=12, color=COLORS['dark'])
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels(['X Mean', 'X Std', 'Y Mean', 'Y Std'])
    ax9.legend()
    ax9.grid(True, alpha=0.3, linestyle='--')
    ax9.set_facecolor(COLORS['light'])
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n📊 Analytical Distribution Construction Results:")
    print("=" * 70)
    print(f"Original p(x,y) - X: μ={stats_original[0]:.4f}, σ={stats_original[1]:.4f}")
    print(f"Original p(x,y) - Y: μ={stats_original[2]:.4f}, σ={stats_original[3]:.4f}")
    print(f"Mixed p'(x,y)_x - X: μ={stats_mixed_x[0]:.4f}, σ={stats_mixed_x[1]:.4f}")
    print(f"Mixed p'(x,y)_x - Y: μ={stats_mixed_x[2]:.4f}, σ={stats_mixed_x[3]:.4f}")
    print(f"Mixed p'(x,y)_y - X: μ={stats_mixed_y[0]:.4f}, σ={stats_mixed_y[1]:.4f}")
    print(f"Mixed p'(x,y)_y - Y: μ={stats_mixed_y[2]:.4f}, σ={stats_mixed_y[3]:.4f}")
    
    # Calculate correlation coefficients
    original_corr = torch.corrcoef(torch.stack([original_joint_samples[:, 0], 
                                               original_joint_samples[:, 1]]))[0, 1].item()
    mixed_x_corr = torch.corrcoef(torch.stack([mixed_joint_x_samples[:, 0], 
                                              mixed_joint_x_samples[:, 1]]))[0, 1].item()
    mixed_y_corr = torch.corrcoef(torch.stack([mixed_joint_y_samples[:, 0], 
                                              mixed_joint_y_samples[:, 1]]))[0, 1].item()
    
    print(f"Original distribution correlation: {original_corr:.4f}")
    print(f"Mixed distribution_x correlation: {mixed_x_corr:.4f}")
    print(f"Mixed distribution_y correlation: {mixed_y_corr:.4f}")

def compute_distribution_similarity_metrics(original_samples, generated_samples, num_bins=50):
    """
    Calculate various similarity metrics between two distributions
    
    Args:
        original_samples: Original distribution samples (N, 2)
        generated_samples: Generated distribution samples (N, 2)
        num_bins: Number of bins for histogram calculation
    
    Returns:
        dict: Dictionary containing various similarity metrics
    """
    print("\n📊 Computing distribution similarity metrics...")
    print("=" * 60)
    
    if isinstance(original_samples, torch.Tensor):
        original_samples = original_samples.cpu().numpy()
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.cpu().numpy()
    
    metrics = {}
    
    print("📈 Computing statistical measures...")
    original_x_mean, original_x_std = np.mean(original_samples[:, 0]), np.std(original_samples[:, 0])
    original_y_mean, original_y_std = np.mean(original_samples[:, 1]), np.std(original_samples[:, 1])
    generated_x_mean, generated_x_std = np.mean(generated_samples[:, 0]), np.std(generated_samples[:, 0])
    generated_y_mean, generated_y_std = np.mean(generated_samples[:, 1]), np.std(generated_samples[:, 1])
    mean_diff_x = abs(original_x_mean - generated_x_mean)
    mean_diff_y = abs(original_y_mean - generated_y_mean)
    std_diff_x = abs(original_x_std - generated_x_std)
    std_diff_y = abs(original_y_std - generated_y_std)
    
    metrics['statistical'] = {
        'original_x_mean': original_x_mean,
        'original_x_std': original_x_std,
        'original_y_mean': original_y_mean,
        'original_y_std': original_y_std,
        'generated_x_mean': generated_x_mean,
        'generated_x_std': generated_x_std,
        'generated_y_mean': generated_y_mean,
        'generated_y_std': generated_y_std,
        'mean_diff_x': mean_diff_x,
        'mean_diff_y': mean_diff_y,
        'std_diff_x': std_diff_x,
        'std_diff_y': std_diff_y
    }
    
    print("🔗 Computing correlation measures...")
    original_corr = np.corrcoef(original_samples[:, 0], original_samples[:, 1])[0, 1]
    generated_corr = np.corrcoef(generated_samples[:, 0], generated_samples[:, 1])[0, 1]
    corr_diff = abs(original_corr - generated_corr)
    
    metrics['correlation'] = {
        'original_correlation': original_corr,
        'generated_correlation': generated_corr,
        'correlation_difference': corr_diff
    }
    
    print("🌊 Computing 2D Wasserstein distances...")
    try:
        from scipy.stats import wasserstein_distance
        wasserstein_x = wasserstein_distance(original_samples[:, 0], generated_samples[:, 0])
        wasserstein_y = wasserstein_distance(original_samples[:, 1], generated_samples[:, 1])
        
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment
        
        n_samples = min(500, len(original_samples), len(generated_samples))
        original_sub = original_samples[np.random.choice(len(original_samples), n_samples, replace=False)]
        generated_sub = generated_samples[np.random.choice(len(generated_samples), n_samples, replace=False)]
        
        dist_matrix = cdist(original_sub, generated_sub, metric='euclidean')
        
        row_indices, col_indices = linear_sum_assignment(dist_matrix)
        
        wasserstein_2d = dist_matrix[row_indices, col_indices].sum() / n_samples
        
    except ImportError:
        print("⚠️ scipy.optimize not available, using 1D projection for 2D Wasserstein")
        original_2d_proj = np.sqrt(original_samples[:, 0]**2 + original_samples[:, 1]**2)
        generated_2d_proj = np.sqrt(generated_samples[:, 0]**2 + generated_samples[:, 1]**2)
        wasserstein_2d = wasserstein_distance(original_2d_proj, generated_2d_proj)
        wasserstein_x = wasserstein_distance(original_samples[:, 0], generated_samples[:, 0])
        wasserstein_y = wasserstein_distance(original_samples[:, 1], generated_samples[:, 1])
    
    metrics['wasserstein'] = {
        'wasserstein_x': wasserstein_x,
        'wasserstein_y': wasserstein_y,
        'wasserstein_2d': wasserstein_2d
    }
    
    print("🎯 Computing Maximum Mean Discrepancy (MMD)...")
    
    def rbf_kernel(X, Y, gamma=1.0):
        """RBF kernel function"""
        X_norm = np.sum(X**2, axis=1, keepdims=True)
        Y_norm = np.sum(Y**2, axis=1, keepdims=True)
        dist = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
        return np.exp(-gamma * dist)
    
    def mmd_rbf(X, Y, gamma=1.0):
        """Compute MMD distance"""
        XX = rbf_kernel(X, X, gamma)
        YY = rbf_kernel(Y, Y, gamma)
        XY = rbf_kernel(X, Y, gamma)
        
        mmd = np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
        return mmd
    
    n_samples = min(1000, len(original_samples), len(generated_samples))
    original_sub = original_samples[np.random.choice(len(original_samples), n_samples, replace=False)]
    generated_sub = generated_samples[np.random.choice(len(generated_samples), n_samples, replace=False)]
    
    mmd_2d = mmd_rbf(original_sub, generated_sub, gamma=1.0)
    
    metrics['mmd'] = {
        'mmd_2d': mmd_2d
    }
    
    print("⚡ Computing Energy Distance...")
    
    def energy_distance(X, Y):
        """Compute energy distance"""
        n, m = len(X), len(Y)
        XX_dist = 0
        for i in range(n):
            for j in range(i+1, n):
                XX_dist += np.linalg.norm(X[i] - X[j])
        XX_dist = 2 * XX_dist / (n * (n-1)) if n > 1 else 0
        
        YY_dist = 0
        for i in range(m):
            for j in range(i+1, m):
                YY_dist += np.linalg.norm(Y[i] - Y[j])
        YY_dist = 2 * YY_dist / (m * (m-1)) if m > 1 else 0
        
        XY_dist = 0
        for i in range(n):
            for j in range(m):
                XY_dist += np.linalg.norm(X[i] - Y[j])
        XY_dist = XY_dist / (n * m)
        
        return 2 * XY_dist - XX_dist - YY_dist
    
    energy_2d = energy_distance(original_sub, generated_sub)
    
    metrics['energy'] = {
        'energy_2d': energy_2d
    }
    
    return metrics

def print_similarity_metrics(metrics, logger=None):
    """
    Print distribution similarity metrics results and log them
    
    Args:
        metrics: Dictionary of similarity metrics
        logger: Logger instance, if None then only print to console
    """
    def log_and_print(message):
        print(message)
        if logger:
            logger.info(message)
    
    log_and_print("\n" + "="*80)
    log_and_print("📊 DISTRIBUTION SIMILARITY METRICS RESULTS")
    log_and_print("="*80)
    
    log_and_print("\n📈 Statistical Measures:")
    log_and_print("-" * 40)
    stat = metrics['statistical']
    log_and_print(f"X Mean - Original: {stat['original_x_mean']:.4f}, Generated: {stat['generated_x_mean']:.4f}, Diff: {stat['mean_diff_x']:.4f}")
    log_and_print(f"X Std  - Original: {stat['original_x_std']:.4f}, Generated: {stat['generated_x_std']:.4f}, Diff: {stat['std_diff_x']:.4f}")
    log_and_print(f"Y Mean - Original: {stat['original_y_mean']:.4f}, Generated: {stat['generated_y_mean']:.4f}, Diff: {stat['mean_diff_y']:.4f}")
    log_and_print(f"Y Std  - Original: {stat['original_y_std']:.4f}, Generated: {stat['generated_y_std']:.4f}, Diff: {stat['std_diff_y']:.4f}")
    
    log_and_print("\n🔗 Correlation Measures:")
    log_and_print("-" * 40)
    corr = metrics['correlation']
    log_and_print(f"Correlation - Original: {corr['original_correlation']:.4f}, Generated: {corr['generated_correlation']:.4f}, Diff: {corr['correlation_difference']:.4f}")
    
    log_and_print("\n🌊 Wasserstein Distances:")
    log_and_print("-" * 40)
    wass = metrics['wasserstein']
    log_and_print(f"Wasserstein X: {wass['wasserstein_x']:.4f}")
    log_and_print(f"Wasserstein Y: {wass['wasserstein_y']:.4f}")
    log_and_print(f"Wasserstein 2D: {wass['wasserstein_2d']:.4f}")
    
    # MMD
    log_and_print("\n🎯 Maximum Mean Discrepancy (MMD):")
    log_and_print("-" * 40)
    mmd = metrics['mmd']
    log_and_print(f"MMD 2D: {mmd['mmd_2d']:.4f}")
    
    log_and_print("\n⚡ Energy Distance:")
    log_and_print("-" * 40)
    energy = metrics['energy']
    log_and_print(f"Energy Distance 2D: {energy['energy_2d']:.4f}")
    
    log_and_print("="*80)

def visualize_similarity_metrics(metrics, target_name="Distribution Similarity"):
    """
    Visualize distribution similarity metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.patch.set_facecolor(COLORS['light'])
    
    ax1 = axes[0, 0]
    stat = metrics['statistical']
    categories = ['X Mean', 'X Std', 'Y Mean', 'Y Std']
    original_values = [stat['original_x_mean'], stat['original_x_std'], 
                      stat['original_y_mean'], stat['original_y_std']]
    generated_values = [stat['generated_x_mean'], stat['generated_x_std'], 
                       stat['generated_y_mean'], stat['generated_y_std']]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x_pos - width/2, original_values, width, label='Original', 
            color=COLORS['primary'], alpha=0.8)
    ax1.bar(x_pos + width/2, generated_values, width, label='Generated', 
            color=COLORS['secondary'], alpha=0.8)
    
    ax1.set_title('Statistical Measures Comparison', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax1.set_xlabel('Statistics', fontsize=12, color=COLORS['dark'])
    ax1.set_ylabel('Value', fontsize=12, color=COLORS['dark'])
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_facecolor(COLORS['light'])
    
    ax2 = axes[0, 1]
    distance_metrics = ['Wasserstein\n2D', 'MMD\n2D', 'Energy\n2D']
    distance_values = [metrics['wasserstein']['wasserstein_2d'], 
                      metrics['mmd']['mmd_2d'],
                      metrics['energy']['energy_2d']]
    
    colors = [COLORS['primary'], COLORS['info'], COLORS['warning']]
    
    bars = ax2.bar(range(len(distance_metrics)), distance_values, color=colors, alpha=0.8)
    ax2.set_title('2D Distance Measures', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax2.set_xlabel('Metrics', fontsize=12, color=COLORS['dark'])
    ax2.set_ylabel('Value', fontsize=12, color=COLORS['dark'])
    ax2.set_xticks(range(len(distance_metrics)))
    ax2.set_xticklabels(distance_metrics, rotation=45)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_facecolor(COLORS['light'])
    
    ax3 = axes[1, 0]
    corr = metrics['correlation']
    corr_categories = ['Original', 'Generated']
    corr_values = [corr['original_correlation'], corr['generated_correlation']]
    
    bars = ax3.bar(corr_categories, corr_values, color=[COLORS['primary'], COLORS['secondary']], alpha=0.8)
    ax3.set_title('Correlation Comparison', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax3.set_ylabel('Correlation Coefficient', fontsize=12, color=COLORS['dark'])
    ax3.set_ylim(-1, 1)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_facecolor(COLORS['light'])
    
    for bar, value in zip(bars, corr_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.03,
                f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    ax4 = axes[1, 1]
    wass_metrics = ['Wasserstein\nX', 'Wasserstein\nY']
    wass_values = [metrics['wasserstein']['wasserstein_x'], 
                   metrics['wasserstein']['wasserstein_y']]
    
    bars = ax4.bar(range(len(wass_metrics)), wass_values, color=[COLORS['primary'], COLORS['secondary']], alpha=0.8)
    ax4.set_title('1D Wasserstein Distances', fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax4.set_xlabel('Dimensions', fontsize=12, color=COLORS['dark'])
    ax4.set_ylabel('Wasserstein Distance', fontsize=12, color=COLORS['dark'])
    ax4.set_xticks(range(len(wass_metrics)))
    ax4.set_xticklabels(wass_metrics)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_facecolor(COLORS['light'])
    
    plt.tight_layout()
    plt.show()

def run_conditional_flow_matching_experiment(mu_x=0.0, sigma_x=1.0, mu_y=0.0, sigma_y=1.0, rho=0.7, mix_ratio=0.3, complexity_level='simple'):
    """
    Run M2PDE conditional Flow Matching experiment with full original data functionality
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    experiment_name = f"m2pde_{complexity_level}_mix_{mix_ratio}"
    logger, log_filename = setup_logging(experiment_name)
    
    print(f"\n🔬 Starting M2PDE Conditional Flow Matching Experiment")
    print(f"Parameters: μ_x={mu_x}, σ_x={sigma_x}, μ_y={mu_y}, σ_y={sigma_y}, ρ={rho}, mix_ratio={mix_ratio}")
    print("=" * 80)
    
    logger.info(f"Starting M2PDE Conditional Flow Matching Experiment")
    logger.info(f"Parameters: μ_x={mu_x}, σ_x={sigma_x}, μ_y={mu_y}, σ_y={sigma_y}, ρ={rho}, mix_ratio={mix_ratio}")
    logger.info(f"Complexity level: {complexity_level}")
    logger.info(f"Device: {device}")
    
    print("\n📐 Constructing analytical joint distribution...")
    analytical_joint = AnalyticalJointDistribution(mu_x, sigma_x, mu_y, sigma_y, rho, complexity_level=complexity_level)
    
    print("\n🎯 Constructing mixed marginal distributions...")
    mixed_marginal = MixedMarginalDistribution(analytical_joint, mix_ratio)
    
    print("\n📊 Sampling various distributions...")
    original_joint_samples = analytical_joint.sample_joint(10000)
    mixed_joint_x_samples = mixed_marginal.construct_mixed_joint_x(10000)
    mixed_joint_y_samples = mixed_marginal.construct_mixed_joint_y(10000)
    
    print("\n📈 Visualizing distribution construction results...")
    visualize_analytical_distributions(analytical_joint, mixed_marginal, 
                                     original_joint_samples, mixed_joint_x_samples, mixed_joint_y_samples)
    
    print("\n🚀 Training M2PDE conditional Flow Matching models on two datasets...")
    print("  - vtx model trained on p'(x,y)_x (mixed joint based on p'(x) and p(y|x))")
    print("  - vty model trained on p'(x,y)_y (mixed joint based on p'(y) and p(x|y))")
    model_vtx = FlowMatchingNetVtx(input_dim=1, cond_dim=1, hidden_dim=256)
    model_vty = FlowMatchingNetVty(input_dim=1, cond_dim=1, hidden_dim=256)
    losses_vtx, losses_vty = train_flow_matching_conditional_two_datasets(
        model_vtx, model_vty, mixed_joint_x_samples, mixed_joint_y_samples, 
        num_epochs=30, device=device
    )
    
    print("\n🎬 Generating samples with M2PDE algorithm...")
    generated_samples = sample_conditional_flow_matching(model_vtx, model_vty, mix_ratio=mix_ratio, 
                                                       num_samples=10000, num_steps=100, num_external_loops=1,
                                                       device=device, visualize=True, target_name="M2PDE Conditional")
    
    print("\n📊 Computing distribution similarity metrics...")
    similarity_metrics = compute_distribution_similarity_metrics(original_joint_samples, generated_samples)
    
    print_similarity_metrics(similarity_metrics, logger)
    
    print("\n📈 Visualizing similarity metrics...")
    visualize_similarity_metrics(similarity_metrics, "M2PDE Conditional Distribution Similarity")
    
    print("\n💾 Saving distribution plots to m2pde folder...")
    save_distribution_plots(original_joint_samples, generated_samples, "M2PDE Conditional", mix_ratio)
    print("\n📊 Visualizing M2PDE results...")
    visualize_2d_flow_matching_results(original_joint_samples, generated_samples, "M2PDE Conditional", losses_vtx, original_joint_samples)
    
    return (model_vtx, model_vty, analytical_joint, mixed_marginal, 
            original_joint_samples, mixed_joint_x_samples, mixed_joint_y_samples, 
            generated_samples, losses_vtx, losses_vty, similarity_metrics)

if __name__ == "__main__":
    print("🎨 M2PDE Conditional Flow Matching Experiment: Learning Joint Distribution from Conditional Probabilities")
    print("=" * 80)
    
    complexity_level = 'complex'
    
    print(f"\n🔬 Running M2PDE Conditional Flow Matching Experiment...")
    print("=" * 50)
    
    for mix_ratio in [0.1]:
        m2pde_results = run_conditional_flow_matching_experiment(
                mu_x=0.0, sigma_x=1.0, 
                mu_y=0.0, sigma_y=1.5, 
                rho=0.7, mix_ratio=mix_ratio, complexity_level=complexity_level
        )
    
    complexity_level = 'complex'
    
    print(f"\n🔬 Running M2PDE Conditional Flow Matching Experiment...")
    print("=" * 50)
    
    for mix_ratio in [0.3]:
        m2pde_results = run_conditional_flow_matching_experiment(
                mu_x=0.0, sigma_x=1.0, 
                mu_y=0.0, sigma_y=1.5, 
                rho=0.7, mix_ratio=mix_ratio, complexity_level=complexity_level
        )
    
    print("\n✅ M2PDE Conditional Flow Matching Experiment Completed!")