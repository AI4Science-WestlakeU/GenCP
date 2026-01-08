import torch
import numpy as np
import re

import os
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
printed_keys = set()

class RangeNormalizer(object):
    def __init__(self, dataset, mode='train', batch_size=512):
        super(RangeNormalizer, self).__init__()
        dataset_path = os.path.dirname(os.path.dirname(dataset.files_path[0]))
        
        try:
            self.max_inputs, self.min_inputs, self.max_targets, self.min_targets \
                                                = torch.load(os.path.join(dataset_path, f"max_min_{mode}.pt"))
        except:
            raise ValueError(f"max_min_{mode}.pt not found")
            # self.max_inputs, self.min_inputs, self.max_targets, self.min_targets \
            #                                     = self.compute_min_max(dataset, batch_size)
            # torch.save((self.max_inputs, self.min_inputs, self.max_targets, self.min_targets), 
            #            os.path.join(dataset_path, f"max_min_{mode}.pt"))
        
        self.range_inputs = self.max_inputs - self.min_inputs
        self.range_targets = self.max_targets - self.min_targets
        
        self.range_inputs = torch.where(self.range_inputs == 0, torch.ones_like(self.range_inputs), self.range_inputs)
        self.range_targets = torch.where(self.range_targets == 0, torch.ones_like(self.range_targets), self.range_targets)

    def preprocess(self, x, y):
        x_norm = 2 * (x - self.min_inputs) / self.range_inputs - 1
        y_norm = 2 * (y - self.min_targets) / self.range_targets - 1
        return x_norm, y_norm

    def postprocess(self, x, y):
        x_denorm = (x + 1) * self.range_inputs / 2 + self.min_inputs
        y_denorm = (y + 1) * self.range_targets / 2 + self.min_targets
        return x_denorm, y_denorm

    def compute_min_max(self, dataset, batch_size):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        max_inputs, min_inputs = None, None
        max_targets, min_targets = None, None
        
        for inputs, targets, grid_x, grid_y, attrs in tqdm(loader, desc="Computing min and max"):
            c1, c2 = inputs.size(-1), targets.size(-1)
            inputs = inputs.view(-1, c1)
            targets = targets.view(-1, c2)

            batch_max_inputs = inputs.max(dim=0)[0]
            batch_min_inputs = inputs.min(dim=0)[0]
            batch_max_targets = targets.max(dim=0)[0]
            batch_min_targets = targets.min(dim=0)[0]

            if max_inputs is None:
                max_inputs, min_inputs = batch_max_inputs, batch_min_inputs
                max_targets, min_targets = batch_max_targets, batch_min_targets
            else:
                max_inputs = torch.max(max_inputs, batch_max_inputs)
                min_inputs = torch.min(min_inputs, batch_min_inputs)
                max_targets = torch.max(max_targets, batch_max_targets)
                min_targets = torch.min(min_targets, batch_min_targets)

        return max_inputs, min_inputs, max_targets, min_targets
    

class DoubleCylinderDataset(Dataset):

    def __init__(self, dataset_path, length=50, input_size=1, output_size=1, stride=10, mode='train', stage='fluid', num_delta_t=1, dt=1, use_cache=True, cache_dir_name=None):
        """
        Initialize FluidZero dataset

        Args:
            dataset_path: Dataset root path
            length: Sequence length
            input_size: Input sequence length
            output_size: Output sequence length
            stride: Sliding window stride
            mode: Dataset mode ('train', 'val', 'test')
            stage: Stage ('fluid', 'structure', 'couple')
            num_delta_t: Number of timesteps between input and output
            dt: Timestep interval (for non-continuous sampling)
            use_cache: Whether to use cached data
            cache_dir_name: Cache directory name, auto-generated if None
        """
        # Check if we should use cached data
        self.use_cache = use_cache
        if use_cache:
            from utils import check_cache_exists, generate_cache_dir_name
            from .cached_dataset import CachedDataset

            if cache_dir_name is None:
                # Create a dummy args object with the needed parameters
                class DummyArgs:
                    def __init__(self, length, input_step, output_step, stride, stage, num_delta_t, dt):
                        self.length = length
                        self.input_step = input_step
                        self.output_step = output_step
                        self.stride = stride
                        self.stage = stage
                        self.num_delta_t = num_delta_t
                        self.dt = dt

                dummy_args = DummyArgs(length, input_size, output_size, stride, stage, num_delta_t, dt)
                cache_dir_name = generate_cache_dir_name(dummy_args, "double_cylinder")

            if check_cache_exists(cache_dir_name, mode):
                print(f"Using cached dataset for {mode} mode")
                # Replace self with CachedDataset instance
                cached_dataset = CachedDataset(cache_dir_name, mode)
                # Copy all attributes from cached dataset
                self.__dict__.update(cached_dataset.__dict__)
                self.__class__ = CachedDataset
                return

        if stage == 'fluid':
            file_path = 'double_cylinder/flow_condition_on_cylinder/' 
        elif stage == 'structure':
            file_path = 'double_cylinder/cylinder_condition_on_flow/' # couple
        elif stage == 'couple' or stage == 'joint':
            file_path = 'double_cylinder/couple/' 
        else:
            raise ValueError(f"Invalid stage: {stage}")
        
        self.file_path = file_path
        self.mode = mode
        self.length = length
        self.stride = stride
        self.files_path = os.listdir(os.path.join(dataset_path, file_path, mode))
        self.files_path.sort()
        self.files_path = [os.path.join(dataset_path, file_path, mode, file) for file in self.files_path]
        self.input_size = input_size
        self.output_size = output_size
        self.stage = stage
        self.num_delta_t = num_delta_t
        self.dt = dt
        self.single_file_num_samples = self.get_num_samples()
        self.num_samples = self.single_file_num_samples * len(self.files_path)
    
    
    def get_num_samples(self):
        """Calculate number of samples in a single file, considering dt interval"""
        required_length = self.input_size * self.dt + self.num_delta_t + self.output_size * self.dt
        return max(0, (self.length - required_length) // self.stride + 1)
        # return (self.length - (self.input_size + self.output_size)) // self.stride + 1

    def get_file_index(self, index):
        """Get file index from global index"""
        return index // self.single_file_num_samples
    
    def get_sample_index(self, index):
        """Get sample index from global index"""
        return index % self.single_file_num_samples
    
    def split_data(self, data, start, end, dt):
        """
        Override split_data method to adapt to FluidZero data format
        Extract pressure, velocity fields and SDF data
        """
        x_velocity = torch.from_numpy(data['data']['velocity_x'][start:end:dt]) # .permute(0,2,1)
        y_velocity = torch.from_numpy(data['data']['velocity_y'][start:end:dt]) # .permute(0,2,1)
        pressure = torch.from_numpy(data['data']['pressure'][start:end:dt]) # .permute(0,2,1)
        # print("pressure:",pressure.max(), pressure.min())
        # sdf = torch.from_numpy(data['data']['sdf'][start:end]).permute(0,2,1)/100
        sdf = torch.from_numpy(data['data']['sdf'][start:end:dt])/100
        
        sdf_mask = torch.where(sdf > 0.02, torch.tensor(1.0), torch.tensor(0.0))
        x_velocity = x_velocity * sdf_mask
        y_velocity = y_velocity * sdf_mask
        pressure = pressure * sdf_mask

        sdf = sdf[:, 1:129, 1:129]
        x_velocity = x_velocity[:, 1:129, 1:129]
        y_velocity = y_velocity[:, 1:129, 1:129]
        pressure = pressure[:, 1:129, 1:129]
        
        return torch.stack([x_velocity, y_velocity, pressure, sdf], dim=-1)
    
    def get_sample(self, data, i):
        """Get input and output samples"""
        x_start = i * self.stride
        x_end = x_start + self.input_size * self.dt
        y_start = x_end + self.num_delta_t
        y_end = y_start + self.output_size * self.dt
        x = self.split_data(data, x_start, x_end, self.dt)
        y = self.split_data(data, y_start, y_end, self.dt)

        return x, y
    
    def get_metadata(self, f):
        """
        Override get_metadata method to adapt to FluidZero metadata format
        """
        grid_x = torch.from_numpy(f['metadata']['grid']['x_coordinates'][:])
        grid_y = torch.from_numpy(f['metadata']['grid']['y_coordinates'][:])
        attrs = {}
        for key, value in f['metadata'].attrs.items():
            if key not in ['trajectory_id', 'creation_time', 'data_type'] and not isinstance(value, str):
                value = np.array([value])
                attrs[key] = torch.from_numpy(value).float()   
                
        return grid_x, grid_y, attrs

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        """Get a sample from the dataset"""
        file_index = self.get_file_index(index)
        sample_index = self.get_sample_index(index)
        file_path = self.files_path[file_index]
        
        with h5py.File(file_path, 'r') as f:
            x, y = self.get_sample(f, sample_index)
            grid_x, grid_y, attrs = self.get_metadata(f)
            
        return x, y, grid_x, grid_y, attrs
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import os
    import h5py
    
    dataset = DoubleCylinderDataset(
        dataset_path='/path/to/dataset',
        length=50,
        input_size=1,
        output_size=1,
        stride=1,
        mode='train',
        dt=3
    )
    print(f"Dataset length: {len(dataset)}")
    print(f"Input shape: {dataset[0][0].shape}")
    print(f"Output shape: {dataset[0][1].shape}")
    print(f"Grid X shape: {dataset[0][2].shape}")
    print(f"Grid Y shape: {dataset[0][3].shape}")
    print(f"Attributes: {dataset[0][4]}") 
    
    x, _, grid_x, grid_y, _ = dataset[0]
    # x: [input_size, H, W, 4]
    x0 = x[0]  # [H, W, 4]
    u = x0[..., 0].cpu().numpy()
    v = x0[..., 1].cpu().numpy()
    p = x0[..., 2].cpu().numpy()
    sdf = x0[..., 3].cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    im0 = axes[0, 0].imshow(u, cmap='RdBu_r')
    axes[0, 0].set_title('u')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(v, cmap='RdBu_r')
    axes[0, 1].set_title('v')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(p, cmap='viridis')
    axes[1, 0].set_title('p')
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(sdf, cmap='coolwarm')
    axes[1, 1].set_title('sdf')
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualization_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'test_field_turek_hron_data.png')
    plt.savefig(out_path, dpi=200)
    print(f"Saved visualization to: {out_path}")