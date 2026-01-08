import torch
import numpy as np
from .dataset import BaseDataset

class TurekHronDataset(BaseDataset):
    """
    Turek-Hron dataset class for FSI data.
    Inherits from BaseDataset, handles fluidzero_data format.
    """
    
    def __init__(self, dataset_path, length=50, input_size=1, output_size=1, stride=10, mode='train', stage='fluid', num_delta_t=1, dt=1):
        """
        Initialize Turek-Hron dataset.
        
        Args:
            dataset_path: Dataset root path
            length: Sequence length
            input_size: Input sequence length
            output_size: Output sequence length
            stride: Sliding window stride
            mode: Dataset mode ('train', 'val', 'test')
            stage: Stage ('fluid', 'structure', 'couple')
            num_delta_t: Time steps between input and output
            dt: Time step interval (for non-continuous sampling)
        """
        if stage == 'fluid':
            file_path = 'turek_hron/flow_condition_on_beam/' 
        elif stage == 'structure':
            file_path = 'turek_hron/beam_condition_on_flow/'
        elif stage == 'couple' or stage == 'joint':
            file_path = 'turek_hron/couple/' 
        else:
            raise ValueError(f"Invalid stage: {stage}")
        super().__init__(file_path, dataset_path, length, input_size, output_size, stride, mode, stage, num_delta_t, dt)
    
    def split_data(self, data, start, end, dt):
        """
        Override split_data to adapt to FluidZero data format.
        Extract pressure, velocity fields and SDF data.
        """
        x_velocity = torch.from_numpy(data['data']['velocity_x'][start:end:dt])
        y_velocity = torch.from_numpy(data['data']['velocity_y'][start:end:dt])
        pressure = torch.from_numpy(data['data']['pressure'][start:end:dt])
        sdf = torch.from_numpy(data['data']['sdf'][start:end:dt])/100
        
        sdf_mask = torch.where(sdf > 0.02, torch.tensor(1.0), torch.tensor(0.0))
        x_velocity = x_velocity * sdf_mask
        y_velocity = y_velocity * sdf_mask
        pressure = pressure * sdf_mask

        sdf = sdf[:, 26:134, 66:154]
        x_velocity = x_velocity[:, 26:134, 66:154]
        y_velocity = y_velocity[:, 26:134, 66:154]
        pressure = pressure[:, 26:134, 66:154]
        
        return torch.stack([x_velocity, y_velocity, pressure, sdf], dim=-1)
    
    def get_metadata(self, f):
        """
        Override get_metadata to adapt to FluidZero metadata format.
        """
        grid_x = torch.from_numpy(f['metadata']['grid']['x_coordinates'][:])
        grid_y = torch.from_numpy(f['metadata']['grid']['y_coordinates'][:])
        attrs = {}
        for key, value in f['metadata'].attrs.items():
            if key not in ['trajectory_id', 'creation_time', 'data_type'] and not isinstance(value, str):
                value = np.array([value])
                attrs[key] = torch.from_numpy(value).float()   
                
        return grid_x, grid_y, attrs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import os
    import h5py
    
    dataset = TurekHronDataset(
        dataset_path='/path/to/gencp_dataset',
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
    x0 = x[0]
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