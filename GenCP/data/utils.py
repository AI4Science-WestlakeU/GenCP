import numpy as np
import h5py
import os
from tqdm import tqdm
from datetime import datetime

def load_trajectory_data(trajectory_id):
    """Load all data for a single trajectory"""
    data = {}
    state_file = os.path.join("state", f"{trajectory_id}.npz")
    if os.path.exists(state_file):
        with np.load(state_file) as state_data:
            state_array = state_data[state_data.files[0]]
            data['u'] = state_array[:, 0, :, :, 0]
            data['v'] = state_array[:, 0, :, :, 1]
            data['p'] = state_array[:, 0, :, :, 2] 
    sdf_file = os.path.join("sdf_data", f"sdf_{trajectory_id}.npz")
    if os.path.exists(sdf_file):
        with np.load(sdf_file) as sdf_data:
            sdf_array = sdf_data['sdf']
            data['sdf'] = sdf_array[:, :, :, 0]
    param_files = {
        'massR_C': os.path.join("massR_C", f"{trajectory_id}.npz"),
        'epi_C': os.path.join("epi_C", f"{trajectory_id}.npz"),
        'Re': os.path.join("Re", f"{trajectory_id}.npz")
    }
    for param_name, param_file in param_files.items():
        if os.path.exists(param_file):
            with np.load(param_file) as param_data:
                param_array = param_data[param_data.files[0]]
                data[param_name] = param_array
    return data

def create_h5_from_trajectory(trajectory_id, output_dir="integrated_h5_data"):
    """Convert a single trajectory to H5 format"""
    print(f"Processing trajectory {trajectory_id}")
    trajectory_data = load_trajectory_data(trajectory_id)
    if not trajectory_data:
        print(f"  Trajectory {trajectory_id} data incomplete, skipping")
        return None
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"trajectory_{trajectory_id}.h5")
    with h5py.File(output_file, 'w') as f:
        data_group = f.create_group('data')
        metadata_group = f.create_group('metadata')
        time_steps = trajectory_data['u'].shape[0]
        height, width = trajectory_data['u'].shape[1], trajectory_data['u'].shape[2]
        print(f"  Time steps: {time_steps}")
        print(f"  Grid size: {height} x {width}")
        chunk_size = (1, min(256, height), min(256, width))
        compression = 'gzip'
        compression_opts = 1
        for field_name in ['u', 'v', 'p', 'sdf']:
            if field_name in trajectory_data:
                field_data = trajectory_data[field_name]
                dataset = data_group.create_dataset(field_name,
                                                  shape=field_data.shape,
                                                  dtype=np.float32,
                                                  chunks=chunk_size,
                                                  compression=compression,
                                                  compression_opts=compression_opts)
                dataset[:] = field_data
                if field_name == 'u':
                    dataset.attrs['description'] = 'Velocity x-component'
                    dataset.attrs['units'] = 'm/s'
                elif field_name == 'v':
                    dataset.attrs['description'] = 'Velocity y-component'
                    dataset.attrs['units'] = 'm/s'
                elif field_name == 'p':
                    dataset.attrs['description'] = 'Pressure field'
                    dataset.attrs['units'] = 'Pa'
                elif field_name == 'sdf':
                    dataset.attrs['description'] = 'Signed Distance Function'
                    dataset.attrs['units'] = 'm'
        params_group = data_group.create_group('parameters')
        for param_name in ['massR_C', 'epi_C', 'Re']:
            if param_name in trajectory_data:
                param_data = trajectory_data[param_name]
                params_group.create_dataset(param_name, data=param_data)
        metadata_group.attrs['trajectory_id'] = trajectory_id
        metadata_group.attrs['time_steps'] = time_steps
        metadata_group.attrs['grid_height'] = height
        metadata_group.attrs['grid_width'] = width
        metadata_group.attrs['creation_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_group.attrs['description'] = 'Double cylinder flow simulation trajectory'
        grid_group = metadata_group.create_group('grid')
        x_coords = np.linspace(0, width, width)
        y_coords = np.linspace(0, height, height)
        grid_group.create_dataset('x_coordinates', data=x_coords)
        grid_group.create_dataset('y_coordinates', data=y_coords)
    print(f"  Saved: {output_file}")
    return output_file

def integrate_all_trajectories():
    """Integrate all trajectories into H5 format"""
    print("="*60)
    print("Integrating trajectory data into H5 format")
    print("="*60)
    state_dir = "state"
    trajectory_ids = []
    for file in sorted(os.listdir(state_dir)):
        if file.endswith('.npz'):
            trajectory_id = file.replace('.npz', '')
            trajectory_ids.append(trajectory_id)
    print(f"Found {len(trajectory_ids)} trajectories")
    output_files = []
    for trajectory_id in trajectory_ids:
        output_file = create_h5_from_trajectory(trajectory_id)
        if output_file:
            output_files.append(output_file)
    print(f"\nIntegration complete!")
    print(f"Successfully processed {len(output_files)} trajectories")
    print(f"Output directory: integrated_h5_data")
    return output_files

def verify_h5_file(h5_file):
    """Verify H5 file contents"""
    print(f"\nVerifying file: {h5_file}")
    with h5py.File(h5_file, 'r') as f:
        print(f"File structure:")
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}, Shape: {obj.shape}, Dtype: {obj.dtype}")
                if 'description' in obj.attrs:
                    print(f"    Description: {obj.attrs['description']}")
            elif isinstance(obj, h5py.Group):
                print(f"  Group: {name}")
        f.visititems(print_structure)
        print(f"\nMetadata:")
        metadata = f['metadata']
        for key, value in metadata.attrs.items():
            print(f"  {key}: {value}")
            
if __name__ == "__main__":
    output_files = integrate_all_trajectories()
    if output_files:
        verify_h5_file(output_files[0]) 