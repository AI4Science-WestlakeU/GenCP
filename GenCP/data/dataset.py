import numpy as np
import torch
import os
import sys
import pdb
import h5py
from torch.utils.data import Dataset

class BaseDataset(Dataset):
    """
    Base dataset class for fluid datasets.
    All specific datasets inherit from this class.
    """
    def __init__(self, file_path, dataset_path, length, input_size, output_size, stride, mode='train', stage='fluid', num_delta_t=0, dt=1):
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
        """Calculate number of samples in a single file, considering dt interval."""
        required_length = self.input_size * self.dt + self.num_delta_t + self.output_size * self.dt
        return max(0, (self.length - required_length) // self.stride + 1)
        # return (self.length - (self.input_size + self.output_size)) // self.stride + 1

    def get_file_index(self, index):
        """Get file index from global index."""
        return index // self.single_file_num_samples
    
    def get_sample_index(self, index):
        """Get sample index from global index."""
        return index % self.single_file_num_samples
    
    def split_data(self, data, start, end, dt):
        """
        Extract fluid field data from dataset.
        Subclasses can override this method to adapt to different data formats.
        """
        
        # if type == "x" or self.mode != "train":
        #     x_velocity = torch.from_numpy(data['data']['velocity_x'][start:end])
        #     y_velocity = torch.from_numpy(data['data']['velocity_y'][start:end])
        #     sdf = torch.from_numpy(data['data']['sdf'][start:end])
        #     pressure = torch.from_numpy(data['data']['pressure'][start:end])
        # else:    
        #     if self.stage == 'fluid':
        #         x_velocity = torch.from_numpy(data['data']['velocity_x'][start:end])
        #         y_velocity = torch.from_numpy(data['data']['velocity_y'][start:end])
        #         sdf = torch.from_numpy(data['data']['sdf'][start:end])
        #         pressure = torch.from_numpy(data['data']['pressure'][start:end])
        #     elif self.stage == 'structure':
        #         x_velocity = torch.from_numpy(data['data']['velocity_x'][start:end])
        #         y_velocity = torch.from_numpy(data['data']['velocity_y'][start:end])
        #         sdf = torch.from_numpy(data['data']['sdf'][start:end])
        #         pressure = torch.from_numpy(data['data']['pressure'][start:end])    
                
        x_velocity = torch.from_numpy(data['data']['velocity_x'][start:end])
        y_velocity = torch.from_numpy(data['data']['velocity_y'][start:end])
        sdf = torch.from_numpy(data['data']['sdf'][start:end])
        pressure = torch.from_numpy(data['data']['pressure'][start:end])
        
        return torch.stack([x_velocity, y_velocity, pressure, sdf], dim=-1)
    
    def get_sample(self, data, i):
        """Get input and output samples."""
        x_start = i * self.stride
        x_end = x_start + self.input_size * self.dt
        y_start = x_end + self.num_delta_t
        y_end = y_start + self.output_size * self.dt
        x = self.split_data(data, x_start, x_end, self.dt)
        y = self.split_data(data, y_start, y_end, self.dt)

        return x, y

    def get_metadata(self, f):
        """
        Get metadata.
        Subclasses can override this method to adapt to different metadata formats.
        """
        grid_x = torch.from_numpy(f['metadata']['grid']['x_coordinates'][:])
        grid_y = torch.from_numpy(f['metadata']['grid']['y_coordinates'][:])
        attrs = {}
        for key, value in f['metadata'].attrs.items():
            if key != "creation_time":
                value = np.array([value])
                attrs[key] = torch.from_numpy(value).float()
        return grid_x, grid_y, attrs

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        """Get a sample from the dataset."""
        file_index = self.get_file_index(index)
        sample_index = self.get_sample_index(index)
        file_path = self.files_path[file_index]
        
        with h5py.File(file_path, 'r') as f:
            x, y = self.get_sample(f, sample_index)
            grid_x, grid_y, attrs = self.get_metadata(f)
            
        return x, y, grid_x, grid_y, attrs
            
        
if __name__ == '__main__':
    print("BaseDataset base class created")
    print("Use specific dataset implementation classes, e.g., TurekHronDataset")
    