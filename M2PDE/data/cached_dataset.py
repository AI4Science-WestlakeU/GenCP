import torch
import os
import pickle
from torch.utils.data import Dataset


class CachedDataset(Dataset):
    """Dataset class that loads preprocessed data from cache"""

    def __init__(self, cache_dir_name, mode='train'):
        """
        Initialize cached dataset

        Args:
            cache_dir_name: Name of cache directory
            mode: 'train', 'val', or 'test'
        """
        from utils import get_cache_path

        self.cache_path = get_cache_path(cache_dir_name, mode)
        self.mode = mode

        # Load metadata
        metadata_path = os.path.join(self.cache_path, 'metadata.pkl')
        if not os.path.exists(metadata_path):
            raise ValueError(f"Cache metadata not found at {metadata_path}")

        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        self.num_samples = self.metadata['length']
        self.files_path = self.metadata['files_path']  # Add files_path for RangeNormalizer compatibility

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """Load cached sample"""
        from utils import load_cached_sample

        sample_data = load_cached_sample(self.cache_path, index)

        # Return in same format as original dataset: (x, y, grid_x, grid_y, attrs)
        return (
            sample_data['x'],
            sample_data['y'],
            sample_data['grid_x'],
            sample_data['grid_y'],
            sample_data['attrs']
        )
