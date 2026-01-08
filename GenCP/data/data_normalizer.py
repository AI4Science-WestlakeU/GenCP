import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class IdentityNormalizer(object):
    def __init__(self, device):
        super(IdentityNormalizer, self).__init__()
        self.device = device

    def preprocess(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        return x, y

    def postprocess(self, x, y):
        x, y = x.to(self.device), y.to(self.device)
        return x, y
    
class GaussianNormalizer(object):
    def __init__(self, dataset, device, mode='train', batch_size=512):
        super(GaussianNormalizer, self).__init__()
        self.device = device
        # print([attr for attr in dir(dataset) if not attr.startswith('__')])
        dataset_path = os.path.dirname(os.path.dirname(dataset.files_path[0]))
        
        try:
            self.mean_inputs, self.mean_targets, self.std_inputs, self.std_targets \
                                                = torch.load(os.path.join(dataset_path, f"mean_std_{mode}.pt"))
        except:
            self.mean_inputs, self.mean_targets, self.std_inputs, self.std_targets \
                                                = self.compute_mean_std(dataset, batch_size)
            torch.save((self.mean_inputs, self.mean_targets, self.std_inputs, self.std_targets), 
                       os.path.join(dataset_path, f"mean_std_{mode}.pt"))

        self.mean_inputs = self.mean_inputs.to(device)
        self.mean_targets = self.mean_targets.to(device)
        self.std_inputs = self.std_inputs.to(device)
        self.std_targets = self.std_targets.to(device)
        self.std_inputs = torch.where(self.std_inputs==0, torch.ones_like(self.std_inputs), self.std_inputs)
        self.std_targets = torch.where(self.std_targets==0, torch.ones_like(self.std_targets), self.std_targets)

    def preprocess(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        x_norm = (x - self.mean_inputs) / self.std_inputs
        y_norm = (y - self.mean_targets) / self.std_targets
        return x_norm, y_norm

    def postprocess(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        x_denorm = x * self.std_inputs + self.mean_inputs
        y_denorm = y * self.std_targets + self.mean_targets
        return x_denorm, y_denorm

    def compute_mean_std(self, dataset, batch_size):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=12)

        n_samples = 0
        mean_inputs, mean_targets = 0., 0.
        var_inputs, var_targets = 0., 0. 
        for inputs, targets, grid_x, grid_y, attrs in tqdm(loader, desc="Computing mean and std"):
            b, c1, c2 = inputs.size(0), inputs.size(-1), targets.size(-1)
            inputs = inputs.view(b, -1, c1)
            targets = targets.view(b, -1, c2)

            batch_mean = inputs.mean(dim=1)
            batch_var = inputs.var(dim=(0,1), unbiased=False)
            mean_inputs += batch_mean.sum(0)
            var_inputs += batch_var * b

            batch_mean = targets.mean(dim=1)
            batch_var = targets.var(dim=(0,1), unbiased=False)
            mean_targets += batch_mean.sum(0)
            var_targets += batch_var * b

            n_samples += b

        mean_inputs = mean_inputs / n_samples
        mean_targets = mean_targets / n_samples
        var_inputs = var_inputs / n_samples
        var_targets = var_targets / n_samples

        std_inputs = var_inputs ** 0.5
        std_targets = var_targets ** 0.5

        return mean_inputs, mean_targets, std_inputs, std_targets


class RangeNormalizer(object):
    def __init__(self, dataset, device, mode='train', batch_size=512):
        super(RangeNormalizer, self).__init__()
        self.device = device
        dataset_path = os.path.dirname(os.path.dirname(dataset.files_path[0]))
        
        try:
            self.max_inputs, self.min_inputs, self.max_targets, self.min_targets \
                                                = torch.load(os.path.join(dataset_path, f"max_min_{mode}.pt"))
        except:
            self.max_inputs, self.min_inputs, self.max_targets, self.min_targets \
                                                = self.compute_min_max(dataset, batch_size)
            torch.save((self.max_inputs, self.min_inputs, self.max_targets, self.min_targets), 
                       os.path.join(dataset_path, f"max_min_{mode}.pt"))
        
        self.max_inputs = self.max_inputs.to(device)
        self.min_inputs = self.min_inputs.to(device)
        self.max_targets = self.max_targets.to(device)
        self.min_targets = self.min_targets.to(device)
        
        self.range_inputs = self.max_inputs - self.min_inputs
        self.range_targets = self.max_targets - self.min_targets
        
        self.range_inputs = torch.where(self.range_inputs == 0, torch.ones_like(self.range_inputs), self.range_inputs)
        self.range_targets = torch.where(self.range_targets == 0, torch.ones_like(self.range_targets), self.range_targets)

    def preprocess(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        x_norm = 2 * (x - self.min_inputs) / self.range_inputs - 1
        y_norm = 2 * (y - self.min_targets) / self.range_targets - 1
        return x_norm, y_norm

    def postprocess(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
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
