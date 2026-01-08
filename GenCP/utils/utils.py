import torch
import numpy as np
import yaml
import os
import datetime
import logging

from einops import rearrange
from itertools import product
import pdb

# Conditional import for TensorBoard (avoid environment issues)
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except (ImportError, ValueError) as e:
    print(f"Warning: TensorBoard not available: {e}")
    HAS_TENSORBOARD = False
    SummaryWriter = None
from torchvision.datasets.utils import download_url
pretrained_models = {'SiT-XL-2-256x256.pt'}

# config utils
def add_args_from_config(parser):
    # Parse CLI first to get --config path
    args_cli = parser.parse_args()
    existing_args = {action.dest for action in parser._actions}

    with open(args_cli.config, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader) or {}

    # For keys already defined in parser, override defaults with config values
    for key, value in config.items():
        if key in existing_args:
            parser.set_defaults(**{key: value})
        else:
            # For new keys, add them as arguments
            parser.add_argument(f'--{key}', type=type(value), default=value)

    # Parse again with updated defaults to produce final args
    return parser.parse_args()


# def save_config_from_args(args):
#     config_dict = {k: v for k, v in vars(args).items() if k != 'config'}
    
#     config_dir = args.results_path
#     os.makedirs(config_dir, exist_ok=True)  # Ensure the directory exists
#     config_file_path = os.path.join(config_dir, 'config.yaml')
#     with open(config_file_path, 'w') as file:
#         yaml.dump(config_dict, file, default_flow_style=False)


#training utils
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(exp_path, is_use_tb, rank=0):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = os.path.join(exp_path, f"training_{current_time}.log")
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    if rank == 0:
        file_handler = logging.FileHandler(log_filename, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    logger.propagate = False
    
    if is_use_tb and rank == 0:
        writer = SummaryWriter(log_dir=exp_path, filename_suffix=f"training_tf{current_time}")
    else:
        writer = None
    return writer, logger

def rel_l2_loss(pred, target):
    b = pred.size(0)
    pred = pred.reshape(b, -1)
    target = target.reshape(b, -1)
    return (torch.norm(pred - target, dim=1) / torch.norm(target, dim=1))

def mse_loss(pred, target):
    """Returns per-sample MSE with shape [batch]."""
    b = pred.size(0)
    pred = pred.reshape(b, -1)
    target = target.reshape(b, -1)
    return torch.mean((pred - target) ** 2, dim=1)

def apply_mask(self, xt, x0):
    if self.stage == 'fluid':
        xt_mask = torch.ones_like(xt)
        xt_mask[..., -1:] = 0 # reserve u_noise, v_noise, pressure_noise
        xt = xt * xt_mask 
        x0_mask = torch.ones_like(x0)
        # x0_mask[..., -1:] = 0 # u, v, sdf, pressure
        x0 = x0 * x0_mask
        
    elif self.stage == 'structure':
        xt_mask = torch.ones_like(xt)
        xt_mask[..., :-1] = 0 # reserve sdf_noise
        xt = xt * xt_mask
        x0_mask = torch.ones_like(x0) 
        x0_mask[..., :-2] = 0 # reserve sdf, pressure
        x0 = x0 * x0_mask
        
    elif self.stage == 'fsi':
        xt = xt
        x0 = x0
        
    return xt, x0
    
def loss_with_mask(pred, target, loss, channel, stage="fluid", dataset_name='turek_hron_data'):
    """
    Compute masked loss for different datasets and stages.
    
    For NTcouple: Joint evolution paradigm, only compute loss on target portion
    For DoubleCylinder/TurekHron: Selectively mask different channels based on stage
    """
    
    if dataset_name == 'ntcouple':
        if channel >= 1:
            target_target = target[..., -channel:]
            
            b = target_target.size(0)
            pred_flat = pred.reshape(b, -1)
            target_flat = target_target.reshape(b, -1)
            
            return loss(pred_flat, target_flat).mean()
        else:
            raise ValueError(f"Invalid channel={channel} for ntcouple")
    elif channel == 3 or channel == 1:
        if stage == "fluid":
            mask = torch.ones_like(pred)
            target = target[..., :-1]
            # mask[..., -1:] = 0 # reserve u_noise, v_noise, pressure_noise
        elif stage == "structure":
            mask = torch.ones_like(pred)
            target = target[..., -1:]
            # mask[..., :-1] = 0 # reserve sdf_noise
        elif stage == "joint":
            mask = torch.ones_like(pred)
    elif channel == 4:
        if stage == "fluid":
            mask = torch.ones_like(pred)
            mask[..., -1:] = 0 # reserve u_noise, v_noise, pressure_noise
        elif stage == "structure":
            mask = torch.ones_like(pred)
            mask[..., :-1] = 0 # reserve sdf_noise
        elif stage == "joint":
            mask = torch.ones_like(pred)
    else:
        # Default case for other channel counts
        mask = torch.ones_like(pred)
                    
    # print(pred.shape, target.shape, mask.shape)
    b = pred.size(0)
    # print(pred.shape, target.shape, mask.shape)
    pred = (pred*mask).reshape(b, -1)
    target = (target*mask).reshape(b, -1)
    
    return loss(pred, target).mean()
            

def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP", "cos", "Cos"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)
    
def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument("--sampling-method", type=str, default="dopri5", help="blackbox ODE solver methods; for full list check https://github.com/rtqichen/torchdiffeq")
    group.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance")
    group.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance")
    group.add_argument("--reverse", action="store_true")
    group.add_argument("--likelihood", action="store_true")

def parse_sde_args(parser):
    group = parser.add_argument_group("SDE arguments")
    group.add_argument("--sampling-method", type=str, default="Euler", choices=["Euler", "Heun"])
    group.add_argument("--diffusion-form", type=str, default="constant", \
                        choices=["constant", "SBDM", "sigma", "linear", "decreasing", "increasing", "increasing-decreasing"],\
                        help="form of diffusion coefficient in the SDE")
    group.add_argument("--diffusion-coeff", type=float, default=0.1, \
                        help="diffusion coefficient strength for SDE sampling")
    group.add_argument("--diffusion-norm", type=float, default=1.0)
    group.add_argument("--last-step", type=str, default="euler", choices=["euler", "deterministic", "Mean", "Tweedie"],\
                        help="form of last step taken in the SDE")
    group.add_argument("--last-step-size", type=float, default=0.04, \
                        help="size of the last step taken")
    
def find_model(model_name):
    """
    Finds a pre-trained SiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    # if model_name in pretrained_models:  
    #     return download_model(model_name)
    # else:  
    #     assert os.path.isfile(model_name), f'Could not find SiT checkpoint at {model_name}'
    print(f"Loading model from {model_name}")
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint
    
def download_model(model_name):
    """
    Downloads a pre-trained SiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f'pretrained_models/{model_name}'
    if not os.path.isfile(local_path):
        os.makedirs('pretrained_models', exist_ok=True)
        web_path = f'https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0'
        download_url(web_path, 'pretrained_models', filename=model_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


# def merge_x_cond(xt, input, dataset_name, stage):
#     if dataset_name == "ntcouple_data":
#         if stage == 'fuel':
#             # get noised data and noised condition
#             x = xt['fuel'] # shape: [B, 1, 16, 64, 8]
#             cond1 = xt['neu'][...,:8]
#             cond2 = xt['fluid'][:,:1,...,:1]
#             # expand the cond2 along the last dimension into the same shape as xt
#             cond2 = cond2.repeat(1,1,1,1,8)
#             x = torch.cat([x, cond1, cond2], dim=1) # shape: [B, 3, 16, 64, 8]
            
#         elif stage == 'fluid':
#             x = xt['fluid'] # shape: [B, 4, 16, 64, 12]
#             cond1 = xt['fuel'][...,:1]
#             cond1 = cond1.repeat(1,1,1,1,12)
#             x = torch.cat([x, cond1], dim=1) # shape: [B, 5, 16, 64, 12]
            
#         elif stage == 'neu':
#             x = xt['neu'] # shape: [B, 1, 16, 64, 20]
#             # fix the bc in x
#             x[..., :1] = input['bc']
#             cond1 = xt['fuel'][...,:8]
#             cond2 = xt['fluid'][:,:1]
#             cond = torch.cat([cond1, cond2], dim=-1)
#             x = torch.cat([x, cond], dim=1) # shape: [B, 2, 16, 64, 20]
            
#         elif stage == 'couple':
#             pdb.set_trace()
#         return x