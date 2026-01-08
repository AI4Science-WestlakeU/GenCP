import os, sys
import torch
import torch.nn as nn
import numpy as np
import tqdm
import logging
import time
import argparse
import datetime
import shutil
import yaml
import json
from accelerate import Accelerator, DistributedDataParallelKwargs
import pdb


from data.turek_hron_dataset import TurekHronDataset
from data.double_cylinder_dataset import DoubleCylinderDataset
from data.ntcouple_dataset import NTcoupleDataset
from data.ntcouple_normalizer import NTcoupleNormalizer
from model.cno_surrogate import CNO3d as CNO3d_surrogate
from model.fno_surrogate import FNO3d as FNO3d_surrogate
from model.SiT_FNO import SiT_FNO
from model.sit_fno_surrogate import SiT_FNO as SiT_FNO_surrogate
from model.fno import FNO3d
from model.cno import CNO3d
from utils.utils import set_seed, add_args_from_config, setup_logging, mse_loss, rel_l2_loss, find_model, parse_transport_args, loss_with_mask

from collections import OrderedDict
from copy import deepcopy
import torchcfm

# NOTE: simple workaround for preparing input for surrogate model without mask
def prepare_data_for_surrogate(stage, input, target, dataset_name=None):
    """
    Align surrogate inputs/targets with model expectations.
    For NTcouple data, tensors already follow (B, C, T, H, W) so we simply
    permute to (B, T, H, W, C). For double-cylinder style data we retain the
    original workaround that concatenates conditioning channels.
    """
    if dataset_name == "ntcouple":
        if input.dim() != 5 or target.dim() != 5:
            raise ValueError("NTcouple surrogate expects 5D tensors (B, C, T, H, W).")
        input = input.permute(0, 2, 3, 4, 1).contiguous()
        target = target.permute(0, 2, 3, 4, 1).contiguous()
        return input, target

    input = input.repeat(1, target.shape[1] // input.shape[1], 1, 1, 1) # (b, 5, h, w, 4) -> (b, 10, h, w, 4)
    if stage == "fluid":
        cond = target[..., -1:] # (b, 10, h, w, 1)
        input = torch.cat([input, cond], dim=-1) # (b, 10, h, w, 5)
        target = target[..., :-1] # (b, 10, h, w, 3)
    elif stage == "structure":
        cond = target[..., :-1] # (b, 10, h, w, 3)
        input = torch.cat([input, cond], dim=-1) # (b, 10, h, w, 7)
        target = target[..., -1:] # (b, 10, h, w, 1)
    elif stage == "couple":
        pass
    else:
        raise ValueError(f"Invalid stage: {stage}")
    
    return input, target


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    # 1. Update parameters
    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        
    # 2. Sync buffers (e.g. BatchNorm statistics)
    model_buffers = dict(model.named_buffers())
    for name, buffer in ema_model.named_buffers():
        if name in model_buffers:
            buffer.copy_(model_buffers[name])


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Trainer Class                                #
#################################################################################

class Trainer:
    def __init__(self, args):
        self.args = args
        self.use_accelerator = getattr(args, 'use_accelerator', False)
        
        if self.use_accelerator:
            self.setup_accelerator()
        else:
            self.setup_device()
            
        self.setup_paths()
        self.setup_logging()
        self.setup_datasets()
        self.setup_model()
        self.setup_training_components()
        self.setup_transport()
        
        # Training state
        self.train_steps = 0
        self.log_steps = 0
        self.start_time = time.time()
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_rel_l2_loss = float('inf')
        self.current_epoch = 0
        
        if self.args.ckpt is not None:
            self.load_training_state()
        else:
            latest_checkpoint_path = os.path.join(self.exp_path, 'latest_checkpoint.pth')
            if os.path.exists(latest_checkpoint_path):
                self.args.ckpt = latest_checkpoint_path
                self.load_training_state()
                self.print_info(f"Auto-resuming from {latest_checkpoint_path}")
        
    def setup_accelerator(self):
        """Initialize Accelerator"""
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        
        self.accelerator = Accelerator(
            mixed_precision=getattr(self.args, 'mixed_precision', 'no'),
            gradient_accumulation_steps=getattr(self.args, 'gradient_accumulation_steps', 1),
            log_with=None,
            kwargs_handlers=[ddp_kwargs]
        )
        
        self.device = self.accelerator.device
        set_seed(self.args.seed)
        
    def setup_device(self):
        """Setup device (without accelerator)"""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        set_seed(self.args.seed)
        self.accelerator = None
        
    def setup_paths(self):
        """Setup experiment paths"""
        folder_name = self.args.stage + '_' + self.args.model_name
        if getattr(self.args, 'use_surrogate', False):
            folder_name += '_surrogate'
        
        if self.use_accelerator and self.accelerator.is_main_process:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.exp_path = os.path.join(self.args.results_path, self.args.dataset_name, 
                                        folder_name, current_time)
            os.makedirs(self.exp_path, exist_ok=True)
            
        elif not self.use_accelerator:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.exp_path = os.path.join(self.args.results_path, self.args.dataset_name, 
                                        folder_name, current_time)
            os.makedirs(self.exp_path, exist_ok=True)
        else:
            self.exp_path = None
        
        if self.use_accelerator:
            self.accelerator.wait_for_everyone()
            if not self.accelerator.is_main_process:
                self.exp_path = os.path.join(self.args.results_path, self.args.dataset_name, 
                                            folder_name, 'temp')
                if not os.path.exists(self.exp_path):
                    base_path = os.path.join(self.args.results_path, self.args.dataset_name, 
                                            folder_name)
                    if os.path.exists(base_path):
                        dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
                        if dirs:
                            self.exp_path = os.path.join(base_path, sorted(dirs)[-1])
        
        if (self.use_accelerator and self.accelerator.is_main_process) or not self.use_accelerator:
            try:
                if hasattr(self.args, 'config') and self.args.config and os.path.exists(self.args.config):
                    config_save_path = os.path.join(self.exp_path, 'config.yaml')
                    shutil.copy2(self.args.config, config_save_path)
                    self.print_info(f'Original config file saved to: {config_save_path}')
                
                args_dict = vars(self.args)
                args_save_path = os.path.join(self.exp_path, 'training_args.json')
                with open(args_save_path, 'w', encoding='utf-8') as f:
                    json.dump(args_dict, f, indent=2, ensure_ascii=False, default=str)
                self.print_info(f'Training arguments saved to: {args_save_path}')
                
            except Exception as e:
                self.print_info(f'Failed to save configuration files: {e}')
                
    def setup_logging(self):
        """Setup logging system"""
        if self.use_accelerator:
            if self.accelerator.is_main_process:
                self.writer, self.logger = setup_logging(self.exp_path, self.args.is_use_tb, rank=0)
                self.logger.info(f'args: {self.args}')
                self.logger.info("Starting training...")
            else:
                self.writer, self.logger = setup_logging(self.exp_path, self.args.is_use_tb, 
                                                       rank=self.accelerator.process_index)
        else:
            self.writer, self.logger = setup_logging(self.exp_path, self.args.is_use_tb)
            self.logger.info(f'args: {self.args}')
            self.logger.info("Starting training...")
        
        def log_info(message):
            if self.use_accelerator:
                if self.accelerator.is_main_process:
                    self.logger.info(message)
                else:
                    print(f"[Rank {self.accelerator.process_index}] {message}")
            else:
                self.logger.info(message)
        
        self.log_info = log_info
        
    def print_info(self, message):
        """Print information"""
        if self.use_accelerator:
            self.accelerator.print(message)
        else:
            print(message)
        
    def setup_datasets(self):
        """Setup datasets"""
        if self.args.dataset_name == 'turek_hron_data':
            train_dataset = TurekHronDataset(dataset_path=self.args.dataset_path, 
                                                    length=self.args.length, 
                                                    input_size=self.args.input_step, 
                                                    output_size=self.args.output_step, 
                                                    stride=self.args.stride, mode='train',
                                                    stage=self.args.stage, num_delta_t=self.args.num_delta_t,
                                                    dt=self.args.dt)
            val_dataset = TurekHronDataset(dataset_path=self.args.dataset_path, 
                                                  length=self.args.length, 
                                                  input_size=self.args.input_step, 
                                                  output_size=self.args.output_step, 
                                                  stride=self.args.stride, mode='val', 
                                                  stage=self.args.stage, num_delta_t=self.args.num_delta_t,
                                                  dt=self.args.dt)
        elif self.args.dataset_name == 'double_cylinder_data':
            train_dataset = DoubleCylinderDataset(dataset_path=self.args.dataset_path, 
                                                    length=self.args.length, 
                                                    input_size=self.args.input_step, 
                                                    output_size=self.args.output_step, 
                                                    stride=self.args.stride, mode='train',
                                                    stage=self.args.stage, num_delta_t=self.args.num_delta_t,
                                                    dt=self.args.dt)
            val_dataset = DoubleCylinderDataset(dataset_path=self.args.dataset_path, 
                                                  length=self.args.length, 
                                                  input_size=self.args.input_step, 
                                                  output_size=self.args.output_step, 
                                                  stride=self.args.stride, mode='val', 
                                                  stage=self.args.stage, num_delta_t=self.args.num_delta_t,
                                                  dt=self.args.dt)
        elif self.args.dataset_name == 'ntcouple':
            # NTcouple dataset - 3D nuclear-thermal coupling data
            field = getattr(self.args, 'field', self.args.stage)
            train_split = getattr(self.args, 'split', 'decouple')  # decouple, decouple for decoupled
            n_samples = getattr(self.args, 'n_data_set', None)
            
            train_dataset = NTcoupleDataset(
                field=field,
                split=train_split,
                n_samples=n_samples,
                data_root=self.args.dataset_path if hasattr(self.args, 'dataset_path') else None,
                normalize=True  # Dataset-level normalization (required for proper field concatenation)
            )
            
            # For validation, use a smaller subset from couple_val split
            val_split = 'decouple_val' # 'decouple'
            val_n_samples = min(100, n_samples) if n_samples else 100  # Use 100 samples for validation
            
            val_dataset = NTcoupleDataset(
                field=field,
                split=val_split,
                n_samples=val_n_samples,
                data_root=self.args.dataset_path if hasattr(self.args, 'dataset_path') else None,
                normalize=True
            )
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset_name}")
              
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                           batch_size=self.args.train_batch_size, 
                                                           shuffle=True, pin_memory=True, 
                                                           num_workers=self.args.num_workers,
                                                           drop_last=True)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                         batch_size=self.args.test_batch_size,
                                                         shuffle=False, pin_memory=True, 
                                                         num_workers=self.args.num_workers,
                                                         drop_last=True)
        self.print_info(f"Data loaded from {self.args.dataset_path}")
        
    def setup_model(self):
        """Setup model and data normalizer"""
        if self.args.dataset_name == 'ntcouple':
            # NTcouple: Data is already normalized in dataset (required for field concatenation)
            # But we provide normalizer interface for inference-time physical quantity conversion
            field = getattr(self.args, 'field', self.args.stage)
            
            class NTcoupleNormalizerWrapper:
                """
                Wrapper for NTcouple that provides normalizer interface.
                Data is already normalized in dataset, so preprocess is pass-through.
                But normalize/renormalize methods are available for inference use.
                """
                def __init__(self, field, device):
                    self.field = field
                    self.device = device
                    self.normalizer = NTcoupleNormalizer(field=field, device=device)
                
                def preprocess(self, x, y):
                    """Data already normalized in dataset - just move to device."""
                    return x.to(self.device), y.to(self.device)
                
                def postprocess(self, x, y):
                    """Denormalize for evaluation/visualization."""
                    # Use simplified interface without component/context
                    x_denorm = self.normalizer.renormalize(x, field=self.field)
                    y_denorm = self.normalizer.renormalize(y, field=self.field)
                    return x_denorm, y_denorm
                
                def normalize(self, x, field=None):
                    """Normalize physical quantity (for inference use)."""
                    field = field or self.field
                    return self.normalizer.normalize(x, field=field)
                
                def renormalize(self, x, field=None):
                    """Renormalize to physical units (for inference use)."""
                    field = field or self.field
                    return self.normalizer.renormalize(x, field=field)
            
            self.data_normalizer = NTcoupleNormalizerWrapper(field, self.device)
            self.print_info(f"Using NTcoupleNormalizer for field '{field}' (data pre-normalized in dataset)")
        else:
            from data.data_normalizer import RangeNormalizer
            self.data_normalizer = RangeNormalizer(self.train_dataloader.dataset, 
                                                device=self.device, mode='train', 
                                                batch_size=self.args.train_batch_size)

        if self.args.use_torchcfm:
            if self.args.model_name == 'SiT_FNO':
                class_dropout_prob = float(getattr(self.args, 'dropout', 0.1))
                self.model = SiT_FNO(input_size=self.args.input_size, 
                                depth=self.args.depth, 
                                hidden_size=self.args.hidden_size, 
                                patch_size=self.args.patch_size, 
                                num_heads=self.args.num_heads, 
                                in_channels=self.args.in_channels, 
                                out_channels=self.args.out_channels,
                                x0_is_use_noise=self.args.x0_is_use_noise, 
                                dataset_name=self.args.dataset_name, 
                                stage=self.args.stage,
                                modes=self.args.modes,
                                class_dropout_prob=class_dropout_prob)

            elif self.args.model_name == 'CNO':
                self.model = CNO3d(in_dim=self.args.in_dim, 
                                out_dim=self.args.out_dim, 
                                in_size=self.args.in_size, 
                                N_layers=self.args.depth,
                                channel_multiplier=self.args.channel_multiplier,
                                dataset_name=self.args.dataset_name,
                                x0_is_use_noise=self.args.x0_is_use_noise,
                                stage=self.args.stage)
                
            else:
                raise ValueError(f"Model {self.args.model_name} not supported")
        
        elif self.args.use_surrogate:
            if self.args.model_name == 'CNO':
                self.model = CNO3d_surrogate(in_dim=self.args.in_dim, 
                                   out_dim=self.args.out_dim, 
                                   in_size=self.args.in_size, 
                                   N_layers=self.args.depth,
                                   channel_multiplier=self.args.channel_multiplier
                                   )
            elif self.args.model_name == 'SiT_FNO':
                self.model = SiT_FNO_surrogate(input_size=self.args.input_size, 
                                depth=self.args.depth, 
                                hidden_size=self.args.hidden_size, 
                                patch_size=self.args.patch_size, 
                                num_heads=self.args.num_heads, 
                                in_channels=self.args.in_channels, 
                                out_channels=self.args.out_channels,
                                modes=self.args.modes)
        else:
            raise ValueError(f"Model {self.args.model_name} not supported")
        
        if not self.use_accelerator:
            self.model = self.model.to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        self.print_info(f"Number of parameters: {num_params}")
        
    def setup_training_components(self):
        """Setup training components"""
        betas = (0.9, 0.99) if self.args.dataset_name == 'ntcouple' else (0.9, 0.999)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=betas)
        
        if self.args.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        elif self.args.scheduler == 'cosine':
            eta_min = float(getattr(self.args, 'eta_min', 0))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.args.T_max,
                eta_min=eta_min
            )
            self.print_info(f"Using CosineAnnealingLR: T_max={self.args.T_max}, eta_min={eta_min}")
        else:
            raise ValueError(f"Scheduler {self.args.scheduler} not supported")
        
        self.criterion = nn.MSELoss(reduction='none')

        if self.use_accelerator:
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.scheduler = \
                self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader, 
                                       self.val_dataloader, self.scheduler)

        if self.use_accelerator:
            self.ema = deepcopy(self.accelerator.unwrap_model(self.model)).to(self.accelerator.device)
        else:
            self.ema = deepcopy(self.model).to(self.device)
            
        requires_grad(self.ema, False)

        if self.args.ckpt is not None:
            ckpt_path = self.args.ckpt
            state_dict = find_model(ckpt_path)
            
            if self.use_accelerator:
                self.accelerator.unwrap_model(self.model).load_state_dict(state_dict["model_state_dict"])
            else:
                self.model.load_state_dict(state_dict["model_state_dict"])
            
            self.ema.load_state_dict(state_dict["ema"])
            self.optimizer.load_state_dict(state_dict["opt"])
            
            if "scheduler" in state_dict:
                self.scheduler.load_state_dict(state_dict["scheduler"])
        
        if self.args.is_resume:
            if self.use_accelerator:
                self.accelerator.unwrap_model(self.model).load_checkpoint(self.args.checkpoint_path)
            else:
                self.model.load_checkpoint(self.args.checkpoint_path)
            self.print_info(f"Checkpoint {self.args.checkpoint_path} loaded.")

        if self.args.ckpt is None and not self.args.is_resume:
            if self.use_accelerator:
                update_ema(self.ema, self.accelerator.unwrap_model(self.model), decay=0)
            else:
                update_ema(self.ema, self.model, decay=0)
        self.model.train()
        self.ema.eval()
        
    def load_training_state(self):
        """Load complete training state from checkpoint"""
        ckpt_path = self.args.ckpt
        state_dict = find_model(ckpt_path)
        
        if self.use_accelerator:
            self.accelerator.unwrap_model(self.model).load_state_dict(state_dict["model_state_dict"])
        else:
            self.model.load_state_dict(state_dict["model_state_dict"])
        
        self.ema.load_state_dict(state_dict["ema"])
        self.optimizer.load_state_dict(state_dict["opt"])
        
        if "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        
        if "training_state" in state_dict:
            training_state = state_dict["training_state"]
            self.train_steps = training_state.get("train_steps", 0)
            self.log_steps = training_state.get("log_steps", 0)
            self.current_epoch = training_state.get("current_epoch", 0)
            self.best_epoch = training_state.get("best_epoch", 0)
            self.best_val_loss = training_state.get("best_val_loss", float('inf'))
                self.best_val_rel_l2_loss = training_state.get("best_val_rel_l2_loss", float('inf'))
                self.start_time = training_state.get("start_time", time.time())
        
        if "args" in state_dict:
            checkpoint_args = state_dict["args"]
            for key, value in checkpoint_args.items():
                if not hasattr(self.args, key) or key in ['ckpt', 'is_resume', 'checkpoint_path']:
                    setattr(self.args, key, value)
        
        self.print_info(f"Resumed training state from checkpoint:")
        self.print_info(f"  - Training steps: {self.train_steps}")
        self.print_info(f"  - Current epoch: {self.current_epoch}")
        self.print_info(f"  - Best epoch: {self.best_epoch}")
        self.print_info(f"  - Best validation MSE: {self.best_val_loss:.6f}")
        self.print_info(f"  - Best validation REL L2: {self.best_val_rel_l2_loss:.6f}")
        
    def load_from_config(self, config_path):
        """Load parameters from config file"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            for key, value in config.items():
                if hasattr(self.args, key):
                    setattr(self.args, key, value)
            
            self.print_info(f"Loaded parameters from config file {config_path}")
            return True
        return False
        
    def setup_transport(self):
        """Setup transport model"""
        if self.args.use_torchcfm:
            from torchcfm import ExactOptimalTransportConditionalFlowMatcher, ConditionalFlowMatcher
            self.cfm = ConditionalFlowMatcher(sigma=self.args.sigma, dataset_name=self.args.dataset_name)
            self.transport = None
            self.transport_sampler = None
            self.diffusion = None
            self.log_info("Using torchcfm as training backend")

        elif self.args.use_surrogate:
            self.cfm = None
            self.transport = None
            self.transport_sampler = None
            self.diffusion = None
            self.log_info("Using surrogate model as training backend")

            
    def train_step(self, input, target, attrs):
        """Single training step"""
        if self.use_accelerator:
            if self.accelerator.sync_gradients:
                self.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()
 
        input, target = self.data_normalizer.preprocess(input, target)
        
        if self.args.use_torchcfm:
            if self.args.dataset_name != "ntcouple":
                x0 = torch.randn_like(target) if self.args.x0_is_use_noise else input.clone()
                t, xt, ut = self.cfm.sample_location_and_conditional_flow(x0, target)

                vt = self.model(xt, t, input, attrs)

            # NTcouple: joint evolution paradigm
            # Core idea: [conditioning, target] evolve in joint space, only compute loss on target
            if self.args.dataset_name == 'ntcouple':
                # Dataset returns (B, C, T, H, W), convert to (B, T, H, W, C)
                input = input.permute(0, 2, 3, 4, 1).contiguous()
                target = target.permute(0, 2, 3, 4, 1).contiguous()
                
                # 1. Build joint state space: Z = [C, Y]
                joint_state = torch.cat([input, target], dim=-1)
                
                # 2. Joint evolution: add noise to entire joint space
                x0 = torch.randn_like(joint_state) if self.args.x0_is_use_noise else joint_state.clone()
                t, xt, ut = self.cfm.sample_location_and_conditional_flow(x0, joint_state)
                
                # xt: (B, T, H, W, C_cond+C_target) - noisy joint state
                # ut: (B, T, H, W, C_cond+C_target) - true joint velocity field
                
                # 3. Model predicts velocity field of entire joint space
                # Note: CNO model expects x0 not None, pass a dummy x0 (single time step)
                # Model internally concats [dummy_x0, xt] along time dimension, but this doesn't affect joint evolution
                
                C_cond = input.shape[-1]
                C_target = target.shape[-1]

                use_clean_bc = getattr(self.args, 'use_clean_bc', True)
                if use_clean_bc and self.args.stage == 'neutron' and C_cond == 2:
                    xt_cond = xt[..., :C_cond]
                    xt_target = xt[..., C_cond:]
                    xt_cond_clean_bc = torch.cat([xt_cond[..., 0:1], input[..., 1:2]], dim=-1)
                    xt = torch.cat([xt_cond_clean_bc, xt_target], dim=-1)
                
                use_clean_left_bc = getattr(self.args, 'use_clean_left_bc_for_solid', False)
                if use_clean_left_bc and self.args.stage == 'solid' and C_cond == 3:
                    xt_cond = xt[..., :C_cond]
                    xt_target = xt[..., C_cond:]
                    xt_cond_clean_left_bc = torch.cat([xt_cond[..., 0:2], input[..., 2:3]], dim=-1)
                    xt = torch.cat([xt_cond_clean_left_bc, xt_target], dim=-1)
                
                dummy_x0 = xt[:, :1, :, :, :]
                
                t_model = t * 1000.0
                
                vt = self.model(xt, t_model, cond=attrs)
            
            # if self.args.dataset_name == 'double_cylinder_data':
            #     channel = getattr(self.args, 'out_dim', 4)
            # elif self.args.dataset_name == 'turek_hron_data':
            channel = getattr(self.args, 'out_channels', 4)
            # else:
                # raise ValueError("Unknown dataset_name")
            
            loss = loss_with_mask(vt, ut, mse_loss, channel, self.args.stage, self.args.dataset_name)
        
        elif self.args.use_surrogate:
            if self.args.model_name in ["UNet3d", "CNO", "SiT_FNO"]:
                input, target = prepare_data_for_surrogate(self.args.stage, input, target, self.args.dataset_name)
                input = input.permute(0, 4, 1, 2, 3) # (b, t, h, w, c) -> (b, c, t, h, w)
                pred = self.model(input)
                pred = pred.permute(0, 2, 3, 4, 1) # (b, c, t, h, w) -> (b, t, h, w, c)
            elif self.args.model_name == "FNO":
                input, target = prepare_data_for_surrogate(self.args.stage, input, target, self.args.dataset_name)
                pred = self.model(input)
            else: 
                raise NotImplementedError("Model not supported")
            
            loss = mse_loss(pred, target).mean()
            
        if self.train_steps % self.args.log_every == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.log_info(f"Step {self.train_steps}, Loss: {loss.item():.6f}, LR: {current_lr:.6e}")
            
            if ((self.use_accelerator and self.accelerator.is_main_process) or not self.use_accelerator) and self.args.is_use_tb and self.writer is not None:
                self.writer.add_scalar("train/loss", loss.item(), self.train_steps)
                self.writer.add_scalar("train/learning_rate", current_lr, self.train_steps)

        if self.use_accelerator:
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            else:
                grad_norm = 0.0
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        decay = 0.995
        if self.use_accelerator:
            update_ema(self.ema, self.accelerator.unwrap_model(self.model), decay=decay)
        else:
            update_ema(self.ema, self.model, decay=decay)
        
        return loss.item(), grad_norm
        
    def save_checkpoint(self, epoch):
        """Save checkpoint"""
        if (self.use_accelerator and self.accelerator.is_main_process) or not self.use_accelerator:
            training_state = {
                "train_steps": self.train_steps,
                "log_steps": self.log_steps,
                "current_epoch": epoch,
                "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss,
                "best_val_rel_l2_loss": self.best_val_rel_l2_loss,
                "start_time": self.start_time
            }
            
            checkpoint = {
                'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict() if self.use_accelerator else self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "opt": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "training_state": training_state,
                "args": self.args
            }
            latest_checkpoint_path = f"{self.exp_path}/latest_checkpoint.pth"
            if self.use_accelerator:
                self.accelerator.save(checkpoint, latest_checkpoint_path)
            else:
                torch.save(checkpoint, latest_checkpoint_path)
            
            checkpoint_path = f"{self.exp_path}/model_{epoch:02d}.pth"
            if self.use_accelerator:
                self.accelerator.save(checkpoint, checkpoint_path)
            else:
                torch.save(checkpoint, checkpoint_path)
                
            self.log_info(f"Saved checkpoint to {checkpoint_path}")
            self.log_info(f"Saved latest checkpoint to {latest_checkpoint_path}")
            
        if self.use_accelerator:
            self.accelerator.wait_for_everyone()
    
    def save_best_model(self, epoch, val_loss, val_rel_l2_loss):
        """Save best model (including EMA and main model)"""
        if (self.use_accelerator and self.accelerator.is_main_process) or not self.use_accelerator:
            training_state = {
                "train_steps": self.train_steps,
                "log_steps": self.log_steps,
                "current_epoch": epoch,
                "best_epoch": self.best_epoch,
                "best_val_loss": self.best_val_loss,
                "best_val_rel_l2_loss": self.best_val_rel_l2_loss,
                "start_time": self.start_time
            }
            
            best_ema_checkpoint = {
                'model_state_dict': self.ema.state_dict(),
                "ema": self.ema.state_dict(),
                "opt": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "training_state": training_state,
                "args": self.args,
                "val_loss": val_loss,
                "val_rel_l2_loss": val_rel_l2_loss,
                "epoch": epoch
            }
            
            best_main_checkpoint = {
                'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict() if self.use_accelerator else self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "opt": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "training_state": training_state,
                "args": self.args,
                "val_loss": val_loss,
                "val_rel_l2_loss": val_rel_l2_loss,
                "epoch": epoch
            }
            
            best_ema_path = f"{self.exp_path}/best_ema_model.pth"
            if self.use_accelerator:
                self.accelerator.save(best_ema_checkpoint, best_ema_path)
            else:
                torch.save(best_ema_checkpoint, best_ema_path)
            
            best_main_path = f"{self.exp_path}/best_main_model.pth"
            if self.use_accelerator:
                self.accelerator.save(best_main_checkpoint, best_main_path)
            else:
                torch.save(best_main_checkpoint, best_main_path)
                
            self.log_info(f"Saved best EMA model to: {best_ema_path}")
            self.log_info(f"Saved best main model to: {best_main_path}")
            self.log_info(f"Best validation MSE: {val_loss:.6f}, Best validation REL L2: {val_rel_l2_loss:.6f}, Epoch: {epoch}")
            
        # if self.use_accelerator:
        #     self.accelerator.wait_for_everyone()
        
    def sample(self, z, target_norm, model_kwargs, model=None):
        
        if model is None:
            model = self.model

        if self.args.use_torchcfm:
            if self.args.dataset_name == 'ntcouple':
                cfm = torchcfm.ConditionalFlowMatcher()
                num_steps = getattr(self.args, 'num_sampling_steps', 10)
                t0, t1 = (0.0, 1.0)
                t_grid = torch.linspace(t0, t1, num_steps, device=z.device)
                dt = (t_grid[1] - t_grid[0]).item() if num_steps > 1 else 1.0
                
                input_norm = model_kwargs.get("x0").to(self.device).float()
                joint_ref = torch.cat([input_norm, target_norm], dim=-1)
                
                x = z.clone().to(self.device)
                
                for t in t_grid[:-1]:
                    tb = t.expand(x.shape[0]).to(self.device)
                    tb_model = tb * 1000.0
                    
                    _, xt_joint, _ = cfm.sample_location_and_conditional_flow(
                        z.float().to(self.device), 
                        joint_ref.float().to(self.device), 
                        tb
                    )

                    if self.args.stage == "neutron":
                        yt = xt_joint[..., :-1]
                        x = torch.cat([yt, x[..., -1:]], dim=-1)
                    elif self.args.stage == "solid":
                        yt = xt_joint[..., :-1]
                        x = torch.cat([yt, x[..., -1:]], dim=-1)
                    elif self.args.stage == "fluid":
                        yt = xt_joint[..., :-4]
                        x = torch.cat([yt, x[..., -4:]], dim=-1)
                    else:
                        raise NotImplementedError(f"Stage '{self.args.stage}' not supported")
                    
                    C_cond = input_norm.shape[-1]
                    C_target = target_norm.shape[-1]

                    use_clean_bc = getattr(self.args, 'use_clean_bc', True)
                    if use_clean_bc and self.args.stage == 'neutron' and C_cond == 2:
                        x_cond = x[..., :C_cond]
                        x_target = x[..., C_cond:]
                        x_cond_clean_bc = torch.cat([x_cond[..., 0:1], input_norm[..., 1:2]], dim=-1)
                        x = torch.cat([x_cond_clean_bc, x_target], dim=-1)
                    
                    use_clean_left_bc = getattr(self.args, 'use_clean_left_bc_for_solid', False)
                    if use_clean_left_bc and self.args.stage == 'solid' and C_cond == 3:
                        x_cond = x[..., :C_cond]
                        x_target = x[..., C_cond:]
                        x_cond_clean_left_bc = torch.cat([x_cond[..., 0:2], input_norm[..., 2:3]], dim=-1)
                        x = torch.cat([x_cond_clean_left_bc, x_target], dim=-1)
                    
                    vt = model(x, tb_model, cond=model_kwargs.get("cond"))
                    
                    if self.args.stage == "neutron":
                        x[..., -1:] = x[..., -1:] + vt * dt
                    elif self.args.stage == "solid":
                        x[..., -1:] = x[..., -1:] + vt * dt
                    elif self.args.stage == "fluid":
                        x[..., -4:] = x[..., -4:] + vt * dt
                
                C_target = target_norm.shape[-1]
                return x[..., -C_target:]
            
            cfm = torchcfm.ConditionalFlowMatcher()
            num_steps = getattr(self.args, 'num_sampling_steps', 10)
            t0, t1 = (0.0, 1.0)

            t_grid = torch.linspace(t0, t1, num_steps, device=z.device)

            dt = (t_grid[1] - t_grid[0]).item() if num_steps > 1 else 1.0
            
            x = z.clone().to(self.device)
                
            for t in t_grid[:-1]:
                tb = t.expand(x.shape[0]).to(x)
                _, xt, _ = cfm.sample_location_and_conditional_flow(z.float().to(self.device), target_norm.float().to(self.device), tb)
                    
                if self.args.stage == "fluid":
                    yt = xt[..., -1:]
                    x = torch.cat([x[..., :-1], yt], dim=-1)
                elif self.args.stage == "structure":
                    yt = xt[..., :-1]
                    x = torch.cat([yt, x[..., -1:]], dim=-1)
                elif self.args.stage == "joint":
                    yt = xt[..., :]
                    x = x
      
                vt = model(x.to(self.device), tb.to(self.device), model_kwargs.get("x0").to(self.device), model_kwargs.get("cond"))

                channel = getattr(self.args, 'out_channels', 4)
                if channel == 3 or channel == 1:
                    if self.args.stage == "fluid":
                        x[..., :-1] = x[..., :-1] + vt * dt
                    elif self.args.stage == "structure":
                        x[..., -1:] = x[..., -1:] + vt * dt
                    elif self.args.stage == "joint":
                        x = x + vt * dt
                elif channel == 4:
                    if self.args.stage == "fluid":
                        x[..., :-1] = x[..., :-1] + vt[..., :-1] * dt
                    elif self.args.stage == "structure":
                        x[..., -1:] = x[..., -1:] + vt[..., -1:] * dt
                    elif self.args.stage == "joint":
                        x = x + vt * dt
                    
            pred = x
              
        elif self.args.use_surrogate:
            if self.args.model_name in ["UNet3d", "CNO", "SiT_FNO"]:
                z, _ = prepare_data_for_surrogate(self.args.stage, z, target_norm, self.args.dataset_name)
                z = z.permute(0, 4, 1, 2, 3)
                pred = self.model(z)
                pred = pred.permute(0, 2, 3, 4, 1)
            elif self.args.model_name == "FNO":
                z, _ = prepare_data_for_surrogate(self.args.stage, z, target_norm, self.args.dataset_name)
                pred = self.model(z)
            else: 
                raise NotImplementedError("Model not supported")
        
            if self.args.stage == "fluid":
                pred_ = target_norm.clone()
                pred_[..., :-1] = pred
                pred = pred_
            elif self.args.stage == "structure":
                pred_ = target_norm.clone()
                pred_[..., -1:] = pred
                pred = pred_
            elif self.args.stage == "joint":
                pred = pred
            else:
                raise NotImplementedError("Stage not supported")
        
        return pred

    def validate(self, epoch):
        """Validate (validate one batch only)"""
        if self.args.dataset_name == 'ntcouple':
            val_model = self.model # ema
        else:
            val_model = self.model
            
        val_model.eval()

        with torch.no_grad():
            val_loss = 0.
            val_rel_l2_loss = 0.
            val_loss_train = 0.
            val_rel_l2_loss_train = 0.

            for input, target, grid_x, grid_y, attrs in self.val_dataloader:
                if self.args.dataset_name == 'ntcouple':
                    input = input.to(self.device).float()
                    target = target.to(self.device).float()

                    if self.args.use_surrogate:
                        input_sur, target_sur = prepare_data_for_surrogate(
                            self.args.stage, input, target, self.args.dataset_name
                        )
                        input_sur = input_sur.permute(0, 4, 1, 2, 3)
                        pred = val_model(input_sur)
                        pred = pred.permute(0, 2, 3, 4, 1)
                        val_loss = mse_loss(pred, target_sur)
                        val_rel_l2_loss = rel_l2_loss(pred, target_sur)
                        break
                    
                    input = input.permute(0, 2, 3, 4, 1).contiguous()
                    target = target.permute(0, 2, 3, 4, 1).contiguous()
                    
                    input_norm = input
                    target_norm = target
                    
                    joint_ref = torch.cat([input_norm, target_norm], dim=-1)
                    z = torch.randn_like(joint_ref)
                    
                    model_kwargs = dict(x0=input_norm, cond=attrs)
                    sample_norm = self.sample(z, target_norm, model_kwargs, model=val_model)
                    
                    sample_denorm_BCTHW = sample_norm.permute(0, 4, 1, 2, 3)
                    target_denorm_BCTHW = target_norm.permute(0, 4, 1, 2, 3)
                    input_BCTHW = input_norm.permute(0, 4, 1, 2, 3)
                    
                    _, sample_denorm_BCTHW = self.data_normalizer.postprocess(sample_denorm_BCTHW, sample_denorm_BCTHW)
                    _, target_denorm_BCTHW = self.data_normalizer.postprocess(target_denorm_BCTHW, target_denorm_BCTHW)
                    
                    samples_denorm = sample_denorm_BCTHW.permute(0, 2, 3, 4, 1)
                    target_denorm = target_denorm_BCTHW.permute(0, 2, 3, 4, 1)
                    
                    val_loss = mse_loss(samples_denorm, target_denorm)
                    
                    if self.args.stage == "fluid" and samples_denorm.shape[-1] > 1:
                        samples_BCTHW = samples_denorm.permute(0, 4, 1, 2, 3)
                        target_BCTHW = target_denorm.permute(0, 4, 1, 2, 3)
                        channel_rel_l2 = []
                        for c in range(samples_BCTHW.shape[1]):
                            c_rel_l2 = rel_l2_loss(samples_BCTHW[:, c:c+1], target_BCTHW[:, c:c+1])
                            channel_rel_l2.append(c_rel_l2)
                        val_rel_l2_loss = torch.stack(channel_rel_l2).mean(dim=0)
                    else:
                        val_rel_l2_loss = rel_l2_loss(samples_denorm, target_denorm)
                    
                else:
                    if self.args.use_surrogate:
                        z = input.clone().to(self.device).float()
                    else:
                        if self.args.x0_is_use_noise:
                            z = torch.randn_like(target)
                        else:
                            z = input.clone()
                    
                    input_norm, target_norm = self.data_normalizer.preprocess(input, target)
                        
                    model_kwargs = dict(x0=input_norm, cond=attrs)
                    sample_norm = self.sample(z, target_norm, model_kwargs)

                    _, samples_denorm = self.data_normalizer.postprocess(input_norm, sample_norm)
                    
                    if self.args.stage == "fluid":
                        val_loss = mse_loss(samples_denorm[..., :1], target[..., :1].to(self.device))
                        val_rel_l2_loss = rel_l2_loss(samples_denorm[..., :1], target[..., :1].to(self.device))
                    elif self.args.stage == "structure":
                        val_loss = mse_loss(samples_denorm[..., -1:], target[..., -1:].to(self.device))
                        val_rel_l2_loss = rel_l2_loss(samples_denorm[..., -1:], target[..., -1:].to(self.device))
                    elif self.args.stage == "joint":
                        val_loss = mse_loss(samples_denorm, target.to(self.device))
                        val_rel_l2_loss = rel_l2_loss(samples_denorm, target.to(self.device))

                break

            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.mean().item()
            if isinstance(val_rel_l2_loss, torch.Tensor):
                val_rel_l2_loss = val_rel_l2_loss.mean().item()

            is_best = val_rel_l2_loss < self.best_val_rel_l2_loss
            if is_best:
                self.best_epoch = epoch
                self.best_val_loss = val_loss
                self.best_val_rel_l2_loss = val_rel_l2_loss
                self.save_best_model(epoch, val_loss, val_rel_l2_loss)
                
            self.log_info(f"Epoch {epoch}, val loss: {val_loss:.5f}, val rel l2 loss: {val_rel_l2_loss:.5f}")
            self.log_info(f"Epoch {epoch}, train loss: {val_loss_train:.5f}, train rel l2 loss: {val_rel_l2_loss_train:.5f}")
            
            if is_best:
                self.log_info(f"New best model! Validation MSE: {val_loss:.5f}, Best validation REL L2: {val_rel_l2_loss:.5f} (saved based on REL L2)")

            if ((self.use_accelerator and self.accelerator.is_main_process) or not self.use_accelerator) and self.args.is_use_tb and self.writer is not None:
                self.writer.add_scalar("val_loss", val_loss, epoch)
                self.writer.add_scalar("val_rel_l2_loss", val_rel_l2_loss, epoch)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar("learning_rate", current_lr, epoch)

            return val_loss, val_rel_l2_loss
            
    def train_by_epochs(self):
        """Training mode based on epochs"""
        self.print_info(f"Start training on {self.device}")
        start_epoch = self.current_epoch + 1
        pbar = tqdm.tqdm(range(start_epoch, self.args.epochs + 1), 
                        disable=self.use_accelerator and not self.accelerator.is_local_main_process)
        
        for epoch in pbar:
            self.model.train()
            total_loss = 0.
            
            self.log_info(f"Beginning epoch {epoch}...")
            for input, target, grid_x, grid_y, attrs in self.train_dataloader:
                loss, grad_norm = self.train_step(input, target, attrs)
                total_loss += loss
                pbar.set_postfix(loss=loss, grad_norm=f"{grad_norm:.3f}")
                
                self.log_steps += 1
                self.train_steps += 1
                
                if self.train_steps % self.args.ckpt_every == 0 and self.train_steps > 0:
                    self.save_checkpoint(epoch)
                    
                if (not self.args.use_torchcfm) and self.train_steps % self.args.sample_every == 0 and self.train_steps > 0:
                    self.print_info("Generating EMA samples...")
                    self.print_info("Generating EMA samples done.")
            
            average_loss = total_loss / len(self.train_dataloader)
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            self.log_info(f"Epoch {epoch}, training loss: {average_loss:.5f}, lr: {current_lr:.6f}")
                
            if ((self.use_accelerator and self.accelerator.is_main_process) or not self.use_accelerator) and self.args.is_use_tb and self.writer is not None:
                self.writer.add_scalar("train_loss", average_loss, epoch)
                self.writer.add_scalar("learning_rate", current_lr, epoch)

            if epoch % self.args.test_interval == 0 and ((self.use_accelerator and self.accelerator.is_main_process) or not self.use_accelerator):
                self.validate(epoch)
                
        self.finish_training()
        
    def train_by_iterations(self, total_iterations):
        """Training mode based on iteration count."""
        self.print_info(f"Start training on {self.device} for {total_iterations} iterations")
        
        start_iteration = self.train_steps
        remaining_iterations = total_iterations - start_iteration
        
        if remaining_iterations <= 0:
            self.print_info(f"Training completed. Current steps: {self.train_steps}, Target steps: {total_iterations}")
            return
            
        pbar = tqdm.tqdm(range(remaining_iterations), 
                        disable=self.use_accelerator and not self.accelerator.is_local_main_process)

        data_iter = iter(self.train_dataloader)

        for iteration in pbar:
            self.model.train()

            try:
                input, target, grid_x, grid_y, attrs = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                input, target, grid_x, grid_y, attrs = next(data_iter)

            loss, grad_norm = self.train_step(input, target, attrs)
            pbar.set_postfix(loss=loss, grad_norm=f"{grad_norm:.3f}")
            
            self.log_steps += 1
            self.train_steps += 1
            
            self.scheduler.step()
            
            if self.train_steps % self.args.ckpt_every == 0 and self.train_steps > 0:
                self.save_checkpoint(iteration)
                
            if (not self.args.use_torchcfm) and self.train_steps % self.args.sample_every == 0 and self.train_steps > 0:
                self.print_info("Generating EMA samples...")
                self.print_info("Generating EMA samples done.")
            
            if iteration % self.args.test_interval == 0 and iteration > 0:
                current_lr = self.scheduler.get_last_lr()[0]
                self.log_info(f"Iteration {iteration}, lr: {current_lr:.6f}")
                
                self.validate(iteration)

                self.model.train()
                
        self.finish_training()
        
    def finish_training(self):
        """Finish training"""
        end_time = time.time()
        self.log_info(f"Training complete, best epoch is {self.best_epoch}, best val loss is {self.best_val_loss:.6f}, best val rel l2 loss is {self.best_val_rel_l2_loss:.6f}, time cost is {end_time-self.start_time} s")
        self.log_info(f"Results saved at {self.exp_path}")
        
        if (self.use_accelerator and self.accelerator.is_main_process) or not self.use_accelerator:
            best_ema_path = f"{self.exp_path}/best_ema_model.pth"
            best_main_path = f"{self.exp_path}/best_main_model.pth"
            
            if not os.path.exists(best_ema_path) or not os.path.exists(best_main_path):
                self.log_info("Saving final model as best model at end of training")
                self.save_best_model(self.current_epoch, self.best_val_loss, self.best_val_rel_l2_loss)

        if (self.use_accelerator and self.accelerator.is_main_process) or not self.use_accelerator:
            if self.args.is_use_tb and self.writer is not None:
                self.writer.close()
            for handler in self.logger.handlers:
                handler.flush()
                if isinstance(handler, logging.FileHandler):
                    handler.close()

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    if hasattr(args, 'resume') and args.resume:
        if not args.resume_path:
            raise ValueError("Must specify --resume-path when using --resume")
        
        if not os.path.exists(args.resume_path):
            raise ValueError(f"Resume path does not exist: {args.resume_path}")
        
        latest_checkpoint = os.path.join(args.resume_path, 'latest_checkpoint.pth')
        if os.path.exists(latest_checkpoint):
            args.ckpt = latest_checkpoint
            args.results_path = os.path.dirname(args.resume_path)
            print(f"Resuming from {latest_checkpoint}")
        else:
            checkpoint_files = [f for f in os.listdir(args.resume_path) if f.startswith('model_') and f.endswith('.pth')]
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                latest_checkpoint = os.path.join(args.resume_path, checkpoint_files[-1])
                args.ckpt = latest_checkpoint
                args.results_path = os.path.dirname(args.resume_path)
                print(f"Resuming from {latest_checkpoint}")
            else:
                raise ValueError(f"No checkpoint files found in {args.resume_path}")
        
        config_path = os.path.join(args.resume_path, 'config.yaml')
        if os.path.exists(config_path):
            print(f"Loading config from {config_path}")
    
    trainer = Trainer(args)
    
    if hasattr(args, 'train_mode') and args.train_mode == 'iteration':
        total_iterations = getattr(args, 'total_iterations', 10000)
        trainer.train_by_iterations(total_iterations)
    else:
        trainer.train_by_epochs()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Training Configurations")

    parser.add_argument("--config", type=str, default="configs/train_neu.yaml") 
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--results-dir", type=str, default="results")
    
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=10000)
    parser.add_argument("--cfg-scale", type=float, default=0)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a custom SiT checkpoint")
    
    parser.add_argument("--resume", action="store_true", help="Resume training from previous experiment")
    parser.add_argument("--resume-path", type=str, default=None, help="Path to experiment to resume")
    
    parser.add_argument("--train-mode", type=str, default="epoch", choices=["epoch", "iteration"], 
                       help="Training mode: epoch or iteration")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["step", "cosine"], help="Learning rate scheduler")
    parser.add_argument("--use-torchcfm", default=True, action="store_true", help="Use torchcfm training branch")

    parser.add_argument("--num-sampling-steps", type=int, default=10, help="Number of sampling steps")
    
    parser.add_argument("--use-accelerator", action="store_true", help="Use accelerator for distributed training")
    parser.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision training type")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2, help="Gradient accumulation steps")
    
    
    
    parse_transport_args(parser)
    
    args = add_args_from_config(parser)
    
    from model.SiT import SiT
    
    main(args)
                    joint_ref = torch.cat([input_norm, target_norm], dim=-1)
                    z = torch.randn_like(joint_ref)
                    
                    model_kwargs = dict(x0=input_norm, cond=attrs)
                    # Use val_model (EMA for ntcouple) for sampling
                    sample_norm = self.sample(z, target_norm, model_kwargs, model=val_model)
                    
                    # Denormalize for metric calculation (convert back to (B, C, T, H, W) format)
                    sample_denorm_BCTHW = sample_norm.permute(0, 4, 1, 2, 3)
                    target_denorm_BCTHW = target_norm.permute(0, 4, 1, 2, 3)
                    input_BCTHW = input_norm.permute(0, 4, 1, 2, 3)
                    
                    # Use normalizer for denormalization (must pass correct BCTHW format)
                    _, sample_denorm_BCTHW = self.data_normalizer.postprocess(sample_denorm_BCTHW, sample_denorm_BCTHW)
                    _, target_denorm_BCTHW = self.data_normalizer.postprocess(target_denorm_BCTHW, target_denorm_BCTHW)
                    
                    # Convert back to (B, T, H, W, C) for metric calculation
                    samples_denorm = sample_denorm_BCTHW.permute(0, 2, 3, 4, 1)
                    target_denorm = target_denorm_BCTHW.permute(0, 2, 3, 4, 1)
                    
                    # Compute NTcouple validation loss
                    val_loss = mse_loss(samples_denorm, target_denorm)
                    
                    # For fluid field, use per-channel average rel_l2 (more fair across different physical quantities)
                    if self.args.stage == "fluid" and samples_denorm.shape[-1] > 1:
                        # Convert to (B, C, T, H, W) for per-channel calculation
                        samples_BCTHW = samples_denorm.permute(0, 4, 1, 2, 3)
                        target_BCTHW = target_denorm.permute(0, 4, 1, 2, 3)
                        channel_rel_l2 = []
                        for c in range(samples_BCTHW.shape[1]):
                            c_rel_l2 = rel_l2_loss(samples_BCTHW[:, c:c+1], target_BCTHW[:, c:c+1])
                            channel_rel_l2.append(c_rel_l2)
                        # Average across channels, then across batch
                        val_rel_l2_loss = torch.stack(channel_rel_l2).mean(dim=0)
                    else:
                        val_rel_l2_loss = rel_l2_loss(samples_denorm, target_denorm)
                    
                else:
                    # Double Cylinder and other datasets: original logic
                    # Prepare initial noise or input
                    if self.args.use_surrogate:
                        z = input.clone().to(self.device).float()
                    else:
                        if self.args.x0_is_use_noise:
                            z = torch.randn_like(target)
                        else:
                            z = input.clone()
                    
                    # Data preprocessing
                    input_norm, target_norm = self.data_normalizer.preprocess(input, target)
                        
                    model_kwargs = dict(x0=input_norm, cond=attrs)
                    sample_norm = self.sample(z, target_norm, model_kwargs)

                    # Double Cylinder and other datasets need denormalization
                    _, samples_denorm = self.data_normalizer.postprocess(input_norm, sample_norm)
                    
                    # Compute loss (select different channels based on stage)
                    if self.args.stage == "fluid":
                        val_loss = mse_loss(samples_denorm[..., :1], target[..., :1].to(self.device))
                        val_rel_l2_loss = rel_l2_loss(samples_denorm[..., :1], target[..., :1].to(self.device))
                    elif self.args.stage == "structure":
                        val_loss = mse_loss(samples_denorm[..., -1:], target[..., -1:].to(self.device))
                        val_rel_l2_loss = rel_l2_loss(samples_denorm[..., -1:], target[..., -1:].to(self.device))
                    elif self.args.stage == "joint":
                        val_loss = mse_loss(samples_denorm, target.to(self.device))
                        val_rel_l2_loss = rel_l2_loss(samples_denorm, target.to(self.device))

                break

            # Average results across batch
            if isinstance(val_loss, torch.Tensor):
                val_loss = val_loss.mean().item()
            if isinstance(val_rel_l2_loss, torch.Tensor):
                val_rel_l2_loss = val_rel_l2_loss.mean().item()

            is_best = val_rel_l2_loss < self.best_val_rel_l2_loss
            if is_best:
                self.best_epoch = epoch
                self.best_val_loss = val_loss
                self.best_val_rel_l2_loss = val_rel_l2_loss
                self.save_best_model(epoch, val_loss, val_rel_l2_loss)
                
            self.log_info(f"Epoch {epoch}, val loss: {val_loss:.5f}, val rel l2 loss: {val_rel_l2_loss:.5f}")
            self.log_info(f"Epoch {epoch}, train loss: {val_loss_train:.5f}, train rel l2 loss: {val_rel_l2_loss_train:.5f}")
            
            if is_best:
                self.log_info(f"New best model! Val MSE: {val_loss:.5f}, Best Val REL L2: {val_rel_l2_loss:.5f} (saved based on REL L2)")

            if ((self.use_accelerator and self.accelerator.is_main_process) or not self.use_accelerator) and self.args.is_use_tb and self.writer is not None:
                self.writer.add_scalar("val_loss", val_loss, epoch)
                self.writer.add_scalar("val_rel_l2_loss", val_rel_l2_loss, epoch)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar("learning_rate", current_lr, epoch)

                
            return val_loss, val_rel_l2_loss
            
    def train_by_epochs(self):
        """Epoch-based training mode"""
        self.print_info(f"Start training on {self.device}")
        start_epoch = self.current_epoch + 1
        pbar = tqdm.tqdm(range(start_epoch, self.args.epochs + 1), 
                        disable=self.use_accelerator and not self.accelerator.is_local_main_process)
        
        for epoch in pbar:
            self.model.train()
            total_loss = 0.
            
            self.log_info(f"Beginning epoch {epoch}...")
            for input, target, grid_x, grid_y, attrs in self.train_dataloader:
                # input: b, 1, h, w, 4
                # target: b, 9, h, w, 4
                loss, grad_norm = self.train_step(input, target, attrs)
                total_loss += loss
                pbar.set_postfix(loss=loss, grad_norm=f"{grad_norm:.3f}")
                
                self.log_steps += 1
                self.train_steps += 1
                
                if self.train_steps % self.args.ckpt_every == 0 and self.train_steps > 0:
                    self.save_checkpoint(epoch)
                    
                if (not self.args.use_torchcfm) and self.train_steps % self.args.sample_every == 0 and self.train_steps > 0:
                    self.print_info("Generating EMA samples...")
                    self.print_info("Generating EMA samples done.")
            
            average_loss = total_loss / len(self.train_dataloader)
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            self.log_info(f"Epoch {epoch}, training loss: {average_loss:.5f}, lr: {current_lr:.6f}")
                
            if ((self.use_accelerator and self.accelerator.is_main_process) or not self.use_accelerator) and self.args.is_use_tb and self.writer is not None:
                self.writer.add_scalar("train_loss", average_loss, epoch)
                self.writer.add_scalar("learning_rate", current_lr, epoch)

            if epoch % self.args.test_interval == 0 and ((self.use_accelerator and self.accelerator.is_main_process) or not self.use_accelerator):
                self.validate(epoch)
                
        self.finish_training()
        
    def train_by_iterations(self, total_iterations):
        """Training mode based on iteration count."""
        self.print_info(f"Start training on {self.device} for {total_iterations} iterations")
        
        start_iteration = self.train_steps
        remaining_iterations = total_iterations - start_iteration
        
        if remaining_iterations <= 0:
            self.print_info(f"Training completed. Current steps: {self.train_steps}, Target steps: {total_iterations}")
            return
            
        pbar = tqdm.tqdm(range(remaining_iterations), 
                        disable=self.use_accelerator and not self.accelerator.is_local_main_process)

        # Use standard dataloader iteration, restart when dataset ends
        data_iter = iter(self.train_dataloader)

        for iteration in pbar:
            self.model.train()

            try:
                input, target, grid_x, grid_y, attrs = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                input, target, grid_x, grid_y, attrs = next(data_iter)

            loss, grad_norm = self.train_step(input, target, attrs)
            pbar.set_postfix(loss=loss, grad_norm=f"{grad_norm:.3f}")
            
            self.log_steps += 1
            self.train_steps += 1
            
            # Update learning rate every iteration
            self.scheduler.step()
            
            # Save checkpoint
            if self.train_steps % self.args.ckpt_every == 0 and self.train_steps > 0:
                self.save_checkpoint(iteration)
                
            # Generate samples
            if (not self.args.use_torchcfm) and self.train_steps % self.args.sample_every == 0 and self.train_steps > 0:
                self.print_info("Generating EMA samples...")
                self.print_info("Generating EMA samples done.")
            
            if iteration % self.args.test_interval == 0 and iteration > 0:
                current_lr = self.scheduler.get_last_lr()[0]
                self.log_info(f"Iteration {iteration}, lr: {current_lr:.6f}")
                
                self.validate(iteration)

                self.model.train()
                
        self.finish_training()
        
    def finish_training(self):
        """Finish training"""
        end_time = time.time()
        self.log_info(f"Training complete, best epoch is {self.best_epoch}, best val loss is {self.best_val_loss:.6f}, best val rel l2 loss is {self.best_val_rel_l2_loss:.6f}, time cost is {end_time-self.start_time} s")
        self.log_info(f"Results saved at {self.exp_path}")
        
        if (self.use_accelerator and self.accelerator.is_main_process) or not self.use_accelerator:
            best_ema_path = f"{self.exp_path}/best_ema_model.pth"
            best_main_path = f"{self.exp_path}/best_main_model.pth"
            
            if not os.path.exists(best_ema_path) or not os.path.exists(best_main_path):
                self.log_info("Save final model as best model at end of training")
                self.save_best_model(self.current_epoch, self.best_val_loss, self.best_val_rel_l2_loss)

        if (self.use_accelerator and self.accelerator.is_main_process) or not self.use_accelerator:
            if self.args.is_use_tb and self.writer is not None:
                self.writer.close()
            for handler in self.logger.handlers:
                handler.flush()
                if isinstance(handler, logging.FileHandler):
                    handler.close()

#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    if hasattr(args, 'resume') and args.resume:
        if not args.resume_path:
            raise ValueError("Must specify --resume-path when using --resume")
        
        if not os.path.exists(args.resume_path):
            raise ValueError(f"Resume path does not exist: {args.resume_path}")
        
        latest_checkpoint = os.path.join(args.resume_path, 'latest_checkpoint.pth')
        if os.path.exists(latest_checkpoint):
            args.ckpt = latest_checkpoint
            args.results_path = os.path.dirname(args.resume_path)
            print(f"Resuming training from {latest_checkpoint}")
        else:
            checkpoint_files = [f for f in os.listdir(args.resume_path) if f.startswith('model_') and f.endswith('.pth')]
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                latest_checkpoint = os.path.join(args.resume_path, checkpoint_files[-1])
                args.ckpt = latest_checkpoint
                args.results_path = os.path.dirname(args.resume_path)
                print(f"Resuming training from {latest_checkpoint}")
            else:
                raise ValueError(f"No checkpoint files found in {args.resume_path}")
        
        config_path = os.path.join(args.resume_path, 'config.yaml')
        if os.path.exists(config_path):
            print(f"Loading config from {config_path}")
    
    # Create trainer
    trainer = Trainer(args)
    
    # Select training mode based on parameters
    if hasattr(args, 'train_mode') and args.train_mode == 'iteration':
        # Training based on iteration count
        total_iterations = getattr(args, 'total_iterations', 10000)
        trainer.train_by_iterations(total_iterations)
    else:
        # Training based on epochs
        trainer.train_by_epochs()


if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Training Configurations")

    parser.add_argument("--config", type=str, default="configs/train_neu.yaml") 
    parser.add_argument("--gpu", type=int, default=0)

    # parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    
    # parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=10000)
    parser.add_argument("--cfg-scale", type=float, default=0)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a custom SiT checkpoint")
    
    parser.add_argument("--resume", action="store_true", help="Resume training from previous experiment")
    parser.add_argument("--resume-path", type=str, default=None, help="Path to experiment to resume")
    
    parser.add_argument("--train-mode", type=str, default="epoch", choices=["epoch", "iteration"], 
                       help="Training mode: epoch or iteration")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["step", "cosine"], help="Learning rate scheduler")
    parser.add_argument("--use-torchcfm", default=True, action="store_true", help="Use torchcfm training branch")

    parser.add_argument("--num-sampling-steps", type=int, default=10, help="Number of sampling steps")
    
    parser.add_argument("--use-accelerator", action="store_true", help="Use accelerator for distributed training")
    parser.add_argument("--mixed-precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision training type")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2, help="Gradient accumulation steps")
    
    
    
    parse_transport_args(parser)
    
    args = add_args_from_config(parser)
    
    # args = parser.parse_args()

    from model.SiT import SiT
    
    main(args)
