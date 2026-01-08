import argparse
from torch import nn
import torch.nn.functional as F
import os
from datetime import datetime

from paradigms.diffusion import GaussianDiffusion
from train import Trainer
from models.unet3d import Unet3D
from models.sit_fno import SiT_FNO
from models.cno import CNO3d
from utils import create_res, set_seed, save_config_from_args, get_parameter_net
from data.double_cylinder import DoubleCylinderDataset, RangeNormalizer

from filepath import ABSOLUTE_PATH

import torch
import random

from functools import partial

def prepare_data(stage, input, target):

    input = input.repeat(1, target.shape[1] // input.shape[1], 1, 1, 1) # (b, 3, h, w, 4) -> (b, 12, h, w, 4)
    if stage == "fluid":
        cond = target[..., -1:] # (b, 12, h, w, 1)
        cond = torch.cat([input, cond], dim=-1) # (b, 12, h, w, 5)
        data = target[..., :-1] # (b, 12, h, w, 3)
    elif stage == "structure":
        cond = target[..., :-1] # (b, 12, h, w, 3)
        cond = torch.cat([input, cond], dim=-1) # (b, 12, h, w, 7)
        data = target[..., -1:] # (b, 12, h, w, 1)
    elif stage == "couple":
        pass
    else:
        raise ValueError(f"Invalid stage: {stage}")
    
    return cond, data


def collate_fn(batch, data_prepare_fn, normalizer):
    cond_list = []
    data_list = []
    for cond, data, _, _, _ in batch:
        cond_list.append(cond)
        data_list.append(data)
    cond = torch.stack(cond_list, dim=0)
    data = torch.stack(data_list, dim=0)
    cond, data = normalizer.preprocess(cond, data)
    cond, data = data_prepare_fn(cond, data)
    cond = cond.permute(0, 4, 1, 2, 3)
    data = data.permute(0, 4, 1, 2, 3)
    return data, cond


def forward_function():
    def func_train(model: nn.Module, batch):
        data, cond_tensor = batch
        loss = model(data, cond_tensor)
        return loss

    def func_val(model: nn.Module, batch, loss_fn=F.mse_loss):
        device = next(model.parameters()).device
        data, cond_tensor = batch
        data = data.to(device)
        cond_tensor = cond_tensor.to(device)
        batchsize = data.shape[0]
        outputs_p = model.sample(batchsize, cond_tensor)
        loss = loss_fn(data, outputs_p)
        return loss

    return func_train, func_val


def parse_tuple(s):
    """Parse a string like '108,88' into a tuple (108, 88)"""
    return tuple(int(x.strip()) for x in s.split(','))

def parse_t_or_f(s):
    """Parse a string like 't' or 'f' into a boolean"""
    if s.strip().lower() in ['t', 'true', '1']:
        return True
    elif s.strip().lower() in ['f', 'false', '0']:
        return False
    else:
        raise ValueError(f"Invalid boolean string: {s}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train simple distribution model (one-tensor cond)")
    parser.add_argument("--exp_id", default="double_cylinder", type=str, help="experiment folder id")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--batchsize", default=8, type=int, help="size of dataset")
    parser.add_argument("--num_steps", default=100000, type=int, help="training epoch")
    parser.add_argument("--diffusion_step", default=250, type=int, help="diffusion_step")
    parser.add_argument("--checkpoint", default=1000, type=int, help="save and sample period")
    parser.add_argument("--overall_results_path", default="results", type=str, help="where to create result folder")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--dataset", default="decouple", type=str, help="dataset, option: decouple, decouple")
    parser.add_argument("--mixed_precision_type", default="no", type=str, help="mixed precision type")
    parser.add_argument("--model_type", default="Unet", type=str, help="Unet or ViT or FNO")
    parser.add_argument("--gradient_accumulate_every", default=1, type=int, help="gradient accumulate every")
    parser.add_argument("--num_workers", default=32, type=int, help="number of workers")
    parser.add_argument("--prefetch_factor", default=16, type=int, help="prefetch factor")
    
    parser.add_argument("--paradigm", default="diffusion", type=str, help="diffusion or surrogate")
    
    # model
    parser.add_argument("--input_size", default="128,128", type=parse_tuple, help="input size as tuple (height,width)")
    
    # model cno
    parser.add_argument("--n_layers", default=2, type=int, help="depth")
    parser.add_argument("--channel_multiplier", default=16, type=int, help="channel multiplier")
    
    # model sit_fno
    parser.add_argument("--depth", default=6, type=int, help="depth")
    parser.add_argument("--hidden_size", default=128, type=int, help="hidden size")
    parser.add_argument("--patch_size", default="2,2", type=parse_tuple, help="patch size as tuple (height,width)")
    parser.add_argument("--num_heads", default=4, type=int, help="number of heads")
    parser.add_argument("--modes", default=4, type=int, help="modes")

    # model unet
    parser.add_argument("--dim", default=8, type=int, help="encode dim")
    
    # dataset
    parser.add_argument("--length", default=999, type=int, help="length")
    parser.add_argument("--input_step", default=3, type=int, help="input step")
    parser.add_argument("--output_step", default=12, type=int, help="output step")
    parser.add_argument("--stride", default=1, type=int, help="stride")
    parser.add_argument("--stage", default="fluid", type=str, help="stage")
    parser.add_argument("--num_delta_t", default=0, type=int, help="num delta t")
    parser.add_argument("--dt", default=10, type=int, help="dt")
    
    # validation
    parser.add_argument("--ddim_validation", default=True, type=parse_t_or_f, help="ddim validation")
    parser.add_argument("--ddim_sampling_timesteps", default=10, type=int, help="ddim sampling timesteps")
    parser.add_argument("--num_val_batches", default=4, type=int, help="num val batches")
    parser.add_argument("--val_batchsize", default=32, type=int, help="val batchsize")
    parser.add_argument("--use_validation", default=True, type=parse_t_or_f, help="use validation")

    # caching arguments
    parser.add_argument("--use_cache", default=True, type=parse_t_or_f, help="use data caching")
    parser.add_argument("--force_recreate_cache", default=False, type=parse_t_or_f, help="force recreate cache even if exists")

    args = parser.parse_args()

    set_seed(args.seed)
    results_path = create_res(args.overall_results_path, folder_name=args.exp_id)
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage = args.stage
    paradigm = args.paradigm
    model_type = args.model_type
    results_folder = os.path.join(
        results_path, paradigm + model_type + stage, time_stamp
    )
    print(f"results_folder: {results_folder}")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    save_config_from_args(args, results_folder)

    # Handle data caching
    from utils import generate_cache_dir_name, check_cache_exists, create_data_cache

    cache_dir_name = generate_cache_dir_name(args, "double_cylinder")

    if args.use_cache:
        # Check and create cache for training data
        if not check_cache_exists(cache_dir_name, 'train') or args.force_recreate_cache:
            print("Creating cache for training data...")
            temp_train_dataset = DoubleCylinderDataset(dataset_path=ABSOLUTE_PATH,
                                             length=args.length,
                                             input_size=args.input_step,
                                             output_size=args.output_step,
                                             stride=args.stride,
                                             mode='train',
                                             stage=args.stage,
                                             num_delta_t=args.num_delta_t,
                                             dt=args.dt,
                                             use_cache=False)  # Don't use cache when creating cache
            create_data_cache(temp_train_dataset, cache_dir_name, 'train')

        # # Check and create cache for validation data
        # if not check_cache_exists(cache_dir_name, 'val') or args.force_recreate_cache:
        #     print("Creating cache for validation data...")
        #     temp_val_dataset = DoubleCylinderDataset(dataset_path=ABSOLUTE_PATH,
        #                                      length=args.length,
        #                                      input_size=args.input_step,
        #                                      output_size=args.output_step,
        #                                      stride=args.stride,
        #                                      mode='val',
        #                                      stage=args.stage,
        #                                      num_delta_t=args.num_delta_t,
        #                                      dt=args.dt,
        #                                      use_cache=False)  # Don't use cache when creating cache
        #     create_data_cache(temp_val_dataset, cache_dir_name, 'val')

    train_dataset = DoubleCylinderDataset(dataset_path=ABSOLUTE_PATH,
                                     length=args.length,
                                     input_size=args.input_step,
                                     output_size=args.output_step,
                                     stride=args.stride,
                                     mode='train',
                                     stage=args.stage,
                                     num_delta_t=args.num_delta_t,
                                     dt=args.dt,
                                     use_cache=args.use_cache)
    val_dataset_original = DoubleCylinderDataset(dataset_path=ABSOLUTE_PATH,
                                     length=args.length,
                                     input_size=args.input_step,
                                     output_size=args.output_step,
                                     stride=args.stride,
                                     mode='val',
                                     stage=args.stage,
                                     num_delta_t=args.num_delta_t,
                                     dt=args.dt,
                                     use_cache=False)  # Don't use cache for validation

    size_val = args.num_val_batches * args.val_batchsize
    indices = random.sample(range(len(val_dataset_original)), size_val)
    val_dataset = torch.utils.data.Subset(val_dataset_original, indices)
    
    example_data = train_dataset[0]
    data_prepare_fn = partial(prepare_data, args.stage)
    cond, data = data_prepare_fn(example_data[0].unsqueeze(0), example_data[1].unsqueeze(0))

    cond = cond.permute(0, 4, 1, 2, 3)
    data = data.permute(0, 4, 1, 2, 3)
    
    if model_type == "Unet":
        # cond_channels from prepared tensor
            model = Unet3D(
                dim=args.dim,
                out_dim=data.shape[1],
                cond_channels=cond.shape[1],
                dim_mults=(1, 2, 4),
                use_sparse_linear_attn=False,
                attn_dim_head=16,
                expects_x=True,
            )
    elif model_type == "SiT_FNO":
        model = SiT_FNO(input_size=args.input_size, 
                depth=args.depth, 
                hidden_size=args.hidden_size, 
                patch_size=args.patch_size, 
                num_heads=args.num_heads, 
                in_channels=data.shape[1] + cond.shape[1],
                out_channels=data.shape[1],
                modes=args.modes)
    elif model_type == "CNO":
        model = CNO3d(in_dim=data.shape[1] + cond.shape[1],
                    out_dim=data.shape[1],
                    in_size=max(args.input_size),
                    N_layers=args.n_layers,
                    channel_multiplier=args.channel_multiplier)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    sampling_timesteps = args.ddim_sampling_timesteps if args.ddim_validation else args.diffusion_step
    GenModel = GaussianDiffusion(model, seq_length=tuple(data.shape[1:]), timesteps=args.diffusion_step, auto_normalize=False, sampling_timesteps=sampling_timesteps)

    train_normalizer = RangeNormalizer(train_dataset, batch_size=args.batchsize)
    val_normalizer = RangeNormalizer(val_dataset_original, batch_size=args.batchsize)

    get_parameter_net(GenModel)

    train_function, val_function = forward_function()
    train = Trainer(
        model=GenModel,
        data_train=train_dataset,
        train_function=train_function,
        val_function=val_function,
        data_val=val_dataset,
        train_lr=args.lr,
        train_num_steps=args.num_steps,
        train_batch_size=args.batchsize,
        save_every=args.checkpoint,
        results_folder=results_folder,
        mixed_precision_type=args.mixed_precision_type,
        gradient_accumulate_every=args.gradient_accumulate_every,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        train_collate_fn=partial(collate_fn, data_prepare_fn=data_prepare_fn, normalizer=train_normalizer),
        val_collate_fn=partial(collate_fn, data_prepare_fn=data_prepare_fn, normalizer=val_normalizer),
        val_batch_size=args.val_batchsize,
        use_validation=args.use_validation,
    )

    train.train()


