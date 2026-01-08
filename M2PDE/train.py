import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from accelerate import Accelerator
from ema_pytorch import EMA

from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

import csv


__version__ = "1.0.0"


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def cycle(dl):
    while True:
        for data in dl:
            yield data


def exists(x):
    return x is not None


class Trainer(object):

    def __init__(
        self,
        model,
        data_train,
        data_val,
        train_function,
        val_function,
        train_batch_size=16,
        gradient_accumulate_every=2,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_every=1000,
        num_samples=25,
        results_folder="./results",
        mixed_precision_type="no",
        split_batches=True,
        max_grad_norm=1.0,
        loss_fn=F.mse_loss,
        num_workers=4,
        prefetch_factor=4,
        train_collate_fn=None,
        val_collate_fn=None,
        val_batch_size=None,
        use_validation=True,
        use_tensorboard=True,
    ):
        super().__init__()
        self.use_validation = use_validation
        self.use_tensorboard = use_tensorboard
        self.loss_fn = loss_fn
        self.train_function = train_function
        self.val_function = val_function
        # accelerator
        self.accelerator = Accelerator(
            split_batches=split_batches, mixed_precision=mixed_precision_type
        )

        # model

        self.model = model

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), "number of samples must have an integer square root"
        self.num_samples = num_samples
        self.save_every = save_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        dl = (
            DataLoader(
                data_train,
                batch_size=train_batch_size,
                shuffle=True,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=True,
                collate_fn=train_collate_fn,
            )
            if not isinstance(data_train, DataLoader)
            else data_train
        )
        if not exists(val_batch_size):
            val_batch_size = train_batch_size * 10
        self.data_val = (
            DataLoader(
                data_val,
                batch_size=val_batch_size,
                shuffle=False,
                collate_fn=val_collate_fn,
            )
            if not isinstance(data_val, DataLoader)
            else data_val
        )
        
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        self.opt = Adam(model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # tensorboard writer and CSV logging
        if self.use_tensorboard and self.accelerator.is_main_process:
            self.writer = SummaryWriter(log_dir=str(self.results_folder / "tensorboard"))
            self.accelerator.print("Tensorboard logging enabled")

            # Set up CSV logging as backup/fallback
            self.csv_log_file = self.results_folder / "training_log.csv"
            self.csv_writer = None
            self._init_csv_logging()
        else:
            self.writer = None
            self.csv_writer = None

        # step counter state

        self.step = 0
        self.record = []  # add: milestone, train loss, test loss per checkpoint
        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def _init_csv_logging(self):
        """Initialize CSV logging file"""
        if self.accelerator.is_main_process and self.use_tensorboard:
            with open(self.csv_log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'train_loss', 'learning_rate', 'val_loss'])

    def _log_to_csv(self, step, train_loss=None, learning_rate=None, val_loss=None):
        """Log metrics to CSV file"""
        if self.accelerator.is_main_process and self.use_tensorboard:
            with open(self.csv_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, train_loss or '', learning_rate or '', val_loss or ''])

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            "version": __version__,
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))
        np.save(str(self.results_folder / "record.npy"), np.array(self.record))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f"model-{milestone}.pt"), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if "version" in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.dl)
                    with self.accelerator.autocast():
                        loss = self.train_function(self.model, batch) # , self.loss_fn)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f"loss: {total_loss:.6f}")

                # Log training loss to tensorboard and CSV
                if self.writer is not None:
                    self.writer.add_scalar("train/loss", total_loss, self.step)
                self._log_to_csv(self.step, train_loss=total_loss)

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1

                # Log learning rate to tensorboard and CSV
                current_lr = self.opt.param_groups[0]['lr']
                if self.writer is not None:
                    self.writer.add_scalar("train/learning_rate", current_lr, self.step)
                self._log_to_csv(self.step, learning_rate=current_lr)
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_every == 0:
                        milestone = self.step // self.save_every
                        if self.use_validation:
                            self.ema.ema_model.eval()
                            with torch.no_grad():
                                loss_val = 0
                                for batch in self.data_val:
                                    loss_val += self.val_function(self.ema.ema_model, batch, self.loss_fn)
                                # print("mse in validation data: ", loss_val / (len(self.data_val)))
                        else:
                            loss_val = torch.tensor(0.0)
                        self.record.append([milestone, total_loss, loss_val.item()])

                        # Log validation loss to tensorboard and CSV
                        if self.writer is not None:
                            self.writer.add_scalar("val/loss", loss_val.item(), self.step)
                        self._log_to_csv(self.step, val_loss=loss_val.item())

                        self.save(milestone)
                pbar.update(1)

        accelerator.print("training complete")
        # Only summarize on main process; other ranks have empty records
        if accelerator.is_main_process and len(self.record) > 0:
            self.record = torch.tensor(self.record)
            min_index = torch.argmin(self.record[:, 2])
            accelerator.print(
                "Minimum validation error is: ", self.record[min_index, 2], " in milestone: ", self.record[min_index, 0]
            )

        # Close tensorboard writer
        if self.writer is not None:
            self.writer.close()
