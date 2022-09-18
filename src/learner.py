import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from os.path import join
from datetime import datetime

from dataloader import get_train_val_dataloaders
from dirs import *
from model import Diffpro_diffwave
from utils import nested_map


class DiffproLearner:
    def __init__(self, output_dir, model, train_dl, val_dl, optimizer, params):
        self.output_dir = output_dir
        self.log_dir = f"{output_dir}/logs"
        self.checkpoint_dir = f"{output_dir}/chkpts"
        os.makedirs(self.log_dir)
        os.makedirs(self.checkpoint_dir)

        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.optimizer = optimizer
        self.params = params

        self.step = 0
        self.epoch = 0
        self.summary_writer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.autocast = torch.cuda.amp.autocast(enabled=params.fp16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=params.fp16)

        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))

    def _write_summary(self, step, losses: dict, type):
        """type: train or val"""
        summary_losses = losses
        summary_losses["grad_norm"] = self.grad_norm
        writer = self.summary_writer or SummaryWriter(self.log_dir, purge_step=step)
        writer.add_scalars(type, losses, step)
        writer.flush()
        self.summary_writer = writer

    def state_dict(self):
        model_state = self.model.state_dict()
        return {
            "step": self.step,
            "epoch": self.epoch,
            "model":
                {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in model_state.items()
                },
            "optimizer":
                {
                    k: v.cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in self.optimizer.state_dict().items()
                },
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step = state_dict["step"]
        self.epoch = state_dict["epoch"]
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.scaler.load_state_dict(state_dict["scaler"])

    def restore_from_checkpoint(self, fname="weights"):
        try:
            fpath = f"{self.checkpoint_dir}/{fname}.pt"
            checkpoint = torch.load(fpath)
            self.load_state_dict(checkpoint)
            print(f"restored from checkpoint {fpath}!")
            return True
        except FileNotFoundError:
            print("No checkpoint found. Starting from scratch...")
            return False

    def save_to_checkpoint(self, fname="weights"):
        save_name = f"{fname}-{self.epoch}.pt"
        save_fpath = f"{self.checkpoint_dir}/{save_name}"
        link_fpath = f"{self.checkpoint_dir}/{fname}.pt"
        torch.save(self.state_dict(), save_fpath)
        if os.path.islink(link_fpath):
            os.unlink(link_fpath)
        os.symlink(save_name, link_fpath)

    def train(self, max_epoch=None):
        self.model.train()
        while True:
            self.epoch = self.step // len(self.train_dl)
            if max_epoch is not None and self.epoch >= max_epoch:
                return

            for batch in tqdm(self.train_dl, desc=f"Epoch {self.epoch}"):
                batch = nested_map(
                    batch, lambda x: x.to(self.device)
                    if isinstance(x, torch.Tensor) else x
                )
                losses = self.train_step(batch)
                # check NaN
                for loss_value in list(losses.values()):
                    if isinstance(loss_value,
                                  torch.Tensor) and torch.isnan(loss_value).any():
                        raise RuntimeError(
                            f"Detected NaN loss at step {self.step}, epoch {self.epoch}"
                        )
                if self.step % 50 == 0:
                    self._write_summary(self.step, losses, "train")
                if self.step % 5000 == 0:
                    self.valid()
                self.step += 1

            # valid
            self.valid()

    def valid(self):
        losses = None
        for batch in self.val_dl:
            batch = nested_map(
                batch, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x
            )
            current_losses = self.val_step(batch)
            losses = losses or current_losses
            for k, v in current_losses.items():
                losses[k] += v
        assert losses is not None
        for k, v in losses.items():
            losses[k] /= len(self.val_dl)
        self._write_summary(self.step, losses, "val")

        self.save_to_checkpoint()

    def train_step(self, batch):
        # people say this is the better way to set zero grad
        # instead of self.optimizer.zero_grad()
        for param in self.model.parameters():
            param.grad = None

        pnotree_x, pnotree_y = batch

        # here forward the model
        with self.autocast:
            loss_dict = self.model.get_loss_dict(pnotree_x, pnotree_y, self.noise_level)

        loss = loss_dict["loss"]
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad.clip_grad_norm_(
            self.model.parameters(), self.params.max_grad_norm or 1e9
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss_dict

    def val_step(self, batch):
        with torch.no_grad():
            pnotree_x, pnotree_y = batch
            with self.autocast:
                loss_dict = self.model.get_loss_dict(pnotree_x, pnotree_y)
        return loss_dict


def train(params, output_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Diffpro_diffwave(params, pt_pnotree_model_path=PT_PNOTREE_PATH).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    train_dl, val_dl = get_train_val_dataloaders(params.batch_size, params)
    if output_dir is not None:
        os.makedirs(f"{output_dir}", exist_ok=True)
        output_dir = f"{output_dir}/{datetime.now().strftime('%m-%d_%H%M%S')}"
    else:
        output_dir = f"result/{datetime.now().strftime('%m-%d_%H%M%S')}"
    learner = DiffproLearner(output_dir, model, train_dl, val_dl, optimizer, params)
    learner.train(max_epoch=params.max_epoch)
