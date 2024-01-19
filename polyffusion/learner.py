import os
from typing import Optional

import torch
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from dirs import *
from utils import nested_map


class PolyffusionLearner:
    def __init__(
        self, output_dir, model, train_dl, val_dl, optimizer, params, param_scheduler
    ):
        self.output_dir = output_dir
        self.log_dir = f"{output_dir}/logs"
        self.checkpoint_dir = f"{output_dir}/chkpts"
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.optimizer = optimizer
        self.params = params
        self.param_scheduler = param_scheduler  # teacher-forcing stuff
        self.step = 0
        self.epoch = 0
        self.grad_norm = 0.0
        self.summary_writer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.autocast = torch.cuda.amp.autocast(enabled=params.fp16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=params.fp16)

        self.best_val_loss = torch.tensor([1e10], device=self.device)

        # restore if directory exists
        if os.path.exists(self.output_dir):
            self.restore_from_checkpoint()
            print("found previous output directory")
        else:
            os.makedirs(self.output_dir)
            os.makedirs(self.log_dir)
            os.makedirs(self.checkpoint_dir)
            OmegaConf.save(self.params, f"{output_dir}/params.yaml")

        print(OmegaConf.to_yaml(self.params))
        wandb.init(project=f"Polyff-{output_dir}".replace("/", "-"), config=params)

    def _write_summary(self, losses: dict, scheduled_params: Optional[dict], type):
        """type: train or val"""
        summary_losses = losses
        summary_losses["grad_norm"] = self.grad_norm
        if scheduled_params is not None:
            for k, v in scheduled_params.items():
                summary_losses[f"sched_{k}"] = v
        writer = self.summary_writer or SummaryWriter(
            self.log_dir, purge_step=self.step
        )
        writer.add_scalars(type, summary_losses, self.step)
        writer.flush()
        self.summary_writer = writer

        wandb_losses = {}
        for k, v in summary_losses.items():
            wandb_losses[f"{type}/{k}"] = v
        wandb.log(wandb_losses)

    def state_dict(self):
        model_state = self.model.state_dict()
        return {
            "step": self.step,
            "epoch": self.epoch,
            "model": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_state.items()
            },
            "optimizer": {
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
            print(f"Restored from checkpoint {fpath} --> {fname}-{self.epoch}.pt!")
            return True
        except FileNotFoundError:
            print("No checkpoint found. Starting from scratch...")
            return False

    def _link_checkpoint(self, save_name, link_fpath):
        if os.path.islink(link_fpath):
            os.unlink(link_fpath)
        os.symlink(save_name, link_fpath)

    def write_epoch_info(self):
        with open(f"{self.checkpoint_dir}/info.txt", "w") as f:
            f.write(str(self.epoch))
            f.write(str(self.best_val_loss))

    def save_to_checkpoint(self, fname="weights", is_best=False):
        save_fpath = f"{self.checkpoint_dir}/{fname}.pt"
        save_best_fpath = f"{self.checkpoint_dir}/{fname}_best.pt"
        # link_best_fpath = f"{self.checkpoint_dir}/{fname}_best.pt"
        # link_fpath = f"{self.checkpoint_dir}/{fname}.pt"
        torch.save(self.state_dict(), save_fpath)
        # self._link_checkpoint(save_name, link_fpath)
        if is_best:
            # self._link_checkpoint(save_name, link_best_fpath)
            torch.save(self.state_dict(), save_best_fpath)
            self.write_epoch_info()

    def train(self, max_epoch=None):
        self.model.train()

        while True:
            if self.param_scheduler is not None:
                self.param_scheduler.train()
            if max_epoch is not None and self.epoch >= max_epoch:
                return

            for _step, batch in enumerate(
                tqdm(self.train_dl, desc=f"Epoch {self.epoch}")
            ):
                batch = nested_map(
                    batch,
                    lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x,
                )
                losses, scheduled_params = self.train_step(batch)
                # check NaN
                for loss_value in list(losses.values()):
                    if (
                        isinstance(loss_value, torch.Tensor)
                        and torch.isnan(loss_value).any()
                    ):
                        raise RuntimeError(
                            f"Detected NaN loss at step {self.step}, epoch {self.epoch}"
                        )
                self.step += 1
                if self.step % 100 == 0:
                    self._write_summary(losses, scheduled_params, "train")
                if _step % 5000 == 4999:
                    break
                    # self.model.eval()
                    # self.valid()
                    # self.model.train()
            self.epoch += 1

            # valid
            self.model.eval()
            self.valid()
            self.model.train()

    def valid(self):
        if self.param_scheduler is not None:
            self.param_scheduler.eval()
        losses = None
        for batch in self.val_dl:
            batch = nested_map(
                batch, lambda x: x.to(self.device) if isinstance(x, torch.Tensor) else x
            )
            current_losses, _ = self.val_step(batch)
            if losses is None:
                losses = current_losses
            else:
                for k, v in current_losses.items():
                    losses[k] += v
        assert losses is not None
        for k, v in losses.items():
            losses[k] /= len(self.val_dl)
        self._write_summary(losses, None, "val")

        if self.best_val_loss >= losses["loss"]:
            self.best_val_loss = losses["loss"]
            self.save_to_checkpoint(is_best=True)
        else:
            self.save_to_checkpoint(is_best=False)

    def train_step(self, batch):
        # people say this is the better way to set zero grad
        # instead of self.optimizer.zero_grad()
        for param in self.model.parameters():
            param.grad = None

        # here forward the model
        with self.autocast:
            if self.param_scheduler is not None:
                scheduled_params = self.param_scheduler.step()
                loss_dict = self.model.get_loss_dict(
                    batch, self.step, **scheduled_params
                )
            else:
                scheduled_params = None
                loss_dict = self.model.get_loss_dict(batch, self.step)

        loss = loss_dict["loss"]
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad.clip_grad_norm_(
            self.model.parameters(), self.params.max_grad_norm or 1e9
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss_dict, scheduled_params

    def val_step(self, batch):
        with torch.no_grad():
            with self.autocast:
                if self.param_scheduler is not None:
                    scheduled_params = self.param_scheduler.step()
                    loss_dict = self.model.get_loss_dict(
                        batch, self.step, **scheduled_params
                    )
                else:
                    scheduled_params = None
                    loss_dict = self.model.get_loss_dict(batch, self.step)

        return loss_dict, scheduled_params
