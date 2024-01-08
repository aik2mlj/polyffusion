import json
from pathlib import Path

import lightning
import torch

from dirs import *


class LightningLearner(lightning.LightningModule):
    def __init__(self, output_dir, model, optimizer, params, param_scheduler):
        self.model_name = Path(output_dir).parent.name
        self.output_dir = output_dir
        self.log_dir = f"{output_dir}/logs"
        self.checkpoint_dir = f"{output_dir}/chkpts"
        self.model = model
        self.optimizer = optimizer
        self.params = params
        self.param_scheduler = param_scheduler  # teacher-forcing stuff

        print(json.dumps(self.params, sort_keys=True, indent=4))

        self.save_hyperparameters("output_dir", "params", "param_scheduler")

    def training_step(self, batch, batch_idx):
        if self.param_scheduler is not None:
            scheduled_params = self.param_scheduler.step()
            loss_dict = self.model.get_loss_dict(
                batch, self.global_step, **scheduled_params
            )
        else:
            scheduled_params = None
            loss_dict = self.model.get_loss_dict(batch, self.global_step)

        # check NaN
        for loss_value in list(loss_dict.values()):
            if isinstance(loss_value, torch.Tensor) and torch.isnan(loss_value).any():
                raise RuntimeError(
                    f"Detected NaN loss at step {self.global_step}, epoch {self.epoch}"
                )

        for k, v in loss_dict.items():
            loss_dict[f"train/{k}"] = v
        self.log_dict(loss_dict, prog_bar=True)

        loss = loss_dict["loss"]
        return loss

    def validation_step(self, batch, batch_idx):
        if self.param_scheduler is not None:
            scheduled_params = self.param_scheduler.step()
            loss_dict = self.model.get_loss_dict(
                batch, self.global_step, **scheduled_params
            )
        else:
            scheduled_params = None
            loss_dict = self.model.get_loss_dict(batch, self.global_step)

        for k, v in loss_dict.items():
            loss_dict[f"val/{k}"] = v
        self.log_dict(loss_dict)

    def configure_optimizers(self):
        return self.optimizer
