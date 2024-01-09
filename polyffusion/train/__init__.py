import json
import os
from datetime import datetime
from shutil import copy2
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from learner import PolyffusionLearner
from lightning_learner import LightningLearner
from params import AttrDict


class TrainConfig:
    model: torch.nn.Module
    train_dl: DataLoader
    val_dl: DataLoader
    optimizer: Optimizer

    def __init__(self, params, param_scheduler, output_dir) -> None:
        self.model_name = params.model_name
        self.params = params
        self.param_scheduler = param_scheduler

        self.resume = False
        if os.path.exists(f"{output_dir}/chkpts/last.ckpt"):
            print("Checkpoint already exists.")
            if input("Resume training? (y/n)") == "y":
                self.resume = True
            else:
                print("Aborting...")
                exit(0)
        else:
            output_dir = f"{output_dir}/{datetime.now().strftime('%y-%m-%d_%H%M%S')}"
            print(f"Creating new log folder as {output_dir}")
            os.makedirs(output_dir)

        self.output_dir = output_dir
        self.log_dir = f"{output_dir}/logs"
        self.checkpoint_dir = f"{output_dir}/chkpts"
        self.time_stamp = Path(output_dir).name

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # json dump
        if os.path.exists(f"{output_dir}/params.json"):
            with open(f"{output_dir}/params.json", "r+") as params_file:
                old_params = AttrDict(json.load(params_file))

                # The "weights" attribute is a tuple in AttrDict, but saved as a list. To compare these two, we make them both tuples:
                if "weights" in old_params:
                    old_params["weights"] = tuple(old_params["weights"])

                if old_params != self.params:
                    print("New params differ, using new params could break things.")
                    if (
                        input(
                            "Do you want to keep the old params file (y/n)? The model will still be trained on new params regardless."
                        )
                        == "y"
                    ):
                        time_stamp = datetime.now().strftime("%y-%m-%d_%H%M%S")
                        copy2(
                            f"{output_dir}/params.json",
                            f"{output_dir}/old_params_{time_stamp}.json",
                        )
                    params_file.seek(0)
                    json.dump(self.params, params_file)
                    params_file.truncate()
        else:
            json.dump(self.params, open(f"{output_dir}/params.json", "w"))

    def train(self):
        total_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_parameters}")
        print(json.dumps(self.params, sort_keys=True, indent=4))

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            monitor="val/loss",
            filename="epoch{epoch}-val_loss{val/loss:.6f}",
            save_last=True,
            save_top_k=3,
            auto_insert_metric_name=False,
        )
        logger = WandbLogger(
            project=f"Polyff-{self.model_name}",
            save_dir=self.log_dir,
            name=self.time_stamp,
        )
        trainer = Trainer(
            default_root_dir=self.output_dir,
            callbacks=[checkpoint_callback],
            max_epochs=self.params.max_epoch,
            logger=logger,
            precision="16-mixed" if self.params.fp16 else "32-true",
        )
        learner = LightningLearner(
            self.model,
            self.optimizer,
            self.params,
            self.param_scheduler,
        )
        trainer.fit(
            learner,
            self.train_dl,
            self.val_dl,
            ckpt_path="last" if self.resume else None,
        )
