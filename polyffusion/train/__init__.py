import os
from datetime import datetime
from pathlib import Path
from shutil import copy2

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from lightning_learner import LightningLearner
from utils import convert_json_to_yaml


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
            os.makedirs(output_dir, exist_ok=True)

        self.output_dir = output_dir
        self.log_dir = f"{output_dir}/logs"
        self.checkpoint_dir = f"{output_dir}/chkpts"
        self.time_stamp = Path(output_dir).name

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # json to yaml (compatibility)
        if os.path.exists(f"{output_dir}/params.json"):
            convert_json_to_yaml(f"{output_dir}/params.json")

        if os.path.exists(f"{output_dir}/params.yaml"):
            old_params = OmegaConf.load(f"{output_dir}/params.yaml")

            # The "weights" attribute is a tuple in AttrDict, but saved as a list. To compare these two, we make them both tuples:
            # if "weights" in old_params:
            #     old_params["weights"] = tuple(old_params["weights"])

            if old_params != self.params:
                print("New params differ, using new params could break things.")
                if (
                    input(
                        "Do you want to keep the old params file (y/n)? The model will be trained on new params regardless."
                    )
                    == "y"
                ):
                    time_stamp = datetime.now().strftime("%y-%m-%d_%H%M%S")
                    copy2(
                        f"{output_dir}/params.yaml",
                        f"{output_dir}/old_params_{time_stamp}.yaml",
                    )
                    print(f"Old params saved as old_params_{time_stamp}.yaml")
        # save params
        OmegaConf.save(self.params, f"{output_dir}/params.yaml")

    def train(self):
        total_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total parameters: {total_parameters}")
        print(OmegaConf.to_yaml(self.params))

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
