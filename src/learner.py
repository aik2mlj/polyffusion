import numpy as np
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter

from dataloader import get_train_val_dataloaders


class DiffproLearner:
    def __init__(self, output_dir, model, train_dl, val_dl, optimizer):
        os.makedirs(output_dir)
        self.output_dir = output_dir
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.optimizer = optimizer

        self.step = 0
        self.summary_writer = None

    def _write_summary(self, step, losses):
        writer = self.summary_writer or SummaryWriter(self.output_dir, purge_step=step)
        for loss_name, loss in losses:
            writer.add_scalar(loss_name, loss, step)
        writer.add_scalar('grad_norm', self.grad_norm, step)
        writer.flush()
        self.summary_writer = writer
