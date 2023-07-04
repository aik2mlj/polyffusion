import torch
import sys
import os

# from stable_diffusion.model.autoencoder import Autoencoder, Encoder, Decoder
from . import *
from data.dataloader import get_train_val_dataloaders
from dl_modules import ChordEncoder, ChordDecoder
from models.model_chd_8bar import Chord_8Bar
from train.scheduler import TeacherForcingScheduler, ParameterScheduler


class Chord8bar_TrainConfig(TrainConfig):
    def __init__(self, params, output_dir) -> None:
        # Teacher-forcing rate for Chord VAE training
        tfr_chd = params.tfr_chd
        tfr_chd_scheduler = TeacherForcingScheduler(*tfr_chd)
        params_dict = dict(tfr_chd=tfr_chd_scheduler)
        param_scheduler = ParameterScheduler(**params_dict)

        super().__init__(params, param_scheduler, output_dir)

        self.chord_enc = ChordEncoder(
            input_dim=params.chd_input_dim,
            hidden_dim=params.chd_hidden_dim,
            z_dim=params.chd_z_dim
        )
        self.chord_dec = ChordDecoder(
            input_dim=params.chd_input_dim,
            z_input_dim=params.chd_z_input_dim,
            hidden_dim=params.chd_hidden_dim,
            z_dim=params.chd_z_dim,
            n_step=params.chd_n_step
        )
        self.model = Chord_8Bar(
            self.chord_enc,
            self.chord_dec,
        ).to(self.device)
        # Create dataloader
        self.train_dl, self.val_dl = get_train_val_dataloaders(params.batch_size)
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.learning_rate
        )
