# This file is unused

import torch

from data.dataloader import get_custom_train_val_dataloaders, get_train_val_dataloaders
from dirs import *
from models.model_autoencoder import Polyffusion_Autoencoder
from stable_diffusion.model.autoencoder import Autoencoder, Decoder, Encoder

from . import TrainConfig


class Autoencoder_TrainConfig(TrainConfig):
    model: Autoencoder
    optimizer: torch.optim.Adam

    def __init__(self, params, output_dir, data_dir=None) -> None:
        super().__init__(params, None, output_dir)
        encoder = Encoder(
            in_channels=params.in_channels,
            z_channels=params.z_channels,
            channels=params.channels,
            channel_multipliers=params.channel_multipliers,
            n_resnet_blocks=params.n_res_blocks,
        )

        decoder = Decoder(
            out_channels=params.out_channels,
            z_channels=params.z_channels,
            channels=params.channels,
            channel_multipliers=params.channel_multipliers,
            n_resnet_blocks=params.n_res_blocks,
        )

        autoencoder = Autoencoder(
            encoder=encoder,
            decoder=decoder,
            emb_channels=params.emb_channels,
            z_channels=params.z_channels,
        )

        self.model = Polyffusion_Autoencoder(autoencoder)

        # Create dataloader
        if data_dir is None:
            self.train_dl, self.val_dl = get_train_val_dataloaders(
                params.batch_size, params.num_workers, params.pin_memory
            )
        else:
            self.train_dl, self.val_dl = get_custom_train_val_dataloaders(
                params.batch_size,
                data_dir,
                num_workers=params.num_workers,
                pin_memory=params.pin_memory,
            )

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.learning_rate
        )
