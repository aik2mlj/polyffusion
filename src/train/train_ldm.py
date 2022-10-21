import torch
from argparse import ArgumentParser
import sys

sys.path.insert(0, "..")

# from stable_diffusion.model.autoencoder import Autoencoder, Encoder, Decoder
from . import *
from stable_diffusion.model.unet import UNetModel
from stable_diffusion.latent_diffusion import LatentDiffusion
from model_sdf import Diffpro_SDF
from dataloader import get_train_val_dataloaders
# from ..params_sdf import params


class LDM_TrainConfig(TrainConfig):
    def __init__(self, params, output_dir, use_autoencoder=False) -> None:
        super().__init__(params, output_dir)
        self.autoencoder = None

        if use_autoencoder:
            # encoder = Encoder(
            #     in_channels=2,
            #     z_channels=4,
            #     channels=64,
            #     channel_multipliers=[1, 2, 4, 4],
            #     n_resnet_blocks=2
            # )

            # decoder = Decoder(
            #     out_channels=2,
            #     z_channels=4,
            #     channels=64,
            #     channel_multipliers=[1, 2, 4, 4],
            #     n_resnet_blocks=2
            # )

            # self.autoencoder = Autoencoder(
            #     emb_channels=4, encoder=encoder, decoder=decoder, z_channels=4
            # ).to(self.device)
            raise NotImplementedError

        self.unet_model = UNetModel(
            in_channels=params.in_channels,
            out_channels=params.out_channels,
            channels=params.channels,
            attention_levels=params.attention_levels,
            n_res_blocks=params.n_res_blocks,
            channel_multipliers=params.channel_multipliers,
            n_heads=params.n_heads,
            tf_layers=params.tf_layers,
            d_cond=params.d_cond
        )

        self.ldm_model = LatentDiffusion(
            linear_start=params.linear_start,
            linear_end=params.linear_end,
            n_steps=params.n_steps,
            latent_scaling_factor=params.latent_scaling_factor,
            autoencoder=self.autoencoder,
            unet_model=self.unet_model
        )

        self.model = Diffpro_SDF(self.ldm_model,
                                 cond_mode=params.cond_mode).to(self.device)
        # Create dataloader
        self.train_dl, self.val_dl = get_train_val_dataloaders(params.batch_size)
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.learning_rate
        )


# if __name__ == "__main__":
#     parser = ArgumentParser(description='train (or resume training) a Diffpro model')
#     parser.add_argument(
#         "--output_dir",
#         default=None,
#         help='directory in which to store model checkpoints and training logs'
#     )
#     args = parser.parse_args()
#     config = LDM_TrainConfig(params, args.output_dir)
#     config.train()
