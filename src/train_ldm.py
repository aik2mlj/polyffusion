import torch
from datetime import datetime

from learner import DiffproLearner
# from stable_diffusion.model.autoencoder import Autoencoder, Encoder, Decoder
from stable_diffusion.model.unet import UNetModel
from stable_diffusion.latent_diffusion import LatentDiffusion
from model_sdf import Diffpro_SDF
from dataloader import get_train_val_dataloaders
from dirs import *
from params_sdf import params


class LDM_TrainConfig():
    optimizer: torch.optim.Adam

    def __init__(self, params, use_autoencoder=False) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)
        self.autoencoder = None
        self.params = params
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

        self.model = Diffpro_SDF(self.ldm_model).to(self.device)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(total_params)

        # Create dataloader
        self.train_dl, self.val_dl = get_train_val_dataloaders(params.batch_size)
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.learning_rate
        )

    def train(self, output_dir=None):
        if output_dir is not None:
            os.makedirs(f"{output_dir}", exist_ok=True)
            output_dir = f"{output_dir}/{datetime.now().strftime('%m-%d_%H%M%S')}"
        else:
            output_dir = f"result/{datetime.now().strftime('%m-%d_%H%M%S')}"
        learner = DiffproLearner(
            output_dir, self.model, self.train_dl, self.val_dl, self.optimizer,
            self.params
        )
        learner.train(max_epoch=self.params.max_epoch)


from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description='train (or resume training) a Diffpro model')
    parser.add_argument(
        "--output_dir",
        default=None,
        help='directory in which to store model checkpoints and training logs'
    )
    args = parser.parse_args()
    config = LDM_TrainConfig(params)
    config.train(args.output_dir)
