from data.dataloader import get_custom_train_val_dataloaders, get_train_val_dataloaders
from ddpm import DenoiseDiffusion
from ddpm.unet import UNet
from models.model_ddpm import Polyffusion_DDPM

from . import *


class DDPM_TrainConfig(TrainConfig):
    # U-Net model for $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
    eps_model: UNet
    # [DDPM algorithm](index.html)
    diffusion: DenoiseDiffusion

    # Adam optimizer
    optimizer: torch.optim.Adam

    def __init__(self, params, output_dir, data_dir=None):
        super().__init__(params, None, output_dir)

        self.eps_model = UNet(
            image_channels=params.image_channels,
            n_channels=params.n_channels,
            ch_mults=params.channel_multipliers,
            is_attn=params.is_attention,
        )

        # Create [DDPM class](index.html)
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=params.n_steps,
        )

        self.model = Polyffusion_DDPM(self.diffusion, params)
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
            self.eps_model.parameters(), lr=params.learning_rate
        )
