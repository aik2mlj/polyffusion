from argparse import ArgumentParser
from learner import *

from params import params


class Configs():
    # U-Net model for $\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$
    eps_model: UNet
    # [DDPM algorithm](index.html)
    diffusion: DenoiseDiffusion

    # Adam optimizer
    optimizer: torch.optim.Adam

    def __init__(self, params):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)
        self.eps_model = UNet(
            image_channels=params.image_channels,
            n_channels=params.n_channels,
            ch_mults=params.channel_multipliers,
            is_attn=params.is_attention,
        ).to(self.device)

        # Create [DDPM class](index.html)
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=params.n_steps,
            device=self.device,
        )

        self.model = Diffpro_DDPM(self.diffusion, params).to(self.device)

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(total_params)
        # Create dataloader
        self.train_dl, self.val_dl = get_train_val_dataloaders(params.batch_size)
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.eps_model.parameters(), lr=params.learning_rate
        )

    def train(self, params, output_dir=None):
        if output_dir is not None:
            os.makedirs(f"{output_dir}", exist_ok=True)
            output_dir = f"{output_dir}/{datetime.now().strftime('%m-%d_%H%M%S')}"
        else:
            output_dir = f"result/{datetime.now().strftime('%m-%d_%H%M%S')}"
        learner = DiffproLearner(
            output_dir, self.model, self.train_dl, self.val_dl, self.optimizer, params
        )
        learner.train(max_epoch=params.max_epoch)


if __name__ == "__main__":
    parser = ArgumentParser(description='train (or resume training) a Diffpro model')
    parser.add_argument(
        "--output_dir",
        default=None,
        help='directory in which to store model checkpoints and training logs'
    )
    args = parser.parse_args()
    config = Configs(params)
    config.train(params, args.output_dir)
