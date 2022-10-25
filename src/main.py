from argparse import ArgumentParser
from params.params_sdf import params as params_sdf
from params.params_ddpm import params as params_ddpm
from params.params_autoencoder import params as params_autoencoder

from train.train_ldm import LDM_TrainConfig
from train.train_autoencoder import Autoencoder_TrainConfig
from train.train_ddpm import DDPM_TrainConfig

if __name__ == "__main__":
    parser = ArgumentParser(description='train (or resume training) a Diffpro model')
    parser.add_argument(
        "--output_dir",
        default=None,
        help='directory in which to store model checkpoints and training logs'
    )
    parser.add_argument("--model", help="which model to train (autoencoder, ldm, ddpm)")
    args = parser.parse_args()
    if args.model == "ldm":
        config = LDM_TrainConfig(params_sdf, args.output_dir)
    elif args.model == "ddpm":
        config = DDPM_TrainConfig(params_ddpm, args.output_dir)
    elif args.model == "autoencoder":
        config = Autoencoder_TrainConfig(params_autoencoder, args.output_dir)
    else:
        raise NotImplementedError
    config.train()
