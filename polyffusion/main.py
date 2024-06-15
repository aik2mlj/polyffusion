from argparse import ArgumentParser

from omegaconf import OmegaConf

from train.train_autoencoder import Autoencoder_TrainConfig
from train.train_chd_8bar import Chord8bar_TrainConfig
from train.train_ddpm import DDPM_TrainConfig
from train.train_ldm import LDM_TrainConfig

if __name__ == "__main__":
    parser = ArgumentParser(
        description="train (or resume training) a Polyffusion model"
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="directory in which to store model checkpoints and training logs",
    )
    parser.add_argument(
        "--data_dir", default=None, help="directory of custom training data, in npzs"
    )
    parser.add_argument(
        "--pop909_use_track",
        default="0,1,2",
        help="which tracks to use for pop909 (default dataset) training. (0: melody, 1: bridge, 2: piano accompaniment)",
    )
    parser.add_argument("--model", help="which model to train (autoencoder, ldm, ddpm)")
    args = parser.parse_args()

    use_track = [int(x) for x in args.pop909_use_track.split(",")]

    params = OmegaConf.load(f"polyffusion/params/{args.model}.yaml")

    if args.model.startswith("sdf"):
        use_musicalion = "musicalion" in args.model
        config = LDM_TrainConfig(
            params,
            args.output_dir,
            use_musicalion,
            use_track=use_track,
            data_dir=args.data_dir,
        )
    elif args.model == "ddpm":
        config = DDPM_TrainConfig(params, args.output_dir, data_dir=args.data_dir)
    elif args.model == "autoencoder":
        config = Autoencoder_TrainConfig(
            params, args.output_dir, data_dir=args.data_dir
        )
    elif args.model == "chd_8bar":
        config = Chord8bar_TrainConfig(params, args.output_dir, data_dir=args.data_dir)
    else:
        raise NotImplementedError
    config.train()
