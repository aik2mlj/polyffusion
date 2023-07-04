from argparse import ArgumentParser
from params.params_sdf import params as params_sdf
from params.params_sdf_chd8bar import params as params_sdf_chd8bar
from params.params_sdf_pnotree import params as params_sdf_pnotree
from params.params_sdf_txt import params as params_sdf_txt
from params.parmas_sdf_txtvnl import params as params_sdf_txtvnl
from params.params_sdf_concat import params as params_sdf_concat

from params.params_ddpm import params as params_ddpm
from params.params_autoencoder import params as params_autoencoder
from params.params_chd_8bar import params as params_chd_8bar

from train.train_ldm import LDM_TrainConfig
from train.train_autoencoder import Autoencoder_TrainConfig
from train.train_ddpm import DDPM_TrainConfig
from train.train_chd_8bar import Chord8bar_TrainConfig

if __name__ == "__main__":
    parser = ArgumentParser(
        description='train (or resume training) a Polyffusion model'
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help='directory in which to store model checkpoints and training logs'
    )
    parser.add_argument(
        "--pop909_use_track", help='which tracks to use for pop909 training'
    )
    parser.add_argument("--model", help="which model to train (autoencoder, ldm, ddpm)")
    args = parser.parse_args()

    use_track = [0, 1, 2]
    if args.pop909_use_track is not None:
        use_track = [int(x) for x in args.pop909_use_track.split(",")]

    if args.model == "ldm_chdvnl":
        config = LDM_TrainConfig(params_sdf, args.output_dir, use_track=use_track)
    elif args.model == "ldm_chd8bar":
        config = LDM_TrainConfig(
            params_sdf_chd8bar, args.output_dir, use_track=use_track
        )
    elif args.model == "ldm_pnotree":
        config = LDM_TrainConfig(
            params_sdf_pnotree, args.output_dir, use_track=use_track
        )
    elif args.model == "ldm_txt":
        config = LDM_TrainConfig(params_sdf_txt, args.output_dir, use_track=use_track)
    elif args.model == "ldm_txtvnl":
        config = LDM_TrainConfig(
            params_sdf_txtvnl, args.output_dir, use_track=use_track
        )
    elif args.model == "ldm_concat":
        config = LDM_TrainConfig(
            params_sdf_concat, args.output_dir, use_track=use_track
        )
    elif args.model == "ldm_musicalion_pnotree":
        config = LDM_TrainConfig(
            params_sdf_pnotree, args.output_dir, use_musicalion=True
        )
    elif args.model == "ldm_musicalion_txt":
        config = LDM_TrainConfig(params_sdf_txt, args.output_dir, use_musicalion=True)
    elif args.model == "ddpm":
        config = DDPM_TrainConfig(params_ddpm, args.output_dir)
    elif args.model == "autoencoder":
        config = Autoencoder_TrainConfig(params_autoencoder, args.output_dir)
    elif args.model == "chd_8bar":
        config = Chord8bar_TrainConfig(params_chd_8bar, args.output_dir)
    else:
        raise NotImplementedError
    config.train()
