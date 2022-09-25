from argparse import ArgumentParser

from learner import Configs
from params import params

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
