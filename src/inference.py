import torch
from os.path import join
from argparse import ArgumentParser
from params import params
from datetime import datetime
from dataset import DataSampleNpz
from dirs import *
from utils import estx_to_midi_file
from model import Diffpro
import pickle


def predict(model_dir, is_sampling=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    song_fn, pnotree_x, pnotree_y = choose_song_from_val_dl()
    pnotree_x, pnotree_y = pnotree_x.to(device), pnotree_y.to(device)
    model = Diffpro.load_trained(model_dir, params).to(device)
    y_prd, _, _ = model.infer(pnotree_x, is_sampling=is_sampling)
    output_stamp = f"inf_[{song_fn}]_{datetime.now().strftime('%m-%d_%H%M%S')}.mid"
    estx_to_midi_file(y_prd, f"exp/x_{output_stamp}")


def choose_song_from_val_dl():
    split_fpath = join(TRAIN_SPLIT_DIR, "musicalion.pickle")
    with open(split_fpath, "rb") as f:
        split = pickle.load(f)
    print(split[1])
    num = int(input("choose one:"))
    song_fn = split[1][num]
    print(song_fn)

    song = DataSampleNpz(song_fn)
    pnotree_x, pnotree_y = song.get_whole_song_data()
    estx_to_midi_file(pnotree_x, "exp/origin_x.mid")
    estx_to_midi_file(pnotree_y, "exp/origin_y.mid")
    return song_fn, pnotree_x, pnotree_y


if __name__ == "__main__":
    parser = ArgumentParser(description='inference a Diffpro model')
    parser.add_argument(
        "--model_dir", help='directory in which trained model checkpoints are stored'
    )
    args = parser.parse_args()
    predict(args.model_dir)
