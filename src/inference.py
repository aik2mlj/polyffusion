import torch
from os.path import join
from argparse import ArgumentParser
from params import params
from datetime import datetime
from dataset import DataSampleNpz
from dirs import *
from utils import estx_to_midi_file, output_to_numpy
from model import Diffpro_diffwave, Diffpro
from params import params
import pickle
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def predict_diffwave(model_dir, fast_sampling=False):
    song_fn, pnotree_x, pnotree_y = choose_song_from_val_dl()
    pnotree_x, pnotree_y = pnotree_x.to(device), pnotree_y.to(device)
    model = Diffpro_diffwave.load_trained(model_dir, params).to(device)
    model.eval()
    model.params.override(params)

    with torch.no_grad():
        # Change in notation from the DiffWave paper for fast sampling.
        # DiffWave paper -> Implementation below
        # --------------------------------------
        # alpha -> talpha
        # beta -> training_noise_schedule
        # gamma -> alpha
        # eta -> beta
        training_noise_schedule = np.array(model.params.noise_schedule)
        inference_noise_schedule = np.array(
            model.params.inference_noise_schedule
        ) if fast_sampling else training_noise_schedule

        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) \
                            / (talpha_cum[t]**0.5 - talpha_cum[t + 1]**0.5)
                    T.append(t + twiddle)
                    break
        T = np.array(T, dtype=np.float32)

        if not model.params.unconditional:
            z_y_prd = torch.randn(pnotree_y.shape[0], model.params.z_dim, device=device)
        else:
            z_y_prd = torch.randn(1, model.params.z_dim, device=device)
        noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n]**0.5
            c2 = beta[n] / (1 - alpha_cum[n])**0.5
            z_y_prd = c1 * (
                z_y_prd -
                c2 * model(z_y_prd, torch.tensor([T[n]], device=device), pnotree_x)
            )
            if n > 0:
                noise = torch.randn_like(z_y_prd)
                sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                z_y_prd += sigma * noise
            z_y_prd = torch.clamp(z_y_prd, -1.0, 1.0)

        # pianotree decoder
        recon_pitch, recon_dur = model.pnotree_dec(z_y_prd, True, None, None, 0, 0)
        y_prd, _, _ = output_to_numpy(recon_pitch, recon_dur)

    output_stamp = f"inf_[{song_fn}]_{datetime.now().strftime('%m-%d_%H%M%S')}.mid"
    estx_to_midi_file(y_prd, f"exp/x_{output_stamp}")


def predict(model_dir, is_sampling=False):
    song_fn, pnotree_x, pnotree_y = choose_song_from_val_dl()
    pnotree_x, pnotree_y = pnotree_x.to(device), pnotree_y.to(device)
    model = Diffpro.load_trained(model_dir, params).to(device)
    model.eval()
    y_prd, _, _ = model.infer(pnotree_x, is_sampling=is_sampling)
    output_stamp = f"inf_[{song_fn}]_{datetime.now().strftime('%m-%d_%H%M%S')}.mid"
    estx_to_midi_file(y_prd, f"exp/x_{output_stamp}")


def choose_song_from_val_dl():
    split_fpath = join(TRAIN_SPLIT_DIR, "split_dict.pickle")
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
    predict_diffwave(args.model_dir, fast_sampling=False)
