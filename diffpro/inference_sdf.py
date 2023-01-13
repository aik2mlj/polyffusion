"""
---
title: Denoising Diffusion Probabilistic Models (DDPM) Sampling
summary: >
 Annotated PyTorch implementation/tutorial of
 Denoising Diffusion Probabilistic Models (DDPM) Sampling
 for stable diffusion model.
---

# Denoising Diffusion Probabilistic Models (DDPM) Sampling

For a simpler DDPM implementation refer to our [DDPM implementation](../../ddpm/index.html).
We use same notations for $\alpha_t$, $\beta_t$ schedules, etc.
"""

from typing import Optional, List
import numpy as np

import numpy as np
import torch
import json
import random

from labml import monit
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.model.unet import UNetModel
from models.model_sdf import Diffpro_SDF
# from params_sdf import params
from params import AttrDict

from os.path import join
from argparse import ArgumentParser
import pickle
from tqdm import tqdm
from datetime import datetime

from data.dataset import DataSampleNpz
from data.dataset_musicalion import DataSampleNpz_Musicalion
from dirs import *
from utils import (
    prmat2c_to_midi_file, show_image, chd_to_midi_file, estx_to_midi_file,
    load_pretrained_pnotree_enc_dec, load_pretrained_txt_enc,
    load_pretrained_chd_enc_dec, prmat_to_midi_file, show_image, prmat2c_to_prmat
)
from sampler_sdf import SDFSampler
from polydis_aftertouch import PolydisAftertouch
from data.datasample import DataSample
from data.midi_to_data import get_data_for_single_midi

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = ArgumentParser(description='inference a Diffpro model')


def get_data_preprocessed(song):
    prmat2c, pnotree, chord, prmat = song.get_whole_song_data()
    prmat2c_np = prmat2c.squeeze().cpu().numpy()
    pnotree_np = pnotree.cpu().numpy()
    prmat2c_to_midi_file(prmat2c_np, "exp/o_prmat2c.mid")
    estx_to_midi_file(pnotree_np, "exp/o_pnotree.mid")
    if chord is not None:
        chord_np = chord.cpu().numpy()
        chd_to_midi_file(chord_np, "exp/o_chord.mid")
        return prmat2c.to(device), pnotree.to(device), chord.to(device
                                                               ), prmat.to(device)
    else:
        return prmat2c.to(device), pnotree.to(device), None, prmat.to(device)


def choose_song_from_val_dl():
    split_fpath = join(TRAIN_SPLIT_DIR, "pop909.pickle")
    with open(split_fpath, "rb") as f:
        split = pickle.load(f)
    print(split[1])
    num = int(input("choose one from pop909:"))
    song_fn = split[1][num]
    print(song_fn)

    song = DataSampleNpz(song_fn)
    return *get_data_preprocessed(song), song_fn


def choose_song_from_val_dl_musicalion():
    split_fpath = join(TRAIN_SPLIT_DIR, "musicalion.pickle")
    with open(split_fpath, "rb") as f:
        split = pickle.load(f)
    print(split[1])
    num = int(input("choose one from musicalion:"))
    song_fn = split[1][num]
    print(song_fn)

    song = DataSampleNpz_Musicalion(song_fn)
    return *get_data_preprocessed(song), song_fn


class Experiments:
    def __init__(self, params, sampler: SDFSampler) -> None:
        self.params = params
        self.sampler = sampler

    def predict(
        self,
        cond: torch.Tensor,
        uncond_scale=1.,
        is_autoreg=False,
        polydis_recon=False,
        polydis_chd=None
    ):
        n_samples = cond.shape[0]
        shape = [
            n_samples, self.params.out_channels, self.params.img_h, self.params.img_w
        ]
        uncond_cond = (-torch.ones_like(cond)).to(device)  # a bunch of -1
        print(f"predicting {shape} with uncond_scale = {uncond_scale}")
        self.sampler.model.eval()
        with torch.no_grad():
            if is_autoreg:
                half_len = self.params.img_h // 2
                single_shape = [
                    1, self.params.out_channels, self.params.img_h, self.params.img_w
                ]
                last = None  # this is the last inpainted 4-bar
                mask = torch.zeros(single_shape, device=device)
                # the first half is masked for inpainting
                mask[:, :, 0 : half_len, :] = 1.

                # squeeze the first two dimensions of the condition,
                # for convenience in getting an arbitrary 8-bar (i.e. [1, 4, 128])
                print(cond.shape)  # [#B, 4, 128]
                cond_sqz = cond.view(
                    cond.shape[0] * cond.shape[1], cond.shape[2]
                )  # FIXME: cond
                print(cond_sqz.shape)  # [#B * 4, 128]
                cond_len = cond.shape[-2]  # 4
                uncond_cond_seg = uncond_cond[0].unsqueeze(0)

                gen = []  # the generated
                for idx in range(n_samples * 2 - 1):  # inpaint a 4-bar each time
                    if idx == 0:
                        x0 = self.sampler.sample(
                            single_shape,
                            cond[idx].unsqueeze(0),
                            uncond_scale=uncond_scale,
                            uncond_cond=uncond_cond_seg
                        )
                        gen.append(x0[:, :, 0 : half_len, :])
                    else:
                        assert last is not None
                        t_idx = self.params.n_steps - 1
                        orig_noise = torch.randn(last.shape, device=device)
                        xt = self.sampler.q_sample(last, t_idx, orig_noise)
                        cond_start_idx = idx * (cond_len // 2)
                        cond_seg = cond_sqz[cond_start_idx : cond_start_idx +
                                            cond_len, :].unsqueeze(0)
                        x0 = sampler.paint(
                            xt,
                            cond_seg,
                            t_idx,
                            orig=last,
                            mask=mask,
                            orig_noise=orig_noise,
                            uncond_scale=uncond_scale,
                            uncond_cond=uncond_cond_seg
                        )
                    last = torch.zeros_like(x0)
                    # the last fixed half should have not changed
                    # assert x0[:, :, 0 : half_len, :] == last[:, :, 0 : half_len, :]
                    new_inpainted_half = x0[:, :, half_len :, :]
                    last[:, :, 0 : half_len, :] = new_inpainted_half
                    gen.append(new_inpainted_half)

                gen = torch.cat(gen, dim=0)
                print(f"piano_roll: {gen.shape}")
                assert gen.shape[0] == n_samples * 2
                # gen = gen.view(n_samples, gen.shape[1], half_len * 2, gen.shape[-1])
                # print(f"piano_roll: {gen.shape}")

            else:
                gen = self.sampler.sample(
                    shape, cond, uncond_scale=uncond_scale, uncond_cond=uncond_cond
                )

        output_stamp = f"sdf+{args.dataset}_[scale={uncond_scale},{'autoreg' if is_autoreg else ''}]_{datetime.now().strftime('%m-%d_%H%M%S')}"
        prmat = gen.cpu().numpy()
        prmat2c_to_midi_file(prmat, f"exp/{output_stamp}.mid")
        if polydis_recon:
            aftertouch = PolydisAftertouch()
            prmat = prmat2c_to_prmat(prmat)
            print(prmat.shape)
            prmat_to_midi_file(prmat, f"exp/{output_stamp}_prmat.mid")
            prmat = torch.from_numpy(prmat)
            chd = polydis_chd
            aftertouch.reconstruct(prmat, chd, f"exp/{output_stamp}")
        return gen

    def inpaint(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t_start: int,
        orig: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        orig_noise: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.,
        uncond_cond: Optional[torch.Tensor] = None,
    ):
        pass

    def show_q_imgs(self, prmat2c):
        if int(args.length) > 0:
            prmat2c = prmat2c[: int(args.length)]
        show_image(prmat2c, f"exp/img/q0.png")
        for step in self.sampler.time_steps:
            s1 = step + 1
            if s1 % 100 == 0 or (s1 <= 100 and s1 % 25 == 0):
                noised = self.sampler.q_sample(prmat2c, step)
                show_image(noised, f"exp/img/q{s1}.png")


if __name__ == "__main__":
    parser.add_argument(
        "--model_dir", help='directory in which trained model checkpoints are stored'
    )
    parser.add_argument(
        "--uncond_scale",
        default=1.,
        help="unconditional scale for classifier-free guidance"
    )
    parser.add_argument("--seed", help="use specific seed for inference")
    parser.add_argument(
        "--is_autoreg",
        action="store_true",
        help="autoregressively inpaint the music segments"
    )
    parser.add_argument(
        "--show_image",
        action="store_true",
        help="whether to show the images of generated piano-roll"
    )
    parser.add_argument(
        "--polydis_recon",
        action="store_true",
        help=
        "whether to use polydis to reconstruct the generated midi from diffusion model"
    )
    parser.add_argument(
        "--dataset",
        default="pop909",
        help="choose from which dataset (pop909, musicalion)"
    )
    parser.add_argument(
        "--from_midi", help="choose condition from a specific midi file"
    )
    parser.add_argument(
        "--only_q_imgs",
        action="store_true",
        help="only show q_sample results (for testing)"
    )
    parser.add_argument("--length", default=0, help="the generated length (in 8-bars)")
    args = parser.parse_args()

    if args.seed is not None:
        SEED = int(args.seed)
        print(f"fixed SEED = {SEED}")
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    # params ready
    with open(f"{args.model_dir}/params.json", "r") as params_file:
        params = json.load(params_file)
    params = AttrDict(params)

    # model ready
    autoencoder = None
    unet_model = UNetModel(
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

    ldm_model = LatentDiffusion(
        linear_start=params.linear_start,
        linear_end=params.linear_end,
        n_steps=params.n_steps,
        latent_scaling_factor=params.latent_scaling_factor,
        autoencoder=autoencoder,
        unet_model=unet_model
    )

    pnotree_enc, pnotree_dec = None, None
    chord_enc, chord_dec = None, None
    txt_enc = None
    if params.cond_type == "pnotree":
        pnotree_enc, pnotree_dec = load_pretrained_pnotree_enc_dec(
            PT_PNOTREE_PATH, 20, device
        )
    elif params.cond_type == "chord":
        if params.use_chd_enc:
            chord_enc, chord_dec = load_pretrained_chd_enc_dec(
                PT_CHD_8BAR_PATH, params.chd_input_dim, params.chd_z_input_dim,
                params.chd_hidden_dim, params.chd_z_dim, params.chd_n_step
            )
    elif params.cond_type == "txt":
        txt_enc = load_pretrained_txt_enc(
            PT_POLYDIS_PATH, params.txt_emb_size, params.txt_hidden_dim,
            params.txt_z_dim, params.txt_num_channel
        )
    else:
        raise NotImplementedError

    model = Diffpro_SDF.load_trained(
        ldm_model, f"{args.model_dir}/chkpts/weights_best.pt", params.cond_type,
        params.cond_mode, chord_enc, chord_dec, pnotree_enc, pnotree_dec, txt_enc
    ).to(device)
    sampler = SDFSampler(
        model.ldm,
        is_autocast=params.fp16,
        is_show_image=args.show_image,
    )
    expmt = Experiments(params, sampler)

    # input ready
    if args.from_midi is not None:
        song_fn = args.from_midi
        data = get_data_for_single_midi(args.from_midi, f"exp/chords_extracted.out")
        data_sample = DataSample(data)
        prmat2c, pnotree, chd, prmat = get_data_preprocessed(data_sample)
    elif args.dataset == "musicalion":
        prmat2c, pnotree, chd, prmat, song_fn = choose_song_from_val_dl_musicalion(
        )  # here chd is None
        assert params.cond_type != "chord"
    elif args.dataset == "pop909":
        prmat2c, pnotree, chd, prmat, song_fn = choose_song_from_val_dl()
    else:
        raise NotImplementedError
    print(f"using the {params.cond_type} of midi file: {song_fn}")

    # for demonstrating diffusion process
    if args.only_q_imgs:
        expmt.show_q_imgs(prmat2c)
        exit(0)

    # conditions ready
    polydis_chd = None
    if params.cond_type == "pnotree":
        cond = model._encode_pnotree(pnotree)
        pnotree_recon = model._decode_pnotree(cond)
        estx_to_midi_file(pnotree_recon, f"exp/pnotree_recon.mid")
    elif params.cond_type == "chord":
        # print(chd.shape)
        assert chd is not None
        cond = model._encode_chord(chd)
        # print(chd_enc.shape)
        polydis_chd = chd.view(-1, 8, 36)  # 2-bars
        # print(polydis_chd.shape)
    elif params.cond_type == "txt":
        cond = model._encode_txt(prmat)
    else:
        raise NotImplementedError

    if int(args.length) > 0:
        cond = cond[: int(args.length)]
        print(f"selected cond shape: {cond.shape}")

    # generate!
    expmt.predict(
        cond,
        uncond_scale=float(args.uncond_scale),
        is_autoreg=args.is_autoreg,
        polydis_recon=args.polydis_recon,
        polydis_chd=polydis_chd
    )
