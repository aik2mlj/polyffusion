import torch
from argparse import ArgumentParser
import sys

sys.path.insert(0, "..")

# from stable_diffusion.model.autoencoder import Autoencoder, Encoder, Decoder
from . import *
from stable_diffusion.model.unet import UNetModel
from stable_diffusion.latent_diffusion import LatentDiffusion
from models.model_sdf import Diffpro_SDF
from dataloader import get_train_val_dataloaders
from dl_modules import ChordEncoder, ChordDecoder
from dirs import PT_A2S_PATH


def load_pretrained_chd_enc_dec(fpath, input_dim, hidden_dim, z_dim):
    chord_enc = ChordEncoder(input_dim, hidden_dim, z_dim)
    chord_dec = ChordDecoder(z_dim=z_dim)
    checkpoint = torch.load(fpath)
    from collections import OrderedDict
    enc_chkpt = OrderedDict()
    dec_chkpt = OrderedDict()
    for k, v in checkpoint.items():
        part = k.split('.')[0]
        name = '.'.join(k.split('.')[1 :])
        if part == "chord_enc":
            enc_chkpt[name] = v
        elif part == "chord_dec":
            dec_chkpt[name] = v
    chord_enc.load_state_dict(enc_chkpt)
    chord_dec.load_state_dict(dec_chkpt)
    return chord_enc, chord_dec


class LDM_TrainConfig(TrainConfig):
    def __init__(self, params, output_dir, use_autoencoder=False) -> None:
        super().__init__(params, output_dir)
        self.autoencoder = None

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

        if params.use_chd_enc:
            self.chord_enc, self.chord_dec = load_pretrained_chd_enc_dec(
                PT_A2S_PATH, params.chd_input_dim, params.chd_hidden_dim,
                params.chd_z_dim
            )
        else:
            self.chord_enc, self.chord_dec = None, None

        self.model = Diffpro_SDF(
            self.ldm_model,
            cond_mode=params.cond_mode,
            chord_enc=self.chord_enc,
            chord_dec=self.chord_dec
        ).to(self.device)
        # Create dataloader
        self.train_dl, self.val_dl = get_train_val_dataloaders(params.batch_size)
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.learning_rate
        )
