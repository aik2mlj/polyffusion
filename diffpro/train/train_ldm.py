import torch
from argparse import ArgumentParser
import sys
import os

# from stable_diffusion.model.autoencoder import Autoencoder, Encoder, Decoder
from . import *
from stable_diffusion.model.unet import UNetModel
from stable_diffusion.latent_diffusion import LatentDiffusion
from models.model_sdf import Diffpro_SDF
from data.dataloader import get_train_val_dataloaders
from data.dataloader_musicalion import get_train_val_dataloaders as get_train_val_dataloaders_musicalion
from dl_modules import ChordEncoder, ChordDecoder, TextureEncoder
from dirs import PT_A2S_PATH, PT_CHD_8BAR_PATH, PT_PNOTREE_PATH, PT_POLYDIS_PATH
from utils import load_pretrained_pnotree_enc_dec, load_pretrained_chd_enc_dec, load_pretrained_txt_enc


class LDM_TrainConfig(TrainConfig):
    def __init__(
        self,
        params,
        output_dir,
        use_autoencoder=False,
        use_musicalion=False,
        use_track=[0, 1, 2],
    ) -> None:
        super().__init__(params, None, output_dir)
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

        self.pnotree_enc, self.pnotree_dec = None, None
        self.chord_enc, self.chord_dec = None, None
        self.txt_enc = None
        if params.cond_type == "pnotree":
            self.pnotree_enc, self.pnotree_dec = load_pretrained_pnotree_enc_dec(
                PT_PNOTREE_PATH, 20, self.device
            )
        elif params.cond_type == "chord":
            if params.use_enc:
                self.chord_enc, self.chord_dec = load_pretrained_chd_enc_dec(
                    PT_CHD_8BAR_PATH, params.chd_input_dim, params.chd_z_input_dim,
                    params.chd_hidden_dim, params.chd_z_dim, params.chd_n_step
                )
        elif params.cond_type == "txt":
            if params.use_enc:
                self.txt_enc = load_pretrained_txt_enc(
                    PT_POLYDIS_PATH, params.txt_emb_size, params.txt_hidden_dim,
                    params.txt_z_dim, params.txt_num_channel
                )
        else:
            raise NotImplementedError
        self.model = Diffpro_SDF(
            self.ldm_model,
            cond_type=params.cond_type,
            cond_mode=params.cond_mode,
            chord_enc=self.chord_enc,
            chord_dec=self.chord_dec,
            pnotree_enc=self.pnotree_enc,
            pnotree_dec=self.pnotree_dec,
            txt_enc=self.txt_enc,
            concat_blurry=params.concat_blurry
            if hasattr(params, 'concat_blurry') else False,
            concat_ratio=params.concat_ratio if hasattr(params, 'concat_ratio') else 1 /
            8,
        ).to(self.device)
        # Create dataloader
        if use_musicalion:
            self.train_dl, self.val_dl = get_train_val_dataloaders_musicalion(
                params.batch_size, params.num_workers, params.pin_memory
            )
        else:
            self.train_dl, self.val_dl = get_train_val_dataloaders(
                params.batch_size,
                params.num_workers,
                params.pin_memory,
                use_track=use_track
            )

        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.learning_rate
        )
