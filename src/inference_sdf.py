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
from stable_diffusion.sampler import DiffusionSampler
from stable_diffusion.model.unet import UNetModel
from models.model_sdf import Diffpro_SDF
# from params_sdf import params
from params import AttrDict

from os.path import join
from argparse import ArgumentParser
import pickle
from tqdm import tqdm
from datetime import datetime

from dataset import DataSampleNpz
from dirs import *
from utils import prmat2c_to_midi_file, show_image, chd_to_midi_file, chd_to_onehot
from train.train_ldm import load_pretrained_chd_enc_dec

SEED = 7890
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


class DDPMSampler(DiffusionSampler):
    """
    ## DDPM Sampler

    This extends the [`DiffusionSampler` base class](index.html).

    DDPM samples images by repeatedly removing noise by sampling step by step from
    $p_\theta(x_{t-1} | x_t)$,

    \begin{align}

    p_\theta(x_{t-1} | x_t) &= \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \tilde\beta_t \mathbf{I} \big) \\

    \mu_t(x_t, t) &= \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
                         + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t \\

    \tilde\beta_t &= \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t \\

    x_0 &= \frac{1}{\sqrt{\bar\alpha_t}} x_t -  \Big(\sqrt{\frac{1}{\bar\alpha_t} - 1}\Big)\epsilon_\theta \\

    \end{align}
    """

    model: LatentDiffusion

    def __init__(
        self, model: LatentDiffusion, params, is_show_image=False, is_autoreg=False
    ):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """
        super().__init__(model)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Sampling steps $1, 2, \dots, T$
        self.time_steps = np.asarray(list(range(self.n_steps)), dtype=np.int32)

        self.params = params

        self.is_show_image = is_show_image

        self.is_autoreg = is_autoreg

        self.autocast = torch.cuda.amp.autocast(enabled=params.fp16)

        with torch.no_grad():
            # $\bar\alpha_t$
            alpha_bar = self.model.alpha_bar
            # $\beta_t$ schedule
            beta = self.model.beta
            #  $\bar\alpha_{t-1}$
            alpha_bar_prev = torch.cat([alpha_bar.new_tensor([1.]), alpha_bar[:-1]])

            # $\sqrt{\bar\alpha}$
            self.sqrt_alpha_bar = alpha_bar**.5
            # $\sqrt{1 - \bar\alpha}$
            self.sqrt_1m_alpha_bar = (1. - alpha_bar)**.5
            # $\frac{1}{\sqrt{\bar\alpha_t}}$
            self.sqrt_recip_alpha_bar = alpha_bar**-.5
            # $\sqrt{\frac{1}{\bar\alpha_t} - 1}$
            self.sqrt_recip_m1_alpha_bar = (1 / alpha_bar - 1)**.5

            # $\frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t$
            variance = beta * (1. - alpha_bar_prev) / (1. - alpha_bar)
            # Clamped log of $\tilde\beta_t$
            self.log_var = torch.log(torch.clamp(variance, min=1e-20))
            # $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$
            self.mean_x0_coef = beta * (alpha_bar_prev**.5) / (1. - alpha_bar)
            # $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}$
            self.mean_xt_coef = (1. - alpha_bar_prev) * ((1 - beta)**
                                                         0.5) / (1. - alpha_bar)

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        step: int,
        repeat_noise: bool = False,
        temperature: float = 1.,
        uncond_scale: float = 1.,
        uncond_cond: Optional[torch.Tensor] = None
    ):
        """
        ### Sample $x_{t-1}$ from $p_\theta(x_{t-1} | x_t)$

        :param x: is $x_t$ of shape `[batch_size, channels, height, width]`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param t: is $t$ of shape `[batch_size]`
        :param step: is the step $t$ as an integer
        :repeat_noise: specified whether the noise should be same for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """

        # Get $\epsilon_\theta$
        with self.autocast:
            e_t = self.get_eps(
                x, t, c, uncond_scale=uncond_scale, uncond_cond=uncond_cond
            )

        # Get batch size
        bs = x.shape[0]

        # $\frac{1}{\sqrt{\bar\alpha_t}}$
        sqrt_recip_alpha_bar = x.new_full(
            (bs, 1, 1, 1), self.sqrt_recip_alpha_bar[step]
        )
        # $\sqrt{\frac{1}{\bar\alpha_t} - 1}$
        sqrt_recip_m1_alpha_bar = x.new_full(
            (bs, 1, 1, 1), self.sqrt_recip_m1_alpha_bar[step]
        )

        # Calculate $x_0$ with current $\epsilon_\theta$
        #
        # $$x_0 = \frac{1}{\sqrt{\bar\alpha_t}} x_t -  \Big(\sqrt{\frac{1}{\bar\alpha_t} - 1}\Big)\epsilon_\theta$$
        x0 = sqrt_recip_alpha_bar * x - sqrt_recip_m1_alpha_bar * e_t

        # $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$
        mean_x0_coef = x.new_full((bs, 1, 1, 1), self.mean_x0_coef[step])
        # $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}$
        mean_xt_coef = x.new_full((bs, 1, 1, 1), self.mean_xt_coef[step])

        # Calculate $\mu_t(x_t, t)$
        #
        # $$\mu_t(x_t, t) = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
        #    + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$
        mean = mean_x0_coef * x0 + mean_xt_coef * x
        # $\log \tilde\beta_t$
        log_var = x.new_full((bs, 1, 1, 1), self.log_var[step])

        # Do not add noise when $t = 1$ (final step sampling process).
        # Note that `step` is `0` when $t = 1$)
        if step == 0:
            noise = 0
        # If same noise is used for all samples in the batch
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1 :]), device=self.device)
        # Different noise for each sample
        else:
            noise = torch.randn(x.shape, device=self.device)

        # Multiply noise by the temperature
        noise = noise * temperature

        # Sample from,
        #
        # $$p_\theta(x_{t-1} | x_t) = \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \tilde\beta_t \mathbf{I} \big)$$
        x_prev = mean + (0.5 * log_var).exp() * noise

        #
        return x_prev, x0, e_t

    @torch.no_grad()
    def q_sample(
        self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None
    ):
        """
        ### Sample from $q(x_t|x_0)$

        $$q(x_t|x_0) = \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)$$

        :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        :param index: is the time step $t$ index
        :param noise: is the noise, $\epsilon$
        """

        # Random noise, if noise is not specified
        if noise is None:
            noise = torch.randn_like(x0, device=self.device)

        # Sample from $\mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)$
        return self.sqrt_alpha_bar[index] * x0 + self.sqrt_1m_alpha_bar[index] * noise

    @torch.no_grad()
    def sample(
        self,
        shape: List[int],
        cond: torch.Tensor,
        repeat_noise: bool = False,
        temperature: float = 1.,
        x_last: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.,
        uncond_cond: Optional[torch.Tensor] = None,
        t_start: int = 0,
    ):
        """
        ### Sampling Loop

        :param shape: is the shape of the generated images in the
            form `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param x_last: is $x_T$. If not provided random noise will be used.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        :param skip_steps: is the number of time steps to skip $t'$. We start sampling from $T - t'$.
            And `x_last` is then $x_{T - t'}$.
        """

        # Get device and batch size
        bs = shape[0]

        # Get $x_T$
        x = x_last if x_last is not None else torch.randn(shape, device=self.device)

        # Time steps to sample at $T - t', T - t' - 1, \dots, 1$
        time_steps = np.flip(self.time_steps)[t_start :]

        # Sampling loop
        for step in monit.iterate('Sample', time_steps):
            # Time step $t$
            ts = x.new_full((bs, ), step, dtype=torch.long)

            # Sample $x_{t-1}$
            x, pred_x0, e_t = self.p_sample(
                x,
                cond,
                ts,
                step,
                repeat_noise=repeat_noise,
                temperature=temperature,
                uncond_scale=uncond_scale,
                uncond_cond=uncond_cond
            )

            s1 = step + 1
            if self.is_show_image:
                if s1 % 100 == 0 or (s1 <= 100 and s1 % 25 == 0):
                    show_image(x, f"exp/x{s1}.jpg")
                    prmat_x = x.cpu().numpy()
                    # prmat2c_to_midi_file(prmat_x, f"exp/x{s1}.mid")

        # Return $x_0$
        return x

    @torch.no_grad()
    def paint(
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
        """
        ### Painting Loop

        :param x: is $x_{S'}$ of shape `[batch_size, channels, height, width]`
        :param cond: is the conditional embeddings $c$
        :param t_start: is the sampling step to start from, $S'$
        :param orig: is the original image in latent page which we are in paining.
            If this is not provided, it'll be an image to image transformation.
        :param mask: is the mask to keep the original image.
        :param orig_noise: is fixed noise to be added to the original image.
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """
        # Get  batch size
        bs = x.shape[0]

        # Time steps to sample at $\tau_{S`}, \tau_{S' - 1}, \dots, \tau_1$
        time_steps = np.flip(self.time_steps[: t_start])

        for i, step in monit.enum('Paint', time_steps):
            # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
            # index = len(time_steps) - i - 1
            # Time step $\tau_i$
            ts = x.new_full((bs, ), step, dtype=torch.long)

            # Sample $x_{\tau_{i-1}}$
            x, _, _ = self.p_sample(
                x, cond, ts, step, uncond_scale=uncond_scale, uncond_cond=uncond_cond
            )

            # Replace the masked area with original image
            if orig is not None:
                assert mask is not None
                # Get the $q_{\sigma,\tau}(x_{\tau_i}|x_0)$ for original image in latent space
                orig_t = self.q_sample(orig, step, noise=orig_noise)
                # Replace the masked area
                x = orig_t * mask + x * (1 - mask)

            s1 = step + 1
            if self.is_show_image:
                if s1 % 100 == 0 or (s1 <= 100 and s1 % 25 == 0):
                    show_image(x, f"exp/x{s1}.jpg")
                    prmat_x = x.cpu().numpy()
                    # prmat2c_to_midi_file(prmat_x, f"exp/x{s1}.mid")
        return x

    def predict(self, cond: torch.Tensor, uncond_scale=1.):
        n_samples = cond.shape[0]
        shape = [
            n_samples, self.params.out_channels, self.params.img_h, self.params.img_w
        ]
        uncond_cond = (-torch.ones_like(cond)).to(self.device)  # a bunch of -1
        print(f"predicting {shape} with uncond_scale = {uncond_scale}")
        self.model.eval()
        with torch.no_grad():
            if self.is_autoreg:
                half_len = self.params.img_h // 2
                single_shape = [
                    1, self.params.out_channels, self.params.img_h, self.params.img_w
                ]
                last = None  # this is the last inpainted 4-bar
                mask = torch.zeros(single_shape, device=self.device)
                mask[:, :,
                     0 : half_len, :] = 1.  # the first half is masked for inpainting

                # squeeze the first two dimensions of the condition,
                # for convenience in getting an arbitrary 8-bar (i.e. [1, 4, 128])
                print(cond.shape)  # [#B, 4, 128]
                cond_sqz = cond.view(cond.shape[0] * cond.shape[1], cond.shape[2])
                print(cond_sqz.shape)  # [#B * 4, 128]
                cond_len = cond.shape[-2]  # 4

                gen = []  # the generated
                for idx in range(n_samples * 2 - 1):  # inpaint a 4-bar each time
                    if idx == 0:
                        x0 = self.sample(
                            single_shape,
                            cond[idx].unsqueeze(0),
                            uncond_scale=uncond_scale,
                            uncond_cond=uncond_cond
                        )
                        gen.append(x0[:, :, 0 : half_len, :])
                    else:
                        assert last is not None
                        t_idx = self.params.n_steps - 1
                        orig_noise = torch.randn(last.shape, device=self.device)
                        xt = self.q_sample(last, t_idx, orig_noise)
                        cond_start_idx = idx * (cond_len // 2)
                        cond_seg = cond_sqz[cond_start_idx : cond_start_idx +
                                            cond_len, :].unsqueeze(0)
                        x0 = self.paint(
                            xt,
                            cond_seg,
                            t_idx,
                            orig=last,
                            mask=mask,
                            orig_noise=orig_noise,
                            uncond_scale=uncond_scale,
                            uncond_cond=uncond_cond
                        )
                    last = torch.zeros_like(x0)
                    new_inpainted_half = x0[:, :, half_len :, :]
                    last[:, :, 0 : half_len, :] = new_inpainted_half
                    gen.append(new_inpainted_half)

                gen = torch.cat(gen, dim=0)
                print(f"piano_roll: {gen.shape}")
                assert gen.shape[0] == n_samples * 2
                gen.view(n_samples, gen.shape[1], half_len * 2, gen.shape[-1])
                print(f"piano_roll: {gen.shape}")

            else:
                gen = self.sample(
                    shape, cond, uncond_scale=uncond_scale, uncond_cond=uncond_cond
                )

        if self.is_show_image:
            show_image(gen, "exp/x0.jpg")
        prmat_x = gen.cpu().numpy()
        output_stamp = f"sdf+pop909_[scale:{uncond_scale},autoreg={self.is_autoreg}]_{datetime.now().strftime('%m-%d_%H%M%S')}"
        prmat2c_to_midi_file(prmat_x, f"exp/{output_stamp}.mid")
        return gen
        # else:
        # song_fn, x_init, _ = choose_song_from_val_dl()
        # x0 = self.sample(n_samples, init_cond=x_init, init_step=init_step)
        # show_image(x0, "exp/x0.jpg")
        # prmat_x = x0.squeeze().cpu().numpy()
        # output_stamp = f"sdf+pop909_init_[{song_fn}]_{datetime.now().strftime('%m-%d_%H%M%S')}"
        # prmat2c_to_midi_file(prmat_x, f"exp/{output_stamp}.mid")
        # return x0
        # raise NotImplementedError


def choose_song_from_val_dl():
    split_fpath = join(TRAIN_SPLIT_DIR, "pop909.pickle")
    with open(split_fpath, "rb") as f:
        split = pickle.load(f)
    print(split[1])
    num = int(input("choose one:"))
    song_fn = split[1][num]
    print(song_fn)

    song = DataSampleNpz(song_fn)
    prmat, density, chord = song.get_whole_song_data()
    prmat_np = prmat.squeeze().cpu().numpy()
    chord_np = chord.cpu().numpy()
    prmat2c_to_midi_file(prmat_np, "exp/origin_x.mid")
    chd_to_midi_file(chord_np, "exp/chord.mid")
    return song_fn, prmat, chord


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ArgumentParser(description='inference a Diffpro model')
    parser.add_argument(
        "--model_dir", help='directory in which trained model checkpoints are stored'
    )
    parser.add_argument("--uncond_scale", default=1., help="unconditional scale")
    parser.add_argument("--is_autoreg", default=False, help="is autoregressive")
    parser.add_argument(
        "--show_image",
        default=False,
        help="whether to show the images of generated piano-roll"
    )
    args = parser.parse_args()

    with open(f"{args.model_dir}/params.json", "r") as params_file:
        params = json.load(params_file)
    params = AttrDict(params)
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

    if params.use_chd_enc:
        chord_enc, chord_dec = load_pretrained_chd_enc_dec(
            PT_A2S_PATH, params.chd_input_dim, params.chd_hidden_dim, params.chd_z_dim
        )
    else:
        chord_enc, chord_dec = None, None

    model = Diffpro_SDF.load_trained(
        ldm_model, f"{args.model_dir}/chkpts", params.cond_mode, chord_enc, chord_dec
    ).to(device)
    config = DDPMSampler(
        model.ldm, params, is_show_image=args.show_image, is_autoreg=args.is_autoreg
    )

    _, _, cond = choose_song_from_val_dl()
    print(cond.shape)
    cond = torch.Tensor(np.array([chd_to_onehot(chord) for chord in cond])).to(device)
    cond = model._encode_chord(cond)
    # cond = cond[: 3]
    print(cond.shape)
    config.predict(cond, uncond_scale=float(args.uncond_scale))
