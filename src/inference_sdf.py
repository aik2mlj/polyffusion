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
import torch

from labml import monit
from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.sampler import DiffusionSampler
from stable_diffusion.model.unet import UNetModel
from model_sdf import Diffpro_SDF
from params_sdf import params

from os.path import join
from argparse import ArgumentParser
import pickle
from tqdm import tqdm
from datetime import datetime

from dataset import DataSampleNpz
from dirs import *
from utils import prmat2c_to_midi_file, show_image, chd_to_midi_file, chd_to_onehot


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

    def __init__(self, model: LatentDiffusion, params):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """
        super().__init__(model)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Sampling steps $1, 2, \dots, T$
        self.time_steps = np.asarray(list(range(self.n_steps)))

        self.params = params

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
            self.sqrt_1m_alpha_bar = alpha_bar**.5
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
    def sample(
        self,
        shape: List[int],
        cond: torch.Tensor,
        repeat_noise: bool = False,
        temperature: float = 1.,
        x_last: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.,
        uncond_cond: Optional[torch.Tensor] = None,
        skip_steps: int = 0,
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
        time_steps = np.flip(self.time_steps)[skip_steps :]

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
            if s1 % 100 == 0 or (s1 <= 100 and s1 % 25 == 0):
                show_image(x, f"exp/x{s1}.jpg")
                prmat_x = x.squeeze().cpu().numpy()
                prmat2c_to_midi_file(prmat_x, f"exp/x{s1}.mid")

        # Return $x_0$
        return x

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
        e_t = self.get_eps(x, t, c, uncond_scale=uncond_scale, uncond_cond=uncond_cond)

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

    def predict(self, cond: torch.Tensor, init_cond=False, init_step=None):
        n_samples = cond.shape[0]
        shape = [
            n_samples, self.params.out_channels, self.params.img_h, self.params.img_w
        ]
        print(shape)
        self.model.eval()
        with torch.no_grad():
            if not init_cond:
                x0 = self.sample(shape, cond)
                show_image(x0, "exp/x0.jpg")
                prmat_x = x0.squeeze().cpu().numpy()
                output_stamp = f"sdf+pop909_[cond]_{datetime.now().strftime('%m-%d_%H%M%S')}"
                prmat2c_to_midi_file(prmat_x, f"exp/{output_stamp}.mid")
                return x0
            else:
                # song_fn, x_init, _ = choose_song_from_val_dl()
                # x0 = self.sample(n_samples, init_cond=x_init, init_step=init_step)
                # show_image(x0, "exp/x0.jpg")
                # prmat_x = x0.squeeze().cpu().numpy()
                # output_stamp = f"sdf+pop909_init_[{song_fn}]_{datetime.now().strftime('%m-%d_%H%M%S')}"
                # prmat2c_to_midi_file(prmat_x, f"exp/{output_stamp}.mid")
                # return x0
                raise NotImplementedError


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
    args = parser.parse_args()
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

    model = Diffpro_SDF.load_trained(ldm_model, args.model_dir).to(device)
    config = DDPMSampler(model.ldm, params)

    _, _, cond = choose_song_from_val_dl()
    print(cond.shape)
    cond = torch.Tensor(np.array([chd_to_onehot(chord) for chord in cond])).to(device)
    config.predict(cond, init_cond=False, init_step=100)
