from typing import List, Optional

import numpy as np
import torch
from labml import monit

from stable_diffusion.latent_diffusion import LatentDiffusion
from stable_diffusion.sampler import DiffusionSampler
from utils import show_image


class SDFSampler(DiffusionSampler):
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
        self,
        model: LatentDiffusion,
        is_show_image=False,
    ):
        """
        :param model: is the model to predict noise $\epsilon_\text{cond}(x_t, c)$
        """
        super().__init__(model)

        # Sampling steps $1, 2, \dots, T$
        self.time_steps = np.asarray(list(range(self.n_steps)), dtype=np.int32)

        self.is_show_image = is_show_image

        with torch.no_grad():
            # $\bar\alpha_t$
            alpha_bar = self.model.alpha_bar
            # $\beta_t$ schedule
            beta = self.model.beta
            #  $\bar\alpha_{t-1}$
            alpha_bar_prev = torch.cat([alpha_bar.new_tensor([1.0]), alpha_bar[:-1]])

            # $\sqrt{\bar\alpha}$
            self.sqrt_alpha_bar = alpha_bar**0.5
            # $\sqrt{1 - \bar\alpha}$
            self.sqrt_1m_alpha_bar = (1.0 - alpha_bar) ** 0.5
            # $\frac{1}{\sqrt{\bar\alpha_t}}$
            self.sqrt_recip_alpha_bar = alpha_bar**-0.5
            # $\sqrt{\frac{1}{\bar\alpha_t} - 1}$
            self.sqrt_recip_m1_alpha_bar = (1 / alpha_bar - 1) ** 0.5

            # $\frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t$
            variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            # Clamped log of $\tilde\beta_t$
            self.log_var = torch.log(torch.clamp(variance, min=1e-20))
            # $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$
            self.mean_x0_coef = beta * (alpha_bar_prev**0.5) / (1.0 - alpha_bar)
            # $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}$
            self.mean_xt_coef = (
                (1.0 - alpha_bar_prev) * ((1 - beta) ** 0.5) / (1.0 - alpha_bar)
            )

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        step: int,
        repeat_noise: bool = False,
        temperature: float = 1.0,
        uncond_scale: float = 1.0,
        uncond_cond: Optional[torch.Tensor] = None,
        cond_concat=None,
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
        if cond_concat is not None:
            e_t = self.get_eps(
                torch.concat([x, cond_concat], dim=1),
                t,
                c,
                uncond_scale=uncond_scale,
                uncond_cond=uncond_cond,
            )
        else:
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
            noise = torch.randn((1, *x.shape[1:]), device=x.device)
        # Different noise for each sample
        else:
            noise = torch.randn(x.shape, device=x.device)

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
            noise = torch.randn_like(x0, device=x0.device)

        # Sample from $\mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)$
        return self.sqrt_alpha_bar[index] * x0 + self.sqrt_1m_alpha_bar[index] * noise

    @torch.no_grad()
    def sample(
        self,
        shape: List[int],
        cond: torch.Tensor,
        repeat_noise: bool = False,
        temperature: float = 1.0,
        x_last: Optional[torch.Tensor] = None,
        uncond_scale: float = 1.0,
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
        x = x_last if x_last is not None else torch.randn(shape, device=cond.device)

        # Time steps to sample at $T - t', T - t' - 1, \dots, 1$
        time_steps = np.flip(self.time_steps)[t_start:]

        # Sampling loop
        for step in monit.iterate("Sample", time_steps):
            # Time step $t$
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample $x_{t-1}$
            x, pred_x0, e_t = self.p_sample(
                x,
                cond,
                ts,
                step,
                repeat_noise=repeat_noise,
                temperature=temperature,
                uncond_scale=uncond_scale,
                uncond_cond=uncond_cond,
            )

            s1 = step + 1
            if self.is_show_image:
                if s1 % 100 == 0 or (s1 <= 100 and s1 % 25 == 0):
                    show_image(x, f"exp/img/x{s1}.png")

        # Return $x_0$
        if self.is_show_image:
            show_image(x, "exp/img/x0.png")
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
        uncond_scale: float = 1.0,
        uncond_cond: Optional[torch.Tensor] = None,
        cond_concat=None,
        repaint_n=1,
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
        time_steps = np.flip(self.time_steps[: t_start + 1])
        print(f"RePainting: sampling steps = {repaint_n}")

        for i, step in monit.enum("Paint", time_steps):
            if orig is None:
                # vanilla generation
                ts = x.new_full((bs,), step, dtype=torch.long)

                # Sample $x_{\tau_{i-1}}$
                x, _, _ = self.p_sample(
                    x,
                    cond,
                    ts,
                    step,
                    uncond_scale=uncond_scale,
                    uncond_cond=uncond_cond,
                    cond_concat=cond_concat,
                )
            else:
                # RePaint
                # Replace the masked area with original image
                assert mask is not None
                # Get the $q_{\sigma,\tau}(x_{\tau_i}|x_0)$ for original image in latent space
                x_t = x
                for u in range(repaint_n):
                    # Index $i$ in the list $[\tau_1, \tau_2, \dots, \tau_S]$
                    # index = len(time_steps) - i - 1
                    # Time step $\tau_i$
                    noise = (
                        torch.randn_like(orig, device=orig.device)
                        if step > 0
                        else torch.zeros_like(orig, device=orig.device)
                    )
                    x_kn_tm1 = self.q_sample(orig, step, noise=noise)
                    ts = x_t.new_full((bs,), step, dtype=torch.long)
                    # Sample $x_{\tau_{i-1}}$
                    x_unkn_tm1, _, _ = self.p_sample(
                        x_t,
                        cond,
                        ts,
                        step,
                        uncond_scale=uncond_scale,
                        uncond_cond=uncond_cond,
                        cond_concat=cond_concat,
                    )

                    # Replace the masked area
                    x = x_kn_tm1 * mask + x_unkn_tm1 * (1 - mask)
                    if u < repaint_n - 1 and step > 0:
                        noise = torch.randn_like(orig, device=orig.device)
                        x_t = (
                            1 - self.model.beta[step - 1]
                        ) ** 0.5 * x + self.model.beta[step - 1] * noise

            s1 = step + 1
            if self.is_show_image:
                if s1 % 100 == 0 or (s1 <= 100 and s1 % 25 == 0):
                    show_image(x, f"exp/img/x{s1}.png")

        if self.is_show_image:
            show_image(x, "exp/img/x0.png")
        return x
