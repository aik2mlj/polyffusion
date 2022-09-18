from dl_modules import PianoTreeEncoder, PianoTreeDecoder, NaiveNN
import torch
import torch.nn as nn
from utils import *
import torch.nn.functional as F
from math import sqrt

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    """
    Embedding for the step t
    """
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer(
            'embedding', self._build_embedding(max_steps), persistent=False
        )
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)  # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)  # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation, uncond=False):
        '''
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable spectrogram conditional
        '''
        super().__init__()
        self.dilated_conv = Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation
        )
        self.diffusion_projection = Linear(512, residual_channels)
        if not uncond:  # conditional model
            self.conditioner_projection = Conv1d(1, 2 * residual_channels, 1)
        else:  # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):
        assert (conditioner is None and self.conditioner_projection is None) or \
               (conditioner is not None and self.conditioner_projection is not None)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        if self.conditioner_projection is None:  # using a unconditional model
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class Diffpro_diffwave(nn.Module):
    def __init__(self, params, max_simu_note=20, pt_pnotree_model_path=None):
        super().__init__()
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # load pretrained model
        if pt_pnotree_model_path is not None:
            self.pnotree_enc, self.pnotree_dec = load_pretrained_pnotree_enc_dec(
                pt_pnotree_model_path, max_simu_note, self.device
            )
        else:
            self.pnotree_enc = PianoTreeEncoder(
                self.device, max_simu_note=max_simu_note
            )
            self.pnotree_dec = PianoTreeDecoder(
                self.device, max_simu_note=max_simu_note
            )
        self._disable_grads_for_enc_dec()

        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    params.residual_channels,
                    2**(i % params.dilation_cycle_length),
                    uncond=params.unconditional
                ) for i in range(params.residual_layers)
            ]
        )
        self.skip_projection = Conv1d(
            params.residual_channels, params.residual_channels, 1
        )
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def _disable_grads_for_enc_dec(self):
        for param in self.pnotree_enc.parameters():
            param.requires_grad = False
        for param in self.pnotree_dec.parameters():
            param.requires_grad = False

    def encode_z(self, pnotree, is_sampling):
        dist, _, _ = self.pnotree_enc(pnotree)
        z = dist.rsample() if is_sampling else dist.mean
        return z

    def loss_function(self, noise, predicted):
        loss = nn.L1Loss(noise, predicted)
        return {"loss": loss}

    def forward(self, input, diffusion_step, pnotree_x):
        """
        pnotree_x is the condition
        """
        assert (pnotree_x is None and self.params.unconditional) or \
               (pnotree_x is not None and not self.params.unconditional)
        x = input.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        if self.params.unconditional:  # use conditional model
            z_x = self.encode_z(pnotree_x, is_sampling=True)
        else:
            z_x = None

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, z_x)
            skip = skip_connection if skip is None else skip_connection + skip
        assert skip is not None

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        x = x.squeeze(1)  # NOTE: add squeeze here
        return x

    def get_loss_dict(self, pnotree_x, pnotree_y, noise_level):
        """
        z_y is the stuff the diffusion model needs to learn
        """
        N = pnotree_x.shape[0]
        z_y = self.encode_z(pnotree_y, is_sampling=True)
        t = torch.randint(0, len(self.params.noise_schedule), [N], device=self.device)
        noise_scale = noise_level[t].unsqueeze(1)
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(z_y)
        noisy_z_y = noise_scale_sqrt * z_y + (1.0 - noise_scale)**0.5 * noise

        predicted = self.forward(noisy_z_y, t, pnotree_x)
        return self.loss_function(noise, predicted)


class Diffpro(nn.Module):
    def __init__(self, params, max_simu_note=20, pt_pnotree_model_path=None):
        super(Diffpro, self).__init__()
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # load pretrained model
        if pt_pnotree_model_path is not None:
            self.pnotree_enc, self.pnotree_dec = load_pretrained_pnotree_enc_dec(
                pt_pnotree_model_path, max_simu_note, self.device
            )
        else:
            self.pnotree_enc = PianoTreeEncoder(
                self.device, max_simu_note=max_simu_note
            )
            self.pnotree_dec = PianoTreeDecoder(
                self.device, max_simu_note=max_simu_note
            )
        self.naive_nn = NaiveNN()
        self._disable_grads_for_enc_dec()

    @classmethod
    def load_trained(cls, model_dir, params, max_simu_note=20):
        model = cls(params, max_simu_note, None)
        trained_leaner = torch.load(f"{model_dir}/weights.pt")
        model.load_state_dict(trained_leaner["model"])
        return model

    def _disable_grads_for_enc_dec(self):
        for param in self.pnotree_enc.parameters():
            param.requires_grad = False
        for param in self.pnotree_dec.parameters():
            param.requires_grad = False

    def loss_function(self, pnotree_y, recon_pitch, recon_dur, dist_x):
        pnotree_l, pitch_l, dur_l = self.pnotree_dec.recon_loss(
            pnotree_y, recon_pitch, recon_dur, self.params.weights, False
        )

        # kl losses
        kl_x = kl_with_normal(dist_x)
        kl_l = self.params.beta * (kl_x)

        # TODO: contrastive loss

        loss = pnotree_l + kl_l

        return {
            "loss": loss,
            "pnotree_l": pnotree_l,
            "pitch_l": pitch_l,
            "dur_l": dur_l,
            "kl_l": kl_l,
            "kl_x": kl_x,
            "beta": self.params.beta
        }

    def forward(self, pnotree_x, pnotree_y, tfr1, tfr2):
        # FIXME: teacher-forcing is not needed here?
        dist_x, emb_x, _ = self.pnotree_enc(pnotree_x)

        z_x = dist_x.rsample()

        z = self.naive_nn(z_x)

        # teaching force data
        embedded_pnotree, pnotree_lgths = self.pnotree_dec.emb_x(pnotree_y)

        # pianotree decoder
        recon_pitch, recon_dur = self.pnotree_dec(
            z, False, embedded_pnotree, pnotree_lgths, tfr1, tfr2
        )

        return (recon_pitch, recon_dur, dist_x)

    def get_loss_dict(self, pnotree_x, pnotree_y, tfr1=0, tfr2=0):
        recon_pitch, recon_dur, dist_x = self.forward(pnotree_x, pnotree_y, tfr1, tfr2)

        return self.loss_function(pnotree_y, recon_pitch, recon_dur, dist_x)

    def output_to_numpy(self, recon_pitch, recon_dur):
        est_pitch = recon_pitch.max(-1)[1].unsqueeze(-1)  # (B, 32, 20, 1)
        est_dur = recon_dur.max(-1)[1]  # (B, 32, 11, 5)
        est_x = torch.cat([est_pitch, est_dur], dim=-1)  # (B, 32, 20, 6)
        est_x = est_x.cpu().numpy()
        recon_pitch = recon_pitch.cpu().numpy()
        recon_dur = recon_dur.cpu().numpy()
        return est_x, recon_pitch, recon_dur

    def infer(self, pnotree_x, is_sampling=False):
        with torch.no_grad():
            dist_x, emb_x, _ = self.pnotree_enc(pnotree_x)

            z_x = dist_x.rsample() if is_sampling else dist_x.mean

            z = self.naive_nn(z_x)

            # pianotree decoder
            recon_pitch, recon_dur = self.pnotree_dec(z, True, None, None, 0, 0)

            return self.output_to_numpy(recon_pitch, recon_dur)
