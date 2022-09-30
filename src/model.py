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

        self.cond_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_projection = Linear(512, 64)
        self.fc1 = Linear(512, 512)
        self.fc2 = Linear(512, 512)
        self.fc3 = Linear(512, 512)

        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

        self.loss_fn = nn.L1Loss()

        beta = np.array(self.params.noise_schedule)
        alpha_cum = np.cumprod(1 - beta)
        self.alpha_cum = torch.tensor(alpha_cum.astype(np.float32), device=self.device)
        self.noise_scale = self.alpha_cum.unsqueeze(1)
        print(self.noise_scale.shape)
        self.noise_scale_sqrt = self.noise_scale**0.5

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

    def encode_z(self, pnotree, is_sampling):
        dist, _, _ = self.pnotree_enc(pnotree)
        z = dist.rsample() if is_sampling else dist.mean
        return z

    def decode_z(self, z):
        recon_pitch, recon_dur = self.pnotree_dec(z, True, None, None, 0, 0)
        y_prd, _, _ = output_to_numpy(recon_pitch, recon_dur)
        return y_prd

    def loss_function(self, noise, predicted):
        loss = self.loss_fn(noise, predicted)
        return {"loss": loss}

    def forward(self, input, diffusion_step, pnotree_x):
        """
        pnotree_x is the condition
        """
        assert (pnotree_x is None and self.params.unconditional) or \
               (pnotree_x is not None and not self.params.unconditional)
        x = input.unsqueeze(1)  # (B, 1, 512)
        x = self.input_projection(x)  # (B, 64, 512)
        x = silu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)  # (B, 512)
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        x += diffusion_step
        x = self.fc1(x)
        x = silu(x)
        if not self.params.unconditional:  # use conditional model
            z_x = self.encode_z(pnotree_x, is_sampling=True)
            z_x = z_x.unsqueeze(1)  # (B, 1, 512)
            z_x = self.cond_projection(z_x)  # (B, 64, 512)
            x += z_x
        x = self.fc2(x)
        x = silu(x)
        x = self.fc3(x)
        x = silu(x)

        # skip = None
        # for layer in self.residual_layers:
        #     x, skip_connection = layer(x, diffusion_step, z_x)
        #     skip = skip_connection if skip is None else skip_connection + skip
        # assert skip is not None

        x = self.output_projection(x)
        x = x.squeeze(1)  # NOTE: add squeeze here
        return x

    def get_loss_dict(self, pnotree_x, pnotree_y):
        """
        z_y is the stuff the diffusion model needs to learn
        """
        N = pnotree_x.shape[0]
        z_y = self.encode_z(pnotree_y, is_sampling=True)
        t = torch.randint(0, len(self.params.noise_schedule), [N], device=self.device)
        noisy_z_y, noise = self.q_t(z_y, t)

        if not self.params.unconditional:
            predicted = self.forward(noisy_z_y, t, pnotree_x)
        else:
            predicted = self.forward(noisy_z_y, t, None)
        return self.loss_function(noise, predicted)

    def q_t(self, z, t):
        noise = torch.randn_like(z)
        zt = self.noise_scale_sqrt[t] * z + (1.0 - self.noise_scale[t])**0.5 * noise
        return zt, noise
