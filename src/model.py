from dl_modules import PianoTreeEncoder, PianoTreeDecoder, NaiveNN
import torch
import torch.nn as nn
from utils import *
from ddpm import DenoiseDiffusion
import torch.nn.functional as F


class Diffpro_DDPM(nn.Module):
    def __init__(
        self,
        ddpm: DenoiseDiffusion,
        params,
        max_simu_note=20,
        pt_pnotree_model_path=None
    ):
        super(Diffpro_DDPM, self).__init__()
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
        self.ddpm = ddpm
        self._disable_grads_for_enc_dec()

    @classmethod
    def load_trained(cls, ddpm, model_dir, params, max_simu_note=20):
        model = cls(ddpm, params, max_simu_note, None)
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

    @staticmethod
    def transform_to_2d(z):
        # z: (B, 512,)
        N = z.shape[0]
        z_padded = F.pad(z, (0, 512))
        z_2d = z_padded.view(N, 32, 32)
        z_2d = z_2d.unsqueeze(1)
        return z_2d

    def get_loss_dict(self, pnotree):
        """
        z_y is the stuff the diffusion model needs to learn
        """
        z = self.encode_z(pnotree, is_sampling=True)
        z_2d = self.transform_to_2d(z)
        return {"loss": self.ddpm.loss(z_2d)}
