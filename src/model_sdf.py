import torch
import torch.nn as nn
from utils import *
from stable_diffusion.latent_diffusion import LatentDiffusion
import torch.nn.functional as F


class Diffpro_SDF(nn.Module):
    def __init__(
        self,
        ldm: LatentDiffusion,
    ):
        super(Diffpro_SDF, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ldm = ldm

    @classmethod
    def load_trained(cls, ldm, model_dir):
        model = cls(ldm)
        trained_leaner = torch.load(f"{model_dir}/weights.pt")
        model.load_state_dict(trained_leaner["model"])
        return model

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        return self.ldm.p_sample(xt, t)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        return self.ldm.q_sample(x0, t)

    def get_loss_dict(self, batch, step):
        """
        z_y is the stuff the diffusion model needs to learn
        """
        prmat, _, chord = batch
        return {"loss": self.ldm.loss(prmat, chord)}
