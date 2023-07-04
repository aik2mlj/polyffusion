import torch
import torch.nn as nn
from utils import *
from ddpm import DenoiseDiffusion
import torch.nn.functional as F


class Polyffusion_DDPM(nn.Module):
    def __init__(
        self,
        ddpm: DenoiseDiffusion,
        params,
        max_simu_note=20,
    ):
        super(Polyffusion_DDPM, self).__init__()
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ddpm = ddpm

    @classmethod
    def load_trained(cls, ddpm, model_dir, params, max_simu_note=20):
        model = cls(ddpm, params, max_simu_note)
        trained_leaner = torch.load(f"{model_dir}/weights.pt")
        model.load_state_dict(trained_leaner["model"])
        return model

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        return self.ddpm.p_sample(xt, t)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        return self.ddpm.q_sample(x0, t)

    def get_loss_dict(self, prmat):
        """
        z_y is the stuff the diffusion model needs to learn
        """
        return {"loss": self.ddpm.loss(prmat)}
