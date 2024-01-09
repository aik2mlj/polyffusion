import torch
import torch.nn as nn

from ddpm import DenoiseDiffusion
from utils import *


class Polyffusion_DDPM(nn.Module):
    def __init__(
        self,
        ddpm: DenoiseDiffusion,
        params,
        max_simu_note=20,
    ):
        super(Polyffusion_DDPM, self).__init__()
        self.params = params
        self.ddpm = ddpm

    @classmethod
    def load_trained(cls, ddpm, chkpt_fpath, params, max_simu_note=20):
        model = cls(ddpm, params, max_simu_note)
        trained_leaner = torch.load(chkpt_fpath)
        model.load_state_dict(trained_leaner["model"])
        return model

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        return self.ddpm.p_sample(xt, t)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor):
        return self.ddpm.q_sample(x0, t)

    def get_loss_dict(self, batch, step):
        """
        z_y is the stuff the diffusion model needs to learn
        """
        prmat2c, pnotree, chord, prmat = batch
        return {"loss": self.ddpm.loss(prmat2c)}
