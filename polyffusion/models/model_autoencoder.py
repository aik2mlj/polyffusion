import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_diffusion.model.autoencoder import Autoencoder
from utils import *


class Polyffusion_Autoencoder(nn.Module):
    def __init__(self, autoencoder: Autoencoder):
        super(Polyffusion_Autoencoder, self).__init__()
        self.autoencoder = autoencoder

    @classmethod
    def load_trained(cls, ldm, model_dir):
        model = cls(ldm)
        trained_leaner = torch.load(f"{model_dir}/weights.pt")
        model.load_state_dict(trained_leaner["model"])
        return model

    def get_loss_dict(self, batch, step):
        """
        z_y is the stuff the diffusion model needs to learn
        """
        prmat, _, chord = batch
        # (#B, 2, 128, 128)
        print(f"prmat: {prmat.requires_grad}")
        prmat = F.pad(prmat, [0, 0, 0, 0, 0, 1], "constant", 0)
        print(f"prmat: {prmat.requires_grad}")
        # (#B, 3, 128, 128)
        return self.autoencoder.get_loss_dict(prmat, step)
