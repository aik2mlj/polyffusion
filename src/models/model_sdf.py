import torch
import torch.nn as nn
import sys

sys.path.insert(0, "..")
from utils import *
from stable_diffusion.latent_diffusion import LatentDiffusion
import torch.nn.functional as F
import random


class Diffpro_SDF(nn.Module):
    def __init__(
        self, ldm: LatentDiffusion, cond_mode="cond", chord_enc=None, chord_dec=None
    ):
        """
        cond_mode: {cond, mix, uncond}
            mix: use a special condition for unconditional learning with probability of 0.2
        use_enc: whether to use pretrained chord encoder to generate encoded condition
        """
        super(Diffpro_SDF, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ldm = ldm
        self.cond_mode = cond_mode
        self.chord_enc = chord_enc
        self.chord_dec = chord_dec

        # Freeze params for pretrained chord enc and dec
        if self.chord_enc is not None:
            for param in self.chord_enc.parameters():
                param.requires_grad = False
        if self.chord_dec is not None:
            for param in self.chord_dec.parameters():
                param.requires_grad = False

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

    def _encode_chord(self, chord):
        if self.chord_enc is not None:
            z_list = []
            for chord_seg in chord.split(8, 1):  # (#B, 8, 36) * 4
                # print(f"chord seg {chord_seg.shape}")
                z_seg = self.chord_enc(chord_seg).mean
                # print(f"chord seg z {chord_seg_z.shape}")
                z_list.append(z_seg)
            z = torch.stack(z_list, dim=1)
            return z
        else:
            return chord

    def _decode_chord(self, z):
        if self.chord_dec is not None:
            chord_list = []
            for z_seg in z.split(1, 1):
                z_seg = z_seg.squeeze()
                # print(f"z_seg {z_seg.shape}")
                recon_root, recon_chroma, recon_bass = self.chord_dec(
                    z_seg, inference=True, tfr=0.
                )
                recon_root = F.one_hot(recon_root.max(-1)[-1], num_classes=12)
                recon_chroma = recon_chroma.max(-1)[-1]
                recon_bass = F.one_hot(recon_bass.max(-1)[-1], num_classes=12)
                # print(recon_root.shape, recon_chroma.shape, recon_bass.shape)
                chord_seg = torch.cat([recon_root, recon_chroma, recon_bass], dim=-1)
                # print(f"chord seg {chord_seg.shape}")
                chord_list.append(chord_seg)
            chord = torch.cat(chord_list, dim=1)
            # print(f"chord {chord.shape}")
            return chord
        else:
            return z

    def get_loss_dict(self, batch, step):
        """
        z_y is the stuff the diffusion model needs to learn
        """
        prmat, _, chord = batch
        # chd_to_midi_file(chord, "exp/chd_origin.mid")
        cond = self._encode_chord(chord)
        # recon_chord = self._decode_chord(chord_z)
        # chd_to_midi_file(recon_chord, "exp/chd_recon.mid")
        # exit(0)

        if self.cond_mode == "uncond":
            cond = (-torch.ones_like(cond)).to(self.device)  # a bunch of -1
        elif self.cond_mode == "mix":
            if random.random() < 0.2:
                cond = (-torch.ones_like(cond)).to(self.device)  # a bunch of -1
        return {"loss": self.ldm.loss(prmat, cond)}
