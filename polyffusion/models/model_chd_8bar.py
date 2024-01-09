import torch
import torch.nn as nn

from dl_modules import ChordDecoder, ChordEncoder
from utils import *


class Chord_8Bar(nn.Module):
    def __init__(self, chord_enc: ChordEncoder, chord_dec: ChordDecoder):
        super(Chord_8Bar, self).__init__()
        self.chord_enc = chord_enc
        self.chord_dec = chord_dec

    @classmethod
    def load_trained(cls, chord_enc, chord_dec, model_dir):
        model = cls(chord_enc, chord_dec)
        trained_leaner = torch.load(f"{model_dir}/weights.pt")
        model.load_state_dict(trained_leaner["model"])
        return model

    def chord_loss(self, c, recon_root, recon_chroma, recon_bass):
        loss_fun = nn.CrossEntropyLoss()
        root = c[:, :, 0:12].max(-1)[-1].view(-1).contiguous()
        chroma = c[:, :, 12:24].long().view(-1).contiguous()
        bass = c[:, :, 24:].max(-1)[-1].view(-1).contiguous()

        recon_root = recon_root.view(-1, 12).contiguous()
        recon_chroma = recon_chroma.view(-1, 2).contiguous()
        recon_bass = recon_bass.view(-1, 12).contiguous()
        root_loss = loss_fun(recon_root, root)
        chroma_loss = loss_fun(recon_chroma, chroma)
        bass_loss = loss_fun(recon_bass, bass)
        chord_loss = root_loss + chroma_loss + bass_loss
        return {
            "loss": chord_loss,
            "root": root_loss,
            "chroma": chroma_loss,
            "bass": bass_loss,
        }

    def get_loss_dict(self, batch, step, tfr_chd):
        _, _, chord, _ = batch

        z_chd = self.chord_enc(chord).rsample()
        recon_root, recon_chroma, recon_bass = self.chord_dec(
            z_chd, False, tfr_chd, chord
        )
        return self.chord_loss(chord, recon_root, recon_chroma, recon_bass)
