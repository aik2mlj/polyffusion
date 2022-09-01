from dl_modules import PianoTreeEncoder, PianoTreeDecoder, NaiveNN
import torch
import torch.nn as nn
from utils import *


class Diffpro(nn.Module):
    def __init__(self, pt_pnotree_model_path, params):
        super(Diffpro, self).__init__()
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # load pretrained model
        self.pnotree_enc, self.pnotree_dec = load_pretrained_pnotree_enc_dec(
            pt_pnotree_model_path, 20, self.device
        )
        self.naive_nn = NaiveNN()
        self._disable_grads_for_enc_dec()

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
        """
        FIXME: teacher-forcing is not needed here?
        """
        dist_x, emb_x = self.pnotree_enc(pnotree_x)

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
