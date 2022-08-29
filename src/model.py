from dl_modules import PianoTreeEncoder, PianoTreeDecoder, NaiveNN
import torch
import torch.nn as nn


class Diffpro(nn.Module):
    def __init__(self, pretrained_pnotree_model):
        super(Diffpro, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pnotree_enc = PianoTreeEncoder(self.device)
        self.naive_nn = NaiveNN()
        self.pnotree_dec = PianoTreeDecoder(self.device)
        self._disable_grads_for_enc_dec()

    def _disable_grads_for_enc_dec(self):
        for param in self.pnotree_enc.parameters():
            param.requires_grad = False
        for param in self.pnotree_dec.parameters():
            param.requires_grad = False

    def forward(self, pnotree_x, pnotree_y, tfr1=0, tfr2=0):
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
