from dl_modules import PianoTreeEncoder, PianoTreeDecoder, NaiveNN
import torch
import torch.nn as nn
from utils import *


class Diffpro(nn.Module):
    def __init__(self, params, max_simu_note=20, pt_pnotree_model_path=None):
        super(Diffpro, self).__init__()
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
        self.naive_nn = NaiveNN()
        self._disable_grads_for_enc_dec()

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
        # FIXME: teacher-forcing is not needed here?
        dist_x, emb_x, _ = self.pnotree_enc(pnotree_x)

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

    def output_to_numpy(self, recon_pitch, recon_dur):
        est_pitch = recon_pitch.max(-1)[1].unsqueeze(-1)  # (B, 32, 20, 1)
        est_dur = recon_dur.max(-1)[1]  # (B, 32, 11, 5)
        est_x = torch.cat([est_pitch, est_dur], dim=-1)  # (B, 32, 20, 6)
        est_x = est_x.cpu().numpy()
        recon_pitch = recon_pitch.cpu().numpy()
        recon_dur = recon_dur.cpu().numpy()
        return est_x, recon_pitch, recon_dur

    def infer(self, pnotree_x, is_sampling=False):
        with torch.no_grad():
            dist_x, emb_x, _ = self.pnotree_enc(pnotree_x)

            z_x = dist_x.rsample() if is_sampling else dist_x.mean

            z = self.naive_nn(z_x)

            # pianotree decoder
            recon_pitch, recon_dur = self.pnotree_dec(z, True, None, None, 0, 0)

            return self.output_to_numpy(recon_pitch, recon_dur)
