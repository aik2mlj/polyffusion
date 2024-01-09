import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.utils.rnn import pack_padded_sequence


class PianoTreeEncoder(nn.Module):
    def __init__(
        self,
        max_simu_note=20,
        max_pitch=127,
        min_pitch=0,
        pitch_sos=128,
        pitch_eos=129,
        pitch_pad=130,
        dur_pad=2,
        dur_width=5,
        num_step=32,
        note_emb_size=128,
        enc_notes_hid_size=256,
        enc_time_hid_size=512,
        z_size=512,
    ):
        super(PianoTreeEncoder, self).__init__()

        # Parameters
        # note and time
        self.max_pitch = max_pitch  # the highest pitch in train/val set.
        self.min_pitch = min_pitch  # the lowest pitch in train/val set.
        self.pitch_sos = pitch_sos
        self.pitch_eos = pitch_eos
        self.pitch_pad = pitch_pad
        self.pitch_range = max_pitch - min_pitch + 3  # not including pad.
        self.dur_pad = dur_pad
        self.dur_width = dur_width
        self.note_size = self.pitch_range + dur_width
        self.max_simu_note = max_simu_note  # the max # of notes at each ts.
        self.num_step = num_step  # 32
        self.note_emb_size = note_emb_size
        self.z_size = z_size
        self.enc_notes_hid_size = enc_notes_hid_size
        self.enc_time_hid_size = enc_time_hid_size

        self.note_embedding = nn.Linear(self.note_size, note_emb_size)
        self.enc_notes_gru = nn.GRU(
            note_emb_size,
            enc_notes_hid_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.enc_time_gru = nn.GRU(
            2 * enc_notes_hid_size,
            enc_time_hid_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.linear_mu = nn.Linear(2 * enc_time_hid_size, z_size)
        self.linear_std = nn.Linear(2 * enc_time_hid_size, z_size)

    @property
    def device(self):
        """
        ### Get model device
        """
        return next(iter(self.parameters())).device

    def get_len_index_tensor(self, ind_x):
        """Calculate the lengths ((B, 32), torch.LongTensor) of pgrid."""
        with torch.no_grad():
            lengths = self.max_simu_note - (
                ind_x[:, :, :, 0] - self.pitch_pad == 0
            ).sum(dim=-1)
        return lengths.to("cpu")

    def index_tensor_to_multihot_tensor(self, ind_x):
        """Transfer piano_grid to multi-hot piano_grid."""
        # ind_x: (B, 32, max_simu_note, 1 + dur_width)
        with torch.no_grad():
            dur_part = ind_x[:, :, :, 1:].float()
            out = torch.zeros(
                [
                    ind_x.size(0) * self.num_step * self.max_simu_note,
                    self.pitch_range + 1,
                ],
                dtype=torch.float,
            ).to(self.device)

            out[range(0, out.size(0)), ind_x[:, :, :, 0].reshape(-1)] = 1.0
            out = out.view(-1, 32, self.max_simu_note, self.pitch_range + 1)
            out = torch.cat([out[:, :, :, 0 : self.pitch_range], dur_part], dim=-1)
        return out

    def encoder(self, x, lengths):
        embedded = self.note_embedding(x)
        # x: (B, num_step, max_simu_note, note_emb_size)
        # now x are notes
        x = embedded.view(-1, self.max_simu_note, self.note_emb_size)
        x = pack_padded_sequence(
            x, lengths.view(-1), batch_first=True, enforce_sorted=False
        )
        x = self.enc_notes_gru(x)[-1].transpose(0, 1).contiguous()
        x = x.view(-1, self.num_step, 2 * self.enc_notes_hid_size)
        # now, x is simu_notes.
        x = self.enc_time_gru(x)[-1].transpose(0, 1).contiguous()
        # x: (B, 2, enc_time_hid_size)
        x = x.view(x.size(0), -1)
        mu = self.linear_mu(x)  # (B, z_size)
        std = self.linear_std(x).exp_()  # (B, z_size)
        dist = Normal(mu, std)
        return dist, embedded

    def forward(self, x, return_iterators=False):
        lengths = self.get_len_index_tensor(x)
        x = self.index_tensor_to_multihot_tensor(x)
        dist, embedded_x = self.encoder(x, lengths)
        if return_iterators:
            return dist.mean, dist.scale, embedded_x
        else:
            return dist, embedded_x, lengths
