import random

import torch
from torch import nn


class ChordDecoder(nn.Module):
    def __init__(
        self, input_dim=36, z_input_dim=256, hidden_dim=512, z_dim=256, n_step=8
    ):
        super(ChordDecoder, self).__init__()
        self.z2dec_hid = nn.Linear(z_dim, hidden_dim)
        self.z2dec_in = nn.Linear(z_dim, z_input_dim)
        self.gru = nn.GRU(
            input_dim + z_input_dim, hidden_dim, batch_first=True, bidirectional=False
        )
        self.init_input = nn.Parameter(torch.rand(36))
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.root_out = nn.Linear(hidden_dim, 12)
        self.chroma_out = nn.Linear(hidden_dim, 24)
        self.bass_out = nn.Linear(hidden_dim, 12)
        self.n_step = n_step
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, z_chd, inference, tfr, gt_chd=None):
        # z_chd: (B, z_chd_size)
        bs = z_chd.size(0)
        z_chd_hid = self.z2dec_hid(z_chd).unsqueeze(0)
        z_chd_in = self.z2dec_in(z_chd).unsqueeze(1)
        if inference:
            tfr = 0.0
        token = self.init_input.repeat(bs, 1).unsqueeze(1)
        recon_root = []
        recon_chroma = []
        recon_bass = []

        for t in range(self.n_step):
            chd_t, z_chd_hid = self.gru(torch.cat([token, z_chd_in], dim=-1), z_chd_hid)

            # compute output distribution
            r_root = self.root_out(chd_t)  # (bs, 1, 12)
            r_chroma = self.chroma_out(chd_t).view(bs, 1, 12, 2).contiguous()
            r_bass = self.bass_out(chd_t)  # (bs, 1, 12)

            # write distribution on the list
            recon_root.append(r_root)
            recon_chroma.append(r_chroma)
            recon_bass.append(r_bass)

            # prepare the input to the next step
            if t == self.n_step - 1:
                break
            teacher_force = random.random() < tfr
            if teacher_force and not inference:
                token = gt_chd[:, t].unsqueeze(1)
            else:
                t_root = torch.zeros(bs, 1, 12).to(z_chd.device).float()
                t_root[torch.arange(0, bs), 0, r_root.max(-1)[-1]] = 1.0
                t_chroma = r_chroma.max(-1)[-1].float()
                t_bass = torch.zeros(bs, 1, 12).to(z_chd.device).float()
                t_bass[torch.arange(0, bs), 0, r_bass.max(-1)[-1]] = 1.0
                token = torch.cat([t_root, t_chroma, t_bass], dim=-1)

        recon_root = torch.cat(recon_root, dim=1)
        recon_chroma = torch.cat(recon_chroma, dim=1)
        recon_bass = torch.cat(recon_bass, dim=1)
        # print(recon_root.shape, recon_chroma.shape, recon_bass.shape)
        return recon_root, recon_chroma, recon_bass

    def recon_loss(self, c, recon_root, recon_chroma, recon_bass):
        loss_fun = self.loss_func
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
        return chord_loss, root_loss, chroma_loss, bass_loss
