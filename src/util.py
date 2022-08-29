from dirs import *
import numpy as np
import pickle
import os
import pretty_midi as pm
import torch
from dl_modules import PianoTreeEncoder, PianoTreeDecoder
from collections import OrderedDict


def load_pretrained_pnotree_enc_dec(fpath, max_simu_note, device):
    pnotree_enc = PianoTreeEncoder(device=device, max_simu_note=max_simu_note)
    pnotree_dec = PianoTreeDecoder(device=device, max_simu_note=max_simu_note)
    checkpoint = torch.load(fpath)
    enc_checkpoint = OrderedDict()
    dec_checkpoint = OrderedDict()
    enc_param_list = [
        "note_embedding", "enc_notes_gru", "enc_time_gru", "linear_mu", "linear_std"
    ]
    for k, v in checkpoint.items():
        part = k.split('.')[0]
        # print(part)
        # name = '.'.join(k.split('.')[1 :])
        # print(part, name)
        if part in enc_param_list:
            enc_checkpoint[k] = v
            if part == "note_embedding":
                dec_checkpoint[k] = v
        else:
            dec_checkpoint[k] = v
    pnotree_enc.load_state_dict(enc_checkpoint)
    pnotree_dec.load_state_dict(dec_checkpoint)
    pnotree_enc.to(device)
    pnotree_dec.to(device)
    return pnotree_enc, pnotree_dec


if __name__ == "__main__":
    load_pretrained_pnotree_enc_dec(
        "../PianoTree-VAE/model20/train_20-last-model.pt", 20, None
    )
