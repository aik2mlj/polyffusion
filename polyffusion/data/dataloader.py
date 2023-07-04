import sys
import os
import torch
import random
from torch.utils.data import DataLoader
import numpy as np

from data.dataset import PianoOrchDataset
from utils import (
    pr_mat_pitch_shift, prmat2c_to_midi_file, chd_to_onehot, chd_pitch_shift,
    chd_to_midi_file, estx_to_midi_file, pianotree_pitch_shift, prmat_to_midi_file
)

# SEED = 7890
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# random.seed(SEED)


def collate_fn(batch, shift):
    def sample_shift():
        return np.random.choice(np.arange(-6, 6), 1)[0]

    prmat2c = []
    pnotree = []
    chord = []
    prmat = []
    song_fn = []
    for b in batch:
        # b[0]: seg_pnotree; b[1]: seg_pnotree_y
        seg_prmat2c = b[0]
        seg_pnotree = b[1]
        seg_chord = b[2]
        seg_prmat = b[3]

        if shift:
            shift_pitch = sample_shift()
            seg_prmat2c = pr_mat_pitch_shift(seg_prmat2c, shift_pitch)
            seg_pnotree = pianotree_pitch_shift(seg_pnotree, shift_pitch)
            seg_chord = chd_pitch_shift(seg_chord, shift_pitch)
            seg_prmat = pr_mat_pitch_shift(seg_prmat, shift_pitch)

        seg_chord = chd_to_onehot(seg_chord)

        prmat2c.append(seg_prmat2c)
        pnotree.append(seg_pnotree)
        chord.append(seg_chord)
        prmat.append(seg_prmat)

        if len(b) > 4:
            song_fn.append(b[4])

    prmat2c = torch.Tensor(np.array(prmat2c, np.float32)).float()
    pnotree = torch.Tensor(np.array(pnotree, np.int64)).long()
    chord = torch.Tensor(np.array(chord, np.float32)).float()
    prmat = torch.Tensor(np.array(prmat, np.float32)).float()
    # prmat = prmat.unsqueeze(1)  # (B, 1, 128, 128)
    if len(song_fn) > 0:
        return prmat2c, pnotree, chord, prmat, song_fn
    else:
        return prmat2c, pnotree, chord, prmat


def get_train_val_dataloaders(
    batch_size, num_workers=0, pin_memory=False, debug=False, **kwargs
):
    train_dataset, val_dataset = PianoOrchDataset.load_train_and_valid_sets(
        debug, **kwargs
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size,
        True,
        collate_fn=lambda x: collate_fn(x, shift=True),
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size,
        True,
        collate_fn=lambda x: collate_fn(x, shift=False),
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    print(
        f"Dataloader ready: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}, {kwargs}"
    )
    return train_dl, val_dl


def get_val_dataloader(
    batch_size, num_workers=0, pin_memory=False, debug=False, **kwargs
):
    val_dataset = PianoOrchDataset.load_valid_set(debug, **kwargs)
    val_dl = DataLoader(
        val_dataset,
        batch_size,
        True,
        collate_fn=lambda x: collate_fn(x, shift=False),
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    print(
        f"Dataloader ready: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}, {kwargs}"
    )
    return val_dl


if __name__ == "__main__":
    train_dl, val_dl = get_train_val_dataloaders(16)
    print(len(train_dl))
    for batch in train_dl:
        print(len(batch))
        prmat2c, pnotree, chord, prmat = batch
        print(prmat2c.shape)
        print(pnotree.shape)
        print(chord.shape)
        print(prmat.shape)
        prmat2c = prmat2c.cpu().numpy()
        pnotree = pnotree.cpu().numpy()
        chord = chord.cpu().numpy()
        prmat = prmat.cpu().numpy()
        # chord = [onehot_to_chd(onehot) for onehot in chord]
        prmat2c_to_midi_file(prmat2c, f"exp/dl_prmat2c.mid")
        estx_to_midi_file(pnotree, f"exp/dl_pnotree.mid")
        chd_to_midi_file(chord, "exp/dl_chord.mid")
        prmat_to_midi_file(prmat, f"exp/dl_prmat.mid")
        exit(0)
