import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, f"{os.path.dirname(__file__)}/../")
from data.dataset_musicalion import PianoOrchDataset_Musicalion
from utils import (
    pr_mat_pitch_shift, prmat2c_to_midi_file, denormalize_prmat, chd_to_onehot,
    chd_pitch_shift, onehot_to_chd, chd_to_midi_file, pianotree_pitch_shift,
    estx_to_midi_file
)


def collate_fn(batch):
    def sample_shift():
        return np.random.choice(np.arange(-6, 6), 1)[0]

    prmat = []
    pnotree = []
    song_fn = []
    for b in batch:
        # b[0]: seg_pnotree_x; b[1]: seg_pnotree_y
        seg_prmat = b[0]
        seg_pnotree = b[1]

        shift = sample_shift()
        seg_prmat = pr_mat_pitch_shift(seg_prmat, shift)
        seg_pnotree = pianotree_pitch_shift(seg_pnotree, shift)

        prmat.append(seg_prmat)
        pnotree.append(seg_pnotree)

        if len(b) > 3:
            song_fn.append(b[3])

    prmat = torch.Tensor(np.array(prmat, np.float32)).float()
    pnotree = torch.Tensor(np.array(pnotree, np.int64)).long()
    # prmat_x = prmat_x.unsqueeze(1)  # (B, 1, 128, 128)
    if len(song_fn) > 0:
        return prmat, pnotree, None, song_fn
    else:
        return prmat, pnotree, None


def get_train_val_dataloaders(batch_size, num_workers=4, pin_memory=True, debug=False):
    train_dataset, val_dataset = PianoOrchDataset_Musicalion.load_train_and_valid_sets(
        debug
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size,
        True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size,
        True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return train_dl, val_dl


if __name__ == "__main__":
    train_dl, val_dl = get_train_val_dataloaders(16)
    print(len(train_dl))
    for batch in train_dl:
        print(len(batch))
        prmat, pnotree, _ = batch
        print(prmat.shape)
        print(pnotree.shape)
        prmat = prmat.cpu().numpy()
        pnotree = pnotree.cpu().numpy()
        prmat2c_to_midi_file(prmat, f"exp/test.mid")
        estx_to_midi_file(pnotree, f"exp/test_pnotree.mid")
        exit(0)
