import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset_musicalion import PianoOrchDataset_Musicalion
from utils import (
    estx_to_midi_file,
    pianotree_pitch_shift,
    pr_mat_pitch_shift,
    prmat2c_to_midi_file,
    prmat_to_midi_file,
)


def collate_fn(batch, shift):
    def sample_shift():
        return np.random.choice(np.arange(-6, 6), 1)[0]

    prmat2c = []
    pnotree = []
    prmat = []
    song_fn = []
    for b in batch:
        # b[0]: seg_pnotree; b[1]: seg_pnotree_y
        seg_prmat2c = b[0]
        seg_pnotree = b[1]
        seg_prmat = b[3]

        if shift:
            shift_pitch = sample_shift()
            seg_prmat2c = pr_mat_pitch_shift(seg_prmat2c, shift_pitch)
            seg_pnotree = pianotree_pitch_shift(seg_pnotree, shift_pitch)
            seg_prmat = pr_mat_pitch_shift(seg_prmat, shift_pitch)

        prmat2c.append(seg_prmat2c)
        pnotree.append(seg_pnotree)
        prmat.append(seg_prmat)

        if len(b) > 4:
            song_fn.append(b[4])

    prmat2c = torch.Tensor(np.array(prmat2c, np.float32)).float()
    pnotree = torch.Tensor(np.array(pnotree, np.int64)).long()
    prmat = torch.Tensor(np.array(prmat, np.float32)).float()
    # prmat = prmat.unsqueeze(1)  # (B, 1, 128, 128)
    if len(song_fn) > 0:
        return prmat2c, pnotree, None, prmat, song_fn
    else:
        return prmat2c, pnotree, None, prmat


def get_train_val_dataloaders(batch_size, num_workers=0, pin_memory=False, debug=False):
    train_dataset, val_dataset = PianoOrchDataset_Musicalion.load_train_and_valid_sets(
        debug
    )
    train_dl = DataLoader(
        train_dataset,
        batch_size,
        True,
        collate_fn=lambda x: collate_fn(x, shift=True),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size,
        False,
        collate_fn=lambda x: collate_fn(x, shift=False),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    print(
        f"Dataloader ready: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}"
    )
    return train_dl, val_dl


if __name__ == "__main__":
    train_dl, val_dl = get_train_val_dataloaders(16)
    print(len(train_dl))
    for batch in train_dl:
        print(len(batch))
        prmat2c, pnotree, _, prmat = batch
        print(prmat2c.shape)
        print(pnotree.shape)
        print(prmat.shape)
        prmat2c = prmat2c.cpu().numpy()
        pnotree = pnotree.cpu().numpy()
        prmat = prmat.cpu().numpy()
        prmat2c_to_midi_file(prmat2c, "exp/m_dl_prmat2c.mid")
        estx_to_midi_file(pnotree, "exp/m_dl_pnotree.mid")
        prmat_to_midi_file(prmat, "exp/m_dl_prmat.mid")
        exit(0)
