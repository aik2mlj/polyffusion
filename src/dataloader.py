import torch
from torch.utils.data import DataLoader
from dataset import PianoOrchDataset
from utils import (pr_mat_pitch_shift, prmat2c_to_midi_file, denormalize_prmat)
import numpy as np
from params import params


def collate_fn(batch):
    def sample_shift():
        return np.random.choice(np.arange(-6, 6), 1)[0]

    prmat_x = []
    song_fn = []
    for b in batch:
        # b[0]: seg_pnotree_x; b[1]: seg_pnotree_y
        seg_prmat_x = b[0]

        shift = sample_shift()
        seg_prmat_x = pr_mat_pitch_shift(seg_prmat_x, shift)

        prmat_x.append(seg_prmat_x)

        if len(b) > 2:
            song_fn.append(b[2])

    prmat_x = torch.Tensor(np.array(prmat_x, np.float32)).float()
    # prmat_x = prmat_x.unsqueeze(1)  # (B, 1, 128, 128)
    if len(song_fn) > 0:
        return prmat_x, prmat_x, song_fn
    else:
        return prmat_x, prmat_x


def get_train_val_dataloaders(batch_size, params, debug=False):
    train_dataset, val_dataset = PianoOrchDataset.load_train_and_valid_sets(debug)
    train_dl = DataLoader(
        train_dataset,
        batch_size,
        True,
        collate_fn=collate_fn,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory
    )
    val_dl = DataLoader(
        val_dataset,
        batch_size,
        True,
        collate_fn=collate_fn,
        num_workers=params.num_workers,
        pin_memory=params.pin_memory
    )
    return train_dl, val_dl


if __name__ == "__main__":
    train_dl, val_dl = get_train_val_dataloaders(64, params, debug=True)
    print(len(train_dl))
    for batch in train_dl:
        print(len(batch))
        prmat_x, _, song_fn = batch
        print(prmat_x.shape)
        prmat_x = prmat_x.cpu().numpy()
        prmat2c_to_midi_file(prmat_x, f"exp/test_x.mid", song_fn)
        exit(0)
