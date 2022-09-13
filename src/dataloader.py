import torch
from torch.utils.data import DataLoader
from dataset import PianoOrchDataset
from utils import (pianotree_pitch_shift, estx_to_midi_file)
import numpy as np


def collate_fn(batch):
    def sample_shift():
        return np.random.choice(np.arange(-6, 6), 1)[0]

    pnotree_x = []
    pnotree_y = []
    song_fn = []
    for b in batch:
        # b[0]: seg_pnotree_x; b[1]: seg_pnotree_y
        seg_pnotree_x = b[0]
        seg_pnotree_y = b[1]

        shift = sample_shift()
        seg_pnotree_x = pianotree_pitch_shift(seg_pnotree_x, shift)
        seg_pnotree_y = pianotree_pitch_shift(seg_pnotree_y, shift)

        pnotree_x.append(b[0])
        pnotree_y.append(b[1])

        if len(b) > 2:
            song_fn.append(b[2])

    pnotree_x = torch.Tensor(np.array(pnotree_x)).long()
    pnotree_y = torch.Tensor(np.array(pnotree_y)).long()
    if len(song_fn) > 0:
        return pnotree_x, pnotree_y, song_fn
    else:
        return pnotree_x, pnotree_y


def get_train_val_dataloaders(batch_size, debug=False):
    train_dataset, val_dataset = PianoOrchDataset.load_train_and_valid_sets(debug)
    train_dl = DataLoader(train_dataset, batch_size, True, collate_fn=collate_fn)
    val_dl = DataLoader(val_dataset, batch_size, True, collate_fn=collate_fn)
    return train_dl, val_dl


if __name__ == "__main__":
    train_dl, val_dl = get_train_val_dataloaders(128, debug=True)
    for batch in train_dl:
        print(len(batch))
        pnotree_x, pnotree_y, song_fn = batch
        estx_to_midi_file(pnotree_x, f"exp/test_x.mid", song_fn)
        estx_to_midi_file(pnotree_y, f"exp/test_y.mid", song_fn)
        exit(0)
