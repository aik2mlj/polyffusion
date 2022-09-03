import torch
from torch.utils.data import DataLoader
from dataset import PianoOrchDataset
from utils import (
    pianotree_pitch_shift,
)
import numpy as np


def collate_fn(batch):
    def sample_shift():
        return np.random.choice(np.arange(-6, 6), 1)[0]

    pnotree_x = []
    pnotree_y = []
    for b in batch:
        seg_pnotree_x, seg_pnotree_y = b

        shift = sample_shift()
        seg_pnotree_x = pianotree_pitch_shift(seg_pnotree_x, shift)
        seg_pnotree_y = pianotree_pitch_shift(seg_pnotree_y, shift)

        pnotree_x.append(seg_pnotree_x)
        pnotree_y.append(seg_pnotree_y)

    pnotree_x = torch.Tensor(np.array(pnotree_x))
    pnotree_y = torch.Tensor(np.array(pnotree_y))
    pnotree_x = pnotree_x.long()
    pnotree_y = pnotree_y.long()
    return pnotree_x, pnotree_y


def get_train_val_dataloaders(batch_size):
    train_dataset, val_dataset = PianoOrchDataset.load_train_and_valid_sets()
    train_dl = DataLoader(train_dataset, batch_size, True, collate_fn=collate_fn)
    val_dl = DataLoader(val_dataset, batch_size, True, collate_fn=collate_fn)
    return train_dl, val_dl


if __name__ == "__main__":
    train_dl, val_dl = get_train_val_dataloaders(128)
    for batch in train_dl:
        
