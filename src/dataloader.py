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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pno_tree_x = []
    pno_tree_y = []
    for b in batch:
        seg_pno_tree_x, seg_pno_tree_y = b

        shift = sample_shift()
        seg_pno_tree_x = pianotree_pitch_shift(seg_pno_tree_x, shift)
        seg_pno_tree_y = pianotree_pitch_shift(seg_pno_tree_y, shift)

        pno_tree_x.append(seg_pno_tree_x)
        pno_tree_y.append(seg_pno_tree_y)

    pno_tree_x = torch.Tensor(np.array(pno_tree_x))
    pno_tree_y = torch.Tensor(np.array(pno_tree_y))
    pno_tree_x = pno_tree_x.long().to(device)
    pno_tree_y = pno_tree_y.long().to(device)
    # print(pno_tree_x.shape)
    return pno_tree_x, pno_tree_y


def get_train_val_dataloaders(batch_size):
    train_dataset, val_dataset = PianoOrchDataset.load_train_and_valid_sets()
    train_loader = DataLoader(train_dataset, batch_size, True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size, True, collate_fn=collate_fn)
