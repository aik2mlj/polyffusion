# pyright: reportOptionalSubscript=false

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from dirs import *
from utils import (
    estx_to_midi_file,
    nmat_to_pianotree_repr,
    nmat_to_prmat,
    nmat_to_prmat2c,
    prmat2c_to_midi_file,
    prmat_to_midi_file,
    read_dict,
)

SEG_LGTH = 32
N_BIN = 4
SEG_LGTH_BIN = SEG_LGTH * N_BIN


class DataSampleNpz_Musicalion:
    """
    A pair of song segment stored in .npz format
    containing piano and orchestration versions

    This class aims to get input samples for a single song
    `__getitem__` is used for retrieving ready-made input segments to the model
    it will be called in DataLoader
    """

    def __init__(self, song_fn) -> None:
        self.fpath = os.path.join(MUSICALION_DATA_DIR, song_fn)
        self.song_fn = song_fn
        """
        notes (onset_beat, onset_bin, duration, pitch, velocity)
        start_table : i-th row indicates the starting row of the "notes" array
            at i-th beat.
        db_pos: an array of downbeat beat_ids

        x: orchestra
        y: piano

        dict : each downbeat corresponds to a SEG_LGTH-long segment
            nmat: note matrix (same format as input npz files)
            pr_mat: piano roll matrix (the format for texture decoder)
            pnotree: pnotree format (used for calculating loss & teacher-forcing)
        """

        # self.notes = None
        # self.chord = None
        # self.start_table = None
        # self.db_pos = None

        # self._nmat_dict = None
        # self._pnotree_dict = None
        # self._pr_mat_dict = None
        # self._feat_dict = None

        # def load(self, use_chord=False):
        #     """ load data """

        data = np.load(self.fpath, allow_pickle=True)
        self.notes = data["notes"]
        self.start_table: dict = data["start_table"].item()

        self.db_pos = data["db_pos"]
        self.db_pos_filter = data["db_pos_filter"]
        self.db_pos = self.db_pos[self.db_pos_filter]
        if len(self.db_pos) != 0:
            self.last_db = self.db_pos[-1]

        self._nmat_dict = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._pnotree_dict = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._prmat2c_dict = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._prmat_dict = dict(zip(self.db_pos, [None] * len(self.db_pos)))

    def __len__(self):
        """Return number of complete 8-beat segments in a song"""
        return len(self.db_pos)

    def note_mat_seg_at_db(self, db):
        """
        Select rows (notes) of the note_mat which lie between beats
        [db: db + 8].
        """

        s_ind = self.start_table[db]
        if db + SEG_LGTH_BIN in self.start_table:
            e_ind = self.start_table[db + SEG_LGTH_BIN]
            seg_mats = self.notes[s_ind:e_ind]
        else:
            seg_mats = self.notes[s_ind:]  # NOTE: may be wrong
        return seg_mats.copy()

    @staticmethod
    def reset_db_to_zeros(note_mat, db):
        note_mat[:, 0] -= db

    @staticmethod
    def format_reset_seg_mat(seg_mat):
        """
        The input seg_mat is (N, 5)
            onset, pitch, duration, velocity, program = note
        The output seg_mat is (N, 3). Columns for onset, pitch, duration.
        Onset ranges between range(0, 32).
        """

        output_mat = np.zeros((len(seg_mat), 3), dtype=np.int64)
        output_mat[:, 0] = seg_mat[:, 0]
        output_mat[:, 1] = seg_mat[:, 1]
        output_mat[:, 2] = seg_mat[:, 2]
        return output_mat

    def store_nmat_seg(self, db):
        """
        Get note matrix (SEG_LGTH) of orchestra(x) at db position
        """
        if self._nmat_dict[db] is not None:
            return

        nmat = self.note_mat_seg_at_db(db)
        self.reset_db_to_zeros(nmat, db)

        nmat = self.format_reset_seg_mat(nmat)
        self._nmat_dict[db] = nmat

    def store_prmat2c_seg(self, db):
        """
        Get piano roll format (SEG_LGTH) from note matrices at db position
        """
        if self._prmat2c_dict[db] is not None:
            return

        prmat2c = nmat_to_prmat2c(self._nmat_dict[db], SEG_LGTH_BIN)
        self._prmat2c_dict[db] = prmat2c

    def store_prmat_seg(self, db):
        """
        Get piano roll format (SEG_LGTH) from note matrices at db position
        """
        if self._prmat_dict[db] is not None:
            return

        prmat2c = nmat_to_prmat(self._nmat_dict[db], SEG_LGTH_BIN)
        self._prmat_dict[db] = prmat2c

    def store_pnotree_seg(self, db):
        """
        Get pnotree representation (SEG_LGTH) from nmat
        """
        if self._pnotree_dict[db] is not None:
            return

        self._pnotree_dict[db] = nmat_to_pianotree_repr(
            self._nmat_dict[db], n_step=SEG_LGTH_BIN
        )

    def _store_seg(self, db):
        self.store_nmat_seg(db)
        self.store_prmat2c_seg(db)
        self.store_prmat_seg(db)
        self.store_pnotree_seg(db)

    def _get_item_by_db(self, db):
        """
        Return segments of
            prmat, prmat_y
        """

        self._store_seg(db)

        seg_prmat2c = self._prmat2c_dict[db]
        seg_prmat = self._prmat_dict[db]
        seg_pnotree = self._pnotree_dict[db]
        return seg_prmat2c, seg_pnotree, None, seg_prmat

    def __getitem__(self, idx):
        db = self.db_pos[idx]
        return self._get_item_by_db(db)

    def get_whole_song_data(self):
        """
        used when inference
        """
        prmat2c = []
        pnotree = []
        prmat = []
        idx = 0
        i = 0
        while i < len(self):
            seg_prmat2c, seg_pnotree, _, seg_prmat = self[i]
            prmat2c.append(seg_prmat2c)
            pnotree.append(seg_pnotree)
            prmat.append(seg_prmat)

            idx += SEG_LGTH_BIN
            while i < len(self) and self.db_pos[i] < idx:
                i += 1
        prmat2c = torch.from_numpy(np.array(prmat2c, dtype=np.float32))
        pnotree = torch.from_numpy(np.array(pnotree, dtype=np.int64))
        prmat = torch.from_numpy(np.array(prmat, dtype=np.float32))
        return prmat2c, pnotree, None, prmat


class PianoOrchDataset_Musicalion(Dataset):
    def __init__(self, data_samples, debug=False):
        super(PianoOrchDataset_Musicalion, self).__init__()

        # a list of DataSampleNpz
        self.data_samples = data_samples

        self.lgths = np.array([len(d) for d in self.data_samples], dtype=np.int64)
        self.lgth_cumsum = np.cumsum(self.lgths)
        self.debug = debug

    def __len__(self):
        return self.lgth_cumsum[-1]

    def __getitem__(self, index):
        # song_no is the smallest id that > dataset_item
        song_no = np.where(self.lgth_cumsum > index)[0][0]
        song_item = index - np.insert(self.lgth_cumsum, 0, 0)[song_no]

        song_data = self.data_samples[song_no]
        if self.debug:
            return *song_data[song_item], song_data.song_fn
        else:
            return song_data[song_item]

    @classmethod
    def load_with_song_paths(cls, song_paths, debug=False):
        data_samples = [DataSampleNpz_Musicalion(song_path) for song_path in song_paths]
        return cls(data_samples, debug)

    @classmethod
    def load_train_and_valid_sets(cls, debug=False):
        split = read_dict(os.path.join(TRAIN_SPLIT_DIR, "musicalion.pickle"))
        return cls.load_with_song_paths(split[0], debug), cls.load_with_song_paths(
            split[1], debug
        )

    @classmethod
    def load_with_train_valid_paths(cls, tv_song_paths, **kwargs):
        return cls.load_with_song_paths(
            tv_song_paths[0], **kwargs
        ), cls.load_with_song_paths(tv_song_paths[1], **kwargs)


if __name__ == "__main__":
    test = "ssccm172.npz"
    song = DataSampleNpz_Musicalion(test)
    os.system(f"cp {MUSICALION_DATA_DIR}/{test[:-4]}_flatten.mid exp/copy.mid")
    prmat2c, pnotree, _, prmat = song.get_whole_song_data()
    print(prmat2c.shape)
    print(pnotree.shape)
    print(prmat.shape)
    prmat2c = prmat2c.cpu().numpy()
    pnotree = pnotree.cpu().numpy()
    prmat = prmat.cpu().numpy()
    prmat2c_to_midi_file(prmat2c, "exp/m_prmat2c.mid")
    estx_to_midi_file(pnotree, "exp/m_pnotree.mid")
    prmat_to_midi_file(prmat, "exp/m_prmat.mid")
