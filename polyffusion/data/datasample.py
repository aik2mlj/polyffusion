# pyright: reportOptionalSubscript=false
"""
This is for directly get DataSampleNpz from a midi file when inference.
"""

import sys

import numpy as np
import torch

from data.midi_to_data import get_data_for_single_midi
from dirs import *
from utils import (
    chd_to_midi_file,
    chd_to_onehot,
    estx_to_midi_file,
    nmat_to_pianotree_repr,
    nmat_to_prmat,
    nmat_to_prmat2c,
    prmat2c_to_midi_file,
    prmat_to_midi_file,
)

SEG_LGTH = 32
N_BIN = 4
SEG_LGTH_BIN = SEG_LGTH * N_BIN


class DataSample:
    """
    This class aims to get input samples for a single song
    `__getitem__` is used for retrieving ready-made input segments to the model
    it will be called in DataLoader
    """

    def __init__(self, data) -> None:
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

        self.notes = data["notes"]
        self.start_table: dict = data["start_table"].item()

        self.db_pos = data["db_pos"]
        self.db_pos_filter = data["db_pos_filter"]
        self.db_pos = self.db_pos[self.db_pos_filter]
        if len(self.db_pos) != 0:
            self.last_db = self.db_pos[-1]

        self.chord = data["chord"].astype(np.int32)

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
        chord = self.chord[db // N_BIN : db // N_BIN + SEG_LGTH]
        if chord.shape[0] < SEG_LGTH:
            chord = np.append(
                chord, np.zeros([SEG_LGTH - chord.shape[0], 14], dtype=np.int32), axis=0
            )

        return seg_prmat2c, seg_pnotree, chord, seg_prmat

    def __getitem__(self, idx):
        db = self.db_pos[idx]
        return self._get_item_by_db(db)

    def get_whole_song_data(self):
        """
        used when inference
        """
        prmat2c = []
        pnotree = []
        chord = []
        prmat = []
        idx = 0
        i = 0
        while i < len(self):
            seg_prmat2c, seg_pnotree, seg_chord, seg_prmat = self[i]
            prmat2c.append(seg_prmat2c)
            pnotree.append(seg_pnotree)
            chord.append(chd_to_onehot(seg_chord))
            prmat.append(seg_prmat)

            idx += SEG_LGTH_BIN
            while i < len(self) and self.db_pos[i] < idx:
                i += 1
        prmat2c = torch.from_numpy(np.array(prmat2c, dtype=np.float32))
        pnotree = torch.from_numpy(np.array(pnotree, dtype=np.int64))
        chord = torch.from_numpy(np.array(chord, dtype=np.float32))
        prmat = torch.from_numpy(np.array(prmat, dtype=np.float32))

        return prmat2c, pnotree, chord, prmat


if __name__ == "__main__":
    fpath = sys.argv[1]
    data = get_data_for_single_midi(fpath, sys.argv[2])
    song = DataSample(data)
    prmat2c, pnotree, chord, prmat = song.get_whole_song_data()
    print(prmat2c.shape)
    print(pnotree.shape)
    print(chord.shape)
    print(prmat.shape)
    prmat2c = prmat2c.cpu().numpy()
    pnotree = pnotree.cpu().numpy()
    chord = chord.cpu().numpy()
    prmat = prmat.cpu().numpy()
    prmat2c_to_midi_file(prmat2c, "exp/s_prmat2c.mid")
    estx_to_midi_file(pnotree, "exp/s_pnotree.mid")
    chd_to_midi_file(chord, "exp/s_chord.mid")
    prmat_to_midi_file(prmat, "exp/s_prmat.mid")
