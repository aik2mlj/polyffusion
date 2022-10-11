# pyright: reportOptionalSubscript=false

from torch.utils.data import Dataset
from utils import (
    nmat_to_pianotree_repr, prmat2c_to_midi_file, normalize_prmat, denormalize_prmat,
    nmat_to_prmat2c
)
from utils import read_dict
from dirs import *
import os
import torch
import numpy as np

SEG_LGTH = 32
N_BIN = 4
SEG_LGTH_BIN = SEG_LGTH * N_BIN


class DataSampleNpz:
    """
    A pair of song segment stored in .npz format
    containing piano and orchestration versions

    This class aims to get input samples for a single song
    `__getitem__` is used for retrieving ready-made input segments to the model
    it will be called in DataLoader
    """
    def __init__(self, song_fn, use_track=[1, 2]) -> None:
        self.fpath = os.path.join(POP909_DATA_DIR, song_fn)
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
        self.use_track = use_track  # which tracks to use when converting to prmat2c

        data_x = np.load(self.fpath, allow_pickle=True)
        self.notes_x = np.array(
            data_x["notes"]
        )  # NOTE: here we have 3 tracks: melody, bridge and piano
        self.start_table_x = data_x["start_table"]  # NOTE: same here

        self.db_pos = data_x["db_pos"]
        self.db_pos_filter = data_x["db_pos_filter"]
        self.db_pos = self.db_pos[self.db_pos_filter]
        if len(self.db_pos) != 0:
            self.last_db = self.db_pos[-1]

        self._nmat_dict_x = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._pnotree_dict_x = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._pr_mat_dict_x = dict(zip(self.db_pos, [None] * len(self.db_pos)))
        self._feat_dict_x = dict(zip(self.db_pos, [None] * len(self.db_pos)))

    def __len__(self):
        """Return number of complete 8-beat segments in a song"""
        return len(self.db_pos)

    def note_mats_seg_at_db_x(self, db):
        """
        Select rows (notes) of the note_mat which lie between beats
        [db: db + 8].
        """

        seg_mats = []
        for notes, start_table in zip(self.notes_x, self.start_table_x):
            s_ind = start_table[db]
            if db + SEG_LGTH_BIN in start_table:
                e_ind = start_table[db + SEG_LGTH_BIN]
                note_seg = np.array(notes[s_ind : e_ind])
            else:
                note_seg = np.array(notes[s_ind :])  # NOTE: may be wrong
            if note_seg.size == 0:
                note_seg = np.zeros([0, 5])
            seg_mats.append(note_seg)
        return seg_mats

    @staticmethod
    def cat_note_mats(note_mats):
        return np.concatenate(note_mats, 0)

    @staticmethod
    def reset_db_to_zeros(note_mats, db):
        for mat in note_mats:
            mat[:, 0] -= db

    @staticmethod
    def format_reset_seg_mats(seg_mats):
        """
        The input seg_mat is (N, 5)
            onset, pitch, duration, velocity, program = note
        The output seg_mat is (N, 3). Columns for onset, pitch, duration.
        Onset ranges between range(0, 32).
        """

        output_mats = []
        for seg in seg_mats:
            mat = np.zeros((len(seg), 3), dtype=np.int64)
            mat[:, 0] = seg[:, 0]
            mat[:, 1] = seg[:, 1]
            mat[:, 2] = seg[:, 2]
            output_mats.append(mat)
        return output_mats

    def store_nmat_seg_x(self, db):
        """
        Get note matrix (SEG_LGTH) of orchestra(x) at db position
        """
        if self._nmat_dict_x[db] is not None:
            return

        nmat = self.note_mats_seg_at_db_x(db)
        self.reset_db_to_zeros(nmat, db)

        nmat = self.format_reset_seg_mats(nmat)
        self._nmat_dict_x[db] = nmat

    def store_prmat_seg_x(self, db):
        """
        Get piano roll format (SEG_LGTH) from note matrices at db position
        """
        if self._pr_mat_dict_x[db] is not None:
            return

        prmat = nmat_to_prmat2c(
            self._nmat_dict_x[db], SEG_LGTH_BIN, use_track=self.use_track
        )
        self._pr_mat_dict_x[db] = prmat

    def store_pnotree_seg_x(self, db):
        """
        Get pnotree representation (SEG_LGTH) from nmat
        """
        if self._pnotree_dict_x[db] is not None:
            return

        self._pnotree_dict_x[db] = nmat_to_pianotree_repr(self._nmat_dict_x[db])

    def _store_seg(self, db):
        self.store_nmat_seg_x(db)
        self.store_prmat_seg_x(db)

    def _get_item_by_db(self, db):
        """
        Return segments of
            prmat_x, prmat_y
        """

        self._store_seg(db)

        seg_prmat_x = self._pr_mat_dict_x[db]
        return seg_prmat_x, None

    def __getitem__(self, idx):
        db = self.db_pos[idx]
        return self._get_item_by_db(db)

    def get_whole_song_data(self):
        """
        used when inference
        """
        prmat_x = []
        idx = 0
        i = 0
        while i < len(self):
            seg_prmat_x, _ = self[i]
            prmat_x.append(seg_prmat_x)

            idx += SEG_LGTH_BIN
            while i < len(self) and self.db_pos[i] < idx:
                i += 1
        prmat_x = torch.from_numpy(np.array(prmat_x, dtype=np.float32))
        return prmat_x, prmat_x


class PianoOrchDataset(Dataset):
    def __init__(self, data_samples, debug=False):
        super(PianoOrchDataset, self).__init__()

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
        data_samples = [DataSampleNpz(song_path) for song_path in song_paths]
        return cls(data_samples, debug)

    @classmethod
    def load_train_and_valid_sets(cls, debug=False):
        split = read_dict(os.path.join(TRAIN_SPLIT_DIR, "pop909.pickle"))
        return cls.load_with_song_paths(split[0], debug), cls.load_with_song_paths(
            split[1], debug
        )

    @classmethod
    def load_with_train_valid_paths(cls, tv_song_paths, **kwargs):
        return cls.load_with_song_paths(tv_song_paths[0],
                                        **kwargs), cls.load_with_song_paths(
                                            tv_song_paths[1], **kwargs
                                        )


if __name__ == "__main__":
    test = "660.npz"
    song = DataSampleNpz(test)
    os.system(f"cp {POP909_DATA_DIR}/{test[:-4]}_flated.mid exp/copy_x.mid")
    prmat_x, _ = song.get_whole_song_data()
    print(prmat_x.shape)
    prmat_x = prmat_x.cpu().numpy()
    prmat2c_to_midi_file(prmat_x, "exp/origin_x.mid")
