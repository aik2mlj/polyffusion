import os
from os.path import join

import numpy as np
import pretty_midi as pm
from tqdm import tqdm

ORIGIN_DIR = "data/POP09-PIANOROLL-4-bin-quantization"
NEW_DIR = "data/POP909_4_bin_pnt_8bar"

ONE_BEAT_TIME = 0.5
SEG_LGTH = 32
BEAT = 4
BIN = 4
SEG_LGTH_BIN = SEG_LGTH * BIN


def get_note_matrix(mats):
    """
    (onset_beat, onset_bin, bin, offset_beat, offset_bin, bin, pitch, velocity)
    """
    notes = []

    for mat in mats:
        assert mat[2] == mat[5] == BIN
        onset = mat[0] * BIN + mat[1]
        offset = mat[3] * BIN + mat[4]
        duration = offset - onset
        if duration > 0:
            # this is compulsory because there may be notes
            # with zero duration after adjusting resolution
            notes.append([onset, mat[6], duration, mat[7], 0])
    # sort according to (start, duration)
    # notes.sort(key=lambda x: (x[0] * BIN + x[1], x[2]))
    notes.sort(key=lambda x: (x[0], x[1], x[2]))
    return notes


def dedup_note_matrix(notes):
    """
    remove duplicated notes (because of multiple tracks)
    """

    last = []
    notes_dedup = []
    for i, note in enumerate(notes):
        if i != 0:
            if note[:2] != last[:2]:
                # if start and pitch are not the same
                notes_dedup.append(note)
        else:
            notes_dedup.append(note)
        last = note
    # print(f"dedup: {len(notes) - len(notes_dedup)} : {len(notes)}")

    return notes_dedup


def retrieve_midi_from_nmat(notes, output_fpath):
    """
    retrieve midi from note matrix
    """
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    for note in notes:
        onset, pitch, duration, velocity, program = note
        start = onset * ONE_BEAT_TIME / float(BIN)
        end = start + duration * ONE_BEAT_TIME / float(BIN)
        pm_note = pm.Note(velocity, pitch, start, end)
        piano.notes.append(pm_note)

    midi.instruments.append(piano)
    midi.write(output_fpath)


def get_downbeat_pos_and_filter(notes, beats):
    """
    beats: [0-1, 2/3beat-cnt, 2, 0-3, 4/6beat-cnt, 4]
    """
    end_time = notes[-1][0]
    db_pos = []
    for i, beat in enumerate(beats):
        if beat[3] == 0:
            pos = i * BIN
            db_pos.append(pos)

    # print(db_pos)
    db_pos_filter = []
    for idx, db in enumerate(db_pos):
        if (
            idx + (SEG_LGTH / BEAT) <= len(db_pos)
            and db_pos[idx + 1] - db == BEAT * BIN
        ):
            db_pos_filter.append(True)
        else:
            db_pos_filter.append(False)
    # print(db_pos_filter)
    return db_pos, db_pos_filter


def get_start_table(notes, db_pos):
    """
    i-th row indicates the starting row of the "notes" array at i-th beat.
    """

    # simply add 8-beat padding in case of out-of-range index
    # total_beat = int(music.get_end_time()) + 8
    row_cnt = 0
    start_table = {}
    # for beat in range(total_beat):
    for db in db_pos:
        while row_cnt < len(notes) and notes[row_cnt][0] < db:
            row_cnt += 1
        start_table[db] = row_cnt

    return start_table


def cat_note_mats(note_mats):
    return np.concatenate(note_mats, 0)


if __name__ == "__main__":
    if os.path.exists(NEW_DIR):
        os.system(f"rm -rf {NEW_DIR}")
    os.makedirs(NEW_DIR)

    for piece in tqdm(os.listdir(ORIGIN_DIR)):
        fpath = os.path.join(ORIGIN_DIR, piece)
        f = np.load(fpath)
        melody = get_note_matrix(f["melody"])
        bridge = get_note_matrix(f["bridge"])
        piano = get_note_matrix(f["piano"])
        beats = f["beat"]
        notes = cat_note_mats([melody, bridge, piano])

        retrieve_midi_from_nmat(
            notes, os.path.join(NEW_DIR, piece[:-4] + "_flatten.mid")
        )
        db_pos, db_pos_filter = get_downbeat_pos_and_filter(notes, beats)
        start_table_melody = get_start_table(melody, db_pos)
        start_table_bridge = get_start_table(bridge, db_pos)
        start_table_piano = get_start_table(piano, db_pos)
        np.savez(
            join(NEW_DIR, piece[:-4]),
            notes=[melody, bridge, piano],
            start_table=[start_table_melody, start_table_bridge, start_table_piano],
            db_pos=db_pos,
            db_pos_filter=db_pos_filter,
            chord=f["chord"],
        )
