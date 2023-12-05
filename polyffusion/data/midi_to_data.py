import csv
import sys

import mir_eval
import muspy
import numpy as np
import pretty_midi as pm

from chord_extractor import extract_chords_from_midi_file

# duration of one beat
ONE_BEAT = 0.5
SEG_LGTH = 32
BEAT = 4
BIN = 4
SEG_LGTH_BIN = SEG_LGTH * BIN


def get_note_matrix(music: muspy.Music):
    """
    get note matrix: same format as pop909-4-bin
        for piano, this function simply extracts notes
        for orchestra, this function "flattens" the notes into one single track
    plus an instrument program num appended at the end
    """

    notes = []
    for inst in music.tracks:
        for note in inst.notes:
            onset = int(note.time)
            duration = int(note.duration)
            if duration > 0:
                # this is compulsory because there may be notes
                # with zero duration after adjusting resolution
                notes.append(
                    [
                        onset,
                        note.pitch,
                        duration,
                        note.velocity,
                        inst.program,
                    ]
                )
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
        start = onset * ONE_BEAT / float(BIN)
        end = start + duration * ONE_BEAT / float(BIN)
        pm_note = pm.Note(velocity, pitch, start, end)
        piano.notes.append(pm_note)

    midi.instruments.append(piano)
    midi.write(output_fpath)


def get_chord_matrix(fpath):
    """
    chord matrix [M * 14], each line represent the chord of a beat
    same format as mir_eval.chord.encode():
        root_number(1), semitone_bitmap(12), bass_number(1)
    inputs are generated from junyan's algorithm
    """
    file = csv.reader(open(fpath), delimiter="\t")
    beat_cnt = 0
    chords = []
    for line in file:
        start = float(line[0]) / ONE_BEAT
        end = float(line[1]) / ONE_BEAT
        chord = line[2]

        while beat_cnt < int(round(end)):
            beat_cnt += 1
            # see https://craffel.github.io/mir_eval/#mir_eval.chord.encode
            chd_enc = mir_eval.chord.encode(chord)

            root = chd_enc[0]
            # make chroma and bass absolute
            chroma_bitmap = chd_enc[1]
            chroma_bitmap = np.roll(chroma_bitmap, root)
            bass = (chd_enc[2] + root) % 12

            chord_line = [root]
            for _ in chroma_bitmap:
                chord_line.append(_)
            chord_line.append(bass)

            chords.append(chord_line)
    return chords


def retrieve_midi_from_chd(chords, output_fpath):
    """
    retrieve midi from chords
    """
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    for beat, chord in enumerate(chords):
        root = chord[0]
        chroma = chord[1:13]
        bass = chord[13]

        chroma = np.roll(chroma, -bass)
        c3 = 48
        for i, n in enumerate(chroma):
            if n == 1:
                note = pm.Note(
                    velocity=80,
                    pitch=c3 + i + bass,
                    start=beat * ONE_BEAT,
                    end=(beat + 1) * ONE_BEAT,
                )
                piano.notes.append(note)

    midi.instruments.append(piano)
    midi.write(output_fpath)


def get_downbeat_pos_and_filter(music: muspy.Music, debug_info):
    """
    simply get the downbeat position of the given midi file
    and whether each downbeat is complete
    "complete" means at least one 4/4 measures after it.
    E.g.,
    [1, 2, 3, 4, 1, 2, 3, 4] is complete.
    [1, 2, 3, 1, 2, 3, 4], [1, 2, 3, 4, 1, 2, 3] are not.
    """
    music.infer_barlines_and_beats()
    barlines = music.barlines
    for b in barlines:
        if not float(b.time).is_integer():
            # print("======")
            # print("downbeat not integer!")
            # print(debug_info)
            # BAD_SONGS.add(debug_info)
            return None, None

    db_pos = [int(b.time) for b in barlines]
    # end_pos = int(music.get_end_time() / ONE_BEAT)
    db_pos_diff = np.diff(db_pos).tolist()
    db_pos_diff.append(db_pos_diff[len(db_pos_diff) - 1])
    assert len(db_pos_diff) == len(db_pos)
    db_pos_filter = []
    for i in range(len(db_pos)):
        if db_pos_diff[i] not in {2 * BIN, 4 * BIN, 8 * BIN}:
            db_pos_filter.append(False)
            continue
        length = db_pos_diff[i]
        left = 8 * BIN - length
        idx = i + 1
        bad = False
        while left > 0 and idx < len(db_pos):
            if db_pos_diff[idx] != length:
                bad = True
                break
            left -= length
            idx += 1
        if bad:
            db_pos_filter.append(False)
        else:
            db_pos_filter.append(True)

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


BAD_SONGS = set()


def get_data_for_single_midi(fpath, chdfile_path):
    music = muspy.read_midi(fpath)
    music.adjust_resolution(BIN)
    if len(music.time_signatures) == 0:
        music.time_signatures.append(muspy.TimeSignature(0, 4, 4))

    note_mat = get_note_matrix(music)
    note_mat = dedup_note_matrix(note_mat)
    extract_chords_from_midi_file(fpath, chdfile_path)
    chord = get_chord_matrix(chdfile_path)

    db_pos, db_pos_filter = get_downbeat_pos_and_filter(music, fpath)
    if db_pos is not None:
        start_table = get_start_table(note_mat, db_pos)
        return {
            "notes": np.array(note_mat),
            "start_table": np.array(start_table),
            "db_pos": np.array(db_pos),
            "db_pos_filter": np.array(db_pos_filter),
            "chord": np.array(chord),
        }
    else:
        print("get downbeat error!")


if __name__ == "__main__":
    print(get_data_for_single_midi(sys.argv[1], sys.argv[2]))
