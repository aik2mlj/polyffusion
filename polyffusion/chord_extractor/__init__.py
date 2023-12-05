import csv
import sys

import mir_eval
import numpy as np

from .main import transcribe_cb1000_midi


def get_chord_from_chdfile(fpath, one_beat=0.5, rounding=True):
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
        start = float(line[0])
        end = float(line[1])
        chord = line[2]
        if not rounding:
            assert ((end - start) / one_beat).is_integer()
            beat_num = int((end - start) / one_beat)
        else:
            beat_num = round((end - start) / one_beat)
        for _ in range(beat_num):
            beat_cnt += 1
            # see https://craffel.github.io/mir_eval/#mir_eval.chord.encode
            chd_enc = mir_eval.chord.encode(chord)

            root = chd_enc[0]
            # make chroma and bass absolute
            chroma_bitmap = chd_enc[1]
            chroma_bitmap = np.roll(chroma_bitmap, root)
            bass = (chd_enc[2] + root) % 12

            line = [root]
            for _ in chroma_bitmap:
                line.append(_)
            line.append(bass)

            chords.append(line)
    return np.array(chords, dtype=np.float32)


def extract_chords_from_midi_file(fpath, chdfile_path):
    transcribe_cb1000_midi(fpath, chdfile_path)
    return get_chord_from_chdfile(chdfile_path)


if __name__ == "__main__":
    extract_chords_from_midi_file(sys.argv[1], sys.argv[2])
