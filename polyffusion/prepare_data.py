import os
from argparse import ArgumentParser

import muspy
import numpy as np
from tqdm import tqdm

from data.midi_to_data import *


def force_length(music: muspy.music, bars=8):
    """Loops a MIDI file if it's under the specified number of bars, in place."""
    num_tracks_at_least_bars = sum(
        [
            1 if (track.get_end_time() + 15) // 16 >= bars else 0
            for track in music.tracks
        ]
    )
    if num_tracks_at_least_bars > 0:
        return
    for track in music.tracks:
        timesteps = track.get_end_time()
        old_bars = (timesteps + 15) // 16
        div = bars // old_bars
        for i in range(1, div):
            tmp = track.deepcopy()
            tmp.adjust_time(lambda x: x + i * timesteps)
            track.notes.extend(tmp.notes)


def get_note_matrix_melodies(music, ignore_non_melody=True):
    """Similar to get_note_matrix from data.midi_to_data, with an option to ignore non-melodies."""
    notes = []
    for inst in music.tracks:
        if ignore_non_melody and (inst.is_drum or inst.program >= 113):
            continue
        for note in inst.notes:
            onset = int(note.time)
            duration = int(note.duration)
            if duration > 0:
                notes.append(
                    [
                        onset,
                        note.pitch,
                        duration,
                        note.velocity,
                        inst.program,
                    ]
                )
    notes.sort(key=lambda x: (x[0], x[1], x[2]))
    assert len(notes)  # in case if a MIDI has only non-melodies
    return notes


def prepare_npz(midi_dir, chords_dir, output_dir, force=False, ignore_non_melody=True):
    for dir in [chords_dir, output_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    ttl = 0
    success = 0
    downbeat_errors = 0
    chords_errors = 0
    for root, dirs, files in os.walk(midi_dir):
        for midi in tqdm(files, desc=f"Processing {root}"):
            ttl += 1
            fpath = os.path.join(root, midi)
            chdpath = os.path.join(chords_dir, os.path.splitext(midi)[0] + ".csv")
            music = muspy.read_midi(fpath)
            music.adjust_resolution(4)
            if len(music.time_signatures) == 0:
                music.time_signatures.append(muspy.TimeSignature(0, 4, 4))
            if force:
                force_length(music)

            try:
                note_mat = get_note_matrix_melodies(music, ignore_non_melody)
                note_mat = dedup_note_matrix(note_mat)
                extract_chords_from_midi_file(fpath, chdpath)
                chord = get_chord_matrix(chdpath)
            except:
                chords_errors += 1
                continue

            try:
                db_pos, db_pos_filter = get_downbeat_pos_and_filter(music, fpath)
            except:
                downbeat_errors += 1
                continue
            if db_pos is not None and sum(filter(lambda x: x, db_pos_filter)) != 0:
                start_table = get_start_table(note_mat, db_pos)
                processed_data = {
                    "notes": np.array(note_mat),
                    "start_table": np.array(start_table),
                    "db_pos": np.array(db_pos),
                    "db_pos_filter": np.array(db_pos_filter),
                    "chord": np.array(chord),
                }
                np.savez(
                    os.path.join(output_dir, midi),
                    notes=processed_data["notes"],
                    start_table=processed_data["start_table"],
                    db_pos=processed_data["db_pos"],
                    db_pos_filter=processed_data["db_pos_filter"],
                    chord=processed_data["chord"],
                )
                success += 1
            else:
                downbeat_errors += 1

    print(
        f"""{ttl} tracks processed, {success} succeeded, {chords_errors} chords errors, {downbeat_errors} downbeat errors"""
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="prepare data from midi for a Polyffusion model"
    )
    parser.add_argument(
        "--midi_dir", type=str, help="directory of input midis to preparep"
    )
    parser.add_argument(
        "--chords_dir", type=str, help="directory to store extracted chords"
    )
    parser.add_argument(
        "--npz_dir", type=str, help="directory to store prepared data in npz"
    )
    parser.add_argument(
        "--force_length",
        action="store_true",
        help="to repeat shorter samples into the desired number of bars",
    )
    parser.add_argument(
        "--ignore_non_melody",
        action="store_false",
        help="whether ignore all non-melody instruments. default: true",
    )
    args = parser.parse_args()
    prepare_npz(
        args.midi_dir,
        args.chords_dir,
        args.npz_dir,
        args.force_length,
        args.ignore_non_melody,
    )
