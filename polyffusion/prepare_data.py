import os
import muspy
from tqdm import tqdm
import numpy as np
from data.midi_to_data import *
from data.dataset import PianoOrchDataset
from argparse import ArgumentParser

def force_length(music : muspy.music, bars = 8):
    for track in music.tracks:
        timesteps = track.get_end_time()
        old_bars = (timesteps + 15) // 16
        div = bars // old_bars
        for i in range(1, div):
            tmp = track.deepcopy()
            tmp.adjust_time(lambda x : x + i * timesteps)
            track.notes.extend(tmp.notes)

def prepare_npz(midi_dir, chords_dir, output_dir, force=False):
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
                note_mat = get_note_matrix(music)
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
            if db_pos is not None and sum(filter(lambda x : x, db_pos_filter)) != 0:
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
                chord=processed_data["chord"]
                )
                success += 1
            else:
                downbeat_errors += 1

    print(f"""{ttl} tracks processed, {success} succeeded, {chords_errors} chords errors, {downbeat_errors} downbeat errors""")



if __name__ == "__main__":
    parser = ArgumentParser(
        description='prepare data from midi for a Polyffusion model'
    )
    parser.add_argument(
        "--midi_dir",
        type=str,
        help='directory of input midis to preparep'
    )
    parser.add_argument(
        "--chords_dir",
        type=str,
        help='directory to store extracted chords'
    )
    parser.add_argument(
        "--npz_dir",
        type=str,
        help='directory to store prepared data in npz'
    )
    parser.add_argument(
        "--force_length",
        action="store_true",
        help="whether to repeat shorter samples into the desired number of bars"
    )

    args = parser.parse_args()
    prepare_npz(args.midi_dir, args.chords_dir, args.npz_dir, args.force_length)