from data.dataloader import get_train_val_dataloaders
from utils import prmat_to_midi_file

if __name__ == "__main__":
    # val_dataset = PianoOrchDataset.load_valid_set(use_track=[1, 2])
    # val_dl = get_val_dataloader(1000, use_track=[0, 1, 2])
    train_dl, val_dl = get_train_val_dataloaders()
    print(len(val_dl))
    for i, batch in enumerate(val_dl):
        prmat2c, pnotree, chord, prmat = batch
        prmat_to_midi_file(prmat, "exp/ref_wm.mid")
        break

    # dir = "data/POP909_MIDIs"
    # os.makedirs(dir, exist_ok=True)

    # for i in range(1, 910):
    #     fpath = os.path.join(POP909_DATA_DIR, f"{i:03}.npz")
    #     print(fpath)
    #     if not os.path.exists(fpath):
    #         continue
    #     data = np.load(fpath, allow_pickle=True)
    #     notes = data["notes"]
    #     midi = pm.PrettyMIDI()
    #     piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    #     piano = pm.Instrument(program=piano_program)
    #     one_beat = 0.125
    #     for track in notes:
    #         for note in track:
    #             onset, pitch, duration, velocity, program = note
    #             note = pm.Note(
    #                 velocity=velocity,
    #                 pitch=pitch,
    #                 start=onset * one_beat,
    #                 end=(onset + duration) * one_beat
    #             )
    #             piano.notes.append(note)

    #     midi.instruments.append(piano)
    #     midi.write(f"{dir}/{i:03}.mid")
