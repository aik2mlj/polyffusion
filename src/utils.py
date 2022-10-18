from dirs import *
import numpy as np
import pickle
import os
import pretty_midi as pm
import torch
from dl_modules import PianoTreeEncoder, PianoTreeDecoder
from collections import OrderedDict
from torch.distributions import Normal, kl_divergence
import matplotlib.pyplot as plt


def load_pretrained_pnotree_enc_dec(fpath, max_simu_note, device):
    pnotree_enc = PianoTreeEncoder(device=device, max_simu_note=max_simu_note)
    pnotree_dec = PianoTreeDecoder(device=device, max_simu_note=max_simu_note)
    checkpoint = torch.load(fpath)
    enc_checkpoint = OrderedDict()
    dec_checkpoint = OrderedDict()
    enc_param_list = [
        "note_embedding", "enc_notes_gru", "enc_time_gru", "linear_mu", "linear_std"
    ]
    for k, v in checkpoint.items():
        part = k.split('.')[0]
        # print(part)
        # name = '.'.join(k.split('.')[1 :])
        # print(part, name)
        if part in enc_param_list:
            enc_checkpoint[k] = v
            if part == "note_embedding":
                dec_checkpoint[k] = v
        else:
            dec_checkpoint[k] = v
    pnotree_enc.load_state_dict(enc_checkpoint)
    pnotree_dec.load_state_dict(dec_checkpoint)
    pnotree_enc.to(device)
    pnotree_dec.to(device)
    return pnotree_enc, pnotree_dec


def output_to_numpy(recon_pitch, recon_dur):
    est_pitch = recon_pitch.max(-1)[1].unsqueeze(-1)  # (B, 32, 20, 1)
    est_dur = recon_dur.max(-1)[1]  # (B, 32, 11, 5)
    est_x = torch.cat([est_pitch, est_dur], dim=-1)  # (B, 32, 20, 6)
    est_x = est_x.cpu().numpy()
    recon_pitch = recon_pitch.cpu().numpy()
    recon_dur = recon_dur.cpu().numpy()
    return est_x, recon_pitch, recon_dur


def save_dict(path, dict_file):
    with open(path, "wb") as handle:
        pickle.dump(dict_file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_dict(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def nested_map(struct, map_fn):
    """This is for trasfering into cuda device"""
    if isinstance(struct, tuple):
        return tuple(nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)


def standard_normal(shape):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    if torch.cuda.is_available():
        N.loc = N.loc.cuda()
        N.scale = N.scale.cuda()
    return N


def kl_with_normal(dist):
    shape = dist.mean.size(-1)
    normal = standard_normal(shape)
    kl = kl_divergence(dist, normal).mean()
    return kl


def nmat_to_pianotree_repr(
    nmat,
    n_step=32,
    max_note_count=20,
    dur_pad_ind=2,
    min_pitch=0,
    pitch_sos_ind=128,
    pitch_eos_ind=129,
    pitch_pad_ind=130,
):
    """
    Convert the input note matrix to pianotree representation.
    Input: (N, 3), 3 for onset, pitch, duration. o and d are in time steps.
    """

    pnotree = np.ones((n_step, max_note_count, 6), dtype=np.int64) * dur_pad_ind
    pnotree[:, :, 0] = pitch_pad_ind
    pnotree[:, 0, 0] = pitch_sos_ind

    cur_idx = np.ones(n_step, dtype=np.int64)
    for o, p, d in nmat:
        if o >= n_step:
            continue
        pnotree[o, cur_idx[o], 0] = p - min_pitch

        # e.g., d = 4 -> bin_str = '00011'
        d = min(d, 32)
        bin_str = np.binary_repr(int(d) - 1, width=5)
        pnotree[o, cur_idx[o],
                1 :] = np.fromstring(" ".join(list(bin_str)), dtype=np.int64, sep=" ")

        # FIXME: when more than `max_note_count` notes are played in one step
        if cur_idx[o] < max_note_count - 1:
            cur_idx[o] += 1
        else:
            print(f"more than max_note_count {max_note_count} occur!")

    pnotree[np.arange(0, n_step), cur_idx, 0] = pitch_eos_ind
    return pnotree


def pianotree_pitch_shift(pnotree, shift):
    pnotree = pnotree.copy()
    pnotree[pnotree[:, :, 0] < 128, 0] += shift
    return pnotree


def pr_mat_pitch_shift(pr_mat, shift):
    pr_mat = pr_mat.copy()
    pr_mat = np.roll(pr_mat, shift, -1)
    return pr_mat


def chd_pitch_shift(chd, shift):
    chd = chd.copy()
    chd[:, 0] = (chd[:, 0] + shift) % 12
    chd[:, 1 : 13] = np.roll(chd[:, 1 : 13], shift, axis=-1)
    chd[:, -1] = (chd[:, -1] + shift) % 12
    return chd


def chd_to_onehot(chd):
    n_step = chd.shape[0]
    onehot_chd = np.zeros((n_step, 36), dtype=np.float32)
    onehot_chd[np.arange(n_step), chd[:, 0]] = 1
    onehot_chd[:, 12 : 24] = chd[:, 1 : 13]
    onehot_chd[np.arange(n_step), 24 + chd[:, -1]] = 1
    return onehot_chd


def onehot_to_chd(onehot):
    n_step = onehot.shape[0]
    chd = np.zeros((n_step, 14), dtype=np.float32)
    chd[:, 0] = np.argmax(onehot[:, 0 : 12], axis=1)
    chd[:, 1 : 13] = onehot[:, 12 : 24]
    chd[:, 13] = np.argmax(onehot[:, 24 : 36], axis=1)
    return chd


def nmat_to_pr_mat_repr(nmat, n_step=32):
    pr_mat = np.zeros((n_step, 128), dtype=np.int64)
    for o, p, d in nmat:
        if o < n_step:
            pr_mat[o, p] = d
    return pr_mat


def nmat_to_prmat2c(nmat, n_step=32, use_track=None):
    pr_mat = np.zeros((2, n_step, 128), dtype=np.float32)
    if use_track:
        for track_idx in use_track:
            for o, p, d in nmat[track_idx]:
                if o < n_step:
                    pr_mat[0, o, p] = 1.
                    for dd in range(1, d):
                        if o + dd < n_step:
                            pr_mat[1, o + dd, p] = 1.
    else:
        for o, p, d in nmat:
            if o < n_step:
                pr_mat[0, o, p] = 1.
                for dd in range(1, d):
                    if o + dd < n_step:
                        pr_mat[1, o + dd, p] = 1.
    return pr_mat


def compute_prmat2c_density(prmat2c):
    # only consider onset
    onset = prmat2c[0]
    onset = np.round(onset).astype(np.int8)
    count = np.count_nonzero(onset)
    size = onset.size
    ratio = count / size
    if ratio < 0.004:
        return 0
    elif ratio < 0.008:
        return 1
    elif ratio < 0.012:
        return 2
    else:
        return 3


def normalize_prmat(prmat):
    n_step = prmat.shape[1]
    prmat_norm = prmat.astype(np.float32)
    prmat_norm /= n_step
    return prmat_norm


def denormalize_prmat(prmat_norm):
    n_step = prmat_norm.shape[1]
    prmat_rint = np.rint(prmat_norm * n_step)
    prmat = prmat_rint.astype(np.int64)
    return prmat


def nmat_to_rhy_array(nmat, n_step=32):
    """Compute onset track of from melody note matrix."""
    pr_mat = np.zeros(n_step, dtype=np.int64)
    for o, _, _ in nmat:
        pr_mat[o] = 1
    return pr_mat


def estx_to_midi_file(est_x, fpath, labels=None):
    # print(f"est_x with shape {est_x.shape} to midi file {fpath}")  # (#, 32, 15, 6)
    # pr_mat3d is a (32, max_note_count, 6) matrix. In the last dim,
    # the 0th column is for pitch, 1: 6 is for duration in binary repr. Output is
    # padded with <sos> and <eos> tokens in the pitch column, but with pad token
    # for dur columns.
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    t = 0
    for two_bar_ind, two_bars in enumerate(est_x):
        for step_ind, step in enumerate(two_bars):
            for kth_key in step:
                assert len(kth_key) == 6
                if not (kth_key[0] >= 0 and kth_key[0] <= 127):
                    # rest key
                    # print(f"({two_bar_ind}, {step_ind}, somekey, 0): {kth_key[0]}")
                    continue

                # print(f"({two_bar_ind}, {step_ind}, somekey, 0): {kth_key[0]}")
                dur = (
                    kth_key[5] + (kth_key[4] << 1) + (kth_key[3] << 2) +
                    (kth_key[2] << 3) + (kth_key[1] << 4) + 1
                )
                note = pm.Note(
                    velocity=80,
                    pitch=int(kth_key[0]),
                    start=t + step_ind * 1 / 8,
                    end=min(t + (step_ind + int(dur)) * 1 / 8, t + 4),
                )
                piano.notes.append(note)
        t += 4
    midi.instruments.append(piano)

    if labels is not None:
        midi.lyrics.clear()
        t = 0
        for label in labels:
            midi.lyrics.append(pm.Lyric(label, t))
            t += 4

    midi.write(fpath)


def prmat_to_midi_file(prmat: np.ndarray, fpath, labels=None):
    # prmat: (B, 32, 128)
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    t = 0
    n_step = prmat.shape[1]
    t_bar = int(n_step / 8)
    for bar_ind, bars in enumerate(prmat):
        for step_ind, step in enumerate(bars):
            for key, dur in enumerate(step):
                dur = int(dur)
                if dur > 0:
                    note = pm.Note(
                        velocity=80,
                        pitch=key,
                        start=t + step_ind * 1 / 8,
                        end=min(t + (step_ind + int(dur)) * 1 / 8, t + t_bar)
                    )
                    piano.notes.append(note)
        t += t_bar
    midi.instruments.append(piano)
    if labels is not None:
        midi.lyrics.clear()
        t = 0
        for label in labels:
            midi.lyrics.append(pm.Lyric(label, t))
            t += t_bar
    midi.write(fpath)


def custom_round(x):
    if x > 0.95 and x < 1.05:
        return 1
    else:
        return 0


def prmat2c_to_midi_file(prmat: np.ndarray, fpath, labels=None, is_custom_round=True):
    # prmat2c: (B, 2, 32, 128)
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    t = 0
    n_step = prmat.shape[2]
    t_bar = int(n_step / 8)
    for bar_ind, bars in enumerate(prmat):
        onset = bars[0]
        sustain = bars[1]
        for step_ind, step in enumerate(onset):
            for key, on in enumerate(step):
                if is_custom_round:
                    on = int(custom_round(on))
                else:
                    on = int(round(on))
                if on > 0:
                    dur = 1
                    while step_ind + dur < n_step:
                        if not (int(round(sustain[step_ind + dur, key])) > 0):
                            break
                        dur += 1
                    note = pm.Note(
                        velocity=80,
                        pitch=key,
                        start=t + step_ind * 1 / 8,
                        end=min(t + (step_ind + dur) * 1 / 8, t + t_bar)
                    )
                    piano.notes.append(note)
        t += t_bar
    midi.instruments.append(piano)
    if labels is not None:
        midi.lyrics.clear()
        t = 0
        for label in labels:
            midi.lyrics.append(pm.Lyric(label, t))
            t += t_bar
    midi.write(fpath)


def chd_to_midi_file(chords, output_fpath, one_beat=0.5):
    """
    retrieve midi from chords
    """
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    t = 0.
    for seg in chords:
        for beat, chord in enumerate(seg):
            root = int(chord[0])
            chroma = chord[1 : 13].astype(int)
            bass = int(chord[13])

            chroma = np.roll(chroma, -bass)
            c3 = 48
            for i, n in enumerate(chroma):
                if n == 1:
                    note = pm.Note(
                        velocity=80,
                        pitch=c3 + i + bass,
                        start=t * one_beat,
                        end=(t + 1) * one_beat
                    )
                    piano.notes.append(note)
            t += 1

    midi.instruments.append(piano)
    midi.write(output_fpath)


def show_image(img: torch.Tensor, title=""):
    """Helper function to display an image"""
    # (B, 2, 32, 128)
    img = img.clip(0, 1)
    img = img.cpu().numpy()
    if img.ndim == 4:
        img = np.swapaxes(img, 1, 2)
        img = np.concatenate(img, axis=0)
        img = np.swapaxes(img, 0, 1)
    print(img.shape)
    h = img.shape[1]
    w = img.shape[2]
    img = np.append(img, np.zeros([1, h, w]), axis=0)
    img = img.transpose(2, 1, 0)  # (128, 32, 3)
    plt.imsave(title, img)


if __name__ == "__main__":
    load_pretrained_pnotree_enc_dec(
        "../PianoTree-VAE/model20/train_20-last-model.pt", 20, None
    )
