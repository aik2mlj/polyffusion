import json
import os
import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi as pm
import torch
import torchvision.transforms as T
from omegaconf import OmegaConf
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import interpolate

from dirs import *
from dl_modules import *


def load_pretrained_pnotree_enc_dec(fpath, max_simu_note):
    pnotree_enc = PianoTreeEncoder(max_simu_note=max_simu_note)
    pnotree_dec = PianoTreeDecoder(max_simu_note=max_simu_note)
    checkpoint = torch.load(fpath)
    enc_checkpoint = OrderedDict()
    dec_checkpoint = OrderedDict()
    enc_param_list = [
        "note_embedding",
        "enc_notes_gru",
        "enc_time_gru",
        "linear_mu",
        "linear_std",
    ]
    for k, v in checkpoint.items():
        part = k.split(".")[0]
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
    return pnotree_enc, pnotree_dec


def load_pretrained_chd_enc_dec(
    fpath, input_dim, z_input_dim, hidden_dim, z_dim, n_step
):
    chord_enc = ChordEncoder(input_dim, hidden_dim, z_dim)
    chord_dec = ChordDecoder(input_dim, z_input_dim, hidden_dim, z_dim, n_step)
    checkpoint = torch.load(fpath)
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    from collections import OrderedDict

    enc_chkpt = OrderedDict()
    dec_chkpt = OrderedDict()
    for k, v in checkpoint.items():
        part = k.split(".")[0]
        name = ".".join(k.split(".")[1:])
        if part == "chord_enc":
            enc_chkpt[name] = v
        elif part == "chord_dec":
            dec_chkpt[name] = v
    chord_enc.load_state_dict(enc_chkpt)
    chord_dec.load_state_dict(dec_chkpt)
    return chord_enc, chord_dec


def load_pretrained_txt_enc(fpath, emb_size, hidden_dim, z_dim, num_channel):
    txt_enc = TextureEncoder(emb_size, hidden_dim, z_dim, num_channel)
    checkpoint = torch.load(fpath)
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    from collections import OrderedDict

    enc_chkpt = OrderedDict()
    for k, v in checkpoint.items():
        part = k.split(".")[0]
        name = ".".join(k.split(".")[1:])
        if part == "rhy_encoder":
            enc_chkpt[name] = v
    txt_enc.load_state_dict(enc_chkpt)
    return txt_enc


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
        pnotree[o, cur_idx[o], 1:] = np.fromstring(
            " ".join(list(bin_str)), dtype=np.int64, sep=" "
        )

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
    chd[:, 1:13] = np.roll(chd[:, 1:13], shift, axis=-1)
    chd[:, -1] = (chd[:, -1] + shift) % 12
    return chd


def chd_to_onehot(chd):
    n_step = chd.shape[0]
    onehot_chd = np.zeros((n_step, 36), dtype=np.float32)
    onehot_chd[np.arange(n_step), chd[:, 0]] = 1
    onehot_chd[:, 12:24] = chd[:, 1:13]
    onehot_chd[np.arange(n_step), 24 + chd[:, -1]] = 1
    return onehot_chd


def onehot_to_chd(onehot):
    n_step = onehot.shape[0]
    chd = np.zeros((n_step, 14), dtype=np.float32)
    chd[:, 0] = np.argmax(onehot[:, 0:12], axis=1)
    chd[:, 1:13] = onehot[:, 12:24]
    chd[:, 13] = np.argmax(onehot[:, 24:36], axis=1)
    return chd


def nmat_to_prmat(nmat, n_step=32):
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
                    pr_mat[0, o, p] = 1.0
                    for dd in range(1, d):
                        if o + dd < n_step:
                            pr_mat[1, o + dd, p] = 1.0
    else:
        for o, p, d in nmat:
            if o < n_step:
                pr_mat[0, o, p] = 1.0
                for dd in range(1, d):
                    if o + dd < n_step:
                        pr_mat[1, o + dd, p] = 1.0
    return pr_mat


def prmat2c_to_prmat(prmat2c, n_step=32):
    """
    prmat2c: (N, 2, 32*ratio, 128)
    prmat: (N*ratio, 32, 128)
    """
    if "Tensor" in str(type(prmat2c)):
        prmat2c = prmat2c.cpu().detach().numpy()
    assert prmat2c.ndim == 4
    N = prmat2c.shape[0]
    prmat2c_step = prmat2c.shape[2]  # 128 (normal), 64 (autoregressive)
    ratio = prmat2c_step // n_step
    prmat = np.zeros((N * ratio, n_step, 128), dtype=np.int64)
    # t_bar = int(prmat2c_step / 8)
    # notes = []
    for bar_ind, bars in enumerate(prmat2c):
        onset = bars[0]
        sustain = bars[1]
        for step_ind, step in enumerate(onset):
            for key, on in enumerate(step):
                on = int(round(on))
                if on > 0:
                    dur = 1
                    while step_ind + dur < prmat2c_step:
                        if not (int(round(sustain[step_ind + dur, key])) > 0):
                            break
                        dur += 1
                    prmat[
                        bar_ind * ratio + step_ind // n_step, step_ind % n_step, key
                    ] = dur
    return prmat


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
    if "Tensor" in str(type(est_x)):
        est_x = est_x.cpu().detach().numpy()
    n_step = est_x.shape[1]  # 32, or 128
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
                    kth_key[5]
                    + (kth_key[4] << 1)
                    + (kth_key[3] << 2)
                    + (kth_key[2] << 3)
                    + (kth_key[1] << 4)
                    + 1
                )
                note = pm.Note(
                    velocity=80,
                    pitch=int(kth_key[0]),
                    start=t + step_ind * 1 / 8,
                    end=min(t + (step_ind + int(dur)) * 1 / 8, t + n_step / 8),
                )
                piano.notes.append(note)
        t += n_step / 8
    midi.instruments.append(piano)

    if labels is not None:
        midi.lyrics.clear()
        t = 0
        for label in labels:
            midi.lyrics.append(pm.Lyric(label, t))
            t += n_step / 8

    midi.write(fpath)


def prmat_to_midi_file(prmat, fpath, labels=None):
    # prmat: (B, 32, 128)
    if "Tensor" in str(type(prmat)):
        prmat = prmat.cpu().detach().numpy()
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
                        end=min(t + (step_ind + int(dur)) * 1 / 8, t + t_bar),
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


def check_prmat2c_integrity(prmat2c, is_custom_round=False):
    if "Tensor" in str(type(prmat2c)):
        prmat2c = prmat2c.cpu().detach().numpy()
    round_func = custom_round if is_custom_round else round
    err = 0
    total = 0
    for bar_ind, bars in enumerate(prmat2c):
        onset = bars[0]
        sustain = bars[1]
        for step_ind, step in enumerate(sustain):
            for key, sus in enumerate(step):
                sus = int(round_func(sus))
                if sus > 0 and (
                    step_ind == 0
                    or (
                        int(round_func(onset[step_ind - 1, key])) == 0
                        and int(round_func(sustain[step_ind - 1, key])) == 0
                    )
                ):
                    # no onset, only sustain
                    err += 1
                    total += 1
        for step_ind, step in enumerate(onset):
            for key, on in enumerate(step):
                on = int(round_func(on))
                if on > 0:
                    # a valid note with onset
                    total += 1
    return float(err / total)


def prmat2c_to_midi_file(
    prmat2c, fpath, labels=None, is_custom_round=False, inp_mask=None
):
    # prmat2c: (B, 2, 32, 128)
    if "Tensor" in str(type(prmat2c)):
        prmat2c = prmat2c.cpu().detach().numpy()
    print(f"prmat2c : {prmat2c.shape}")
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    origin = pm.Instrument(program=piano_program)
    inpainted = pm.Instrument(program=piano_program)
    t = 0
    n_step = prmat2c.shape[2]
    t_bar = int(n_step / 8)
    for bar_ind, bars in enumerate(prmat2c):
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
                        end=min(t + (step_ind + dur) * 1 / 8, t + t_bar),
                    )
                    if inp_mask is not None:
                        if inp_mask[bar_ind, 0, step_ind, key] == 0.0:
                            inpainted.notes.append(note)
                        else:
                            origin.notes.append(note)
                    else:
                        origin.notes.append(note)
        t += t_bar
    midi.instruments.append(origin)
    if inp_mask is not None:
        midi.instruments.append(inpainted)
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
    if "Tensor" in str(type(chords)):
        chords = chords.cpu().detach().numpy()
    midi = pm.PrettyMIDI()
    piano_program = pm.instrument_name_to_program("Acoustic Grand Piano")
    piano = pm.Instrument(program=piano_program)
    t = 0.0
    for seg in chords:
        for beat, chord in enumerate(seg):
            if chord.shape[0] == 14:
                root = int(chord[0])
                chroma = chord[1:13].astype(int)
                bass = int(chord[13])
            elif chord.shape[0] == 36:
                root = int(chord[0:12].argmax())
                chroma = chord[12:24].astype(int)
                bass = int(chord[24:].argmax())

            chroma = np.roll(chroma, -bass)
            c3 = 48
            for i, n in enumerate(chroma):
                if n == 1:
                    note = pm.Note(
                        velocity=80,
                        pitch=c3 + i + bass,
                        start=t * one_beat,
                        end=(t + 1) * one_beat,
                    )
                    piano.notes.append(note)
            t += 1

    midi.instruments.append(piano)
    midi.write(output_fpath)


def show_image(img: torch.Tensor, title="", mask=False):
    """Helper function to display an image"""
    # (B, 2, 32, 128)
    print(f"img: {img.shape}")
    img = img.clip(0, 1).clone()
    img = img.cpu().numpy()
    # if mask == True:
    #     img = 1 - img
    if img.ndim == 4:
        img = np.swapaxes(img, 1, 2)
        img = np.concatenate(img, axis=0)
        img = np.swapaxes(img, 0, 1)
    h = img.shape[1]
    w = img.shape[2]
    while img.shape[0] < 3:
        img = np.append(img, np.zeros([1, h, w]), axis=0)
    img = img.transpose(2, 1, 0)  # (128, 32, 3)
    img = np.flip(img, 0)  # flip the pitch axis: lower pitches at the bottom
    if mask == True:
        alpha = np.expand_dims(img[:, :, 0], axis=2)
        img = np.append(img, alpha, axis=2)
    img = np.ascontiguousarray(img)
    print(f"img: {img.shape}")
    plt.imsave(title, img)


def get_blurry_image(img: torch.Tensor, ratio=1 / 8):
    # print(img.shape)
    # show_image(img, "exp/img/original_img.png")
    # size = (img.shape[1] * ratio, img.shape[])

    blurry_img = interpolate(img, scale_factor=ratio, mode="bicubic")
    # show_image(blurry_img, "exp/img/blurry_img_sm.png")
    blurry_img = interpolate(blurry_img, scale_factor=1 / ratio, mode="nearest")
    blurry_img = blurry_img.clip(0, 1)
    # show_image(blurry_img, "exp/img/blurry_img.png")
    # blurry_img[:, 0, :, :] += blurry_img[:, 1, :, :]

    # show_image(blurry_img, "exp/img/blurry_img_.png")
    # print(blurry_img.shape)
    # return blurry_img[:, 0, :, :].unsqueeze(1)
    return blurry_img


def get_blurry_image_2(img):
    # import required libraries

    # read the input image
    # img = img

    # define the transform to blur image
    transform = T.GaussianBlur(kernel_size=(3, 5))

    # blur the input image using the above defined transform
    img = transform(img)
    img.save("exp/img/blurry.png")
    # transform = T.Compose([T.ToTensor()])
    # tensor = transform(img)
    # tensor
    # print(tensor.shape)

    # display the blurred image
    # plt.imsave("exp/img/blurry.png", tensor)


# def get_blurry_image_cv2(img):
#     img = cv2.imread(img)
#     img = cv2.pyrDown(img)
#     img = cv2.pyrDown(img)
#     img = cv2.pyrUp(img)
#     img = cv2.pyrUp(img)
#     cv2.imwrite("exp/img/blurry.png", img)


def convert_json_to_yaml(params_path):
    """Converts json to yaml, return yaml_path"""
    if params_path.endswith(".json"):
        print("Converting json to yaml...")
        with open(params_path, "r") as params_file:
            params = OmegaConf.create(json.load(params_file))
        OmegaConf.save(params, params_path[:-5] + ".yaml")

        if input("Delete old json file? (y/n)") == "y":
            os.remove(params_path)
        params_path = params_path[:-5] + ".yaml"
    return params_path


if __name__ == "__main__":
    # load_pretrained_pnotree_enc_dec(
    #     "../PianoTree-VAE/model20/train_20-last-model.pt", 20, None
    # )
    # img = Image.open("exp/img/original_img.png")
    # transform = T.Compose([T.ToTensor()])
    # tensor = transform(img)
    # tensor = tensor.transpose(1, 2)
    # tensor = tensor.flip(2)
    # get_blurry_image(tensor)
    get_blurry_image_cv2("exp/img/original_img.png")
