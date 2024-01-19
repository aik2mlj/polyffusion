# Polyffusion: A Diffusion Model for Polyphonic Score Generation with Internal and External Controls

- [Paper link](https://arxiv.org/abs/2307.10304)
- Check our [demo page](https://polyffusion.github.io/) and give it a listen!

```
@inproceedings{polyffusion2023,
    author = {Lejun Min and Junyan Jiang and Gus Xia and Jingwei Zhao},
    title = {Polyffusion: A Diffusion Model for Polyphonic Score Generation with Internal and External Controls},
    booktitle = {Proceedings of the 24th International Society for Music Information Retrieval Conference, {ISMIR}},
    year = {2023}
}
```

## Installation

```shell
pip install -r requirements.txt
pip install -e polyffusion
pip install -e polyffusion/chord_extractor
pip isntall -e polyffusion/mir_eval
```

## Some Clarifications

- The abbreviation "sdf" means Stable Diffusion, and "ldm" means Latent Diffusion. Basically they are referring to the same thing. However, we only borrow the cross-attention conditioning mechanism from Latent Diffusion, without utilizing its encoder and decoder. The latter is left for future experiments.
- `prmat2c` in the code is the piano-roll image representation.

## Training

### Preparations

- The extracted features of the dataset POP909 can be accessed [here](https://yukisaki-my.sharepoint.com/:u:/g/personal/aik2_yukisaki_io/EdUovlRZvExJrGatAR8BlTsBDC8udJiuhnIimPuD2PQ3FQ?e=WwD7Dl). Please put it under `/data/` after extraction.

- The needed pre-trained models for training can be accessed [here](https://yukisaki-my.sharepoint.com/:u:/g/personal/aik2_yukisaki_io/Eca406YwV1tMgwHdoepC7G8B5l-4GRBGv7TzrI9OOg3eIA?e=uecJdU). Please put them under `/pretrained/` after extraction.

### Modifications

- You can modify the parameters in the corresponding `*.yaml` files under `/polyffusion/params/`, or create your own.

### Commands

```shell
python polyffusion/main.py --model [model] --output_dir [output_dir]
```

The models can be selected from `/polyffusion/params/[model].yaml`. Here are some cases:

- `sdf_chd8bar`: conditioned on latent chord representations encoded by a pre-trained chord encoder.
- `sdf_txt`: conditioned on latent texture representations encoded by a pre-trained texture encoder.
- `sdf_chdvnl`: conditioned on vanilla chord representations.
- `sdf_txtvnl`: conditioned on vanilla texture representations.
- `ddpm`: vanilla diffusion model from DDPM without conditioning.

Examples:

```shell
python polyffusion/main.py --model sdf_chd8bar --output_dir result/sdf_chd8bar
```

## Trained Checkpoints

If you'd like to test our trained checkpoints, please access the folder [here](https://yukisaki-my.sharepoint.com/:f:/g/personal/aik2_yukisaki_io/EjG0IB8Xb_1CoVfYCmNUB-ABMLVSRqJST4VTrYJxjJFdnw?e=OqmZpp). We suggest to put them under `/result/` after extraction for inference.

## Inference

Please see the helping messages by running

```shell
python polyffusion/inference_sdf.py --help
```

Examples:

```shell
# unconditional generation of length 10x8 bars
python polyffusion/inference_sdf.py --chkpt_path=/path/to/checkpoint --uncond_scale=0. --length=10

# conditional generation using DDIM sampler (default guidance scale = 1)
python polyffusion/inference_sdf.py --chkpt_path=/path/to/checkpoint --ddim --ddim_steps=50 --ddim_eta=0.0 --ddim_discretize=uniform

# conditional generation with guidance scale = 5, conditional chord progressions chosen from a song from POP909 validation set.
python polyffusion/inference_sdf.py --chkpt_path=/path/to/checkpoint --uncond_scale=5.

# conditional iterative inpainting (i.e. autoregressive generation) (default guidance scale = 1)
python polyffusion/inference_sdf.py --chkpt_path=/path/to/checkpoint --autoreg

# unconditional melody generation given accompaniment
python polyffusion/inference_sdf.py --chkpt_path=/path/to/checkpoint --uncond_scale=0. --inpaint_from_midi=/path/to/accompaniment.mid --inpaint_type=above

# accompaniment generation given melody, conditioned on chord progressions of another midi file (default guidance scale = 1)
python polyffusion/inference_sdf.py --chkpt_path=/path/to/checkpoint --inpaint_from_midi=/path/to/melody.mid --inpaint_type=below --from_midi=/path/to/cond_midi.mid
```
