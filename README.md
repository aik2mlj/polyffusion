# Polyffusion: A Diffusion Model for Polyphonic Score Generation with Internal and External Controls

## Installation

```shell
pip install -r requirements.txt
pip install -e diffpro/chord_extractor
pip install -e diffpro
```

## Data
The extracted features of the dataset POP909 can be accessed [here](https://yukisaki-my.sharepoint.com/:u:/g/personal/aik2_yukisaki_io/EdUovlRZvExJrGatAR8BlTsBDC8udJiuhnIimPuD2PQ3FQ?e=WwD7Dl).

Please put it under `/data/` after extraction.

## Pre-trained External Models 
The needed pre-trained models for training can be accessed [here](https://yukisaki-my.sharepoint.com/:u:/g/personal/aik2_yukisaki_io/Eca406YwV1tMgwHdoepC7G8B5l-4GRBGv7TzrI9OOg3eIA?e=uecJdU).

Please put them under `/pretrained/` after extraction.

## Some Clarifications

- The abbreviation "sdf" means Stable Diffusion, and "ldm" means Latent Diffusion. Basically they are referring to the same thing.
- `prmat2c` in the code is the piano-roll image representation.

## Training

```shell
python diffpro/main.py --model [model] --output_dir [output_dir]
```

The models that can be selected (which make sense):
- `ldm_chd8bar`: conditioned on latent chord representations encoded by a pre-trained chord encoder.
- `ldm_txt`: conditioned on latent texture representations encoded by a pre-trained texture encoder.
- `ldm_chdvnl`: conditioned on vanilla chord representations.
- `ldm_txtvnl`: conditioned on vanilla texture representations.
- `ddpm`: vanilla diffusion model from DDPM without conditioning.

Examples:
```shell
python diffpro/main.py --model ldm_chd8bar --output_dir result/ldm_chd8bar
```

## Inference

Please see the helping messages by running
```shell
python diffpro/inference_sdf.py --help
```

Examples:
```shell
# unconditional generation of length 10x8 bars
python diffpro/inference_sdf.py --model_dir=result/ldm_chd8bar --uncond_scale=0. --length=10

# conditional generation with guidance scale = 5, conditional chord progressions chosen from a song from POP909 validation set.
python diffpro/inference_sdf.py --model_dir=result/ldm_chd8bar --uncond_scale=5.

# conditional iterative inpainting (i.e. autoregressive generation) (default guidance scale = 1)
python diffpro/inference_sdf.py --model_dir=result/ldm_chd8bar --autoreg

# unconditional melody generation given accompaniment
python diffpro/inference_sdf.py --model_dir=result/ldm_chd8bar --uncond_scale=0. --inpaint_from_midi=/path/to/accompaniment.mid --inpaint_type=above

# accompaniment generation given melody, conditioned on chord progressions of another midi file (default guidance scale = 1)
python diffpro/inference_sdf.py --model_dir=result/ldm_chd8bar --inpaint_from_midi=/path/to/melody.mid --inpaint_type=below --from_midi=/path/to/cond_midi.mid
```
