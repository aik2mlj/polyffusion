import os

# dataset paths
DATA_DIR = "data/LOP_4_bin_pnt"
TRAIN_SPLIT_DIR = "data/train_split_pnt"

MUSICALION_DATA_DIR = "data/musicalion_solo_piano_4_bin_pnt"
POP909_DATA_DIR = "data/POP909_4_bin_pnt_8bar"

# pretrained path
PT_PNOTREE_PATH = "pretrained/pnotree_20/train_20-last-model.pt"

PT_POLYDIS_PATH = "pretrained/polydis/model_master_final.pt"
PT_A2S_PATH = "pretrained/a2s/a2s-stage3a.pt"

# pretrained chd_8bar
PT_CHD_8BAR_PATH = "pretrained/chd8bar/weights.pt"

# the path to store demo.
DEMO_FOLDER = "./demo"

# the path to save trained model params and tensorboard log.
RESULT_PATH = "./result"

if not os.path.exists(DEMO_FOLDER):
    os.mkdir(DEMO_FOLDER)

if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)
