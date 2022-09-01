import os

# dataset paths
DATA_DIR = "data/LOP_4_bin_pnt"
TRAIN_SPLIT_DIR = "data/train_split_pnt"

# pretrained path
PT_PNOTREE_PATH = "pretrained/pnotree_20/train_20-last-model.pt"

# the path to store demo.
DEMO_FOLDER = "./demo"

# the path to save trained model params and tensorboard log.
RESULT_PATH = "./result"

if not os.path.exists(DEMO_FOLDER):
    os.mkdir(DEMO_FOLDER)

if not os.path.exists(RESULT_PATH):
    os.mkdir(RESULT_PATH)
