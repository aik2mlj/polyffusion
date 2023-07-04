from polydis.model import DisentangleVAE
from polydis.ptvae import RnnEncoder, TextureEncoder, PtvaeEncoder, PtvaeDecoder, RnnDecoder
from utils import estx_to_midi_file
from datetime import datetime

import torch
import pretty_midi as pm
import sys

model_path = "pretrained/polydis/model_master_final.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# readme_fn = "./train.py"
batch_size = 128
length = 16  # 16 bars for inference
# n_epoch = 6
# clip = 1
# parallel = False
# weights = [1, 0.5]
# beta = 0.1
# tf_rates = [(0.6, 0), (0.5, 0), (0.5, 0)]
# lr = 1e-3


class PolydisAftertouch:
    def __init__(self) -> None:
        model = DisentangleVAE.init_model()
        model.load_model(model_path)
        print(f"loaded model {model_path}.")
        self.model = model

    def reconstruct(self, prmat, chd, fn, chd_sample=False):
        chd = chd.to(device).float()
        prmat = prmat.to(device).float()
        est_x = self.model.inference(prmat, chd, sample=False, chd_sample=chd_sample)
        estx_to_midi_file(est_x, fn + "_recon.mid")


if __name__ == "__main__":
    aftertouch = PolydisAftertouch()
