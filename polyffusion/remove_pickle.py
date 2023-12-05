import pickle
import sys

from dirs import *

if __name__ == "__main__":
    remove = sys.argv[1]
    with open(f"{TRAIN_SPLIT_DIR}/pop909.pickle", "rb") as f:
        pic = pickle.load(f)
    assert remove in pic[0] or remove in pic[1]
    if remove in pic[0]:
        pic[0].remove(remove)
    else:
        pic[1].remove(remove)
    with open(f"{TRAIN_SPLIT_DIR}/pop909.pickle", "wb") as f:
        pickle.dump(pic, f)
