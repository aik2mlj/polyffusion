import os

if __name__ == "__main__":
    result = "./result"
    for dir in os.listdir(result):
        dpath = f"{result}/{dir}"
        if dir == "try":
            os.system(f"rm -rf {dpath}/*")
            continue
        for item in os.listdir(dpath):
            item_dpath = f"{dpath}/{item}"
            chkpt_dir = f"{item_dpath}/chkpts"
            if not os.path.exists(f"{chkpt_dir}/weights.pt"):
                os.system(f"ls -l {chkpt_dir}")
                y = input(f"Remove {item_dpath} (y/n)?")
                if y == "y":
                    os.system(f"rm -rf {item_dpath}")
