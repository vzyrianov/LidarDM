import os
import numpy as np

import torch
import cv2
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt
from glob import glob
import shutil
import argparse

def main() -> None:

    parser = argparse.ArgumentParser(prog='formatConverter')
    parser.add_argument("--input_folder")
    parser.add_argument("--output")
    args = parser.parse_args()


    # Input Setup
    all_input = glob(os.path.join(args.input_folder, "out_*"))
    random.shuffle(all_input)

    # Output Setup
    os.makedirs(os.path.join(args.output, "raycast"), exist_ok=True)


    for input_folder in all_input:
        manifest = np.genfromtxt(os.path.join(input_folder, 'manifest.txt'), delimiter=", ", dtype='str')
        seq_id, center_frame_idx = manifest[0], int(manifest[1])

        output_filename = os.path.join(args.output, "raycast", f"{seq_id}_{center_frame_idx:03d}.bin")

        if(os.path.isfile(output_filename)):
            continue
        

        src_fn = os.path.join(input_folder, "raycast", "2.npy")
        if(not os.path.isfile(src_fn)):
            print('ERROR: {src_fn} missing. ')
            continue

        x = np.load(src_fn)
        x = x.astype(np.float32).tofile(output_filename)

        #np.save(output_filename, x)


if __name__ == "__main__":
    main()
