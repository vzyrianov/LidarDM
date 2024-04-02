import os
import sys
from glob import glob
from natsort import natsorted

paths = glob(os.path.join(sys.argv[1], "*", "pcd", "raycast", "2.npy"))
PATHS_FROM = natsorted(paths)

PATHS_TO = sys.argv[2]

i = 0
for p in PATHS_FROM:
    new_path = os.path.join(PATHS_TO, f"{i}.npy")
    os.system(f"cp {p} {new_path}")
    i = i + 1
