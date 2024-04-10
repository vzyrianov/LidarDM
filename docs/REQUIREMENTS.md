# Requirements

**Note**: We have provided a script (`install_lidardm.sh`) to install the following inside a Conda environment.

For the requried packages:
```bash
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install hydra-core lightning tensorboard wandb transformers
pip install matplotlib natsort pykdtree pyntcloud pyquaternion scikit-learn scipy tqdm open3d opencv-python mediapy
pip install diffusers==0.20.2
pip install nksr==1.0.3+pt20cu117           -f https://nksr.huangjh.tech/whl/torch-2.0.0+cu117.html
pip install torch-scatter==2.1.2+pt20cu117  -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install torchmetrics==1.2.0
pip install triton==2.0.0
```

For Waymo preprocessing/training:
```bash
pip install waymo-open-dataset-tf-2-11-0==1.6.0
pip install opencv-python==4.5.4.58
pip install pip install numpy==1.20.3
```

For KITTI-360 preprocessing/training:
```bash
pip install git+https://github.com/autonomousvision/kitti360Scripts.git 
```

For Waymax:
```bash
pip install --upgrade pip
pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax
```

Finally, run: 
```bash
pip install -v -e .
```