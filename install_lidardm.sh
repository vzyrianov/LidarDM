conda create -n lidardm python=3.10.13
conda activate lidardm

# required modules
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
pip install hydra-core
pip install lightning
pip install tensorboard
pip install wandb
pip install transformers
pip install matplotlib
pip install natsort
pip install pykdtree
pip install pyntcloud
pip install pyquaternion
pip install scikit-learn
pip install scipy
pip install tqdm
pip install open3d
pip install opencv-python
pip install diffusers==0.20.2
pip install nksr==1.0.3+pt20cu117           -f https://nksr.huangjh.tech/whl/torch-2.0.0+cu117.html
pip install torch-scatter==2.1.2+pt20cu117  -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install torchmetrics==1.2.0
pip install triton==2.0.0
pip install -v -e . # for lidardm

# for waymo open dataset pre-processing
echo "Do you want to install packages for Waymo Dataset pre-processing? [y/n]"
read waymo_choice
waymo_choice_lowercase=$(echo "$waymo_choice" | tr '[:upper:]' '[:lower:]')
if [ "$waymo_choice_lowercase" = "y" ]; then
  pip install waymo-open-dataset-tf-2-11-0==1.6.0
  pip install opencv-python==4.5.4.58
  pip install numpy==1.20.3 
fi

# for kitti pre-processing
echo "Do you want to install packages for KITTI-360 Dataset pre-processing? [y/n]"
read kitti_choice
kitti_choice_lowercase=$(echo "$" | tr '[:upper:]' '[:lower:]')
if [ "$kitti_choice_lowercase" = "y" ]; then
  pip install git+https://github.com/autonomousvision/kitti360Scripts.git
fi

# for waymax
echo "Do you want to install Waymax? [y/n]"
read waymax_choice
waymax_choice_lowercase=$(echo "$waymax_choice" | tr '[:upper:]' '[:lower:]')
if [ "$waymax_choice_lowercase" = "y" ]; then
  pip install --upgrade pip
  pip install git+https://github.com/waymo-research/waymax.git@main#egg=waymo-waymax
fi
