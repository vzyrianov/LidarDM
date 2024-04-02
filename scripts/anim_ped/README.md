# Pedestrian Generation

## Note

As you can see below, these procedures are rather complicated with a lot of steps, and have only been tested with the exact version of packages below. Thus, we have computed and provided an asset bank using the methods below to animate 100+ pedestrian sequences. It is highly recommended to use that instead. Link to download is in the main README. 

For scientific research purpose only bounded by license: https://smpl-x.is.tue.mpg.de/modellicense

## Prerequisites: 

This already assumes that you have successfly ran [AvatarClip](https://github.com/hongfz16/AvatarCLIP), and have obtained some meshes that you wish to animate. Please follow their instruction to generate the `.ply` files. (their Colab example is really useful!)

## Setup

- **Install Blender**, only tested with Blender 3.6 LTS
  ```code 
  sudo snap install blender --channel=3.6lts/stable --classic
  ```

- **Set up Conda environment** as below:
  ```
  conda create -n smpl python=3.6
  conda activate smpl
  conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit==10.1.243 -c pytorch
  conda install tqdm

  pip install 'smplx[all]'  
  pip install chumpy
  pip install open3d
  ```

- **Install Python FBX SDK**
  - Go to https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-2020-3
  - Download the hyperlink that says `Linux FBX SDK 2020.3.2 Python (gz - 5057Kb)`
  - Unzip the downloaded file and follow the instructions in `Install_FbxSdk.txt`

- **Modify FBX import paths**
  - Edit `export_fbx.py` and `fbx_utils.py` with the FBX path you just extracted:
    ```
    sys.path.append('/path/to/fbxsdk/FBX202031_FBXPYTHONSDK_LINUX/lib/Python37_x64')
    ```

- **Download SMPL model**
  - Download the [SMPl models](https://smpl.is.tue.mpg.de/) (`version 1.1.0 for Python 2.7 (female/male/neutral, 300 shape PCs`) and place it as follows
    ```bash
      anim_ped/
        ├── ...
        └── smpl_models/
              └── smpl/
                    ├── SMPL_FEMALE.pkl
                    ├── SMPL_MALE.pkl
                    └── SMPL_NEUTRAL.pkl
    ```
## Procedures

### 1. Rig the `.ply` files + Convert to FBX

- Make sure that your `.ply` files (which are outputs of AvatarClip), are placed at the `ply_files/` folder
- Run `python export_fbx.py`, which will automatically rig the model and save the exported FBX Binary files to `fbx_files/`

### 2. Get the sample walking motion from Mixamo

**Note**: we only upload one mesh to Mixamo to get their animation that we can then apply to other meshes automatically.  

- Register for [Mixamo](https://www.mixamo.com/#/), search for and choose the Walking animation, and upload one random model in `fbx_files/` 
  
  ![alt text](img/mixamo.png)


- Download the animated model with these options:
   - *Format*: FBX binary
   - *Skin*: with skin
   - *Frames per Second*: 30
   - *Keyframe Reduction*: uniform 

- Move the downloaded FBX file to `animation/`

### 3. Apply animation to the FBX format 

- Before running the scripts, make sure your folder structure is like this
  ```bash
    anim_ped/
        ├── ...
        ├── animation/
        │     └── animated.fbx
        └── fbx_files/
              ├── mesh_1.fbx
              └── ...  
  ```
- Run the Blender script with the following command. Ignore the "Error" that pops up in the terminal. 
    ```bash
    blender -b -P script.py
    ```
- The animated `.obj` sequences will be saved in `outputs/`, where each subfolder contains 30 `.obj` files. You can move all the subfolders of `outputs/` to `lidardm/generated_assets/pedestrian` with
    ```bash
    mv -t /path/to/lidardm/generated_assets/pedestrian outputs/*
    ```

## Credit

This script is built on top of `Avatar2FBX` folder of [AvatarClip](https://github.com/hongfz16/AvatarCLIP/tree/main/Avatar2FBX), which is based on [Smplx2FBX](https://github.com/mrhaiyiwang/Smplx2FBX). Thank the authors of both repositories for the awesome open-source projects!