# LidarDM: Generative LiDAR Simulation in a Generated World

[![Waymax Generation](https://img.shields.io/badge/-Waymax-blue?logo=googlecolab)](https://colab.research.google.com/github/vzyrianov/LidarDM/blob/main/examples/notebook_waymax.ipynb)
[![KITTI Generation](https://img.shields.io/badge/-KITTI360-blue?logo=googlecolab)
](https://colab.research.google.com/github/vzyrianov/LidarDM/blob/main/examples/notebook_kitti.ipynb)
[![KITTI Generation](https://img.shields.io/badge/-arXiv-red?logo=arxiv)
](https://arxiv.org/abs/2404.02903)
[![KITTI Generation](https://img.shields.io/badge/-Page-grey?logo=github)
](https://www.zyrianov.org/lidardm/)


![alt text](docs/assets/teaser.gif)

## Setup


### 1. Setup Conda Environments

The project was tested with Python 3.10.13 and all the package requirements are listed in [REQUIREMENTS.md](docs/REQUIREMENTS.md). To make things easier, we have provided a script to install the enviroment:

```bash
sh install_lidardm.sh
```

### 2. Datasets Download/Preprocess/GT SDF Generation
  
LidarDM supports 2 datasets: 
- KITTI-360 for unconditional generation.
- Waymo Open Dataset for conditional multi-frame generation and downstream tasks.  

For each of the datasets, we provide a preprocess to filter out dynamic objects for every scan. 

#### KITTI-360 

1. Download the [KITTI-360 Dataset](https://www.cvlibs.net/datasets/kitti-360/) into `./_datasets/kitti/`. 
   Make sure to download all of `Raw Velodyne Scans` (`data_3d_raw`), `3D Bounding Boxes` (`data_3d_bboxes`), `Vehicle Poses` (`data_poses`)
2. Run the preprocess script:

    ```bash
    python scripts/dataset/preprocess/kitti360_preprocess.py --dataset_path _datasets/kitti/ --seq 0 1 2 3 4 5 6 7 9 10 --out_dir _datasets/kitti/
    ```
3. Generate Ground Truth SDFs (optional for sampling, required for training): 

    ```bash
    python scripts/dataset/field_generation/kitti360_field_generation.py --dataset_path _datasets/kitti/ --out_dir _datasets/kitti_fields/ --seq 0 1 2 3 4 5 6 7 9 10 --voxel_size 0.15 --grid_size 96 96 6 --save_files
    ```

#### Waymo Open Dataset

1. Download the [Waymo Open Dataset](https://waymo.com/open/download/) Perception Dataset v1.4.2 into `./_datasets/waymo/` 

2. For preprocessing lidar, bounding boxes, and maps, run the following. **Warning**: the entire preprocessed Waymo dataset takes around **2TB** on top of the raw Waymo tfrecords.

    ```bash
    python scripts/dataset/preprocess/waymo_preprocess.py --tfrecord_paths _datasets/waymo/ --split training validation --out_dir _datasets/waymo_preprocessed/
    ```
3. Generate Ground Truth SDFs: 
    ```bash
    python scripts/dataset/field_generation/waymo_field_generation.py --scans _datasets/waymo_preprocessed/ --pose _datasets/waymo_preprocessed/ --splits training validation --out_dir _datasets/waymo_fields/ --voxel_size 0.15 --grid_size 96 96 6 --save_files
    ```

### 3. Assets Generation

We provide a collection of pregenerated assets bank for convenience:

- **Full Version** (200+ vehicle meshes, 150+ pedestrian sequences):
    ```bash 
    wget -O full_assets.tar.xz "https://uofi.box.com/shared/static/4bndbr8l2fgmjb3tdu1gvjts8hqbo76w.xz"
    tar -xvf full_assets.tar.xz
    rm full_assets.tar.xz
    ```
- **Small Version** (20 vehicle meshes, 2 pedestrian sequences):
    ```bash 
    wget -O small_assets.tar.xz "https://uofi.box.com/shared/static/ir42y4a71luia12u17bk5v0fierqrogw.xz"
    tar -xvf small_assets.tar.xz
    rm small_assets.tar.xz
    ```

If you want to generate your own assets, see [ASSET_GENERATION.md](docs/ASSET_GENERATION.md). 

### 4. Folder Structure
Ultimately, the project directory should look like this

```
  lidardm/ 
    ├── ...
    ├── _datasets/
    |     ├── waymo/
    |     |     └── ...
    |     ├── waymo_preprocessed/
    |     |     └── ...
    |     ├── waymo_fields/
    |     |     └── ...
    |     ├── kitti/
    |     |     └── ...
    |     └── kitti_fields/
    |           └── ...
    └── generated_assets/
          ├── pedestrian/
          │     └── pedestrian_1/
          │           ├── 000.obj
          │           └── ...
          └── vehicle/
                ├── car1.obj
                └── ...
```

## Training 

Our pipeline has 2 learnable components: 
- **Scene Generation**: We employ a classifier-free guidance-based latent diffusion model to generate the underlying 3D static world. Refer to [SCENE_GENERATION.md](docs/SCENE_GENERATION.md) for details on training.
- **Raydropping**: After constructing the 4D world and raycasting on it, we train a raydropping network inspired by RangeNet to make our generated LiDAR more realistic. Refer to [RAYDROPPING.md](docs/RAYDROPPING.md) for details on training.



## Inference


### 1. Download Model Weights and Assets

#### Waymo

- Prior to receiving access to the Waymo Weights you are required to have a valid Waymo Open Dataset account with access to the Waymo Open Dataset.

- If you are successfully registered with the Waymo Open dataset, send an email in the following format to receive the Waymo weights. 
    - To: ```vlasz2 AT_SIGN illinois.edu```
    - Subject Line: "lidardm waymo weights"

- You will receive a reply with instructions on getting Map VAE, Scene VAE, Diffusion Model, and raydropping network weights trained on Waymo.

- You will also receive instructions for setting up the Waymo Baseline Model. 


#### KITTI-360

```bash
mkdir pretrained_models && cd pretrained_models

wget -O kitti_weights.zip "https://uofi.box.com/shared/static/tc4hppt38ryy5rsgthiw4q50dxtu1w2f.zip"

unzip kitti_weights.zip

rm kitti_weights.zip
```

### 2. Verify Folder Structure
Ultimately, the project directory should look like this

```
  lidardm/ 
    ├── ...
    └── pretrained_models/
          ├── waymo/ 
          │     ├── raydrop/
          │     │     └── ...
          │     └── scene_gen/
          │           └── ...
          ├── kitti360/ 
          │     ├── raydrop/
          │     │     └── ...
          │     └── scene_gen/
          │           └── ...
          ├── waymo_baseline/
          │     └── ...
          └── waymo_trajbank.npy
```

### 3. Run sampling

We have provided the jupyter notebook in `examples` which provides visualization (like the teaser.gif above), but you can also run larger scale sampling for metrics below.


# Sampling

## KITTI-360 Model

Sample 2000 samples:

```bash
python scripts/diffusion/diffusion_sample_field.py +experiment=kf_s_unet +model.unet.pretrained=../../pretrained_models/kitti360/scene_gen/kfsunet_b.ckpt +sampling.seed_time=True +sampling.outfolder=$PWD/_samples/my_lidardm_kitti +sampling.skiprender=True
```

Perform raycasting on those samples: 
```bash
python lidardm/lidar_generation/scene_composition/scripts/unconditional_comp.py --folder _samples/my_lidardm_kitti/ --waymo_dataset_root _datasets/waymo_preprocessed/
```

The results will be generated under `_samples/my_lidardm_kitti`. Optionally, you can also run the following script to extract the LiDAR for MMD/JSD calculation. 

```bash
mkdir _samples/my_lidardm_kitti_npy

python scripts/util/move_lidar.py _samples/my_lidardm_kitti/ _samples/my_lidardm_kitti_npy
```

Now, the folder `_samples/my_lidardm_kitti_npy` can be provided to the MMD and JSD evaluation script. 

## Waymo Model

### Random Sampling

The following script samples meshes by randomly acquiring conditions from the dataset, and then runs raycasting:

```bash
python scripts/diffusion/diffusion_sample_field.py +experiment=wf_s_unetc +model.unet.pretrained=../../pretrained_models/waymo/scene_gen/wfsunetc.ckpt +sampling.seed_time=True +sampling.outfolder=$PWD/_samples/my_lidardm_waymo +sampling.skiprender=True

python lidardm/lidar_generation/scene_composition/scripts/conditional_comp.py --folder $PWD/_samples/my_lidardm_waymo --waymo_dataset_root _datasets/waymo_preprocessed/
```

### Sampling Dataset

We provide a script to run LidarDM LiDAR generation conditioned on the entire Waymo dataset splits for Sim2Real and Real2Sim evaluation tasks. 

For **test** set generation. 
```bash
mkdir _datasets/lidardm_waymo_sim_test

python scripts/diffusion/diffusion_generate_dataset.py +experiment=wf_s_unetc +model.unet.pretrained=../../pretrained_models/waymo/scene_gen/wfsunetc.ckpt +sampling.outfolder=$PWD/_datasets/lidardm_waymo_sim_test +sampling.skiprender=True +sampling.use_test=True

python lidardm/lidar_generation/scene_composition/scripts/conditional_comp.py --folder $PWD/_datasets/lidardm_waymo_sim_test --waymo_dataset_root _datasets/waymo_preprocessed/
```

For **train** set generation. 
```bash
mkdir _datasets/lidardm_waymo_sim_train

python scripts/diffusion/diffusion_generate_dataset.py +experiment=wf_s_unetc +model.unet.pretrained=../../pretrained_models/waymo/scene_gen/wfsunetc.ckpt +sampling.outfolder=$PWD/_datasets/lidardm_waymo_sim_train +sampling.skiprender=True

python lidardm/lidar_generation/scene_composition/scripts/conditional_comp.py --folder $PWD/_datasets/lidardm_waymo_sim_train --waymo_dataset_root _datasets/waymo_preprocessed/
```

# Evaluating Metrics

### MMD/JSD

The MMD/JSD evaluation script is designed to compare two folders of point clouds (in the format `folder1/*.npy`). We provide a pregenerated set of samples used for comparison in the paper. 

```bash
mkdir _samples && cd _samples

wget -O kitti_uncond.zip https://uofi.box.com/shared/static/ulommpmbq064azfjedkrgzlbmd7drwzv.zip

unzip kitti_uncond.zip
```

The following command executes **MMD** calculation.

```bash
python scripts/metric/eval_set.py --folder1 _samples/kitti_uncond/kitti/ --folder2 _samples/kitti_uncond/lidardm/ --type mmd --folder2_rotations 0
```

To perform **JSD** calculation, replace `--type mmd` with `--type jsd`

### Point2Plane and Chamfer

For a folder of LidarDM generation on Waymo run: 
```bash
python scripts/metric/point2plane.py --input_folder _samples/my_lidardm_waymo/ 
```

To generate samples for the **voxel sequence diffusion model baseline**, see the instructions in [VOXEL_BASELINE.md](docs/VOXEL_BASELINE.md).

To evaluate the results from the baseline model run: 
```bash
python scripts/metric/point2plane.py --is_baseline True --input_folder _samples/my_voxelseq_waymo/
```


### Planner

For instructions on training the LiDAR planner and running planner evaluations refer to [PLANNER.md](docs/PLANNER.md)
