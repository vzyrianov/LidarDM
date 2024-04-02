# Ray-Dropping

## Overview

Our raydrop module employs the same encoder and decoder as [RangeNet++](https://github.com/PRBonn/lidar-bonnetal) (we use [`SqueezeSeqV2`](https://github.com/xuanyuzhou98/SqueezeSegV2) architecture), but modify the decoder head to predict the raydropping probability instead of semantic probabilities. Additionally, instead of using a sigmoid on the output logits, we use Gumbel sigmoid to generate a better-calibrated and more realisitc LiDAR point cloud. To train, we need to have a corresponding training data and ground truth label pairs:

- **training data**: We use NKSR-generated meshes from [SCENE_GENERATION.md](../docs/SCENE_GENERATION.md) and append vehicle and pedestrian meshes on it then raycast on the whole scene. We provide instructions to generate this below. 
- **ground truth label**: Our ground truth label here is just the corresponding raw point cloud.    

**Credit/License**: We thank the author of RangeNet++ for the amazing open-source project. We also included their LICCENSE at `lidardm/lidar_generation/raydropping` 

## Generate Training Data

### Waymo 

```bash
python scripts/dataset/raycast/waymo_raycast.py --split training validation --fields_path _datasets/waymo_fields --dataset_path _datasets/waymo_preprocessed --outdir datasets/waymo_raycast
```

### KITTI-360

```bash
python scripts/dataset/raycast/waymo_raycast.py --seq 0 1 2 3 4 5 6 7 9 10 --fields_path _datasets/kitti_fields --dataset_path _datasets/kitti_preprocessed --outdir datasets/kitti_raycast
```

## Training 

### Waymo
```bash
python lidardm/lidar_generation/raydropping/utils/train.py --arch_cfg=lidardm/lidar_generation/raydropping/config/waymo.yaml
```

### KITTI-360
```bash
python lidardm/lidar_generation/raydropping/utils/train.py --arch_cfg=lidardm/lidar_generation/raydropping/config/kitti360.yaml
```

## Validation

Please refer to the ablation in the supplementary on the usefulness of Raydropping. We also provide the generated LiDAR data with 3 configurations (no raydrop, raydrop with sigmoid, raydrop with gumbel) that we use to run the ablation MMD/JSD, which can be downloaded here:

```bash
TODO: upload zip + metrics code
```