# Scene Generation

## Training

The primary model training codebase is under `lidardm/core/`.

Training runs are launched with `scripts/train.py`.

The specific datasets, loss functions, and models that are used are specified with experiment configs `lidardm/core/configs/experiment/`.

The codebase supports multinode and multigpu training by passing in the argument `++trainer.devices=NUM_DEVICES ++trainer.nodes=NUM_NODES`.


### Conditional Generation (Waymo)

The Waymo pipeline has 3 learnable components: a Map VAE, the Waymo field VAE, and a diffusion model that generates Waymo Fields based on a Map condition. 

- Waymo Map VAE: `python scripts/train.py +experiment=map_vae_waymo`
- Waymo VAE: `python scripts/train.py +experiment=wf_s_vae`
- Waymo Diffusion Model: `python scripts/train.py +experiment=wf_s_unetc` 


### Unconditional Generation (KITTI-360)

- KITTI-360 VAE: `python scripts/train.py +experiment=kf_s_vae`
- KITTI-360 Diffusion Model: `python scripts/train.py +experiment=kf_s_unet`
