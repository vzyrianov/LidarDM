# Voxel Diffusion Baseline

## Setup Weights

To get access to the model weights, see the "Download Model Weights and Assets" section in [README.md](../README.md)

## Run sampling 

```bash
python scripts/diffusion/diffusion_sample_baselines.py +experiment=w_s_unetc_seq +sampling.outfolder=$PWD/_samples/my_waymo_baseline/ +model.unet.pretrained=../../pretrained_models/waymo_baseline/wsunetcseq.ckpt
```