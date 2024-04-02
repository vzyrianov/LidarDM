# Planner Instructions

## Training planner

To train a model on **real** data run: 
```bash
python scripts/train.py +experiment=planning.yaml ++data.loaders.train.dataset.percentage_to_use=0.1
```

To train a model on LidarDM **generated** data run:
```bash
python scripts/train.py +experiment=planning_generated.yaml ++data.loaders.train.dataset.percentage_to_use=0.10
```

Runs will be named based on a timestamp and stored in the `runs/` folder. During training, checkpoints are stored in the `checkpoints/` subfolder. 


To finetune a model on **real** data, modify the following command with the `PATH_TO_PRETRAINED` with the path to a checkpoint file (ends with `.ckpt`) for a planner trained on **generated** data: 

```bash
python scripts/train.py +experiment=planning_resume.yaml ++data.loaders.train.dataset.percentage_to_use=0.10 ++model.pretrained=PATH_TO_PRETRAINED
```

## Evaluating Metrics

After training a planner model, L2 distance and collision percentage can be evaluated for a checkpoint with: 

```bash
python scripts/metric/planning_metrics.py +experiment=planning.yaml +pmetrics.trajbank_dir=../../pretrained_models/waymo_trajbank.npy  +model.pretrained=PATH_TO_PRETRAINED
```