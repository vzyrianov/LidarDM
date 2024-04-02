import os
import numpy as np

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

import torch
import cv2
import random
import os
from datetime import datetime
import matplotlib.pyplot as plt

from lidardm.core.utils import instantiate_data, instantiate_model, instantiate_trainer
from lidardm.core.models.diffusion import ScheduleConfig

from lidardm.visualization import unvoxelize_field, cond_to_rgb_waymo
from lidardm.core.datasets.utils import voxelize

from lidardm.core.schedulers.modified_schedulers import *
from lidardm.core.datasets.utils import unscale_field
from lidardm.visualization.meshify import FieldsMeshifier

CONFIG_PATH = os.path.join(os.getcwd(), "lidardm", "core", "configs")
CONFIG_NAME = "default.yaml"

def sample_to_cuda(sample_cpu):
    sample = {}
    for k in sample_cpu:
        sample[k] = sample_cpu[k][:].cuda()

    return sample

def sample_to_torch(sample_cpu):
    sample = {}
    for k in sample_cpu:
        sample[k] = torch.from_numpy(sample_cpu[k][:]).unsqueeze(0)

    return sample


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    cfg['data']['loaders']['train']['batch_size'] = 1
    cfg['data']['loaders']['train']['shuffle'] = False
    cfg['data']['loaders']['val']['shuffle'] = False
    cfg['data']['loaders']['val']['batch_size'] = 1
    
    GENERATE_VIDEO=False
    CFG_SCALE = 4.0

    OUT_FOLDER = './'
    if('sampling' in cfg and 'outfolder' in cfg['sampling']):
        OUT_FOLDER = cfg['sampling']['outfolder']

    SKIP_RENDERING = False
    if('sampling' in cfg and 'skiprender' in cfg['sampling']):
        SKIP_RENDERING = cfg['sampling']['skiprender']

    SAMPLING_STEPS = 600
    if('sampling' in cfg and 'steps' in cfg['sampling']):
        SAMPLING_STEPS = cfg['sampling']['steps']

    #if('sampling' in cfg and 'seed_time' in cfg['sampling']):
    #    if(cfg['sampling']['seed_time'] == True):
    #        cfg.run.seed = datetime.now().microsecond

    USE_TEST = False
    if('sampling' in cfg and 'use_test' in cfg['sampling']):
        USE_TEST = cfg['sampling']['use_test']




    L.seed_everything(cfg.run.seed, workers=True)

    data: L.LightningDataModule = instantiate_data(cfg)
    model: L.LightningModule = instantiate_model(cfg)

    model = model.model.cuda()

    data_map = {
        'lidar': torch.zeros((1, 1, 1, 1)).cuda()
    }

    
    scheduler = ModifiedEulerAScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear')

    schedule_config = ScheduleConfig()
    schedule_config.last_noise_samples = 10 #80
    schedule_config.second_to_last_noise_samples = 8#50
    schedule_config.use_noise_second_to_last = True
    schedule_config.use_noise_last = False

    SAMPLE_OUT_DIR = 'pc_fields_'
    os.system(f'mkdir {SAMPLE_OUT_DIR}')

    IS_WAYMO = "WaymoFields" in cfg['data']['loaders']['train']['dataset']['_target_']

    selected_dataset = data.train_dataloader().dataset
    if(USE_TEST):
        print('USING THE TESTING SET')
        selected_dataset = data.val_dataloader().dataset

    with torch.no_grad():
        i = 0
        random_order = [*range(0, selected_dataset.__len__())]
        SAMPLES_TO_GENERATE = len(random_order)
        print(f'Length of files: {SAMPLES_TO_GENERATE}')
        while(i < SAMPLES_TO_GENERATE):

            do_again=True
            while(do_again):
                try:
                    os.makedirs(f"{OUT_FOLDER}/out_{i}")
                    do_again=False
                except:
                    do_again=True
                    i = i + 1
                    if(i > SAMPLES_TO_GENERATE):
                        return

            sample_cpu = sample_to_torch(selected_dataset.__getitem__(i))#next(iter(data.train_dataloader()))
            sample = sample_to_cuda(sample_cpu)



            if(IS_WAYMO):
                f, seq_name, center_idx = selected_dataset.get_field(i)
                with open(f"{OUT_FOLDER}/out_{i}/manifest.txt", "w") as file1:
                    file1.write(f"{str(seq_name)}, {str(center_idx)}")


            if(not SKIP_RENDERING):
                os.system(f'mkdir {OUT_FOLDER}/out_{i}/img')
                os.system(f'mkdir {OUT_FOLDER}/out_{i}/vid')
                os.system(f'mkdir {OUT_FOLDER}/out_{i}/cond')
                os.system(f'mkdir {OUT_FOLDER}/out_{i}/ply')

            if((not SKIP_RENDERING) and "bev" in sample.keys()):
                plt.imshow(cond_to_rgb_waymo(sample_cpu['bev'][0].detach().cpu().numpy()))
                plt.savefig(f'{OUT_FOLDER}/out_{i}/cond/cond.png')

            generated_map = model.forward_test(sample, scheduler=scheduler, gumbel_tau=1.0, schedule_config=schedule_config, cfg_scale=CFG_SCALE,
                                               use_gumbel=False, threshold=0.3, num_steps=SAMPLING_STEPS, generate_video=GENERATE_VIDEO)

            for j in range(0, len(generated_map['video_field'])):
                field_recon = unscale_field(generated_map['video_field'][j], selected_dataset.n_min, selected_dataset.n_max)
            
                unvoxelized = unvoxelize_field(torch.ones_like(field_recon), field_recon, cfg['data']['spatial_range'], cfg['data']['voxel_size'])

                device = torch.device("cuda:0")
                meshifier = FieldsMeshifier(device, 0.15, [96, 96, 6])
                mesh = meshifier.generate_mesh(unvoxelized.to(device), f"{OUT_FOLDER}/out_{i}/ply/final.ply")

            i = i + 1

            if(i > SAMPLES_TO_GENERATE):
                break


    
    
if __name__ == "__main__":
    main()
