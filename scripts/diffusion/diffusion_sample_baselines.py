import os
import numpy as np
from datetime import datetime
import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

import torch
import cv2
import os

import matplotlib.pyplot as plt

from lidardm.core.utils import instantiate_data, instantiate_model
from lidardm.core.models.diffusion import ScheduleConfig

from lidardm.visualization import render_open3d, save_open3d_render, unvoxelize

from lidardm.core.schedulers.modified_schedulers import *

CONFIG_PATH = os.path.join(os.getcwd(), "lidardm", "core", "configs")
CONFIG_NAME = "default.yaml"

def sample_to_cuda(sample_cpu):
    sample = {}
    for k in sample_cpu:
        sample[k] = sample_cpu[k][:].cuda()

    return sample


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    GENERATE_VIDEO = False
    GENERATE_MAP_COND_SEQUENCE = False

    cfg['data']['loaders']['train']['batch_size'] = 1
    cfg['data']['loaders']['val']['batch_size'] = 1
    

    OUT_FOLDER = './'
    if('sampling' in cfg and 'outfolder' in cfg['sampling']):
        OUT_FOLDER = cfg['sampling']['outfolder']

    SKIP_RENDERING = True
    #if('sampling' in cfg and 'skiprender' in cfg['sampling']):
    #    SKIP_RENDERING = cfg['sampling']['skiprender']

    SAMPLING_STEPS = 600
    if('sampling' in cfg and 'steps' in cfg['sampling']):
        SAMPLING_STEPS = cfg['sampling']['steps']

    #if('sampling' in cfg and 'seed_time' in cfg['sampling']):
    #    if(cfg['sampling']['seed_time'] == True):
    #        cfg.run.seed = datetime.now().microsecond

    cfg.run.seed = datetime.now().microsecond

    L.seed_everything(cfg.run.seed, workers=True)

    is_conditional = "Conditional" in cfg['model']['_target_']

    data: L.LightningDataModule = instantiate_data(cfg)
    model: L.LightningModule = instantiate_model(cfg)

    model = model.model.cuda()

    # Provides batch size and device information
    data_map = {
        'lidar': torch.zeros((1, 1, 1, 1)).cuda()
    }


    scheduler = ModifiedEulerAScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear')
    #scheduler = ModifiedDPMSolverMultistepScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    

    schedule_config = ScheduleConfig()
    schedule_config.last_noise_samples = 10 #80
    schedule_config.second_to_last_noise_samples = 8#50
    schedule_config.use_noise_second_to_last = True
    schedule_config.use_noise_last = False




    with torch.no_grad():
        SAMPLES_TO_GENERATE = 2000
        i = 0

        for sample_cpu in data.val_dataloader():
    
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

            sample = sample_to_cuda(sample_cpu)

            os.system(f'mkdir {OUT_FOLDER}/out_{i}/img')
            os.system(f'mkdir {OUT_FOLDER}/out_{i}/pts')
            
            
            generated_map = model.forward_test(sample, use_gumbel=False, scheduler=scheduler, gumbel_tau=1.0, threshold=0.5, num_steps=SAMPLING_STEPS, schedule_config=schedule_config)

            for j in range(0, len(generated_map['sample_seq'])):

                unvoxelized = unvoxelize(generated_map['sample_seq'][j][0], cfg['data']['spatial_range'], cfg['data']['voxel_size'])

                #Save for eval
                #np.save(f"{SAMPLE_OUT_DIR}/{i}.npy", unvoxelized.cpu().detach().numpy())

                #plt.imsave(f'out_{i}/img/{j}_bev.png', generated_map["sample_seq"][j][0].sum(0).cpu().detach().numpy())
                torch.save(unvoxelized, f'{OUT_FOLDER}/out_{i}/pts/{j}.pth')

                if (SKIP_RENDERING):
                    bev_img, pts_img, side_img = render_open3d(unvoxelized.cpu().detach().numpy(), spatial_range=cfg['data']['spatial_range'])
               
                    save_open3d_render(f"{OUT_FOLDER}/out_{i}/img/{j}_top.png", bev_img, quality=9)
                    save_open3d_render(f"{OUT_FOLDER}/out_{i}/img/{j}_side.png", side_img, quality=9)
                    save_open3d_render(f"{OUT_FOLDER}/out_{i}/img/{j}_pts.png", pts_img, quality=9) 

            i = i + 1

            if(i > SAMPLES_TO_GENERATE):
                break


if __name__ == "__main__":
    main()
