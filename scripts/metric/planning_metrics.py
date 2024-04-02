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
from lidardm.core.datasets.waymo_planning import WaymoPlanning
from lidardm.visualization.dataset2sample import *
from lidardm.planning.utils import *
from lidardm.visualization.cond2rgb import cond_to_rgb_waymo
from lidardm.core.visualizers.cost_visualizer import *

from lidardm.core.utils import instantiate_data, instantiate_model, instantiate_trainer


CONFIG_PATH = os.path.join(os.getcwd(), "lidardm", "core", "configs")
CONFIG_NAME = "default.yaml"

def convert_pixel_meter(value, pix2m=True):
    '''
    input:
        - value: the value to convert from/to pixel to/from meter
        - pix2m: True if pixel-> meter, False if meter->pixel
    output:
        - converted value
    '''
    if pix2m: return (value - 160)*0.15
    else: return (value/0.15 + 160).astype(np.int32)

def convert_pixel_meter_torch_v(value, pix2m=True, grid_0=320, grid_1=464, voxel_size=0.15):
    '''
    input:
        - value: the value to convert from/to pixel to/from meter
        - pix2m: True if pixel-> meter, False if meter->pixel
    output:
        - converted value
    '''
    return convert_pixel_meter_torch(value, pix2m=pix2m, grid_0=grid_0, grid_1=grid_1, voxel_size=voxel_size)


def convert_local_pixel_to_future_pixel(predicted_pixel, current_pose, future_pose, t):
    '''
    input:
      - predicted_pixel: pixel location of predicted waypoint in current pose ([2,] vector)
      - current_pose: [4x4] matrix
      - future_pose: pose in the future with same timestamp as predicted_pixel [4x4] matrix
    output:
      - predicted waypoint in image frame of future pose
    '''

    pred_loc_meter_xy = convert_pixel_meter_torch_v(predicted_pixel.copy(), pix2m=True)
    pred_loc_meter = np.array([-1*pred_loc_meter_xy[1], pred_loc_meter_xy[0], 0, 1])#np.append(pred_loc_meter_xy, [0,1])
    #the_transform = np.linalg.inv(future_pose) @ current_pose
    the_transform = np.linalg.inv(t) #np.linalg.inv(np.linalg.inv(current_pose) @ future_pose)
    pred_loc_in_future_frame = (the_transform @ pred_loc_meter)[:2]
    pred_loc_in_future_frame = np.array([-1*pred_loc_in_future_frame[1], pred_loc_in_future_frame[0]])
    #print(pred_loc_in_future_frame)
    pred_pixel_in_future_frame = convert_pixel_meter_torch_v(pred_loc_in_future_frame, False)
    return pred_pixel_in_future_frame.astype(np.int32)

def num_collision(predict_trajectory, future_maps, current_pose, future_poses, tn2t4, viz=True):
    '''
    input: 
      - predict_trajectory: List of N one-hot matrix of future waypoints in current_pose coordinate
      - future_maps: Nx640x640x13 corresponding future maps
      - current_pose: 4x4 Matrix of the current timestamp
      - future_poses: List of N ground truth future poses corresponding to 
                      the timestamps of predic_traj 
    output:
      - collision or not: bool 
    '''

    if(viz):
        fig, axs = plt.subplots(2, 10, figsize=(64, 12))

    obstacles = future_maps[:,:,:,5:]
    objects_bitmask = obstacles.sum(3) > 0 # if sum per pixel is > 0 then there is obstacle at that pixel, 10x640x640
    objects_bitmask = np.transpose(objects_bitmask, (1,2,0)) # 640x640x10

    if(viz):
        for i in range(0,objects_bitmask.shape[2]):
            axs[0][i].imshow(objects_bitmask[:,:,i])
        

    traj_bitmask = np.zeros_like(objects_bitmask)
    for i in range(len(predict_trajectory)):
        predicted_pixel = predict_trajectory[i] #np.array([pixel_x, pixel_y])
        pred_x, pred_y = convert_local_pixel_to_future_pixel(predicted_pixel, current_pose, future_poses[i], tn2t4[i])
    
        tmp_image = np.zeros((traj_bitmask.shape[0], traj_bitmask.shape[1], 3))
        cv2.rectangle(tmp_image, (pred_y-16, pred_x-7), (pred_y+12, pred_x+7), color=1, thickness=-1)
        traj_bitmask[:,:,i] = tmp_image.sum(2)
    traj_bitmask = traj_bitmask > 0

    if(viz):
        for i in range(0, traj_bitmask.shape[2]):
            axs[1][i].imshow(traj_bitmask[:,:,i])
    
    if(viz):
        plt.savefig('out.png')
        plt.close()


    return np.any(np.logical_and(traj_bitmask, objects_bitmask), axis=(0,1)).sum()

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    L.seed_everything(cfg.run.seed, workers=True)



    #data: L.LightningDataModule = instantiate_data(cfg)
    model: L.LightningModule = instantiate_model(cfg) 

    model = model.cuda()

    TOTAL_TO_PROCESS = 1000
    if('pmetrics' in cfg and 'total_count' in cfg['pmetrics']):
        TOTAL_TO_PROCESS = cfg['pmetrics']['total_count']
    
    VIZ = False
    if('pmetrics' in cfg and 'viz' in cfg['pmetrics']):
        VIZ = cfg['pmetrics']['viz']

        
    FLAG_USE_TRAJ_BANK = True
    if('pmetrics' in cfg and 'use_trajbank' in cfg['pmetrics']):
        FLAG_USE_TRAJ_BANK = cfg['pmetrics']['use_trajbank']
    
    TRAJ_BANK_DIR = None
    if(FLAG_USE_TRAJ_BANK):
        if('pmetrics' in cfg and 'trajbank_dir' in cfg['pmetrics']):
            TRAJ_BANK_DIR = cfg['pmetrics']['trajbank_dir']
        else:
            raise ValueError("Trajectory bank directory is not specified in the config file.")


        CLUSTER_PATH = TRAJ_BANK_DIR 
        clusters = np.load(CLUSTER_PATH)

    test_loader = WaymoPlanning(
                    root=cfg.data.root,
                    root_generated=cfg.data.root_generated_test,
                    spatial_range=[-48, 48, -48, 48, -3.15, 3.15],
                    voxel_size=[0.15, 0.15, 0.18],
                    use_generated_data=False,
                    planner_dimension=[320,464],
                    is_testing=True,
                    get_eval_metadata=True,
                    skip_overfield=True)
    

    total_count = test_loader.__len__()
    indices = [*range(0, total_count)]
    random.shuffle(indices)
    indices = indices[:TOTAL_TO_PROCESS]
    
    # Uncomment to use hardcoded reproducible indices
    #indices = np.loadtxt("../../_datasets/indices.txt").astype(np.int64).tolist()

    TOTAL_L1 = 0.0
    TOTAL_L2_1S = 0.0
    TOTAL_L2_2S = 0.0
    TOTAL_L2_3S = 0.0
    TOTAL_WAYPOINTS_PROCESSED = 0.0

    TOTAL_COLLISION = 0


    FLAG_CALC_COLLISION = True


    for index in indices:
        evaluation_item = test_loader.__getitem__(index)

        #gt_item = gt_loader.__getitem__(index)
        #gt_cpu = sample_to_torch(gt_item)
        #gt_sample = sample_to_cuda(gt_cpu)
        
        sample_cpu = sample_to_torch(evaluation_item)
        sample = sample_to_cuda(sample_cpu)

        predicted_plan_sample = model.forward(sample)
        predicted_plan_logit = predicted_plan_sample['predicted_plan'][0]

        if(FLAG_USE_TRAJ_BANK):
            predicted_waypoints_meters = logit_grid_to_waypoint_traj_bank(clusters, sample["predicted_plan"][0], apply_softmax=True)
            predicted_waypoints_pixel = [convert_pixel_meter_torch_v(p, pix2m=False) for p in predicted_waypoints_meters]
        else:
            predicted_waypoints_pixel = planning_logit_grid_to_waypoints(sample["predicted_plan"][0])
            predicted_waypoints_meters = [convert_pixel_meter_torch_v(p, pix2m=True) for p in predicted_waypoints_pixel]


        #Method 1
        #gt_waypoints_meters = logit_grid_to_waypoint_traj_bank(clusters, sample["plan"][0])
        
        #Method 2
        gt_plan_pixel = planning_logit_grid_to_waypoints(sample['plan'][0])
        gt_plan_pixel = [x.detach().cpu().numpy() for x in gt_plan_pixel]
        gt_coords = [convert_pixel_meter_torch(p, grid_0=sample['plan'].shape[2], grid_1=sample['plan'].shape[3], pix2m=True) for p in gt_plan_pixel]
        gt_waypoints_meters = gt_coords


        l1_values = [
            np.linalg.norm((gt_waypoints_meters[i] - predicted_waypoints_meters[i]), ord=1) for i in range(0, 10)
        ]

        l2_1s_value = np.linalg.norm(gt_waypoints_meters[2] - predicted_waypoints_meters[2], ord=2)
        l2_2s_value = np.linalg.norm(gt_waypoints_meters[5] - predicted_waypoints_meters[5], ord=2)
        l2_3s_value = np.linalg.norm(gt_waypoints_meters[9] - predicted_waypoints_meters[9], ord=2)

        future_maps = evaluation_item['future_maps']
        future_maps = np.transpose(np.stack(future_maps, axis=0), (0,2,3,1))


        #predicted_trajectory = [
        #                    (predicted_plan_logit[i]==torch.max(predicted_plan_logit[i])).nonzero().detach().cpu().numpy()
        #                    for i in range(0, 10)
        #                    ]
        

        if(FLAG_CALC_COLLISION):
            TOTAL_COLLISION += num_collision(predict_trajectory=predicted_waypoints_pixel, #gt_plan_pixel
                                         future_maps=future_maps,
                                         current_pose=evaluation_item['current_pose'],
                                         future_poses=evaluation_item['future_poses'],
                                         tn2t4 = evaluation_item['tn2t4'],
                                         viz=VIZ)

        if(VIZ):
            fig, ax = plt.subplots(figsize=(6, 6))
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            b = cond_to_rgb_waymo(sample["bev"][0].cpu().detach().numpy())
            ax.imshow(b)

            for points in gt_waypoints_meters:
                m_point = convert_pixel_meter_torch_v(points, pix2m=False)
                ax.plot((m_point[1]), (m_point[0]), 'o', markersize=8)

            for points in predicted_waypoints_meters:
                m_point = convert_pixel_meter_torch_v(points, pix2m=False)
                ax.plot((m_point[1]), (m_point[0]), '+', markersize=8)
            
            ax.set_xlim((0, 464))
            ax.set_ylim((0, 320))

            #fig.show()
            fig.savefig(f"out_{index}.png")
            #fig.close()
            plt.close(fig)
            
            cv = CostVisualizer("", "predicted_plan")
            image = cv.generate_visualization(predicted_plan_sample)
            plt.imshow(image)
            plt.savefig(f"out_{index}_cm.png")


        
        TOTAL_L2_1S += l2_1s_value
        TOTAL_L2_2S += l2_2s_value
        TOTAL_L2_3S += l2_3s_value


        for v in l1_values:
            TOTAL_L1 += v
            TOTAL_WAYPOINTS_PROCESSED += 1
    

    AVERAGE_L1 = TOTAL_L1 / TOTAL_WAYPOINTS_PROCESSED

    AVERAGE_L2_1S = TOTAL_L2_1S / TOTAL_TO_PROCESS
    AVERAGE_L2_2S = TOTAL_L2_2S / TOTAL_TO_PROCESS
    AVERAGE_L2_3S = TOTAL_L2_3S / TOTAL_TO_PROCESS

    COLLISION_RATE = TOTAL_COLLISION / TOTAL_WAYPOINTS_PROCESSED

    model_name = cfg['model']['pretrained']

    print(f'SAMPLE_COUNT: {TOTAL_TO_PROCESS}')
    print(f'USE_TRAJ_BANK: {FLAG_USE_TRAJ_BANK}')
    print(f'MODEL_USED: {model_name}')
    print(f'AVERAGE_L1: {AVERAGE_L1} | COLLISION_RATE: {COLLISION_RATE}')
    print(f'AVERAGE_L2_1sec: {AVERAGE_L2_1S} | AVERAGE_L2_2sec: {AVERAGE_L2_2S}| AVERAGE_L2_3sec: {AVERAGE_L2_3S}')

    
    
if __name__ == "__main__":
    main()
