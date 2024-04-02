from typing import Dict
from .visualizer import * 
from typing import Any, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from lidardm.visualization.cond2rgb import cond_to_rgb_waymo
from lidardm.planning.utils import * 
import numpy as np

__all__ = ["PlanVisualizer"]

class PlanVisualizer(Visualizer):
    def __init__(self, output_key, bev="bev", lidar="lidar_seq", plan="plan", traj_bank_path=None):
        super().__init__(output_key)
        self.bev_key = bev
        self.lidar_key = lidar
        self.plan_key = plan

        if(traj_bank_path is not None):
            self.traj_bank = np.load(traj_bank_path)
        else:
            self.traj_bank = None
    
    def supports_visualization(self, data: Dict[str, Any]) -> bool:
        return ("bev" in data) and ("lidar" in data) and ("plan" in data)

    def generate_visualization(self, data: Dict[str, Any]) -> bool:
        
        w = data['lidar'].shape[2]
        h = data['lidar'].shape[3]

        fig, ax = plt.subplots(figsize=(h/100, w/100))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        #
        # Specific code
        #

        b = cond_to_rgb_waymo(data[self.bev_key][0].cpu().detach().numpy())

        ax.imshow(0.5*(np.stack(3*[(np.sum(np.sum(
            data[self.lidar_key][0].cpu().detach().numpy()[[4]]
            , 0),0)/4).astype(float)], axis=2)) + 0.5*b)

        if(self.traj_bank is None):
            waypoints = planning_logit_grid_to_waypoints(data[self.plan_key][0])
        else: 
            waypoints = logit_grid_to_waypoint_traj_bank_pixel(self.traj_bank, data[self.plan_key][0])

        for points in waypoints:
            ax.plot((points[1]), (points[0]), 'o', markersize=8)
        

        #
        # End. 
        #
        
        
        
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        return image_array[:,:,0:3]
        