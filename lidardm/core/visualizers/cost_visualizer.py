from typing import Dict
from .visualizer import * 
from typing import Any, Dict, Optional, Tuple
import matplotlib.pyplot as plt
from lidardm.visualization.cond2rgb import cond_to_rgb_waymo
from lidardm.planning.utils import * 
import numpy as np

__all__ = ["CostVisualizer"]

class CostVisualizer(Visualizer):
    def __init__(self, output_key, plan="plan"):
        super().__init__(output_key)
        self.plan_key = plan
    
    def supports_visualization(self, data: Dict[str, Any]) -> bool:
        return (self.plan_key in data)

    def generate_visualization(self, data: Dict[str, Any]) -> bool:
        
        b = data[self.plan_key].shape[0]
        c = data[self.plan_key].shape[1]
        w = data[self.plan_key].shape[2]
        h = data[self.plan_key].shape[3]

        logits = data[self.plan_key][0]
        logits = logits.reshape((c, w*h))
        probs = torch.nn.functional.softmax(logits, dim=1)
        probs = probs.reshape((c, w, h))

        fig, axs = plt.subplots(2, 5, figsize=(h/25, w/25))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)



        #
        # Specific code
        #

        for i in range(0, 10):
            axs_0 = i // 5
            axs_1 = i % 5


            b = probs[i].float().cpu().detach().numpy()

            axs[axs_0][axs_1].imshow(b)

        #
        # End. 
        #
        
        
        
        fig.canvas.draw()
        image_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        return image_array[:,:,0:3]
        