
import os
from glob import glob
from typing import Any, Dict, Tuple
from lidardm.core.datasets.utils import encode, decode

import numpy as np
import random
from torch.utils.data import Dataset
from pathlib import PurePath

from pathlib import PurePath
from .utils import voxelize
from natsort import natsorted
from scipy.ndimage import gaussian_filter

from PIL import Image

__all__ = ["WaymoPlanning", "WaymoPlanningVal"]

class WaymoPlanning(Dataset):
    def __init__(
        self,
        root: str,
        root_generated: str,
        spatial_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float],

        planner_dimension: Tuple[float, float],

        use_generated_data: bool,
        is_testing = False,

        plan_clusters_file = None,

        overfit=False,

        get_eval_metadata=False,

        percentage_to_use=1.0,

        skip_overfield=False
    ) -> None:
        self.spatial_range = spatial_range
        self.voxel_size = voxel_size
        self.root = root
        self.root_generated = root_generated
        self.return_dynamic = True
        self.use_generated_data = use_generated_data

        self.planner_dimension = planner_dimension
        
        self.gen_paths = natsorted(glob(os.path.join(self.root_generated, "out_*", "raycast", "2.npy")))
        self.gen_paths = [self.get_path_above(p) for p in self.gen_paths]
        self.gt_path = os.path.join(self.root, "training" if not is_testing else "validation")

        if(plan_clusters_file is None):
            self.plan_clusters = None
        else:
            self.plan_clusters = np.load(plan_clusters_file)

        self.overfit = overfit

        self.get_eval_metadata = get_eval_metadata

        self.percentage_to_use = percentage_to_use
        self.skip_overfield = skip_overfield

    def load_from_filename(self, filename: str):
        lidar = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        lidar = lidar[:,:3]

        return lidar
    
        
    def increment_filename(self, filename, inc):
        s = filename

        pure_path = PurePath(filename)

        new_path = pure_path.with_stem(str(int(pure_path.parts[-1].split('.')[0])+inc))

        
        return str(new_path.as_posix())

    def get_idx(self, filename):
        pure_path = PurePath(filename)
        return int(pure_path.parts[-1].split('.')[0])

    def get_root(self, filename):
        pure_path = PurePath(filename)
        return os.path.join(*pure_path.parts[:-2])

    def get_path_above(self, filename):
        pure_path = PurePath(filename)
        return os.path.join(*pure_path.parts[:-1])
    
    def lidar_to_homogenous(self, lidar):
        return np.concatenate((lidar, np.ones((lidar.shape[0], 1))), axis=1)
    

    def load_map(self, map_filename):
        bev = decode(Image.open(map_filename), 13)
        
        bev[:,:,5:9] = ((bev[:,:,5:9] + bev[:,:,9:]) > 0).astype(bev.dtype)
        bev = bev[:,:,:9]

        bev = np.transpose(bev, axes=(2, 0, 1))
        bev = np.rot90(bev, k=2, axes=(1,2))

        return bev

    def __getitem__(self, index: int) -> Dict[str, Any]:
        
        if(self.overfit):
            index = 1000
        
        if(self.percentage_to_use != 1.0):
            index = index % int(self.__len__() * self.percentage_to_use)

        gen_filename = self.gen_paths[index]
        
        manifest = np.genfromtxt(os.path.join(self.get_path_above(gen_filename), 'manifest.txt'), delimiter=", ", dtype='str')
        seq_id, lidar_idx = str(manifest[0]), int(manifest[1])


        
        #
        # Settings
        #

        OFFSETS_LIDAR = [-2, -1, 0, 1, 2]
        #OFFSETS_LIDAR = [-20, -10, 0, 10, 20]
        PAST_POINT_VIZ = False
        OFFSET_FUTURE_POINTS = list(range(lidar_idx+2+3, lidar_idx+5+30, 3)) #min(all_transforms.shape[0], lidar_idx+30)

        



        gen_lidar = [os.path.join(gen_filename, "0.npy"),
                     os.path.join(gen_filename, "1.npy"),
                     os.path.join(gen_filename, "2.npy"),
                     os.path.join(gen_filename, "3.npy"),
                     os.path.join(gen_filename, "4.npy")
                ]
        
        gt_dir = os.path.join(self.gt_path, seq_id)
        gt_lidar = [
                    os.path.join(gt_dir, "lidar_scan_raw", f"{lidar_idx+OFFSETS_LIDAR[0]}.npy"),
                    os.path.join(gt_dir, "lidar_scan_raw", f"{lidar_idx+OFFSETS_LIDAR[1]}.npy"),
                    os.path.join(gt_dir, "lidar_scan_raw", f"{lidar_idx+OFFSETS_LIDAR[2]}.npy"),
                    os.path.join(gt_dir, "lidar_scan_raw", f"{lidar_idx+OFFSETS_LIDAR[3]}.npy"),
                    os.path.join(gt_dir, "lidar_scan_raw", f"{lidar_idx+OFFSETS_LIDAR[4]}.npy"),
        ]


        #
        # PLAN
        #
        all_transforms = np.loadtxt(os.path.join(gt_dir, 'poses.txt'))[:,1:].reshape(-1, 4,4)
        if(PAST_POINT_VIZ):
            current_transforms = [all_transforms[lidar_idx+OFFSETS_LIDAR[0]],
                                  all_transforms[lidar_idx+OFFSETS_LIDAR[1]],
                                  all_transforms[lidar_idx+OFFSETS_LIDAR[2]],
                                  all_transforms[lidar_idx+OFFSETS_LIDAR[3]],
                                  all_transforms[lidar_idx+OFFSETS_LIDAR[4]]]
        else:
            try:
                current_transforms = [all_transforms[i] for i in OFFSET_FUTURE_POINTS]
            except:
                if(self.overfit):
                    raise Exception("Overfit index fails")
                return self.__getitem__((index + random.randint(30, 10000))
                                         #%self.__len__())
                                         %int(self.__len__() * self.percentage_to_use))


        t4 = all_transforms[lidar_idx+OFFSETS_LIDAR[4]]#current_transforms[0]
        tn2t4_p = [np.linalg.inv(t4) @ tn for tn in current_transforms]
        car_loc = [t @ [0, 0, 0, 1] for t in tn2t4_p]

        car_loc_img = [320+((loc[:2] / self.voxel_size[0])) for loc in car_loc]

        #
        # Plan
        #

        #TODO: Generate in native resolution. 

        plan_img = np.zeros((640, 640, 10))
        for i in range(0, len(car_loc_img)):
            loc_img = car_loc_img[i]
            try:
                plan_img[int(loc_img[0]), int(loc_img[1]), i] = 1.0
            except:
                #plan_img[:,:,i] += 1.0 / (plan_img.shape[0] * plan_img.shape[1])
                continue
            #plan_img[:,:,i] = plan_img[:,:,i] / plan_img[:,:,i].sum()

            
        plan_img = np.rot90(plan_img, k=3, axes=(0,1))
        plan_img = plan_img.transpose((2, 0, 1))

        #
        # Negative Trajectories
        #
        N = 10
        negative_samples = None
        if(self.plan_clusters is not None):
            N_indices = [random.randint(0, self.plan_clusters.shape[0]-1) for _ in range(N)]
            negative_samples = self.plan_clusters[N_indices]



        
        #
        # Lidar
        #

        if(self.use_generated_data):
            lidar = [np.load(x) for x in gen_lidar]
        
            lidar_transforms = [all_transforms[lidar_idx+OFFSETS_LIDAR[0]],
                                all_transforms[lidar_idx+OFFSETS_LIDAR[1]],
                                all_transforms[lidar_idx+OFFSETS_LIDAR[2]],
                                all_transforms[lidar_idx+OFFSETS_LIDAR[3]],
                                all_transforms[lidar_idx+OFFSETS_LIDAR[4]]]
            tn2t4 = [np.linalg.inv(lidar_transforms[4]) @ lidar_transforms[2] for tn in lidar_transforms]
            lidar = [(tn2t4[i] @
                  self.lidar_to_homogenous(lidar[i]).T
                 ).T for i in range(0, 5)]
            
        else:
            lidar = [np.load(x) for x in gt_lidar]
        
            lidar_transforms = [all_transforms[lidar_idx+OFFSETS_LIDAR[0]],
                                all_transforms[lidar_idx+OFFSETS_LIDAR[1]],
                                all_transforms[lidar_idx+OFFSETS_LIDAR[2]],
                                all_transforms[lidar_idx+OFFSETS_LIDAR[3]],
                                all_transforms[lidar_idx+OFFSETS_LIDAR[4]]]
            tn2t4 = [np.linalg.inv(lidar_transforms[4]) @ tn for tn in lidar_transforms]
            lidar = [(tn2t4[i] @ 
                  self.lidar_to_homogenous(lidar[i]).T
                 ).T for i in range(0, 5)]

        voxelized = [voxelize(l[:, :3], spatial_range=self.spatial_range, voxel_size=self.voxel_size).transpose(2,0,1)
                     for l in lidar]
        
        voxelized = [np.rot90(v, k=3, axes=(1,2)) for v in voxelized]


        #
        # Map
        #

        map_filename = gt_lidar[4].replace("lidar_scan_raw", "map").replace(".npy", ".png")
        #bev = decode(Image.open(map_filename), 13)
        
        #bev[:,:,5:9] = ((bev[:,:,5:9] + bev[:,:,9:]) > 0).astype(bev.dtype)
        #bev = bev[:,:,:9]

        #bev = np.transpose(bev, axes=(2, 0, 1))
        #bev = np.rot90(bev, k=2, axes=(1,2))

        bev = self.load_map(map_filename)
        

        # Todo: Infer from ranges. 
        LIDARDM_SHAPE = 640

        h_lower = (LIDARDM_SHAPE - self.planner_dimension[0]) // 2
        h_higher = LIDARDM_SHAPE - h_lower

        w_lower = (LIDARDM_SHAPE - self.planner_dimension[1]) // 2
        w_higher = LIDARDM_SHAPE - w_lower


        lidar_seq = np.stack(voxelized, axis=0)[:, :,h_lower:h_higher, w_lower:w_higher].copy()

        cropped_plan = plan_img[:,h_lower:h_higher, w_lower:w_higher]
        for i in range(cropped_plan.shape[0]):
            if(cropped_plan[i].sum() > 0.1):
                cropped_plan[i] = gaussian_filter(cropped_plan[i], sigma=3)
            else:
                if(self.skip_overfield):    #TODO: Consider how points outside of map are handled. 
                    return self.__getitem__((index + random.randint(30, 10000)) % int(self.__len__() * self.percentage_to_use))
                
                cropped_plan[i] = (1.0 / float(cropped_plan.shape[1] * cropped_plan.shape[2]))
            
            cropped_plan[i] = cropped_plan[i] / cropped_plan[i].sum()

        res = {
            "lidar": lidar_seq[4],
            "lidar_seq": lidar_seq,
            "bev": bev[:,h_lower:h_higher, w_lower:w_higher].copy().astype(voxelized[0].dtype),
            "plan": cropped_plan.copy()
        }

        if(negative_samples is not None):
            res["negative_samples"] = negative_samples
        

        if(self.get_eval_metadata):
            
            res["tn2t4"] = tn2t4_p

            res["current_pose"] = t4 #current_transforms[0]
            res["future_poses"] = current_transforms#[]


            all_maps = []
            #for i in range(lidar_idx, min(all_transforms.shape[0], lidar_idx+10)):
                #current_transforms.append(all_transforms[i])
                #all_maps.append(self.get_map_idx(i)[:,160:480, 160:480])
            all_map_dir = [
                os.path.join(gt_dir, "map", f"{i}.png")
                for i in OFFSET_FUTURE_POINTS
            ]

            all_maps = [self.load_map(f) for f in all_map_dir]
            

            res["future_maps"] = [all_maps[i][:,h_lower:h_higher, w_lower:w_higher].copy().astype(voxelized[0].dtype) for i in range(0, len(all_maps))]

        return res 



    def __len__(self) -> int:
        return len(self.gen_paths)  # if self.split == "val" else 10


class WaymoPlanningVal(Dataset):
    def __init__(
        self,
        root: str,
        spatial_range: Tuple[float, float, float, float, float, float],
        voxel_size: Tuple[float, float, float]
    ) -> None:
        self.spatial_range = spatial_range
        self.voxel_size = voxel_size
        self.root = root
        self.return_dynamic = True

        self.fpaths = glob(os.path.join(self.root, 'validation', "*.tfrecord", "lidar_scan_raw", "*.npy"))
        
        self.fpaths = sorted(self.fpaths)
        fpaths_copy = self.fpaths.copy()
        sequence_paths = glob(os.path.join(self.root, 'validation', "*.tfrecord"))
        sequence = [PurePath(x).parts[-1] for x in sequence_paths]

        for seq_name in sequence:
            subsequence = natsorted([p for p in fpaths_copy if (seq_name in p)])
            for j in range(1, 10):
                self.fpaths.remove(subsequence[-j])

        self.gt_path = os.path.join(self.root, "validation")

    def load_from_filename(self, filename: str):
        lidar = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        lidar = lidar[:,:3]

        return lidar
    
        
    def increment_filename(self, filename, inc):
        s = filename

        pure_path = PurePath(filename)

        new_path = pure_path.with_stem(str(int(pure_path.parts[-1].split('.')[0])+inc))

        
        return str(new_path.as_posix())

    def get_idx(self, filename):
        pure_path = PurePath(filename)
        return int(pure_path.parts[-1].split('.')[0])

    def get_root(self, filename):
        pure_path = PurePath(filename)
        return os.path.join(*pure_path.parts[:-2])

    def get_path_above(self, filename):
        pure_path = PurePath(filename)
        return os.path.join(*pure_path.parts[:-1])

    def get_map_idx(self, index: int) -> np.ndarray:
        lidar_fn = self.fpaths[index]
        map_filename = lidar_fn.replace("lidar_scan_raw", "map").replace(".npy", ".png")
        bev = decode(Image.open(map_filename), 13)
        
        # bev[:,:,5:9] = ((bev[:,:,5:9] + bev[:,:,9:]) > 0).astype(bev.dtype)
        # bev = bev[:,:,:9]

        bev = np.transpose(bev, axes=(2, 0, 1))
        bev = np.rot90(bev, k=2, axes=(1,2))
        return bev

    def __getitem__(self, index: int) -> Dict[str, Any]:
        
        lidar_fn = self.fpaths[index]
        lidar_idx = self.get_idx(lidar_fn)

        seq_folder = self.get_root(lidar_fn)
        
        all_transforms = np.loadtxt(os.path.join(seq_folder, 'poses.txt'))[:,1:].reshape(-1, 4,4)

        current_transforms = []
        all_maps = []
        for i in range(lidar_idx, min(all_transforms.shape[0], lidar_idx+10)):
            current_transforms.append(all_transforms[i])
            all_maps.append(self.get_map_idx(i)[:,160:480, 160:480])

        t0 = current_transforms[0]
        tn2t0 = [np.linalg.inv(t0) @ tn for tn in current_transforms]
        car_loc = [t @ [0, 0, 0, 1] for t in tn2t0]

        car_loc_img = [320+((loc[:2] / self.voxel_size[0])) for loc in car_loc]

        if(np.linalg.norm(car_loc_img[0]-car_loc_img[-1]) < 5):
            return self.__getitem__((index + 31) % self.__len__())

        plan_img = np.zeros((640, 640, 10))
        for i in range(0, len(car_loc_img)):
            loc_img = car_loc_img[i]
            try:
                plan_img[int(loc_img[0]), int(loc_img[1]), i] = 1.0
            except:
                continue

        
        lidar = np.load(lidar_fn)

        voxelized = voxelize(lidar[:, :3], spatial_range=self.spatial_range, voxel_size=self.voxel_size).transpose(2,0,1)
        voxelized = np.rot90(voxelized, k=3, axes=(1,2))

        bev = all_maps[0]
        
        plan_img = np.rot90(plan_img, k=3, axes=(0,1))
        plan_img = plan_img.transpose((2, 0, 1))

        return {
            "lidar": voxelized[:,160:480, 160:480].copy(),
            "bev": bev.copy().astype(voxelized[0].dtype),
            "plan": plan_img[:, 160:480, 160:480].copy(),
            "current_pose": t0,
            "future_poses": current_transforms[1:],
            "future_maps": [all_maps[i].copy().astype(voxelized[0].dtype) for i in range(1, len(all_maps))]
        }

    def __len__(self) -> int:
        return len(self.fpaths)  # if self.split == "val" else 10
