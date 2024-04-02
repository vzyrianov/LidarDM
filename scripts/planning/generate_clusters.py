
from lidardm.planning.path_k_means import * 
from lidardm.core.datasets.waymo_planning import WaymoPlanning
import os
import argparse

def main():
    parser = argparse.ArgumentParser(prog="converter")
    parser.add_argument("--name")
    parser.add_argument("--data_count", type=int)
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--root_generated_dir", type=str)
    args = parser.parse_args()

    dataset = WaymoPlanning(root=args.root_dir,
                    root_generated=args.root_generated_dir,
                    spatial_range=[-48, 48, -48, 48, -3.15, 3.15],
                    voxel_size=[0.15, 0.15, 0.18],
                    planner_dimension=[640,640],
                    use_generated_data=False)


    plan_folder = "_planning"

    output_dir = os.path.join(plan_folder, args.name)
    os.makedirs(output_dir, exist_ok=True)


    if(args.data_count == 0):
        c = dataset.__len__()-1
    else:
        c = args.data_count
        
    all_paths = get_n_random(c, dataset)
    np.save(os.path.join(output_dir, "all_paths.npy"), all_paths)

    clusters = get_clusters(all_paths)
    np.save(os.path.join(output_dir, "clusters.npy"), clusters)


if __name__=="__main__":
    main()