"""

State lattice planner with model predictive trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

- plookuptable.csv is generated with this script:
https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning
/ModelPredictiveTrajectoryGenerator/lookup_table_generator.py

Ref:

- State Space Sampling of Feasible Motions for High-Performance Mobile Robot
Navigation in Complex Environments
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.187.8210&rep=rep1
&type=pdf

"""
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import math
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent))

import trajectory_generator as planner
import motion_model as motion_model

TABLE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/lookup_table.csv"

def search_nearest_one_from_lookup_table(t_x, t_y, t_yaw, lookup_table):
    mind = float("inf")
    minid = -1

    for (i, table) in enumerate(lookup_table):
        dx = t_x - table[0]
        dy = t_y - table[1]
        dyaw = t_yaw - table[2]
        d = math.sqrt(dx ** 2 + dy ** 2 + dyaw ** 2)
        if d <= mind:
            minid = i
            mind = d

    return lookup_table[minid]


def get_lookup_table(table_path):
    return np.loadtxt(table_path, delimiter=',', skiprows=1)


def generate_path(target_states, k0):
    # x, y, yaw, s, km, kf
    lookup_table = get_lookup_table(TABLE_PATH)
    result = []

    for state in target_states:
        bestp = search_nearest_one_from_lookup_table(
            state[0], state[1], state[2], lookup_table)

        target = motion_model.State(x=state[0], y=state[1], yaw=state[2])
        init_p = np.array(
            [np.hypot(state[0], state[1]), bestp[4], bestp[5]]).reshape(3, 1)

        x, y, yaw, p = planner.optimize_trajectory(target, k0, init_p)

        if x is not None:
            print("find good path")
            result.append(
                [x[-1], y[-1], yaw[-1], float(p[0, 0]), float(p[1, 0]), float(p[2, 0])])

    print("finish path generation")
    return result

def get_path(dest, length=-1):
    states = [dest]
    result = generate_path(states, k0=0.0)[0]

    xc, yc, yawc = motion_model.generate_trajectory(
        result[3], result[4], result[5], k0=0.0)
    
    if length==-1:
        return [xc, yc, yawc]
        
    xs   = [xc[i] for i in range(0, len(xc), (len(xc)-1)//length)]
    ys   = [yc[i] for i in range(0, len(yc), (len(yc)-1)//length)]
    yaws = [yawc[i] for i in range(0, len(yc), (len(yc)-1)//length)]

    return [xs, ys, yaws]

def pad_sequence(xs, ys, yaws, target_yaw, length, speed):
    assert len(xs) == len(ys) and len(ys) == len(yaws)

    remaining = length - len(xs)
    
    for _ in range(remaining):
        xs.append(xs[-1] + np.cos(target_yaw) * speed)
        ys.append(ys[-1] + np.sin(target_yaw) * speed)
        yaws.append(target_yaw)
    return xs, ys, yaws
