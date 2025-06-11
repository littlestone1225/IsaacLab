# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def objects_distance(
    env: ManagerBasedRLEnv,
    object1_cfg: SceneEntityCfg = SceneEntityCfg("upper_cube"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("lower_cube"),
) -> torch.Tensor:
    
    object1: RigidObject = env.scene[object1_cfg.name]
    object2: RigidObject = env.scene[object2_cfg.name]
    
    objects_dis = torch.norm(object1.data.root_pos_w[:, :3] - object2.data.root_pos_w[:, :3], dim=-1)

    return objects_dis

def objects_xy_distance(
    env: ManagerBasedRLEnv,
    object1_cfg: SceneEntityCfg = SceneEntityCfg("upper_cube"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("lower_cube"),
) -> torch.Tensor:
    
    object1: RigidObject = env.scene[object1_cfg.name]
    object2: RigidObject = env.scene[object2_cfg.name]
    
    pos_diff = object1.data.root_pos_w - object2.data.root_pos_w
    xy_dis = torch.norm(pos_diff[:, :2], dim=-1)  # XY-plane distance

    return xy_dis

def object_ee_distance(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("upper_cube"),
) -> torch.Tensor:
    
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    cube_pos_w = object.data.root_pos_w
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_dis = torch.norm(cube_pos_w - ee_w, dim=-1)

    return object_ee_dis

def object_is_picked(
    env: ManagerBasedRLEnv, 
    minimal_height: float, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("upper_cube")
) -> torch.Tensor:

    object: RigidObject = env.scene[object_cfg.name]
    #print(object.data.root_pos_w[:, 2].max(),object.data.root_pos_w[:, 2].min())
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def reach_upper_cube_reward(
    env: ManagerBasedRLEnv,
    std: float,
    object1_cfg: SceneEntityCfg = SceneEntityCfg("upper_cube"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("lower_cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:

    object_ee_dis = object_ee_distance(env, ee_frame_cfg, object1_cfg)

    cube_placed = place_upper_cube_reward(env, object1_cfg, object2_cfg).bool()

    reward = torch.where(cube_placed, 
                         (torch.tanh(object_ee_dis / std) - 1)*(10), 
                         (1 - torch.tanh(object_ee_dis / std)))
    return reward

def reach_upper_cube_reward_soarm100(
    env: ManagerBasedRLEnv,
    std: float = 0.1,
    xy_threshold: float = 0.02, 
    height_threshold: float = 0.005, 
    height_diff: float = 0.0234,
    object1_cfg: SceneEntityCfg = SceneEntityCfg("upper_cube"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("lower_cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:

    object_ee_dis = object_ee_distance(env, ee_frame_cfg, object1_cfg)

    cube_placed = place_upper_cube_reward(env, object1_cfg, object2_cfg,xy_threshold, height_threshold,height_diff).bool()

    reward = torch.where(cube_placed, 
                         (torch.tanh(object_ee_dis / std) - 1)*(10), 
                         (1 - torch.tanh(object_ee_dis / std)))

    return reward

def pick_upper_cube_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object1_cfg: SceneEntityCfg = SceneEntityCfg("upper_cube"),
) -> torch.Tensor:
    
    cube_picked = object_is_picked(env, minimal_height, object1_cfg)
    
    return cube_picked

def cubes_distance_reward(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object1_cfg: SceneEntityCfg = SceneEntityCfg("upper_cube"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("lower_cube"),
) -> torch.Tensor:  
    
    cubes_dis = objects_distance(env, object1_cfg, object2_cfg)
    cube_picked = object_is_picked(env, minimal_height, object1_cfg)
    return cube_picked * (1 - torch.tanh(cubes_dis / std))

def cubes_xy_distance_reward(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    object1_cfg: SceneEntityCfg = SceneEntityCfg("upper_cube"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("lower_cube"),
) -> torch.Tensor:  
    
    xy_dis = objects_xy_distance(env, object1_cfg, object2_cfg)
    cube_picked = object_is_picked(env, minimal_height, object1_cfg)
    return cube_picked * (1 - torch.tanh(xy_dis / std))

def place_upper_cube_reward(
    env: ManagerBasedRLEnv,
    object1_cfg, 
    object2_cfg,
    xy_threshold: float=0.05, 
    height_threshold: float=0.01, 
    height_diff: float = 0.0468,
) -> torch.Tensor:

    # Get objects
    object1 = env.scene[object1_cfg.name]
    object2 = env.scene[object2_cfg.name]

    # Compute distances
    pos_diff = object1.data.root_pos_w - object2.data.root_pos_w
    
    xy_dist = torch.norm(pos_diff[:, :2], dim=-1)  # XY-plane distance
    height_dist = torch.abs(pos_diff[:, 2] - height_diff)  # Expected stacking height

    # Check if cube is correctly placed
    cube_placed = torch.where(torch.logical_and(xy_dist < xy_threshold, height_dist < height_threshold),1.0,0.0)

    return cube_placed


def release_upper_cube_reward(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    dis_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object1_cfg: SceneEntityCfg = SceneEntityCfg("upper_cube"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("lower_cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:

    robot = env.scene[robot_cfg.name]    

    # Check if the gripper is open
    gripper_open = torch.where(robot.data.joint_pos[:, -1]+robot.data.joint_pos[:, -2]> 0.065, 1.0, 0.0)

    dis = objects_distance(env, object1_cfg, object2_cfg)
    cubes_closed = torch.where(dis < dis_threshold, 1.0, 0.0)

    cubes_picked = object_is_picked(env, minimal_height, object1_cfg)

    reward = torch.where(
        torch.logical_and(torch.logical_and(cubes_closed, cubes_picked), gripper_open), 
        torch.tanh(object_ee_distance(env, ee_frame_cfg, object2_cfg) / 0.3) + 1.0, 
        0.0
    )

    return reward


def release_upper_cube_reward_soarm100(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    dis_threshold: float,
    std: float = 0.1,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object1_cfg: SceneEntityCfg = SceneEntityCfg("upper_cube"),
    object2_cfg: SceneEntityCfg = SceneEntityCfg("lower_cube"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:

    robot = env.scene[robot_cfg.name]    

    # Check if the gripper is open
    jaw_open = torch.where(robot.data.joint_pos[:, -1]> 0.6, 1.0, 0.0)

    dis = objects_distance(env, object1_cfg, object2_cfg)
    cubes_closed = torch.where(dis < dis_threshold, 1.0, 0.0)
    #print(dis,env.scene[object1_cfg.name].data.root_pos_w[:, 2])
    cubes_picked = object_is_picked(env, minimal_height, object1_cfg)
    #print(cubes_closed,cubes_picked,jaw_open)
    reward = torch.where(
        torch.logical_and(torch.logical_and(cubes_closed, cubes_picked), jaw_open), 
        torch.tanh(object_ee_distance(env, ee_frame_cfg, object2_cfg) / std) + 1.0, 
        0.0
    )

    return reward

def object_move_velocity(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("lower_cube"),
    velocity_threshold=0.05,
    angular_threshold=0.05
)-> torch.Tensor:

    # Get the object handle from the environment
    object: RigidObject = env.scene[object_cfg.name]

    # Retrieve the object's linear and angular velocity
    linear_velocity = torch.norm(object.data.root_lin_vel_w, p=2)
    angular_velocity = torch.norm(object.data.root_ang_vel_w, p=2)

    # Check if the velocities are below the threshold
    moved_velocity = linear_velocity + angular_velocity

    return torch.where(
        (linear_velocity < velocity_threshold) and (angular_velocity < angular_threshold), # Check if the velocities are below the threshold
        0.0, 
        moved_velocity
    )


def object_angular_velocity(
    env: ManagerBasedRLEnv, 
    object_cfg: SceneEntityCfg = SceneEntityCfg("upper_cube"),
    angular_threshold=0.05
)-> torch.Tensor:

    # Get the object handle from the environment
    object: RigidObject = env.scene[object_cfg.name]

    # Retrieve the object's linear and angular velocity
    angular_velocity = torch.norm(object.data.root_ang_vel_w, p=2)


    return torch.where(
        (angular_velocity < angular_threshold), # Check if the velocities are below the threshold
        0.0, 
        angular_velocity
    )