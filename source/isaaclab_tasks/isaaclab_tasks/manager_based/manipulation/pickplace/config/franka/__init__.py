# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-PickPlace-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubePickPlaceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PickPlaceCubePPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-Cube-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubePickPlaceEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:PickPlaceCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
