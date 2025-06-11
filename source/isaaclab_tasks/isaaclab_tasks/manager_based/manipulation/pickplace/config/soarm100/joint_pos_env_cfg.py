# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.pickplace import mdp
from isaaclab_tasks.manager_based.manipulation.pickplace.config.soarm100.pickplace_env_cfg import PickPlaceEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.soarm100 import SOARM100_CFG  # isort: skip


@configclass
class SOARM100CubePickPlaceEnvCfg(PickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set SOARM100 as robot
        self.scene.robot = SOARM100_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (soarm100)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch","Wrist_Roll"], 
            scale=0.5, 
            use_default_offset=True,
        )
        self.actions.jaw_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["Jaw"],
            open_command_expr={"Jaw": 0.6}, #Fully open
            close_command_expr={"Jaw": 0.0}, #Fully closed
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = "Fixed_Jaw"
        self.commands.object_pose.debug_vis = False

        # Set Cube as object
        cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1, # CHANGE: try up to 10~20?
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )

        self.scene.upper_cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.4, 0.4, 0.4),
                rigid_props=cube_properties,
            ),
        )

        self.scene.lower_cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Target",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0, 0.06], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/block.usd",
                scale=(0.5, 0.5, 0.5),
                rigid_props=cube_properties,
            ),
        )
        
        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Base", # CHANGE whole joint set root position
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Fixed_Jaw",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[-0.003, -0.088, 0.0],
                    ),
                ),
            ],
        )


@configclass
class SOARM100CubePickPlaceEnvCfg_PLAY(SOARM100CubePickPlaceEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
