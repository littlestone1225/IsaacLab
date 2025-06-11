# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from ... import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the pick and place scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    upper_cube: RigidObjectCfg | DeformableObjectCfg = MISSING
    # CHANGE target place:
    lower_cube: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(7.0, 7.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object = ObsTerm(func=mdp.object_obs)
        cube_positions = ObsTerm(func=mdp.cube_positions_in_world_frame)
        cube_orientations = ObsTerm(func=mdp.cube_orientations_in_world_frame)
        eef_pos = ObsTerm(func=mdp.ee_frame_pos)
        eef_quat = ObsTerm(func=mdp.ee_frame_quat)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    # CHANGE
    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        is_grasped = ObsTerm(func=mdp.object_grasped,
                             params={"robot_cfg": SceneEntityCfg("robot"), 
                                     "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                                     "object_cfg": SceneEntityCfg("upper_cube")})
        is_stacked = ObsTerm(func=mdp.object_stacked,
                             params={"robot_cfg": SceneEntityCfg("robot"), 
                                     "upper_object_cfg":SceneEntityCfg("upper_cube"), 
                                     "lower_object_cfg":SceneEntityCfg("lower_cube"),}) 


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg() 


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.25, 0.25), "z": (0.0, 0.0)}, # for franka
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("upper_cube"),
        },
    )
    # CHANGE
    reset_target_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.25, 0.25), "z": (0.0, 0.0)}, # for franka
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("lower_cube"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # CHANGE
    reaching_object = RewTerm(
        func=mdp.reach_upper_cube_reward, 
        params={"std": 0.3,
                "ee_frame_cfg": SceneEntityCfg("ee_frame"), 
                "object1_cfg": SceneEntityCfg("upper_cube"),
                "object2_cfg": SceneEntityCfg("lower_cube"),
                },
        weight=1.0)
    
    picking_object = RewTerm(
        func=mdp.pick_upper_cube_reward,
        params={"minimal_height": 0.04, 
                "object1_cfg": SceneEntityCfg("upper_cube"),
                },
        weight=5.0,
    )
    
    objects_distance = RewTerm(
        func=mdp.cubes_distance_reward, 
        params={"std": 0.3,
                "minimal_height": 0.04, # desktop height
                "object1_cfg": SceneEntityCfg("upper_cube"),
                "object2_cfg": SceneEntityCfg("lower_cube"),}, 
        weight=10.0)
    
    objects_xy_distance = RewTerm(
        func=mdp.cubes_xy_distance_reward, 
        params={"std": 0.3,
                "minimal_height": 0.04, # desktop height
                "object1_cfg": SceneEntityCfg("upper_cube"),
                "object2_cfg": SceneEntityCfg("lower_cube"),}, 
        weight=2.0)
    

    released_object = RewTerm(
        func=mdp.release_upper_cube_reward,
        params={"minimal_height": 0.04,
                "dis_threshold": 0.1,
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"), 
                "object1_cfg": SceneEntityCfg("upper_cube"),
                "object2_cfg": SceneEntityCfg("lower_cube")
                },
        weight=50.0,
    )

    placed_object = RewTerm(
        func=mdp.place_upper_cube_reward,
        params={"object1_cfg": SceneEntityCfg("upper_cube"),
                "object2_cfg": SceneEntityCfg("lower_cube")
                },
        weight=10.0,
    )


    # lower cube move penalty
    bumping_lower_cube = RewTerm(
        func=mdp.object_move_velocity,
        weight=-1e-4,
        params={"object_cfg": SceneEntityCfg("lower_cube")}
    )

    # action penalty
    action_rate = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-1e-4
    )

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    upper_cube_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("upper_cube")}
    )

    # CHANGE
    lower_cube_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("lower_cube")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 15000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 15000}
    )


##
# Environment configuration
##


@configclass
class PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the pick and place environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01 # CHANGE: WHY SAME bounce_threshold_velocity setting?
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
