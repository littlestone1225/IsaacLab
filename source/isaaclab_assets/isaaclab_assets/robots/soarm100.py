"""
Reference: https://github.com/TheRobotStudio/SO-ARM100/
motor detail: https://www.feetech.cn/525603.html
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##

SOARM100_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/SOARM100",
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:\\Users\\tiffany.shih\\littlestone\\IsaacLab\\assets\\USD\\so100\\so100.usd", # usd_path
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Rotation": 0.1,     # -1.57, 1.57
            "Pitch": 1.57,       # 0, 3.14  
            "Elbow": 1.57,       # 0, 3.14 
            "Wrist_Pitch": 0.78, # -1.57, 1.57 
            "Wrist_Roll": -0.1,  # -3.14, 3.14
            "Jaw": 0.78,         # 0, 1.57
        },
        joint_vel={".*": 0.0}, # set Initial joint velocities to 0
        pos=(0.0, 0.0, 0.0),  # Adjust if needed
        rot=(0.707, 0.0, 0.0, 0.707), # rotate Counterclockwise 90 degrees
        
    ),
    actuators={        
        # Shoulder rotation moves: ALL mass(~0.8kg total)
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit=1.9,
            velocity_limit=1.5, 
            stiffness=200.0,  # highest moves all mass
            damping=80.0, 
        ),
        # Shoulder pitch moves: Every except base(~0.65kg)
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit=1.9,
            velocity_limit=1.5,  
            stiffness=150,  # Slightly less than rotation
            damping=60.0, 
        ),
        # Elbow moves: lower arm, wrist, Jaw(~0.38kg)
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit=1.9,
            velocity_limit=1.5, 
            stiffness=100.0,  # reduced based on less mass
            damping=50.0,
        ),
        # Wrist pitch moves: wrist, Jaw(~0.24kg)
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit=1.9,
            velocity_limit=1.5,  
            stiffness=80.0,  # reduced for less mass
            damping=40.0,
        ),
        # Wrist roll moves: wrist, Jaw(~0.14kg)
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit=1.9,
            velocity_limit=1.5, 
            stiffness=50.0,  # Low less to move
            damping=30.0,
        ),
        # Jaw moves: Only Jaw(~0.034kg)
        "jaw": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            effort_limit=2.5,    # increased from 1.9 to 2.5 for stronger grip
            velocity_limit=0.2, 
            stiffness=150.0,     # increased from 25.0 to 60.0 for more reliable closing
            damping=50.0,       # increased from 10.0 to 20.0 for stability
        ),
    },
    soft_joint_pos_limit_factor=2.0,
)
