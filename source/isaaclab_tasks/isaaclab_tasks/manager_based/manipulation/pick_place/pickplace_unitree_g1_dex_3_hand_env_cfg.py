# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import tempfile
import torch

import carb
from pink.tasks import FrameTask

import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
#from isaaclab.devices.openxr.retargeters.humanoid.unitree.inspire.g1_upper_body_retargeter import UnitreeG1RetargeterCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_retargeter import (
    G1TriHandUpperBodyRetargeterCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from . import mdp

#from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG  # isort: skip
from isaaclab_assets.robots.unitree import G1_29DOF_CFG #g1_29dof_rev_1_0 is the same robot as g1_29dof, but with different initial pose. We will use the same initial pose for both arms in this environment, so we can use either one of the two configs.

##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.35, 0.45, 0.9996], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"/home/omniverse-2/Documents/Humanoid/AGCO_Coop/Assets/AGCO_KOERBE_usd/Korb_5er_ablage.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=MassPropertiesCfg(
                mass=0.15, #Masse möglichst noch randomisieren in einem realistischen Bereich, damit die Datenvarianz erhöht wird.
            ),
        ),
    )
    
     #pos & rotation should be randomized in scene reset to add variety in the dataset, 
     #but we need a reasonable default value for them to spawn in the right place in the
     #sim and be visible to the cameras.

    # Humanoid robot w/ arms higher
    robot: ArticulationCfg = G1_29DOF_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 1.0),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_joint": 0.0,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_joint": 0.0,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                # -- left/right hand (G1: thumb, index, middle only)
                "*_thumb_*": 0.0,
                "*_index_*": 0.0,
                "*_middle_*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


    # Stereo head cameras for visuomotor dataset collection
    head_left_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RobotHeadLeftCam",
        update_period=0.0,
        height=160,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2.0)),
        offset=CameraCfg.OffsetCfg(pos=(-0.03, 0.12, 1.67675), rot=(-0.19848, 0.9801, 0.0, 0.0), convention="ros"),
    )
    
    head_right_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RobotHeadRightCam",
        update_period=0.0,
        height=160,
        width=256,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 2.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.03, 0.12, 1.67675), rot=(-0.19848, 0.9801, 0.0, 0.0), convention="ros"),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pink_ik_cfg = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_joint",
            ".*_wrist_yaw_joint",
            ".*_wrist_roll_joint",
            ".*_wrist_pitch_joint",
        ],
        hand_joint_names=[
            # G1: Only thumb, index, middle (7 joints per hand)
            "left_hand_thumb_0_joint",  # Thumb base (yaw axis)
            "left_hand_thumb_1_joint",  # Thumb middle (pitch axis)
            "left_hand_thumb_2_joint",  # Thumb tip
            "left_hand_index_0_joint",  # Index finger proximal
            "left_hand_index_1_joint",  # Index finger distal
            "left_hand_middle_0_joint",  # Middle finger proximal
            "left_hand_middle_1_joint",  # Middle finger distal
            "right_hand_thumb_0_joint",  # Thumb base (yaw axis)
            "right_hand_thumb_1_joint",  # Thumb middle (pitch axis)
            "right_hand_thumb_2_joint",  # Thumb tip
            "right_hand_index_0_joint",  # Index finger proximal
            "right_hand_index_1_joint",  # Index finger distal
            "right_hand_middle_0_joint",  # Middle finger proximal
            "right_hand_middle_1_joint",  # Middle finger distal
        ],
        target_eef_link_names={
            "left_wrist": "left_wrist_yaw_link",
            "right_wrist": "right_wrist_yaw_link",
        },
        # the robot in the sim scene we are controlling
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=14,
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    "g1_29dof_rev_1_0_left_wrist_yaw_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=2.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                FrameTask(
                    "g1_29dof_rev_1_0_right_wrist_yaw_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=2.0,  # [cost] / [rad]
                    lm_damping=10,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "g1_29dof_rev_1_0_left_wrist_yaw_link",
                        "g1_29dof_rev_1_0_right_wrist_yaw_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                    gain=0.3,
                ),
            ],
            fixed_input_tasks=[],
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
        ),
        enable_gravity_compensation=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "left_wrist_yaw_link"})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "left_wrist_yaw_link"})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "right_wrist_yaw_link"})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "right_wrist_yaw_link"})

        hand_joint_state = ObsTerm(func=mdp.get_robot_joint_state, params={"joint_names": ["R_.*", "L_.*"]})

        head_left_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("head_left_cam"), "data_type": "rgb", "normalize": False},
        )
        head_right_cam = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("head_right_cam"), "data_type": "rgb", "normalize": False},
        )

        object = ObsTerm(
            func=mdp.object_obs,
            params={"left_eef_link_name": "left_wrist_yaw_link", "right_eef_link_name": "right_wrist_yaw_link"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=mdp.task_done_pick_place, params={"task_link_name": "right_wrist_yaw_link"})


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickPlaceG1Dex3FTPEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the custom Gitterkorb environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Position of the XR anchor in the world frame
    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, 0.0), #modify if needed(height missmatch between real and sim causes issues for hand tracking)
        anchor_rot=(1.0, 0.0, 0.0, 0.0),
    )

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Idle action to hold robot in default pose
    # Action format: [left arm pos (3), left arm quat (4), right arm pos (3), right arm quat (4),
    #                 left hand joint pos (12), right hand joint pos (12)]
    idle_action = torch.tensor([
        # 14 hand joints for EEF control
        -0.1487,
        0.2038,
        1.0952,
        0.707,
        0.0,
        0.0,
        0.707,
        0.1487,
        0.2038,
        1.0952,
        0.707,
        0.0,
        0.0,
        0.707,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ])

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 2

        # Set settings for camera rendering
        self.num_rerenders_on_reset = 3
        self.sim.render.antialiasing_mode = "DLAA"
        self.image_obs_list = ["head_left_cam", "head_right_cam"]

        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.pink_ik_cfg.controller.urdf_path = temp_urdf_output_path
        self.actions.pink_ik_cfg.controller.mesh_path = temp_urdf_meshes_output_path

        self.teleop_devices = DevicesCfg(
            devices={
                "handtracking": OpenXRDeviceCfg(
                    retargeters=[
                        G1TriHandUpperBodyRetargeterCfg(
                            enable_visualization=True,
                            # OpenXR hand tracking has 26 joints per hand
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            # Please confirm that self.actions.pink_ik_cfg.hand_joint_names is consistent with robot.joint_names[-14:]
                            # The order of the joints does matter as it will be used for converting pink_ik actions to final control actions in IsaacLab.
                            hand_joint_names=self.actions.pink_ik_cfg.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
                "manusvive": ManusViveCfg(
                    retargeters=[
                        G1TriHandUpperBodyRetargeterCfg(
                            enable_visualization=True,
                            num_open_xr_hand_joints=2 * 26,
                            sim_device=self.sim.device,
                            hand_joint_names=self.actions.pink_ik_cfg.hand_joint_names,
                        ),
                    ],
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                ),
            },
        )
