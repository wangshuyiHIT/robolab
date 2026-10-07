# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

from robolab.assets.robots import BUMI_CFG, BUMI_STANDING_ROOT_HEIGHT
from robolab.tasks.direct.base import (  # noqa:F401
    BaseEnvCfg,
    CommandRangesCfg,
    EventCfg,
    GRAVEL_TERRAINS_CFG,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    RobotCfg,
    ROUGH_TERRAINS_CFG,
    RewardCfg,
    SceneCfg,
    SceneContextCfg,
)
from robolab.tasks.direct.base import mdp


@configclass
class BumiRewardCfg(RewardCfg):
    # Stronger velocity tracking for straight locomotion.
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=3.0, params={"std": 0.35})
    track_lin_vel_x_exp = RewTerm(func=mdp.track_lin_vel_x_yaw_frame_exp, weight=1.5, params={"std": 0.30})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.2, params={"std": 0.4})
    # Explicit straight-line shaping: kill sideways drift / spinning when cmd is forward.
    lateral_drift_l2 = RewTerm(func=mdp.lateral_drift_l2, weight=-1.5, params={"cmd_y_threshold": 0.05})
    yaw_rate_l2_when_straight = RewTerm(
        func=mdp.yaw_rate_l2_when_straight, weight=-0.8, params={"cmd_yaw_threshold": 0.1}
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.12)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.06)
    energy = RewTerm(func=mdp.energy, weight=-8e-5)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5e-6)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-8e-3)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*ankle_roll.*).*")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # Bumi legs are ~0.73x RPO length; natural step period scales with sqrt of
    # leg length, so the target air time drops from 0.35 s to ~0.30 s.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.8,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"), "threshold": 0.30},
    )
    moving_single_support = RewTerm(
        func=mdp.moving_single_support,
        weight=0.35,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.8,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
        },
    )
    # Bumi weighs ~17.3 kg vs RPO ~33.8 kg; contact-force shaping scaled by ~0.5.
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "threshold": 150,
            "max_reward": 125,
        },
    )
    # Bumi default-pose feet separation is 0.199 m (RPO: 0.145 m); keep the band
    # around the natural stance instead of reusing RPO numbers.
    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "min": 0.16, "max": 0.32},
    )
    # Bumi default-pose knee separation is 0.200 m (RPO: 0.187 m).
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee.*"]), "min": 0.15, "max": 0.30},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-1.2,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    feet_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"])},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.45)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.08,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_leg_yaw.*", ".*_leg_roll.*"])},
    )
    hip_roll_yaw_pair_symmetry = RewTerm(
        func=mdp.paired_joints_mirror_deviation_l1,
        weight=-0.08,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "l_leg_roll_joint",
                    "r_leg_roll_joint",
                    "l_leg_yaw_joint",
                    "r_leg_yaw_joint",
                ],
                preserve_order=True,
            )
        },
    )
    # Penalize same-direction L/R bias while still allowing anti-phase gait/swing.
    leg_sagittal_pair_symmetry = RewTerm(
        func=mdp.paired_joints_mirror_deviation_l1,
        weight=-0.10,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "l_leg_pitch_joint",
                    "r_leg_pitch_joint",
                    "l_knee_pitch_joint",
                    "r_knee_pitch_joint",
                    "l_ankle_pitch_joint",
                    "r_ankle_pitch_joint",
                ],
                preserve_order=True,
            )
        },
    )
    arm_pair_symmetry = RewTerm(
        func=mdp.paired_joints_mirror_deviation_l1,
        weight=-0.20,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "l_arm_pitch_joint",
                    "r_arm_pitch_joint",
                    "l_arm_roll_joint",
                    "r_arm_roll_joint",
                    "l_arm_yaw_joint",
                    "r_arm_yaw_joint",
                    "l_elbow_pitch_joint",
                    "r_elbow_pitch_joint",
                ],
                preserve_order=True,
            )
        },
    )
    # Natural biped gait is contralateral (对角摆臂). Same-side arm/leg swing
    # (同手同脚) is penalized via the complementary error terms inside the reward.
    contralateral_arm_leg = RewTerm(
        func=mdp.contralateral_arm_leg_pitch,
        weight=0.8,
        params={
            "std": 0.35,
            "cmd_threshold": 0.15,
            "vel_weight": 0.15,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "l_arm_pitch_joint",
                    "r_arm_pitch_joint",
                    "l_leg_pitch_joint",
                    "r_leg_pitch_joint",
                ],
                preserve_order=True,
            ),
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist_yaw_joint", ".*_arm_roll.*", ".*_arm_yaw.*", ".*_elbow_pitch.*"],
            )
        },
    )
    # Keep arms near default only weakly so pitch can swing for contralateral gait.
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_arm_pitch.*"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_leg_pitch.*", ".*_knee_pitch.*", ".*_ankle_pitch.*", ".*_ankle_roll.*"],
            )
        },
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.05,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*ankle_roll.*"])},
    )
    upward = RewTerm(func=mdp.upward, weight=0.3)
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.05,
        params={
            "pos_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_arm.*", ".*_elbow.*", "waist_yaw_joint", ".*_leg.*", ".*_knee.*", ".*_ankle.*"],
            ),
            "vel_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_arm.*", ".*_elbow.*", "waist_yaw_joint", ".*_leg.*", ".*_knee.*", ".*_ankle.*"],
            ),
            "pos_weight": 1.0,
            "vel_weight": 0.04,
        },
    )
    # The active Bumi foot collision box spans z=[-0.05, -0.03] m in the
    # ankle-roll frame, so its sole is 0.05 m below the scanner origin.
    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=0.15,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll.*"),
            "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
            "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
            "ankle_height": 0.05,
            "threshold": 0.02,
        },
    )
    swing_feet_clearance = RewTerm(
        func=mdp.swing_feet_clearance,
        weight=0.35,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
            "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
            "ankle_height": 0.05,
            "target_height": 0.03,
        },
    )


@configclass
class BumiFlatEnvCfg(BaseEnvCfg):
    reward = BumiRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.action_space = 21
        self.observation_space = 72
        self.state_space = 129
        self.scene_context.robot = BUMI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene_context.robot.init_state.pos = (0.0, 0.0, BUMI_STANDING_ROOT_HEIGHT)
        self.scene_context.height_scanner.prim_body_name = "base_link"
        self.scene_context.left_feet_scanner_prim_body_name = "l_ankle_roll_link"
        self.scene_context.right_feet_scanner_prim_body_name = "r_ankle_roll_link"
        self.scene_context.feet_scanner_debug_vis = False
        self.scene_context.terrain_type = "generator"
        self.scene_context.terrain_generator = GRAVEL_TERRAINS_CFG
        self.scene_context.height_scanner.enable_height_scan = False
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt=self.sim.dt,
            step_dt=self.decimation * self.sim.dt,
        )
        self.robot.terminate_contacts_body_names = [
            "base_link",
            "waist_yaw_link",
            ".*_leg_pitch_link",
            ".*_leg_roll_link",
            ".*_leg_yaw_link",
            ".*_arm_.*_link",
            ".*_elbow_pitch_link",
        ]
        self.robot.feet_body_names = [".*ankle_roll.*"]
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link", "waist_yaw_link"]
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["base_link", "waist_yaw_link"]
        self.events.scale_link_mass.params["asset_cfg"].body_names = ["l_.*_link", "r_.*_link"]
        self.events.scale_actuator_gains.params["asset_cfg"].joint_names = [".*_joint"]
        self.events.scale_joint_parameters.params["asset_cfg"].joint_names = [".*_joint"]
        self.robot.action_scale = 0.25
        self.noise.noise_scales.joint_vel = 1.75
        self.noise.noise_scales.joint_pos = 0.03
        self.commands.rel_standing_envs = 0.03
        # Prefer mostly forward walking; keep small side/turn commands for robustness.
        self.commands.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.ranges.lin_vel_y = (-0.08, 0.08)
        self.commands.ranges.ang_vel_z = (-0.5, 0.5)
        # Keep resets close to the calibrated flat-foot stance. Large
        # multiplicative noise visibly distorts the crouch before control starts.
        self.events.reset_robot_joints.params["position_range"] = (0.95, 1.05)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.05, 0.05),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.2, 0.2),
        }
        self.events.push_robot.interval_range_s = (15.0, 20.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.25, 0.25),
            "y": (-0.25, 0.25),
            "z": (-0.1, 0.1),
            "roll": (-0.25, 0.25),
            "pitch": (-0.25, 0.25),
            "yaw": (-0.35, 0.35),
        }


@configclass
class BumiRoughEnvCfg(BumiFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.state_space = 316
        self.scene_context.height_scanner.enable_height_scan = True
        self.scene_context.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt=self.sim.dt,
            step_dt=self.decimation * self.sim.dt,
        )
        self.sim.physx.gpu_collision_stack_size = 2**29
        self.reward.ang_vel_xy_l2.weight = -0.05
        self.reward.lin_vel_z_l2.weight = -0.05
