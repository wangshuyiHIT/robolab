# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

from robolab.tasks.direct.attn_enc.rpo_attn_enc_env_cfg import AttnEncCfg
from robolab.tasks.direct.base import CommandRangesCfg, ROUGH_TERRAINS_CFG, SceneCfg, mdp
from robolab.tasks.direct.base.bumi_env_cfg import BumiFlatEnvCfg, BumiRewardCfg


@configclass
class BumiAttnEncRewardCfg(BumiRewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.25, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.05)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    feet_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*ankle_roll.*"]), "min": 0.10, "max": 0.36},
    )
    knee_distance = RewTerm(
        func=mdp.body_distance_y,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_knee.*"]), "min": 0.10, "max": 0.32},
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-0.2,
        params={
            "pos_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_arm.*", ".*_elbow.*", "waist_yaw_joint", ".*_leg.*", ".*_knee.*", ".*_ankle.*"],
            ),
            "vel_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_arm.*", ".*_elbow.*", "waist_yaw_joint", ".*_leg.*", ".*_knee.*", ".*_ankle.*"],
            ),
            "pos_weight": 0.0,
            "vel_weight": 0.04,
        },
    )
    undesired_foothold = RewTerm(
        func=mdp.undesired_foothold,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_roll.*"),
            "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
            "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
            "ankle_height": 0.04,
        },
    )


@configclass
class BumiAttnEncEnvCfg(BumiFlatEnvCfg):
    reward = BumiAttnEncRewardCfg()
    attn_enc = AttnEncCfg(use_attn_enc=True, vel_in_obs=False)

    def __post_init__(self):
        super().__post_init__()
        self.action_space = 21
        self.observation_space = 72
        self.state_space = 135
        self.scene_context.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene_context.height_scanner.enable_height_scan = True
        self.scene_context.height_scanner.enable_height_scan_actor = True
        self.scene_context.height_scanner.resolution = 0.1
        self.scene_context.height_scanner.size = (1.6, 1.0)
        self.scene = SceneCfg(
            config=self.scene_context,
            physics_dt=self.sim.dt,
            step_dt=self.decimation * self.sim.dt,
        )
        self.robot.actor_obs_history_length = 5
        self.robot.critic_obs_history_length = 5
        self.normalization.height_scan_offset = 0.75
        self.sim.physx.gpu_collision_stack_size = 2**29
        self.noise.noise_scales.lin_vel = 0.2
        self.noise.noise_scales.height_scan = 0.025
        self.commands.ranges = CommandRangesCfg(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.6, 0.6),
            ang_vel_z=(-1.57, 1.57),
            heading=(-math.pi, math.pi),
        )
