# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.utils import configclass

from robolab.tasks.direct.base import mdp
from robolab.tasks.direct.base.bumi_env_cfg import BumiFlatEnvCfg, BumiRewardCfg
from robolab.tasks.direct.interrupt.rpo_interrupt_env_cfg import InterruptCfg


@configclass
class BumiInterruptRewardCfg(BumiRewardCfg):
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])},
    )
    joint_deviation_interrupt = RewTerm(
        func=mdp.joint_deviation_interrupt,
        weight=-1.0,
        params={
            "asset_cfg1": SceneEntityCfg(
                "robot",
                joint_names=[".*_arm_roll.*", ".*_arm_yaw.*", ".*_elbow_pitch.*"],
            ),
            "asset_cfg2": SceneEntityCfg("robot", joint_names=[".*_arm_pitch.*"]),
            "weight1": 1.0,
            "weight2": 0.06,
        },
    )
    stand_still = RewTerm(
        func=mdp.stand_still_interrupt,
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
            "interrupt_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow_pitch.*"]),
            "pos_weight": 1.0,
            "vel_weight": 0.04,
        },
    )
    action_penalty = RewTerm(
        func=mdp.action_penalty_interrupt,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_arm.*", ".*_elbow_pitch.*"])},
    )


@configclass
class BumiInterruptEnvCfg(BumiFlatEnvCfg):
    reward = BumiInterruptRewardCfg()
    interrupt = InterruptCfg(
        use_interrupt=True,
        max_curriculum=1.0,
        interrupt_ratio=0.5,
        interrupt_joint_names=[
            "l_arm_pitch_joint",
            "l_arm_roll_joint",
            "l_arm_yaw_joint",
            "l_elbow_pitch_joint",
            "r_arm_pitch_joint",
            "r_arm_roll_joint",
            "r_arm_yaw_joint",
            "r_elbow_pitch_joint",
        ],
        interrupt_scale=[
            4.71,
            2.08,
            3.14,
            2.26,
            4.71,
            2.08,
            3.14,
            2.26,
        ],
        interrupt_lower_bound=[
            -3.14,
            -0.14,
            -1.57,
            -2.26,
            -3.14,
            -1.94,
            -1.57,
            -2.26,
        ],
        interrupt_init_range=0.2,
        interrupt_update_step=30,
        switch_prob=0.005,
    )
    interrupt_vis = InterruptCfg().interrupt_vis

    def __post_init__(self):
        super().__post_init__()
        self.action_space = 21
        self.observation_space = 73
        self.state_space = 130
