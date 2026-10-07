# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots import BUMI_CFG, BUMI_LINKS
from robolab.tasks.manager_based.beyondmimic.rpo_beyondmimic_env_cfg import RPOBeyondMimicEnvCfg


@configclass
class BumiBeyondMimicEnvCfg(RPOBeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BUMI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.motion.motion_file = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "bumi_bm", "yundong1.npz"
        )
        self.commands.motion.anchor_body_name = "waist_yaw_link"
        self.commands.motion.body_names = BUMI_LINKS
        # Bumi motions are generated from Bumi FK and grounded against the active collision geometry.
        self.commands.motion.motion_position_offset = (0.0, 0.0, 0.0)
        self.commands.motion.joint_position_limit_type = "hard"
        self.commands.motion.joint_position_range = (-0.04, 0.04)
        self.commands.motion.pose_range = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (-0.005, 0.005),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.10, 0.10),
        }
        self.commands.motion.velocity_range = {
            "x": (-0.15, 0.15),
            "y": (-0.15, 0.15),
            "z": (-0.05, 0.05),
            "roll": (-0.15, 0.15),
            "pitch": (-0.15, 0.15),
            "yaw": (-0.30, 0.30),
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_com.params["asset_cfg"].body_names = ["base_link", "waist_yaw_link"]
        self.events.scale_link_mass.params["asset_cfg"].body_names = ["l_.*_link", "r_.*_link"]
        self.events.randomize_push_robot.interval_range_s = (3.0, 5.0)
