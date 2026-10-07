# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots import BUMI_CFG, BUMI_LINKS
from robolab.tasks.manager_based.beyondmimic.rpo_getup_mimic_env_cfg import RPOGetupMimicEnvCfg


@configclass
class BumiGetupMimicEnvCfg(RPOGetupMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = BUMI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.commands.motion.motion_file = os.path.join(
            ROBOLAB_ROOT_DIR, "data", "motions", "bumi_bm", "getup_supin2prone.npz"
        )
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = BUMI_LINKS
        self.commands.motion.motion_position_offset = (0.0, 0.0, 0.0)
        self.commands.motion.joint_position_limit_type = "hard"
        self.commands.motion.joint_position_range = (-0.03, 0.03)
        self.commands.motion.pose_range = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "z": (-0.003, 0.003),
            "roll": (-0.03, 0.03),
            "pitch": (-0.03, 0.03),
            "yaw": (-0.05, 0.05),
        }
        self.commands.motion.velocity_range = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.03, 0.03),
            "roll": (-0.10, 0.10),
            "pitch": (-0.10, 0.10),
            "yaw": (-0.15, 0.15),
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.events.base_com.params["asset_cfg"].body_names = ["base_link", "waist_yaw_link"]
        self.events.scale_link_mass.params["asset_cfg"].body_names = ["l_.*_link", "r_.*_link"]
        # Never push a just-reset supine/prone robot at t=0; late-episode pushes remain available.
        self.events.randomize_push_robot.interval_range_s = (3.0, 5.0)
