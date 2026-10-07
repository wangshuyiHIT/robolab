# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from robolab.tasks.manager_based.beyondmimic.agents.rpo_getup_mimic_agent_cfg import RPOGetupMimicPPORunnerCfg


@configclass
class BumiGetupMimicPPORunnerCfg(RPOGetupMimicPPORunnerCfg):
    experiment_name = "bumi_getup_mimic"
    wandb_project = "bumi_getup_mimic"
