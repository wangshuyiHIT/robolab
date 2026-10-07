# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from robolab.tasks.manager_based.beyondmimic.agents.rpo_beyondmimic_agent_cfg import RPOBeyondMimicPPORunnerCfg


@configclass
class BumiBeyondMimicPPORunnerCfg(RPOBeyondMimicPPORunnerCfg):
    experiment_name = "bumi_beyondmimic"
    wandb_project = "bumi_beyondmimic"
