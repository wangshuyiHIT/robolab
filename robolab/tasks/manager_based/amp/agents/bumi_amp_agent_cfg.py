# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from robolab.tasks.manager_based.amp.agents.rpo_amp_agent_cfg import (
    RslRlAmpCfg,
    RslRlOnPolicyRunnerAmpCfg,
    RslRlPpoAmpAlgorithmCfg,
)
from robolab.tasks.manager_based.amp.mdp.symmetry import bumi as bumi_symmetry
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg


@configclass
class BumiAmpRunnerCfg(RslRlOnPolicyRunnerAmpCfg):
    def __post_init__(self):
        self.experiment_name = "bumi_amp"
        self.wandb_project = "bumi_amp"
        self.algorithm = RslRlPpoAmpAlgorithmCfg(
            class_name="PPOAMP",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True,
                use_mirror_loss=True,
                mirror_loss_coeff=0.2,
                data_augmentation_func=bumi_symmetry.compute_symmetric_states,
            ),
            amp_cfg=RslRlAmpCfg(
                disc_obs_buffer_size=100,
                grad_penalty_scale=10.0,
                disc_trunk_weight_decay=1.0e-3,
                disc_linear_weight_decay=1.0e-1,
                disc_learning_rate=1.0e-4,
                disc_max_grad_norm=1.0,
                amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg(
                    hidden_dims=[1024, 512],
                    activation="elu",
                    style_reward_scale=1.5,
                    task_style_lerp=0.6,
                ),
                loss_type="LSGAN",
            ),
        )
