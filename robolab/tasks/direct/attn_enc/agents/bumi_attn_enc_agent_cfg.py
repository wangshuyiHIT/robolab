# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from robolab.tasks.direct.attn_enc.agents.rpo_attn_enc_agent_cfg import (
    RslRlPpoEncActorCriticCfg,
    RslRlPpoEncAlgorithmCfg,
)
from robolab.tasks.direct.base import BaseAgentCfg


@configclass
class BumiAttnEncAgentCfg(BaseAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "bumi_attn_enc"
        self.wandb_project = "bumi_attn_enc"
        self.seed = 42
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"], "perception": ["perception_a", "perception_c"]}
        self.num_steps_per_env = 24
        self.max_iterations = 9001
        self.save_interval = 1000
        self.actor_obs_normalization = True
        self.critic_obs_normalization = True
        self.policy = RslRlPpoEncActorCriticCfg(
            class_name="ActorCriticAttnEnc",
            init_noise_std=1.0,
            noise_std_type="scalar",
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            embedding_dim=32,
            head_num=4,
            map_size=(17, 11),
            map_resolution=0.1,
            actor_history_length=5,
            critic_history_length=5,
            enable_critic_estimation=True,
            estimation_slice=[72, 73, 74],
            estimaiton_hidden_dims=[256, 64],
            enable_obs_encoder=True,
            latent_dim=32,
            obs_encoder_hidden_dims=[256, 128],
        )
        self.algorithm = RslRlPpoEncAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            enable_aux_loss=True,
            aux_loss_coef=0.05,
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=None,
            rnd_cfg=None,
        )
        self.clip_actions = 100.0
