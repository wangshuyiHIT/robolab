# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from functools import lru_cache

import torch
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from robolab.tasks.direct.base import BaseAgentCfg

# Bumi Flat single-frame policy obs:
# ang_vel(3)+gravity(3)+cmd(3)+q(21)+qd(21)+a(21)=72
# Critic adds: lin_vel(3)+feet_contact(2)+feet_force(6)+air_time(2)+feet_h(2)+qdd(21)+tau(21)=129
# Isaac joint order matches scripts/tools/retarget/rpo_to_bumi_motion.BUMI_DOF_NAMES.
_NUM_JOINTS = 21
_POLICY_FRAME_DIM = 72
_CRITIC_FRAME_DIM = 129
_HISTORY = 10

_JOINT_MIRROR_ORDER = [
    1, 0, 2, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19
]
_JOINT_MIRROR_SIGNS = [
    1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1
]


def _joint_mirror(start_idx: int):
    indices = [start_idx + i for i in _JOINT_MIRROR_ORDER]
    return indices, list(_JOINT_MIRROR_SIGNS)


_joint_pos_i, _joint_pos_s = _joint_mirror(9)
_joint_vel_i, _joint_vel_s = _joint_mirror(9 + _NUM_JOINTS)
_action_i, _action_s = _joint_mirror(9 + 2 * _NUM_JOINTS)
_joint_acc_i, _joint_acc_s = _joint_mirror(87)
_joint_tau_i, _joint_tau_s = _joint_mirror(108)

_policy_frame_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8] + _joint_pos_i + _joint_vel_i + _action_i
_policy_frame_signs = [-1, 1, -1, 1, -1, 1, 1, -1, -1] + _joint_pos_s + _joint_vel_s + _action_s

_critic_frame_indices = (
    _policy_frame_indices
    + [72, 73, 74, 76, 75, 80, 81, 82, 77, 78, 79, 84, 83, 86, 85]
    + _joint_acc_i
    + _joint_tau_i
)
_critic_frame_signs = (
    _policy_frame_signs
    + [1, -1, 1, 1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1, 1]
    + _joint_acc_s
    + _joint_tau_s
)

_policy_obs_mirror_indices = []
_critic_obs_mirror_indices = []
for frame in range(_HISTORY):
    p_off = frame * _POLICY_FRAME_DIM
    c_off = frame * _CRITIC_FRAME_DIM
    _policy_obs_mirror_indices.extend(idx + p_off for idx in _policy_frame_indices)
    _critic_obs_mirror_indices.extend(idx + c_off for idx in _critic_frame_indices)

_policy_obs_mirror_signs = _policy_frame_signs * _HISTORY
_critic_obs_mirror_signs = _critic_frame_signs * _HISTORY
_act_mirror_indices = list(_JOINT_MIRROR_ORDER)
_act_mirror_signs = list(_JOINT_MIRROR_SIGNS)


@lru_cache(maxsize=None)
def _signs(device_type_key, values):
    device, dtype = device_type_key
    return torch.tensor(values, device=device, dtype=dtype)


def mirror_policy_observation(policy_obs: torch.Tensor) -> torch.Tensor:
    mirrored = policy_obs[..., _policy_obs_mirror_indices]
    signs = torch.tensor(_policy_obs_mirror_signs, device=policy_obs.device, dtype=policy_obs.dtype)
    return mirrored * signs


def mirror_critic_observation(critic_obs: torch.Tensor) -> torch.Tensor:
    mirrored = critic_obs[..., _critic_obs_mirror_indices]
    signs = torch.tensor(_critic_obs_mirror_signs, device=critic_obs.device, dtype=critic_obs.dtype)
    return mirrored * signs


def mirror_actions(actions: torch.Tensor) -> torch.Tensor:
    mirrored = actions[..., _act_mirror_indices]
    signs = torch.tensor(_act_mirror_signs, device=actions.device, dtype=actions.dtype)
    return mirrored * signs


def data_augmentation_func(env, obs, actions):
    if obs is None:
        obs_aug = None
    else:
        obs_mirror = obs.clone()
        obs_mirror["policy"] = mirror_policy_observation(obs["policy"])
        if "critic" in obs.keys():
            obs_mirror["critic"] = mirror_critic_observation(obs["critic"])
        obs_aug = torch.cat([obs, obs_mirror], dim=0)
    if actions is None:
        actions_aug = None
    else:
        actions_aug = torch.cat((actions, mirror_actions(actions)), dim=0)
    return obs_aug, actions_aug


@configclass
class BumiFlatAgentCfg(BaseAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "bumi_flat"
        self.wandb_project = "bumi_flat"
        self.seed = 42
        self.num_steps_per_env = 24
        self.max_iterations = 9001
        self.save_interval = 1000
        self.actor_obs_normalization = True
        self.critic_obs_normalization = True
        self.algorithm = RslRlPpoAlgorithmCfg(
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
            normalize_advantage_per_mini_batch=False,
            symmetry_cfg=RslRlSymmetryCfg(
                use_data_augmentation=True,
                use_mirror_loss=True,
                mirror_loss_coeff=0.2,
                data_augmentation_func=data_augmentation_func,
            ),
            rnd_cfg=None,
        )
        self.clip_actions = 100.0


@configclass
class BumiRoughAgentCfg(BumiFlatAgentCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "bumi_rough"
        self.wandb_project = "bumi_rough"
        # Rough critic includes height scan; keep reward-side symmetry only for now.
        self.algorithm.symmetry_cfg = None
