# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry helpers for Bumi AMP (21 DoF Isaac runtime order)."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

# Isaac runtime articulation order used by Bumi motion retarget / AMP:
#  0 l_leg_pitch,  1 r_leg_pitch,  2 waist_yaw,
#  3 l_leg_roll,   4 r_leg_roll,
#  5 l_arm_pitch,  6 r_arm_pitch,
#  7 l_leg_yaw,    8 r_leg_yaw,
#  9 l_arm_roll,  10 r_arm_roll,
# 11 l_knee,      12 r_knee,
# 13 l_arm_yaw,   14 r_arm_yaw,
# 15 l_ankle_pitch, 16 r_ankle_pitch,
# 17 l_elbow,     18 r_elbow,
# 19 l_ankle_roll, 20 r_ankle_roll
_NUM_JOINTS = 21
_LEFT_IDS = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19]
_RIGHT_IDS = [1, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# Coronal / transverse joints flip sign under a sagittal-plane mirror.
_FLIP_SIGN_IDS = [2, 3, 4, 7, 8, 9, 10, 13, 14, 19, 20]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        obs_aug["policy"][batch_size : 2 * batch_size] = _transform_policy_obs_left_right(env, obs["policy"])
        obs_aug["critic"][:batch_size] = obs["critic"][:]
        obs_aug["critic"][batch_size : 2 * batch_size] = _transform_critic_obs_left_right(env, obs["critic"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions[:]
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _history_length(env: ManagerBasedRLEnv, group_name: str) -> int:
    cfg = getattr(env, "unwrapped", env).cfg
    history_length = getattr(getattr(cfg.observations, group_name), "history_length", 0)
    return history_length if history_length is not None and history_length > 0 else 1


def _transform_policy_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    history_length = _history_length(env, "policy")
    expected_dim = history_length * (3 + 3 + 3 + _NUM_JOINTS + _NUM_JOINTS + _NUM_JOINTS)
    assert obs.shape[-1] == expected_dim, f"Expected policy obs dim {expected_dim}, got {obs.shape[-1]}."
    obs = obs.clone()
    offset = 0
    term_dim = 3 * history_length
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [-1, 1, -1])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, 1])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, -1])
    offset += term_dim
    term_dim = _NUM_JOINTS * history_length
    for _ in range(3):
        obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(obs[..., offset : offset + term_dim])
        offset += term_dim
    return obs


def _transform_critic_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor) -> torch.Tensor:
    history_length = _history_length(env, "critic")
    expected_dim = history_length * (3 + 3 + 3 + 3 + _NUM_JOINTS + _NUM_JOINTS + _NUM_JOINTS)
    assert obs.shape[-1] == expected_dim, f"Expected critic obs dim {expected_dim}, got {obs.shape[-1]}."
    obs = obs.clone()
    offset = 0
    term_dim = 3 * history_length
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, 1])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [-1, 1, -1])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, 1])
    offset += term_dim
    obs[..., offset : offset + term_dim] = _apply_xyz_sign(obs[..., offset : offset + term_dim], [1, -1, -1])
    offset += term_dim
    term_dim = _NUM_JOINTS * history_length
    for _ in range(3):
        obs[..., offset : offset + term_dim] = _switch_joints_left_right_flat(obs[..., offset : offset + term_dim])
        offset += term_dim
    return obs


def _apply_xyz_sign(obs: torch.Tensor, signs: list[int]) -> torch.Tensor:
    obs_shape = obs.shape
    obs = obs.reshape(*obs_shape[:-1], -1, 3)
    obs = obs * torch.tensor(signs, device=obs.device, dtype=obs.dtype)
    return obs.reshape(obs_shape)


def _switch_joints_left_right_flat(joint_data: torch.Tensor) -> torch.Tensor:
    joint_data_shape = joint_data.shape
    joint_data = joint_data.reshape(*joint_data_shape[:-1], -1, _NUM_JOINTS)
    joint_data = _switch_joints_left_right(joint_data)
    return joint_data.reshape(joint_data_shape)


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    return _switch_joints_left_right(actions.clone())


def _switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    switched = joint_data.clone()
    switched[..., _LEFT_IDS] = joint_data[..., _RIGHT_IDS]
    switched[..., _RIGHT_IDS] = joint_data[..., _LEFT_IDS]
    switched[..., _FLIP_SIGN_IDS] = -1 * switched[..., _FLIP_SIGN_IDS]
    return switched
