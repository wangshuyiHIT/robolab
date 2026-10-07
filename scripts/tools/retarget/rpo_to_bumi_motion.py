# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert prepared RPO motion datasets into Bumi-compatible datasets.

AMP key-body trajectories are rebuilt from the Bumi MuJoCo model so joint
angles and discriminator key-body positions describe the same pose. Joint
angles are range-mapped instead of hard-clipped, and root height is rebuilt
from the Bumi foot collision geometry.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


RPO_LAB_DOF_NAMES = [
    "left_thigh_yaw_joint",
    "right_thigh_yaw_joint",
    "torso_joint",
    "left_thigh_roll_joint",
    "right_thigh_roll_joint",
    "left_arm_pitch_joint",
    "right_arm_pitch_joint",
    "left_thigh_pitch_joint",
    "right_thigh_pitch_joint",
    "left_arm_roll_joint",
    "right_arm_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_arm_yaw_joint",
    "right_arm_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_elbow_yaw_joint",
    "right_elbow_yaw_joint",
]

RPO_URDF_DOF_NAMES = [
    "left_thigh_yaw_joint",
    "left_thigh_roll_joint",
    "left_thigh_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_thigh_yaw_joint",
    "right_thigh_roll_joint",
    "right_thigh_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "torso_joint",
    "left_arm_pitch_joint",
    "left_arm_roll_joint",
    "left_arm_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_yaw_joint",
    "right_arm_pitch_joint",
    "right_arm_roll_joint",
    "right_arm_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_yaw_joint",
]

# Output Bumi motion arrays in Isaac runtime articulation order. BeyondMimic
# compares ``joint_pos`` directly with ``robot.data.joint_pos``.
BUMI_DOF_NAMES = [
    "l_leg_pitch_joint",
    "r_leg_pitch_joint",
    "waist_yaw_joint",
    "l_leg_roll_joint",
    "r_leg_roll_joint",
    "l_arm_pitch_joint",
    "r_arm_pitch_joint",
    "l_leg_yaw_joint",
    "r_leg_yaw_joint",
    "l_arm_roll_joint",
    "r_arm_roll_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_arm_yaw_joint",
    "r_arm_yaw_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
    "l_elbow_pitch_joint",
    "r_elbow_pitch_joint",
    "l_ankle_roll_joint",
    "r_ankle_roll_joint",
]

RPO_LINKS = [
    "base_link",
    "left_thigh_yaw_link",
    "left_thigh_roll_link",
    "left_thigh_pitch_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_thigh_yaw_link",
    "right_thigh_roll_link",
    "right_thigh_pitch_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_arm_pitch_link",
    "left_arm_roll_link",
    "left_arm_yaw_link",
    "left_elbow_pitch_link",
    "left_elbow_yaw_link",
    "right_arm_pitch_link",
    "right_arm_roll_link",
    "right_arm_yaw_link",
    "right_elbow_pitch_link",
    "right_elbow_yaw_link",
]

# Isaac Lab saves ``robot.data.joint_pos`` and ``robot.data.body_pos_w`` in the
# runtime articulation order, not the raw URDF declaration order.  The BM npz
# files produced by ``scripts/tools/beyondmimic/csv_to_npz.py`` therefore use
# these source orders.
RPO_BM_DOF_NAMES = RPO_LAB_DOF_NAMES

RPO_BM_LINKS = [
    "base_link",
    "left_thigh_yaw_link",
    "right_thigh_yaw_link",
    "torso_link",
    "left_thigh_roll_link",
    "right_thigh_roll_link",
    "left_arm_pitch_link",
    "right_arm_pitch_link",
    "left_thigh_pitch_link",
    "right_thigh_pitch_link",
    "left_arm_roll_link",
    "right_arm_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_arm_yaw_link",
    "right_arm_yaw_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_elbow_pitch_link",
    "right_elbow_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_yaw_link",
    "right_elbow_yaw_link",
]

# Output Bumi body arrays in Isaac runtime body order. The command manager then
# slices this array with runtime body indices for its configured body-name order.
BUMI_LINKS = [
    "base_link",
    "l_leg_pitch_link",
    "r_leg_pitch_link",
    "waist_yaw_link",
    "l_leg_roll_link",
    "r_leg_roll_link",
    "l_arm_pitch_link",
    "r_arm_pitch_link",
    "l_leg_yaw_link",
    "r_leg_yaw_link",
    "l_arm_roll_link",
    "r_arm_roll_link",
    "l_knee_pitch_link",
    "r_knee_pitch_link",
    "l_arm_yaw_link",
    "r_arm_yaw_link",
    "l_ankle_pitch_link",
    "r_ankle_pitch_link",
    "l_elbow_pitch_link",
    "r_elbow_pitch_link",
    "l_ankle_roll_link",
    "r_ankle_roll_link",
]

BUMI_AMP_KEY_BODY_NAMES = [
    "l_ankle_roll_link",
    "r_ankle_roll_link",
    "l_knee_pitch_link",
    "r_knee_pitch_link",
    "l_elbow_pitch_link",
    "r_elbow_pitch_link",
]

BUMI_FOOT_BODY_NAMES = ["l_ankle_roll_link", "r_ankle_roll_link"]

DEFAULT_BUMI_MJCF = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "robots"
    / "roboparty"
    / "bumi"
    / "mjcf"
    / "bumi.xml"
)

BUMI_TO_RPO_DOF = {
    "waist_yaw_joint": "torso_joint",
    "l_arm_pitch_joint": "left_arm_pitch_joint",
    "l_arm_roll_joint": "left_arm_roll_joint",
    "l_arm_yaw_joint": "left_arm_yaw_joint",
    "l_elbow_pitch_joint": "left_elbow_pitch_joint",
    "r_arm_pitch_joint": "right_arm_pitch_joint",
    "r_arm_roll_joint": "right_arm_roll_joint",
    "r_arm_yaw_joint": "right_arm_yaw_joint",
    "r_elbow_pitch_joint": "right_elbow_pitch_joint",
    "l_leg_pitch_joint": "left_thigh_pitch_joint",
    "l_leg_roll_joint": "left_thigh_roll_joint",
    "l_leg_yaw_joint": "left_thigh_yaw_joint",
    "l_knee_pitch_joint": "left_knee_joint",
    "l_ankle_pitch_joint": "left_ankle_pitch_joint",
    "l_ankle_roll_joint": "left_ankle_roll_joint",
    "r_leg_pitch_joint": "right_thigh_pitch_joint",
    "r_leg_roll_joint": "right_thigh_roll_joint",
    "r_leg_yaw_joint": "right_thigh_yaw_joint",
    "r_knee_pitch_joint": "right_knee_joint",
    "r_ankle_pitch_joint": "right_ankle_pitch_joint",
    "r_ankle_roll_joint": "right_ankle_roll_joint",
}

BUMI_TO_RPO_LINK = {
    "base_link": "base_link",
    "waist_yaw_link": "torso_link",
    "l_arm_pitch_link": "left_arm_pitch_link",
    "l_arm_roll_link": "left_arm_roll_link",
    "l_arm_yaw_link": "left_arm_yaw_link",
    "l_elbow_pitch_link": "left_elbow_pitch_link",
    "r_arm_pitch_link": "right_arm_pitch_link",
    "r_arm_roll_link": "right_arm_roll_link",
    "r_arm_yaw_link": "right_arm_yaw_link",
    "r_elbow_pitch_link": "right_elbow_pitch_link",
    "l_leg_pitch_link": "left_thigh_pitch_link",
    "l_leg_roll_link": "left_thigh_roll_link",
    "l_leg_yaw_link": "left_thigh_yaw_link",
    "l_knee_pitch_link": "left_knee_link",
    "l_ankle_pitch_link": "left_ankle_pitch_link",
    "l_ankle_roll_link": "left_ankle_roll_link",
    "r_leg_pitch_link": "right_thigh_pitch_link",
    "r_leg_roll_link": "right_thigh_roll_link",
    "r_leg_yaw_link": "right_thigh_yaw_link",
    "r_knee_pitch_link": "right_knee_link",
    "r_ankle_pitch_link": "right_ankle_pitch_link",
    "r_ankle_roll_link": "right_ankle_roll_link",
}

# Bumi's arm yaw / elbow pitch axes point the opposite way from RPO
# (see SIGN_FLIP_JOINTS in scripts/tools/beyondmimic/rpo_npz_to_bumi_npz.py).
SIGN_FLIP_JOINTS = {
    "l_arm_yaw_joint",
    "r_arm_yaw_joint",
    "l_elbow_pitch_joint",
    "r_elbow_pitch_joint",
}

# Bumi joint limits from bumi_edu_pro_collision.urdf.
BUMI_JOINT_LIMITS = {
    "waist_yaw_joint": (-1.57, 1.57),
    "l_arm_pitch_joint": (-3.14, 1.57),
    "l_arm_roll_joint": (-0.14, 1.94),
    "l_arm_yaw_joint": (-1.57, 1.57),
    "l_elbow_pitch_joint": (-2.26, 0.0),
    "r_arm_pitch_joint": (-3.14, 1.57),
    "r_arm_roll_joint": (-1.94, 0.14),
    "r_arm_yaw_joint": (-1.57, 1.57),
    "r_elbow_pitch_joint": (-2.26, 0.0),
    "l_leg_pitch_joint": (-2.09, 2.09),
    "l_leg_roll_joint": (-0.66, 1.57),
    "l_leg_yaw_joint": (-2.53, 2.53),
    "l_knee_pitch_joint": (0.0, 2.24),
    "l_ankle_pitch_joint": (-0.96, 0.44),
    "l_ankle_roll_joint": (-0.17, 0.17),
    "r_leg_pitch_joint": (-2.09, 2.09),
    "r_leg_roll_joint": (-1.57, 0.66),
    "r_leg_yaw_joint": (-2.53, 2.53),
    "r_knee_pitch_joint": (0.0, 2.24),
    "r_ankle_pitch_joint": (-0.96, 0.44),
    "r_ankle_roll_joint": (-0.17, 0.17),
}

RPO_JOINT_LIMITS = {
    "torso_joint": (-3.14, 3.14),
    "left_arm_pitch_joint": (-3.14, 1.57),
    "left_arm_roll_joint": (-0.25, 3.14),
    "left_arm_yaw_joint": (-1.57, 1.57),
    "left_elbow_pitch_joint": (-0.6, 1.57),
    "right_arm_pitch_joint": (-3.14, 1.57),
    "right_arm_roll_joint": (-3.14, 0.25),
    "right_arm_yaw_joint": (-1.57, 1.57),
    "right_elbow_pitch_joint": (-0.6, 1.57),
    "left_thigh_pitch_joint": (-2.094, 0.7854),
    "left_thigh_roll_joint": (-0.2, 1.0),
    "left_thigh_yaw_joint": (-1.0, 0.2),
    "left_knee_joint": (-0.2, 2.5),
    "left_ankle_pitch_joint": (-0.6, 0.6),
    "left_ankle_roll_joint": (-0.5, 0.5),
    "right_thigh_pitch_joint": (-2.094, 0.7854),
    "right_thigh_roll_joint": (-1.0, 0.2),
    "right_thigh_yaw_joint": (-0.2, 1.0),
    "right_knee_joint": (-0.2, 2.5),
    "right_ankle_pitch_joint": (-0.6, 0.6),
    "right_ankle_roll_joint": (-0.5, 0.5),
}

# Horizontal dynamic-similarity scale. Vertical root placement and key-body
# positions are computed from Bumi FK rather than uniformly scaling RPO points.
BUMI_OVER_RPO_HEIGHT = 0.73


def _iter_files(path: Path, suffix: str) -> Iterable[Path]:
    if not path.is_dir():
        raise ValueError(f"Input directory does not exist: {path}")
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix == suffix)


def _ensure_can_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file without --overwrite: {path}")


def _canonicalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Normalize a wxyz track and remove equivalent q/-q discontinuities."""
    quaternions = np.asarray(quaternions, dtype=np.float64).copy()
    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(f"Expected quaternion array with shape (N, 4), got {quaternions.shape}")
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("Motion contains a zero-length root quaternion")
    quaternions /= norms
    for frame_id in range(1, len(quaternions)):
        if np.dot(quaternions[frame_id - 1], quaternions[frame_id]) < 0.0:
            quaternions[frame_id] *= -1.0
    return quaternions.astype(np.float32)


def _quat_apply_wxyz(quaternions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate vectors by matching wxyz quaternions with NumPy broadcasting."""
    quat_vector = quaternions[..., 1:]
    first_cross = np.cross(quat_vector, vectors)
    return vectors + 2.0 * (
        quaternions[..., :1] * first_cross + np.cross(quat_vector, first_cross)
    )


def _safe_bumi_limits(bumi_name: str, limit_margin: float) -> tuple[float, float]:
    target_lower, target_upper = BUMI_JOINT_LIMITS[bumi_name]
    # Do not move a semantic zero away from a zero hard limit (knees/elbows).
    safe_lower = target_lower + limit_margin if target_lower < 0.0 else target_lower
    safe_upper = target_upper - limit_margin if target_upper > 0.0 else target_upper
    return safe_lower, safe_upper


def _map_values_to_bumi_limits(
    values: np.ndarray,
    source_name: str,
    bumi_name: str,
    limit_margin: float,
) -> np.ndarray:
    """Preserve zero and motion shape while fitting the narrower Bumi range."""
    sign = -1.0 if bumi_name in SIGN_FLIP_JOINTS else 1.0
    values = np.asarray(values, dtype=np.float64) * sign

    source_lower, source_upper = RPO_JOINT_LIMITS[source_name]
    source_lower, source_upper = sorted((source_lower * sign, source_upper * sign))
    safe_lower, safe_upper = _safe_bumi_limits(bumi_name, limit_margin)

    negative_scale = 1.0
    if source_lower < 0.0:
        negative_scale = min(1.0, abs(safe_lower) / abs(source_lower)) if safe_lower < 0.0 else 0.0
    positive_scale = 1.0
    if source_upper > 0.0:
        positive_scale = min(1.0, safe_upper / source_upper) if safe_upper > 0.0 else 0.0

    values = np.where(values < 0.0, values * negative_scale, values * positive_scale)
    return np.clip(values, safe_lower, safe_upper)


def _map_composite_hip_rotation(
    array: np.ndarray,
    source_names: list[str],
    side: str,
    limit_margin: float,
) -> dict[str, np.ndarray]:
    """Convert RPO's tilted yaw-roll-pitch hip into Bumi's pitch-roll-yaw hip."""
    source_prefix = "left" if side == "l" else "right"
    yaw = array[:, source_names.index(f"{source_prefix}_thigh_yaw_joint")]
    roll = array[:, source_names.index(f"{source_prefix}_thigh_roll_joint")]
    pitch = array[:, source_names.index(f"{source_prefix}_thigh_pitch_joint")]

    yaw_axis = np.asarray((-0.5, 0.0, -0.86603), dtype=np.float64)
    yaw_axis /= np.linalg.norm(yaw_axis)
    roll_axis = np.asarray((0.86603, 0.0, -0.5), dtype=np.float64)
    roll_axis /= np.linalg.norm(roll_axis)

    source_rotation = (
        Rotation.from_rotvec(yaw[:, None] * yaw_axis).as_matrix()
        @ Rotation.from_rotvec(roll[:, None] * roll_axis).as_matrix()
        @ Rotation.from_rotvec(pitch[:, None] * np.asarray((0.0, 1.0, 0.0))).as_matrix()
    )
    # Upper-case YXZ means Ry(pitch) @ Rx(roll) @ Rz(yaw), matching
    # Bumi's hip chain order.
    mapped = Rotation.from_matrix(source_rotation).as_euler("YXZ")
    result = {}
    for column, component in enumerate(("pitch", "roll", "yaw")):
        bumi_name = f"{side}_leg_{component}_joint"
        result[bumi_name] = np.clip(
            mapped[:, column],
            *_safe_bumi_limits(bumi_name, limit_margin),
        )
    return result


def _map_composite_shoulder_rotation(
    array: np.ndarray,
    source_names: list[str],
    side: str,
    limit_margin: float,
) -> dict[str, np.ndarray]:
    """Account for Bumi's fixed 15/-15/5 degree shoulder mount rotations."""
    source_prefix = "left" if side == "l" else "right"
    pitch = array[:, source_names.index(f"{source_prefix}_arm_pitch_joint")]
    roll = array[:, source_names.index(f"{source_prefix}_arm_roll_joint")]
    yaw = array[:, source_names.index(f"{source_prefix}_arm_yaw_joint")]

    source_rotation = Rotation.from_euler(
        "YXZ",
        np.stack((pitch, roll, -yaw), axis=1),
    ).as_matrix()
    side_sign = 1.0 if side == "l" else -1.0
    zero_mount = Rotation.from_rotvec(
        np.asarray((np.deg2rad(5.0 * side_sign), 0.0, 0.0))
    ).as_matrix()
    inverse_first_mount = Rotation.from_rotvec(
        np.asarray((np.deg2rad(-15.0 * side_sign), 0.0, 0.0))
    ).as_matrix()

    reduced_target = inverse_first_mount @ (source_rotation @ zero_mount)
    mapped = Rotation.from_matrix(reduced_target).as_euler("YXZ")
    mapped[:, 1] += np.deg2rad(10.0 * side_sign)

    result = {}
    for column, component in enumerate(("pitch", "roll", "yaw")):
        bumi_name = f"{side}_arm_{component}_joint"
        result[bumi_name] = np.clip(
            mapped[:, column],
            *_safe_bumi_limits(bumi_name, limit_margin),
        )
    return result


def _remap_dof_array(
    array: np.ndarray,
    source_names: list[str],
    limit_margin: float = 0.01,
) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D dof array, got shape {array.shape}")
    if array.shape[1] != len(source_names):
        raise ValueError(f"Expected {len(source_names)} source dofs, got shape {array.shape}")

    remapped = np.zeros((array.shape[0], len(BUMI_DOF_NAMES)), dtype=np.float32)
    for bumi_index, bumi_name in enumerate(BUMI_DOF_NAMES):
        source_name = BUMI_TO_RPO_DOF[bumi_name]
        values = array[:, source_names.index(source_name)]
        remapped[:, bumi_index] = _map_values_to_bumi_limits(
            values,
            source_name,
            bumi_name,
            limit_margin,
        )

    # RPO and Bumi use different 3-DoF joint orders and fixed joint frames.
    # Matching the composite rotation avoids copying individual Euler angles
    # between incompatible mechanisms.
    composite_values = {}
    for side in ("l", "r"):
        composite_values.update(
            _map_composite_hip_rotation(array, source_names, side, limit_margin)
        )
        composite_values.update(
            _map_composite_shoulder_rotation(array, source_names, side, limit_margin)
        )
    for bumi_name, values in composite_values.items():
        remapped[:, BUMI_DOF_NAMES.index(bumi_name)] = values
    return remapped


class BumiForwardKinematics:
    """Fast key-body FK and foot-collision queries using the Bumi MJCF."""

    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Bumi MJCF does not exist: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        free_joint_ids = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free_joint_ids) != 1:
            raise ValueError(f"Expected one Bumi free joint, found {len(free_joint_ids)}")
        self.root_qpos_address = int(self.model.jnt_qposadr[free_joint_ids[0]])

        self.joint_qpos_addresses = []
        for joint_name in BUMI_DOF_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"Bumi MJCF is missing joint: {joint_name}")
            self.joint_qpos_addresses.append(int(self.model.jnt_qposadr[joint_id]))

        self.root_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        self.key_body_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                for body_name in BUMI_AMP_KEY_BODY_NAMES
            ],
            dtype=np.int32,
        )
        if self.root_body_id < 0 or np.any(self.key_body_ids < 0):
            raise ValueError("Bumi MJCF is missing an AMP key body")

        self.foot_box_corners: list[tuple[int, np.ndarray]] = []
        for body_name in BUMI_FOOT_BODY_NAMES:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            geom_ids = np.flatnonzero(
                (self.model.geom_bodyid == body_id)
                & (self.model.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
            )
            if len(geom_ids) != 1:
                raise ValueError(
                    f"Expected one active foot box on {body_name}, found {len(geom_ids)}"
                )
            geom_id = int(geom_ids[0])
            half_size = self.model.geom_size[geom_id]
            corners = np.asarray(
                [
                    (x, y, z)
                    for x in (-half_size[0], half_size[0])
                    for y in (-half_size[1], half_size[1])
                    for z in (-half_size[2], half_size[2])
                ],
                dtype=np.float64,
            )
            self.foot_box_corners.append((geom_id, corners))

    def evaluate(
        self,
        joint_pos: np.ndarray,
        root_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Bumi key bodies in base frame and lowest sole z relative to root."""
        frame_count = joint_pos.shape[0]
        key_body_pos_b = np.empty(
            (frame_count, len(BUMI_AMP_KEY_BODY_NAMES), 3),
            dtype=np.float32,
        )
        lowest_sole_z = np.empty(frame_count, dtype=np.float32)

        for frame_id in range(frame_count):
            self.data.qpos[:] = 0.0
            root_address = self.root_qpos_address
            self.data.qpos[root_address + 3] = 1.0
            self.data.qpos[self.joint_qpos_addresses] = joint_pos[frame_id]
            mujoco.mj_forward(self.model, self.data)

            root_pos = self.data.xpos[self.root_body_id]
            key_body_pos_b[frame_id] = self.data.xpos[self.key_body_ids] - root_pos

            all_foot_corners_b = []
            for geom_id, local_corners in self.foot_box_corners:
                rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
                corners_b = local_corners @ rotation.T + self.data.geom_xpos[geom_id] - root_pos
                all_foot_corners_b.append(corners_b)
            foot_corners_b = np.concatenate(all_foot_corners_b, axis=0)
            frame_quat = np.broadcast_to(root_quat[frame_id], (len(foot_corners_b), 4))
            foot_corners_w_rel = _quat_apply_wxyz(frame_quat, foot_corners_b)
            lowest_sole_z[frame_id] = foot_corners_w_rel[:, 2].min()

        return key_body_pos_b, lowest_sole_z


def _remap_body_array(array: np.ndarray, source_names: list[str]) -> np.ndarray:
    if array.ndim < 2:
        raise ValueError(f"Expected body array with a body axis, got shape {array.shape}")
    if array.shape[1] != len(source_names):
        raise ValueError(f"Expected {len(source_names)} source bodies, got shape {array.shape}")

    remapped_shape = (array.shape[0], len(BUMI_LINKS), *array.shape[2:])
    remapped = np.zeros(remapped_shape, dtype=array.dtype)
    for bumi_index, bumi_name in enumerate(BUMI_LINKS):
        source_name = BUMI_TO_RPO_LINK[bumi_name]
        remapped[:, bumi_index] = array[:, source_names.index(source_name)]
    return remapped


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values, 1.0 / fps, axis=0).astype(np.float32)


def convert_lab_file(
    source_path: Path,
    output_path: Path,
    kinematics: BumiForwardKinematics,
    limit_margin: float,
    overwrite: bool,
    dry_run: bool,
    strict: bool,
) -> None:
    try:
        try:
            with source_path.open("rb") as stream:
                data = pickle.load(stream)
        except Exception:
            # Some source files are saved with joblib (same fallback as the
            # AMP MotionDataManager loader).
            import joblib

            data = joblib.load(source_path)
    except Exception as exc:
        if strict:
            raise
        print(f"SKIP LAB  {source_path}: {type(exc).__name__}: {exc}")
        return

    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    source_root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    source_key_body_pos = np.asarray(data["key_body_pos"], dtype=np.float64)
    if source_key_body_pos.ndim != 3 or source_key_body_pos.shape[1:] != (6, 3):
        raise ValueError(
            f"Expected six RPO AMP key bodies with shape (N, 6, 3), got "
            f"{source_key_body_pos.shape}: {source_path}"
        )

    root_quat = _canonicalize_quaternions(data["root_rot"])
    dof_pos = _remap_dof_array(
        data["dof_pos"],
        RPO_LAB_DOF_NAMES,
        limit_margin=limit_margin,
    )
    key_body_pos_b, lowest_sole_z_relative = kinematics.evaluate(dof_pos, root_quat)

    # Preserve horizontal path shape under dynamic-similarity scaling. Vertical
    # placement is rebuilt so the Bumi sole follows the source contact/flight
    # clearance instead of scaling an RPO-specific base height.
    root_pos = np.zeros_like(source_root_pos, dtype=np.float32)
    root_pos[:, :2] = (
        source_root_pos[0, :2]
        + BUMI_OVER_RPO_HEIGHT * (source_root_pos[:, :2] - source_root_pos[0, :2])
    )
    source_foot_clearance = np.maximum(
        source_key_body_pos[:, :2, 2].min(axis=1),
        0.0,
    )
    desired_sole_clearance = BUMI_OVER_RPO_HEIGHT * source_foot_clearance
    root_pos[:, 2] = desired_sole_clearance - lowest_sole_z_relative

    root_quat_expanded = root_quat[:, None, :]
    key_body_pos_w = root_pos[:, None, :] + _quat_apply_wxyz(
        root_quat_expanded,
        key_body_pos_b,
    )

    inverse_root_quat = root_quat_expanded.copy()
    inverse_root_quat[..., 1:] *= -1.0
    recovered_key_body_pos_b = _quat_apply_wxyz(
        inverse_root_quat,
        key_body_pos_w - root_pos[:, None, :],
    )
    fk_error = float(np.max(np.abs(recovered_key_body_pos_b - key_body_pos_b)))
    ground_error = float(
        np.max(
            np.abs(
                root_pos[:, 2]
                + lowest_sole_z_relative
                - desired_sole_clearance
            )
        )
    )
    if not all(
        np.isfinite(values).all()
        for values in (root_pos, root_quat, dof_pos, key_body_pos_w)
    ):
        raise ValueError(f"Retargeting produced non-finite values: {source_path}")
    if fk_error > 1.0e-5 or ground_error > 1.0e-5:
        raise ValueError(
            f"Retarget validation failed for {source_path}: "
            f"fk_error={fk_error:.3e}, ground_error={ground_error:.3e}"
        )

    converted = dict(data)
    converted["root_pos"] = root_pos.astype(np.float32)
    converted["root_rot"] = root_quat.astype(np.float32)
    converted["dof_pos"] = dof_pos.astype(np.float32)
    converted["key_body_pos"] = key_body_pos_w.astype(np.float32)

    # The MotionDataManager recomputes velocities, but keep optional cached
    # fields consistent for other consumers.
    if "root_vel" in converted:
        converted["root_vel"] = _finite_difference(root_pos, fps)
    if "dof_vel" in converted:
        converted["dof_vel"] = _finite_difference(dof_pos, fps)

    if not dry_run:
        _ensure_can_write(output_path, overwrite)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as stream:
            pickle.dump(converted, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"LAB  {source_path} -> {output_path} "
        f"(root_z={root_pos[:, 2].min():.3f}..{root_pos[:, 2].max():.3f} m, "
        f"fk_error={fk_error:.1e})"
    )


def convert_bm_file(source_path: Path, output_path: Path, overwrite: bool, dry_run: bool, strict: bool) -> None:
    try:
        with np.load(source_path) as data:
            converted = {name: data[name] for name in data.files}
    except Exception as exc:
        if strict:
            raise
        print(f"SKIP BM   {source_path}: {type(exc).__name__}: {exc}")
        return

    converted["joint_pos"] = _remap_dof_array(
        converted["joint_pos"],
        RPO_BM_DOF_NAMES,
    )
    fps = float(np.asarray(converted["fps"]).reshape(-1)[0])
    converted["joint_vel"] = _finite_difference(converted["joint_pos"], fps)
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        converted[key] = _remap_body_array(converted[key], RPO_BM_LINKS)

    if not dry_run:
        _ensure_can_write(output_path, overwrite)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **converted)
    print(f"BM   {source_path} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpo-lab-dir", default="robolab/data/motions/rpo_lab")
    parser.add_argument("--bumi-lab-dir", default="robolab/data/motions/bumi_lab")
    parser.add_argument("--rpo-bm-dir", default="robolab/data/motions/rpo_bm")
    parser.add_argument("--bumi-bm-dir", default="robolab/data/motions/bumi_bm")
    parser.add_argument(
        "--bumi-mjcf",
        default=str(DEFAULT_BUMI_MJCF),
        help="Bumi MJCF used to rebuild AMP key-body trajectories.",
    )
    parser.add_argument(
        "--joint-limit-margin",
        type=float,
        default=0.01,
        help="Safety margin inside non-zero Bumi hard limits.",
    )
    parser.add_argument("--skip-lab", action="store_true", help="Do not convert rpo_lab pickle files.")
    parser.add_argument("--skip-bm", action="store_true", help="Do not convert rpo_bm npz files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Validate mappings without writing files.")
    parser.add_argument("--strict", action="store_true", help="Fail immediately on unreadable source files.")
    args = parser.parse_args()

    if not args.skip_lab:
        if args.joint_limit_margin < 0.0:
            raise ValueError("--joint-limit-margin must be non-negative")
        kinematics = BumiForwardKinematics(Path(args.bumi_mjcf))
        for source_path in _iter_files(Path(args.rpo_lab_dir), ".pkl"):
            output_path = Path(args.bumi_lab_dir) / source_path.name
            convert_lab_file(
                source_path,
                output_path,
                kinematics,
                args.joint_limit_margin,
                args.overwrite,
                args.dry_run,
                args.strict,
            )

    if not args.skip_bm:
        for source_path in _iter_files(Path(args.rpo_bm_dir), ".npz"):
            output_path = Path(args.bumi_bm_dir) / source_path.name
            convert_bm_file(source_path, output_path, args.overwrite, args.dry_run, args.strict)


if __name__ == "__main__":
    main()
