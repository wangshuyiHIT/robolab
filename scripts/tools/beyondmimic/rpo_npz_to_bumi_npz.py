#!/usr/bin/env python3
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert an RPO BeyondMimic npz into a Bumi-compatible BeyondMimic npz.

The source RPO npz files are saved in Isaac runtime articulation order. This
script maps joints by name, applies the Bumi joint convention fixes, then
replays the mapped pose on the Bumi articulation to regenerate body trajectories
from Bumi forward kinematics.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# Make ``import robolab`` work when the script is launched from the repo root.
ROBOLAB_EXTENSION_DIR = Path(__file__).resolve().parents[3]
if str(ROBOLAB_EXTENSION_DIR) not in sys.path:
    sys.path.insert(0, str(ROBOLAB_EXTENSION_DIR))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input_file", "-i", default="robolab/data/motions/rpo_bm/yundong1.npz")
parser.add_argument("--output_file", "-o", default="robolab/data/motions/bumi_bm/yundong1.npz")
parser.add_argument("--overwrite", action="store_true", help="Overwrite output_file if it already exists.")
parser.add_argument(
    "--root_height_mode",
    choices=("bumi", "source"),
    default="bumi",
    help="Use Bumi default root height plus source z variation, or preserve source root height.",
)
parser.add_argument("--joint_limit_margin", type=float, default=0.02)
parser.add_argument(
    "--ground_motion",
    action="store_true",
    help="Shift the Bumi root trajectory so the lowest configured collision stays near the ground.",
)
parser.add_argument(
    "--ground_clearance",
    type=float,
    default=0.02,
    help="Target clearance in meters for the lowest Bumi collision when --ground_motion is enabled.",
)
parser.add_argument(
    "--ground_smoothing_window",
    type=int,
    default=11,
    help="Odd Savitzky-Golay window used to smooth the vertical root correction; use 1 to disable smoothing.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from robolab.assets.robots import BUMI_CFG


RPO_RUNTIME_DOF_NAMES = [
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

RPO_RUNTIME_BODY_NAMES = [
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

BUMI_RUNTIME_DOF_NAMES = [
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

BUMI_RUNTIME_BODY_NAMES = [
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

SIGN_FLIP_JOINTS = {
    "l_arm_yaw_joint",
    "r_arm_yaw_joint",
    "l_elbow_pitch_joint",
    "r_elbow_pitch_joint",
}


@configclass
class ReplayBumiSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot = BUMI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _central_difference(values: np.ndarray, dt: float) -> np.ndarray:
    return np.gradient(values, dt, axis=0).astype(np.float32)


def _angular_velocity_from_quat_wxyz(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    flat = quat_wxyz.reshape(-1, 4)
    quat_xyzw = flat[:, [1, 2, 3, 0]].reshape(quat_wxyz.shape)
    rotations = Rotation.from_quat(quat_xyzw.reshape(-1, 4)).as_matrix().reshape(*quat_wxyz.shape[:-1], 3, 3)

    angular_velocity = np.zeros((*quat_wxyz.shape[:-1], 3), dtype=np.float32)
    for body_id in range(quat_wxyz.shape[1]):
        body_rot = Rotation.from_matrix(rotations[:, body_id])
        if quat_wxyz.shape[0] < 3:
            angular_velocity[:, body_id] = 0.0
            continue
        rel = body_rot[:-2].inv() * body_rot[2:]
        middle = rel.as_rotvec() / (2.0 * dt)
        angular_velocity[1:-1, body_id] = middle.astype(np.float32)
        angular_velocity[0, body_id] = angular_velocity[1, body_id]
        angular_velocity[-1, body_id] = angular_velocity[-2, body_id]
    return angular_velocity


def _parse_vector(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    vector = np.fromstring(value, sep=" ", dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got {value!r}")
    return vector


def _load_stl_vertices(path: Path) -> np.ndarray:
    """Load STL triangle vertices without adding a mesh-processing dependency."""
    payload = path.read_bytes()
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        if len(payload) == 84 + triangle_count * 50:
            triangle_dtype = np.dtype(
                [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
            )
            triangles = np.frombuffer(payload, dtype=triangle_dtype, count=triangle_count, offset=84)
            return triangles["vertices"].reshape(-1, 3).astype(np.float64)

    vertices = []
    for line in payload.decode(errors="ignore").splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append([float(value) for value in fields[1:]])
    if not vertices:
        raise ValueError(f"Could not read STL vertices: {path}")
    return np.asarray(vertices, dtype=np.float64)


def _load_collision_vertices(urdf_path: Path) -> dict[int, np.ndarray]:
    """Read active Bumi collision geometry in each runtime body frame."""
    robot_xml = ET.parse(urdf_path).getroot()
    collision_vertices: dict[int, np.ndarray] = {}

    for link_xml in robot_xml.findall("link"):
        body_name = link_xml.attrib["name"]
        body_collisions = []
        for collision_xml in link_xml.findall("collision"):
            origin_xml = collision_xml.find("origin")
            origin_xyz = _parse_vector(
                origin_xml.attrib.get("xyz") if origin_xml is not None else None, (0.0, 0.0, 0.0)
            )
            origin_rpy = _parse_vector(
                origin_xml.attrib.get("rpy") if origin_xml is not None else None, (0.0, 0.0, 0.0)
            )
            origin_rotation = Rotation.from_euler("xyz", origin_rpy).as_matrix()

            geometry_xml = collision_xml.find("geometry")
            if geometry_xml is None:
                continue
            mesh_xml = geometry_xml.find("mesh")
            box_xml = geometry_xml.find("box")
            if mesh_xml is not None:
                mesh_path = (urdf_path.parent / mesh_xml.attrib["filename"]).resolve()
                vertices = _load_stl_vertices(mesh_path)
                vertices *= _parse_vector(mesh_xml.attrib.get("scale"), (1.0, 1.0, 1.0))
            elif box_xml is not None:
                half_size = _parse_vector(box_xml.attrib["size"], (0.0, 0.0, 0.0)) / 2.0
                vertices = np.asarray(
                    [
                        (x, y, z)
                        for x in (-half_size[0], half_size[0])
                        for y in (-half_size[1], half_size[1])
                        for z in (-half_size[2], half_size[2])
                    ],
                    dtype=np.float64,
                )
            else:
                raise ValueError(
                    f"Unsupported active collision geometry on {body_name}; expected mesh or box in {urdf_path}"
                )

            body_collisions.append(vertices @ origin_rotation.T + origin_xyz)

        if body_collisions:
            if body_name not in BUMI_RUNTIME_BODY_NAMES:
                raise ValueError(f"Collision link is absent from Bumi runtime body order: {body_name}")
            collision_vertices[BUMI_RUNTIME_BODY_NAMES.index(body_name)] = np.concatenate(body_collisions, axis=0)

    if not collision_vertices:
        raise ValueError(f"No active collision geometry found in {urdf_path}")
    return collision_vertices


def _lowest_collision_z(
    body_pos_w: np.ndarray, body_quat_w: np.ndarray, collision_vertices: dict[int, np.ndarray]
) -> np.ndarray:
    minimum_z = np.full(body_pos_w.shape[0], np.inf, dtype=np.float64)
    for body_index, vertices in collision_vertices.items():
        quat_xyzw = body_quat_w[:, body_index][:, [1, 2, 3, 0]]
        rotation = Rotation.from_quat(quat_xyzw).as_matrix()
        world_z = body_pos_w[:, body_index, 2, None] + rotation[:, 2, :] @ vertices.T
        minimum_z = np.minimum(minimum_z, world_z.min(axis=1))
    return minimum_z


def _align_body_fk_to_ground(
    body_data: dict[str, np.ndarray], fps: float, urdf_path: Path
) -> dict[str, np.ndarray]:
    """Apply a smooth per-frame root-z correction using the active Bumi collisions."""
    body_pos_w = body_data["body_pos_w"].astype(np.float64, copy=True)
    body_quat_w = body_data["body_quat_w"].astype(np.float64, copy=False)
    collision_vertices = _load_collision_vertices(urdf_path)
    minimum_z_before = _lowest_collision_z(body_pos_w, body_quat_w, collision_vertices)
    correction = args_cli.ground_clearance - minimum_z_before

    window = args_cli.ground_smoothing_window
    if window < 1 or (window > 1 and window % 2 == 0):
        raise ValueError("--ground_smoothing_window must be 1 or a positive odd integer")
    if window > body_pos_w.shape[0]:
        window = body_pos_w.shape[0] if body_pos_w.shape[0] % 2 == 1 else body_pos_w.shape[0] - 1
    if window > 1:
        correction = savgol_filter(correction, window_length=window, polyorder=min(3, window - 1), mode="interp")

    # Keep every configured collision out of the ground even when smoothing spans a contact transition.
    minimum_safe_clearance = max(0.0, min(0.005, args_cli.ground_clearance))
    minimum_z_after = minimum_z_before + correction
    if minimum_z_after.min() < minimum_safe_clearance:
        correction += minimum_safe_clearance - minimum_z_after.min()

    body_pos_w[:, :, 2] += correction[:, None]
    body_data["body_pos_w"] = body_pos_w.astype(np.float32)
    body_data["body_lin_vel_w"] = _central_difference(body_data["body_pos_w"], 1.0 / fps)

    minimum_z_after = _lowest_collision_z(body_data["body_pos_w"], body_quat_w, collision_vertices)
    print(
        "[INFO] Ground alignment: "
        f"offset=[{correction.min():.4f}, {correction.max():.4f}] m, "
        f"clearance_before=[{minimum_z_before.min():.4f}, {minimum_z_before.max():.4f}] m, "
        f"clearance_after=[{minimum_z_after.min():.4f}, {minimum_z_after.max():.4f}] m",
        flush=True,
    )
    return body_data


def _map_joint_pos(source_joint_pos: np.ndarray, fps: float, margin: float) -> tuple[np.ndarray, np.ndarray]:
    mapped = np.zeros((source_joint_pos.shape[0], len(BUMI_RUNTIME_DOF_NAMES)), dtype=np.float32)
    for output_id, bumi_name in enumerate(BUMI_RUNTIME_DOF_NAMES):
        source_name = BUMI_TO_RPO_DOF[bumi_name]
        source_id = RPO_RUNTIME_DOF_NAMES.index(source_name)
        values = source_joint_pos[:, source_id].astype(np.float32)
        if bumi_name in SIGN_FLIP_JOINTS:
            values = -values
        lower, upper = BUMI_JOINT_LIMITS[bumi_name]
        lower += margin
        upper -= margin
        if lower > upper:
            lower, upper = BUMI_JOINT_LIMITS[bumi_name]
        mapped[:, output_id] = np.clip(values, lower, upper)
    return mapped, _central_difference(mapped, 1.0 / fps)


def _build_root_state(data: dict[str, np.ndarray], fps: float, default_root_z: float) -> np.ndarray:
    root_id = RPO_RUNTIME_BODY_NAMES.index("base_link")
    root_pos = data["body_pos_w"][:, root_id].astype(np.float32).copy()
    if args_cli.root_height_mode == "bumi":
        root_pos[:, 2] = default_root_z + (root_pos[:, 2] - root_pos[0, 2])
    root_quat = data["body_quat_w"][:, root_id].astype(np.float32)
    root_lin_vel = _central_difference(root_pos, 1.0 / fps)
    root_ang_vel = _angular_velocity_from_quat_wxyz(root_quat[:, None, :], 1.0 / fps)[:, 0]
    return np.concatenate([root_pos, root_quat, root_lin_vel, root_ang_vel], axis=1).astype(np.float32)


def _generate_body_fk(
    root_state: np.ndarray, joint_pos: np.ndarray, joint_vel: np.ndarray, fps: float
) -> dict[str, np.ndarray]:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / fps
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplayBumiSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]

    if list(robot.joint_names) != BUMI_RUNTIME_DOF_NAMES:
        raise RuntimeError(f"Unexpected Bumi joint order: {robot.joint_names}")
    if list(robot.body_names) != BUMI_RUNTIME_BODY_NAMES:
        raise RuntimeError(f"Unexpected Bumi body order: {robot.body_names}")

    body_pos = []
    body_quat = []
    root_state_t = torch.zeros((1, 13), dtype=torch.float32, device=sim.device)
    joint_pos_t = torch.zeros((1, len(BUMI_RUNTIME_DOF_NAMES)), dtype=torch.float32, device=sim.device)
    joint_vel_t = torch.zeros_like(joint_pos_t)

    for frame_id in range(joint_pos.shape[0]):
        root_state_t[0] = torch.as_tensor(root_state[frame_id], dtype=torch.float32, device=sim.device)
        joint_pos_t[0] = torch.as_tensor(joint_pos[frame_id], dtype=torch.float32, device=sim.device)
        joint_vel_t[0] = torch.as_tensor(joint_vel[frame_id], dtype=torch.float32, device=sim.device)
        robot.write_root_state_to_sim(root_state_t)
        robot.write_joint_state_to_sim(joint_pos_t, joint_vel_t)
        sim.render()
        scene.update(sim.get_physics_dt())
        body_pos.append(robot.data.body_pos_w[0].detach().cpu().numpy().copy())
        body_quat.append(robot.data.body_quat_w[0].detach().cpu().numpy().copy())

    body_pos = np.stack(body_pos, axis=0).astype(np.float32)
    body_quat = np.stack(body_quat, axis=0).astype(np.float32)
    body_lin_vel = _central_difference(body_pos, sim_cfg.dt)
    body_ang_vel = _angular_velocity_from_quat_wxyz(body_quat, sim_cfg.dt)
    return {
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
    }


def main() -> None:
    input_path = Path(args_cli.input_file)
    output_path = Path(args_cli.output_file)
    if output_path.exists() and not args_cli.overwrite:
        raise FileExistsError(f"Refusing to overwrite without --overwrite: {output_path}")

    with np.load(input_path) as source:
        data = {key: source[key] for key in source.files}

    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    joint_pos, joint_vel = _map_joint_pos(data["joint_pos"], fps, args_cli.joint_limit_margin)
    default_root_z = float(BUMI_CFG.init_state.pos[2])
    root_state = _build_root_state(data, fps, default_root_z)
    body_data = _generate_body_fk(root_state, joint_pos, joint_vel, fps)
    if args_cli.ground_motion:
        body_data = _align_body_fk_to_ground(body_data, fps, Path(BUMI_CFG.spawn.asset_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        fps=np.asarray([int(round(fps))], dtype=np.int64),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        **body_data,
    )
    print(f"[INFO] Saved Bumi motion: {output_path}", flush=True)
    print(
        f"[INFO] frames={joint_pos.shape[0]} fps={fps:g} joints={joint_pos.shape[1]} "
        f"bodies={body_data['body_pos_w'].shape[1]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
