# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# Copyright (c) 2025-2026, The RoboLab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Bumi BeyondMimic (Bumi-BeyondMimic / Bumi-Getup-Mimic) MuJoCo sim2sim.

Policy observation layout (dim 111, matches beyondmimic_env_cfg.ObservationsCfg.PolicyCfg):
    [ motion joint_pos+joint_vel command (42),
      base_ang_vel (3), projected_gravity (3),
      joint_pos_rel (21), joint_vel_rel (21), last_action (21) ]
Joint-related segments use the Isaac runtime joint order; the mapping to the
MuJoCo (URDF document) order is built by joint name at runtime.
"""

import numpy as np
import mujoco, mujoco_viewer
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from robolab.assets import ISAAC_DATA_DIR
from robolab.assets.robots.roboparty import BUMI_MJCF_PATH, BUMI_URDF_PATH
import torch
import cv2
from pynput import keyboard
from loop_rate_limiters import RateLimiter


# Isaac Lab runtime articulation joint order for Bumi (see rpo_npz_to_bumi_npz.py).
# Motion NPZ joint_pos/joint_vel, the policy action, and the proprio obs all use this order.
BUMI_ISAAC_DOF_NAMES = [
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

# Per-joint gains / limits / default pose, mirrors BUMI_CFG in robolab/assets/robots/roboparty.py
BUMI_KP = {
    ".*_leg_yaw_joint": 60.0,
    ".*_leg_roll_joint": 60.0,
    ".*_leg_pitch_joint": 60.0,
    ".*_knee_pitch_joint": 45.0,
    "waist_yaw_joint": 53.0,
    ".*_ankle_pitch_joint": 10.0,
    ".*_ankle_roll_joint": 10.0,
    ".*_arm_pitch_joint": 12.0,
    ".*_arm_roll_joint": 12.0,
    ".*_arm_yaw_joint": 12.0,
    ".*_elbow_pitch_joint": 12.0,
}
BUMI_KD = {
    ".*_leg_yaw_joint": 2.5,
    ".*_leg_roll_joint": 3.0,
    ".*_leg_pitch_joint": 3.0,
    ".*_knee_pitch_joint": 2.0,
    "waist_yaw_joint": 3.4,
    ".*_ankle_pitch_joint": 0.5,
    ".*_ankle_roll_joint": 0.5,
    ".*_arm_pitch_joint": 0.4,
    ".*_arm_roll_joint": 0.4,
    ".*_arm_yaw_joint": 0.4,
    ".*_elbow_pitch_joint": 0.4,
}
BUMI_TAU_LIMIT = {
    ".*_leg_yaw_joint": 27.0,
    ".*_leg_roll_joint": 27.0,
    ".*_leg_pitch_joint": 60.0,
    ".*_knee_pitch_joint": 60.0,
    "waist_yaw_joint": 27.0,
    ".*_ankle_pitch_joint": 10.0,
    ".*_ankle_roll_joint": 10.0,
    ".*_arm_pitch_joint": 5.5,
    ".*_arm_roll_joint": 5.5,
    ".*_arm_yaw_joint": 5.5,
    ".*_elbow_pitch_joint": 5.5,
}
BUMI_DEFAULT_POS = {
    "l_leg_pitch_joint": -0.1495,
    "r_leg_pitch_joint": -0.1495,
    "l_knee_pitch_joint": 0.3215,
    "r_knee_pitch_joint": 0.3215,
    "l_ankle_pitch_joint": -0.1720,
    "r_ankle_pitch_joint": -0.1720,
    "l_arm_pitch_joint": -0.20,
    "r_arm_pitch_joint": -0.20,
    "l_arm_roll_joint": 0.15,
    "r_arm_roll_joint": -0.15,
    "l_elbow_pitch_joint": -0.35,
    "r_elbow_pitch_joint": -0.35,
}


def _match_value(table: dict, joint_name: str) -> float:
    import re

    for pattern, value in table.items():
        if re.fullmatch(pattern, joint_name):
            return value
    raise KeyError(f"No entry matches joint: {joint_name}")


class cmd:
    camera_follow = True
    reset_requested = False

    @classmethod
    def toggle_camera_follow(cls):
        cls.camera_follow = not cls.camera_follow
        print(f"Camera follow: {cls.camera_follow}")

    @classmethod
    def reset(cls):
        print("Reset")


def on_press(key):
    try:
        if key.char == 'f':
            cmd.toggle_camera_follow()
        elif key.char == '0':
            cmd.reset_requested = True
    except AttributeError:
        pass


def start_keyboard_listener():
    try:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener
    except Exception as exc:  # headless hosts without X display
        print(f"[WARN] Keyboard listener unavailable: {exc}")
        return None


def get_obs(data):
    """Extract base orientation / angular velocity / gravity vector from MuJoCo sensors."""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)  # wxyz -> xyzw
    r = R.from_quat(quat)
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return q, dq, omega, gvec


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg, headless=False, loop=False, motion_file=None):
    def frame_idx(t):
        if loop and num_frames > 0:
            return t % num_frames
        return t if t < num_frames else num_frames - 1

    print("=" * 60)
    print("Keyboard control instructions:")
    print("  0 key: Reset to motion start")
    print("  F key: Toggle camera follow mode")
    print("=" * 60)
    keyboard_listener = start_keyboard_listener()

    motion = np.load(motion_file)
    motion_pos = motion["body_pos_w"]      # (frames, bodies, 3), body 0 = base_link
    motion_quat = motion["body_quat_w"]    # (frames, bodies, 4), wxyz
    m_input_pos = motion["joint_pos"]      # (frames, 21) Isaac order
    m_input_vel = motion["joint_vel"]      # (frames, 21) Isaac order
    num_frames = min(m_input_pos.shape[0], m_input_vel.shape[0], motion_pos.shape[0], motion_quat.shape[0])

    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    data = mujoco.MjData(model)

    # --- name-based joint mapping: Isaac runtime order <-> MuJoCo (URDF) order ---
    mj_joint_names = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        mj_joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j))
    assert len(mj_joint_names) == cfg.robot_config.num_actions, mj_joint_names
    # usd2mj[i] = index in MuJoCo joint vector of the i-th Isaac joint
    usd2mj = np.array([mj_joint_names.index(n) for n in BUMI_ISAAC_DOF_NAMES], dtype=np.int64)

    kps = np.array([_match_value(BUMI_KP, n) for n in mj_joint_names], dtype=np.double)
    kds = np.array([_match_value(BUMI_KD, n) for n in mj_joint_names], dtype=np.double)
    tau_limit = np.array([_match_value(BUMI_TAU_LIMIT, n) for n in mj_joint_names], dtype=np.double)
    default_pos = np.array([BUMI_DEFAULT_POS.get(n, 0.0) for n in mj_joint_names], dtype=np.double)

    # --- initial state from motion frame 0 (important for getup: robot starts lying) ---
    joint0_mj = np.zeros(cfg.robot_config.num_actions, dtype=np.double)
    joint0_mj[usd2mj] = m_input_pos[0, :]
    data.qpos[0:3] = motion_pos[0, 0, :]
    data.qpos[2] += 0.01  # small clearance to avoid initial ground interpenetration
    data.qpos[3:7] = motion_quat[0, 0, :]
    data.qpos[7:] = joint0_mj
    mujoco.mj_forward(model, data)

    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()

    if headless:
        renderer = mujoco.Renderer(model, width=1920, height=1080)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cam = mujoco.MjvCamera()
        cam.distance = 4.0
        cam.azimuth = 45.0
        cam.elevation = -20.0
        cam.lookat = [0, 0, 1]
        out = cv2.VideoWriter('simulation.mp4', fourcc, 1.0 / cfg.sim_config.dt / cfg.sim_config.decimation, (1920, 1080))
    else:
        viewer = mujoco_viewer.MujocoViewer(model, data, mode='window', width=1920, height=1080)
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat = [0, 0, 1]

    num_actions = cfg.robot_config.num_actions
    target_pos = default_pos.copy()
    action = np.zeros(num_actions, dtype=np.double)  # Isaac order
    hist_obs = np.zeros((cfg.robot_config.frame_stack, cfg.robot_config.num_single_obs), dtype=np.double)
    is_first_frame = True

    count_lowlevel = 0
    motion_t = 0
    control_freq = 1.0 / (cfg.sim_config.dt * cfg.sim_config.decimation)
    rate_limiter = RateLimiter(frequency=control_freq, warn=False)
    base_z_err = []
    joint_err = []

    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        if cmd.reset_requested:
            print('Performing reset: restoring qpos/qvel and rewinding motion')
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            cmd.reset()
            cmd.reset_requested = False
            motion_t = 0
            action[:] = 0.0
            is_first_frame = True

        q, dq, omega, gvec = get_obs(data)
        q = q[7:]   # MuJoCo joint order
        dq = dq[6:]

        # 200Hz physics -> 50Hz policy
        if count_lowlevel % cfg.sim_config.decimation == 0:
            idx = frame_idx(motion_t)
            m_input = np.concatenate((m_input_pos[idx, :], m_input_vel[idx, :]), axis=0)
            base_z_err.append(abs(float(data.qpos[2]) - float(motion_pos[idx, 0, 2])))
            joint_err.append(float(np.mean(np.abs(q[usd2mj] - m_input_pos[idx, :]))))

            q_rel_mj = q - default_pos
            q_obs = q_rel_mj[usd2mj]   # -> Isaac order
            dq_obs = dq[usd2mj]

            obs = np.zeros([1, cfg.robot_config.num_single_obs], dtype=np.float32)
            obs[0, 0:42] = m_input
            obs[0, 42:45] = omega
            obs[0, 45:48] = gvec
            obs[0, 48:69] = q_obs
            obs[0, 69:90] = dq_obs
            obs[0, 90:111] = action

            if is_first_frame:
                hist_obs = np.tile(obs, (cfg.robot_config.frame_stack, 1))
                is_first_frame = False
            else:
                hist_obs = np.concatenate((hist_obs[1:], obs.reshape(1, -1)), axis=0)

            policy_input = hist_obs.reshape(1, -1).astype(np.float32)
            with torch.inference_mode():
                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()

            target_q = action * cfg.robot_config.action_scale  # Isaac order
            target_pos[usd2mj] = target_q                      # -> MuJoCo order
            target_pos = target_pos + default_pos

            if headless:
                if cmd.camera_follow:
                    base_pos = data.qpos[0:3].tolist()
                    cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                renderer.update_scene(data, camera=cam)
                out.write(renderer.render())
            else:
                if cmd.camera_follow:
                    base_pos = data.qpos[0:3].tolist()
                    viewer.cam.lookat = [float(base_pos[0]), float(base_pos[1]), float(base_pos[2])]
                viewer.render()

            motion_t += 1
            if not headless:
                rate_limiter.sleep()

        target_vel = np.zeros(num_actions, dtype=np.double)
        tau = pd_control(target_pos, q, kps, target_vel, dq, kds)
        tau = np.clip(tau, -tau_limit, tau_limit)
        data.ctrl = tau
        mujoco.mj_step(model, data)
        count_lowlevel += 1

    if headless:
        out.release()
    else:
        viewer.close()
    if keyboard_listener is not None:
        keyboard_listener.stop()
    print("Simulation finished.")
    if base_z_err:
        print(
            f"[Tracking] base z err mean={np.mean(base_z_err):.3f} m max={np.max(base_z_err):.3f} m | "
            f"joint err mean={np.mean(joint_err):.3f} rad max={np.max(joint_err):.3f} rad"
        )


REPO_ROOT = Path(__file__).resolve().parents[3]

TASK_EXPERIMENTS = {
    'Bumi-BeyondMimic': 'bumi_beyondmimic',
    'Bumi-Getup-Mimic': 'bumi_getup_mimic',
}


def _latest_model_mtime(run_dir):
    models = list(run_dir.glob('model_*.pt'))
    if not models:
        return run_dir.stat().st_mtime
    return max(p.stat().st_mtime for p in models)


def resolve_policy(experiment_name, load_run=None):
    """Find exported/policy.pt: prefer the newest-trained run, else newest exported run."""
    log_root = REPO_ROOT / 'logs' / 'rsl_rl' / experiment_name
    if not log_root.is_dir():
        raise FileNotFoundError(f'No experiment directory: {log_root}')
    runs = sorted((d for d in log_root.iterdir() if d.is_dir()), key=_latest_model_mtime)
    if not runs:
        raise FileNotFoundError(f'No runs in: {log_root}')

    if load_run is not None:
        matches = [d for d in runs if d.name == load_run]
        if not matches:
            raise FileNotFoundError(f'Run not found: {log_root}/{load_run}')
        run_dir = matches[0]
        policy_path = run_dir / 'exported' / 'policy.pt'
        if not policy_path.is_file():
            raise FileNotFoundError(
                f'{policy_path} missing. Run play_bm.py once to export it, or pass --load_model.'
            )
        return policy_path, run_dir

    newest = runs[-1]
    policy_path = newest / 'exported' / 'policy.pt'
    if policy_path.is_file():
        return policy_path, newest

    exported = [d for d in runs if (d / 'exported' / 'policy.pt').is_file()]
    if exported:
        fallback = exported[-1]
        print(
            f'[WARN] Newest run {newest.name} has no exported/policy.pt '
            f'(run play_bm.py once to export). Falling back to {fallback.name}.'
        )
        return fallback / 'exported' / 'policy.pt', fallback
    raise FileNotFoundError(
        f'No exported/policy.pt under {log_root}. Run play_bm.py once to export, or pass --load_model.'
    )


def resolve_motion_file(run_dir):
    """Derive the motion npz used by a run: params/env.yaml first, run name second."""
    import re

    env_yaml = run_dir / 'params' / 'env.yaml'
    if env_yaml.is_file():
        match = re.search(r'motion_file:\s*(\S+)', env_yaml.read_text())
        if match:
            path = Path(match.group(1))
            if path.is_file():
                return path
            print(f'[WARN] motion_file from env.yaml not found on disk: {path}')

    parts = run_dir.name.split('__')
    if len(parts) >= 2:
        candidate = REPO_ROOT / 'robolab' / 'data' / 'motions' / 'bumi_bm' / f'{parts[1]}.npz'
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f'Cannot derive motion file for run {run_dir.name}; pass --motion_file explicitly.'
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Bumi BeyondMimic MuJoCo sim2sim.')
    parser.add_argument('--task', type=str, default='Bumi-BeyondMimic', choices=list(TASK_EXPERIMENTS),
                        help='Task whose latest policy to load when --load_model is not given.')
    parser.add_argument('--load_run', type=str, default=None,
                        help='Specific run directory name; default picks the newest-trained run.')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Explicit JIT policy path (exported/policy.pt); overrides auto discovery.')
    parser.add_argument('--motion_file', type=str, default=None,
                        help='Explicit motion npz; default derives it from the selected run.')
    parser.add_argument('--headless', action='store_true', help='Run without GUI and save simulation.mp4.')
    parser.add_argument('--loop', action='store_true', help='Loop the motion instead of holding the last frame.')
    parser.add_argument('--duration', type=float, default=1000.0, help='Simulation duration in seconds.')
    args = parser.parse_args()

    if args.load_model is not None:
        load_model = args.load_model
        run_dir = None
    else:
        policy_path, run_dir = resolve_policy(TASK_EXPERIMENTS[args.task], args.load_run)
        load_model = str(policy_path)

    if args.motion_file is not None:
        motion_file = args.motion_file
    else:
        if run_dir is None:
            parser.error('--motion_file is required when --load_model is given explicitly.')
        motion_file = str(resolve_motion_file(run_dir))

    print(f'[INFO] policy: {load_model}')
    print(f'[INFO] motion: {motion_file}')
    print(f'[INFO] Bumi URDF source: {BUMI_URDF_PATH}')
    print(f'[INFO] MuJoCo model: {BUMI_MJCF_PATH}')

    class Sim2simCfg:
        class sim_config:
            # Floating-base MJCF companion of BUMI_URDF_PATH (bumi_edu_pro_collision.urdf).
            mujoco_model_path = BUMI_MJCF_PATH
            sim_duration = args.duration
            dt = 0.005
            decimation = 4

        class robot_config:
            frame_stack = 1
            num_single_obs = 111
            num_observations = num_single_obs * frame_stack
            num_actions = 21
            action_scale = 0.25

    policy = torch.jit.load(load_model)
    run_mujoco(policy, Sim2simCfg(), args.headless, args.loop, motion_file)
