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

"""Bumi flat locomotion (Bumi-Flat / Bumi-Rough actor) MuJoCo sim2sim.

Policy observation layout, matching robolab/tasks/direct/base/base_env.py
compute_current_observations() with 10-frame history stacking (oldest first):
    single frame (72) = [ base_ang_vel (3), projected_gravity (3),
                          velocity command vx/vy/wyaw (3),
                          joint_pos_rel (21), joint_vel_rel (21), last_action (21) ]
    policy input = 10 x 72 = 720
Joint segments use the Isaac runtime joint order; mapping to the MuJoCo
(URDF document) order is built by joint name at runtime.

Keyboard: W/S +-vx, A/D +-vy, Q/E +-yaw rate, 0 zero commands, F camera follow.
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pynput import keyboard
from loop_rate_limiters import RateLimiter


# Isaac Lab runtime articulation joint order for Bumi (BFS over the URDF tree).
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

# Gains / limits / default pose, mirrors BUMI_CFG in robolab/assets/robots/roboparty.py
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
    vx = 0.0
    vy = 0.0
    dyaw = 0.0
    camera_follow = True
    reset_requested = False

    @classmethod
    def zero(cls):
        cls.vx, cls.vy, cls.dyaw = 0.0, 0.0, 0.0
        print("Commands zeroed")

    @classmethod
    def toggle_camera_follow(cls):
        cls.camera_follow = not cls.camera_follow
        print(f"Camera follow: {cls.camera_follow}")


def on_press(key):
    try:
        c = key.char
    except AttributeError:
        return
    step_v, step_w = 0.1, 0.1
    if c == 'w':
        cmd.vx += step_v
    elif c == 's':
        cmd.vx -= step_v
    elif c == 'a':
        cmd.vy += step_v
    elif c == 'd':
        cmd.vy -= step_v
    elif c == 'q':
        cmd.dyaw += step_w
    elif c == 'e':
        cmd.dyaw -= step_w
    elif c == '0':
        cmd.zero()
        return
    elif c == 'f':
        cmd.toggle_camera_follow()
        return
    elif c == 'r':
        cmd.reset_requested = True
        return
    else:
        return
    print(f"cmd: vx={cmd.vx:.2f} vy={cmd.vy:.2f} dyaw={cmd.dyaw:.2f}")


def start_keyboard_listener():
    try:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        return listener
    except Exception as exc:  # headless hosts without X display
        print(f"[WARN] Keyboard listener unavailable: {exc}")
        return None


def get_obs(data):
    """Extract base orientation / velocities / gravity vector from MuJoCo sensors."""
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').data[[1, 2, 3, 0]].astype(np.double)  # wxyz -> xyzw
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # base frame linear velocity
    omega = data.sensor('angular-velocity').data.astype(np.double)
    gvec = r.apply(np.array([0.0, 0.0, -1.0]), inverse=True).astype(np.double)
    return q, dq, v, omega, gvec


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def run_mujoco(policy, cfg, headless=False):
    print("=" * 60)
    print("Keyboard control instructions:")
    print("  W/S: +-vx   A/D: +-vy   Q/E: +-yaw rate")
    print("  0: zero commands   R: reset robot   F: toggle camera follow")
    print("=" * 60)
    keyboard_listener = start_keyboard_listener()

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
    usd2mj = np.array([mj_joint_names.index(n) for n in BUMI_ISAAC_DOF_NAMES], dtype=np.int64)

    kps = np.array([_match_value(BUMI_KP, n) for n in mj_joint_names], dtype=np.double)
    kds = np.array([_match_value(BUMI_KD, n) for n in mj_joint_names], dtype=np.double)
    tau_limit = np.array([_match_value(BUMI_TAU_LIMIT, n) for n in mj_joint_names], dtype=np.double)
    default_pos = np.array([BUMI_DEFAULT_POS.get(n, 0.0) for n in mj_joint_names], dtype=np.double)

    # stand at default pose
    data.qpos[2] = 0.48
    data.qpos[7:] = default_pos
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
    tau = np.zeros(num_actions, dtype=np.double)
    hist_obs = np.zeros((cfg.robot_config.frame_stack, cfg.robot_config.num_single_obs), dtype=np.double)
    is_first_frame = True

    count_lowlevel = 0
    control_freq = 1.0 / (cfg.sim_config.dt * cfg.sim_config.decimation)
    rate_limiter = RateLimiter(frequency=control_freq, warn=False)

    # low-frequency logs for plotting
    time_data, cmd_data, actual_vel_data = [], [], []
    target_joint_data, actual_joint_data = [], []

    for step in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):
        if cmd.reset_requested:
            print('Performing reset: restoring qpos/qvel and zeroing commands')
            data.qpos[:] = initial_qpos
            data.qvel[:] = initial_qvel
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            cmd.zero()
            cmd.reset_requested = False
            action[:] = 0.0
            target_pos = default_pos.copy()
            is_first_frame = True

        q, dq, v, omega, gvec = get_obs(data)
        q = q[7:]   # MuJoCo joint order
        dq = dq[6:]

        # 200Hz physics -> 50Hz policy
        if count_lowlevel % cfg.sim_config.decimation == 0:
            q_rel_mj = q - default_pos
            q_obs = q_rel_mj[usd2mj]   # -> Isaac order
            dq_obs = dq[usd2mj]

            obs = np.zeros([1, cfg.robot_config.num_single_obs], dtype=np.float32)
            obs[0, 0:3] = omega
            obs[0, 3:6] = gvec
            obs[0, 6] = cmd.vx
            obs[0, 7] = cmd.vy
            obs[0, 8] = cmd.dyaw
            obs[0, 9:30] = q_obs
            obs[0, 30:51] = dq_obs
            obs[0, 51:72] = action

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

            time_data.append(step * cfg.sim_config.dt)
            cmd_data.append([cmd.vx, cmd.vy, cmd.dyaw])
            actual_vel_data.append([v[0], v[1], omega[2]])
            target_joint_data.append(target_pos.copy())
            actual_joint_data.append(q.copy())

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

    # --- plots ---
    print("Simulation finished. Generating plots...")
    time_data = np.array(time_data)
    cmd_data = np.array(cmd_data)
    actual_vel_data = np.array(actual_vel_data)
    target_joint_data = np.array(target_joint_data)
    actual_joint_data = np.array(actual_joint_data)

    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    labels = [("Vx [m/s]", 0), ("Vy [m/s]", 1), ("Yaw rate [rad/s]", 2)]
    for ax, (label, k) in zip(axes1, labels):
        ax.plot(time_data, cmd_data[:, k], '--', label='Commanded')
        ax.plot(time_data, actual_vel_data[:, k], label='Actual')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True)
    axes1[-1].set_xlabel("Time [s]")
    fig1.suptitle("Commanded vs Actual Base Velocities")
    fig1.tight_layout()
    fig1.savefig("base_velocities.png")

    n_cols = 4
    n_rows = (num_actions + n_cols - 1) // n_cols
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), sharex=True)
    axes2 = axes2.flatten()
    for i in range(num_actions):
        axes2[i].plot(time_data, target_joint_data[:, i], '--', label='Target')
        axes2[i].plot(time_data, actual_joint_data[:, i], label='Actual')
        axes2[i].set_title(mj_joint_names[i], fontsize=8)
        axes2[i].grid(True)
    axes2[0].legend()
    for i in range(num_actions, len(axes2)):
        fig2.delaxes(axes2[i])
    fig2.suptitle("Target vs Actual Joint Positions")
    fig2.tight_layout()
    fig2.savefig("joint_positions.png")
    print("Plots saved: base_velocities.png, joint_positions.png")


REPO_ROOT = Path(__file__).resolve().parents[3]

TASK_EXPERIMENTS = {
    'Bumi-Flat': 'bumi_flat',
    'Bumi-Rough': 'bumi_rough',
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
                f'{policy_path} missing. Run play.py once to export it, or pass --load_model.'
            )
        return policy_path

    newest = runs[-1]
    policy_path = newest / 'exported' / 'policy.pt'
    if policy_path.is_file():
        return policy_path

    exported = [d for d in runs if (d / 'exported' / 'policy.pt').is_file()]
    if exported:
        fallback = exported[-1]
        print(
            f'[WARN] Newest run {newest.name} has no exported/policy.pt '
            f'(run play.py once to export). Falling back to {fallback.name}.'
        )
        return fallback / 'exported' / 'policy.pt'
    raise FileNotFoundError(
        f'No exported/policy.pt under {log_root}. Run play.py once to export, or pass --load_model.'
    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Bumi flat locomotion MuJoCo sim2sim.')
    parser.add_argument('--task', type=str, default='Bumi-Flat', choices=list(TASK_EXPERIMENTS),
                        help='Task whose latest policy to load when --load_model is not given.')
    parser.add_argument('--load_run', type=str, default=None,
                        help='Specific run directory name; default picks the newest-trained run.')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Explicit JIT policy path (exported/policy.pt); overrides auto discovery.')
    parser.add_argument('--headless', action='store_true', help='Run without GUI and save simulation.mp4.')
    parser.add_argument('--duration', type=float, default=60.0, help='Simulation duration in seconds.')
    parser.add_argument('--vx', type=float, default=0.0, help='Initial forward velocity command.')
    parser.add_argument('--vy', type=float, default=0.0, help='Initial lateral velocity command.')
    parser.add_argument('--dyaw', type=float, default=0.0, help='Initial yaw rate command.')
    args = parser.parse_args()

    if args.load_model is not None:
        load_model = args.load_model
    else:
        load_model = str(resolve_policy(TASK_EXPERIMENTS[args.task], args.load_run))
    print(f'[INFO] policy: {load_model}')

    cmd.vx, cmd.vy, cmd.dyaw = args.vx, args.vy, args.dyaw

    class Sim2simCfg:
        class sim_config:
            # Floating-base MJCF companion of BUMI_URDF_PATH (bumi_edu_pro_collision.urdf).
            mujoco_model_path = BUMI_MJCF_PATH
            sim_duration = args.duration
            dt = 0.005
            decimation = 4

        class robot_config:
            frame_stack = 10
            num_single_obs = 72
            num_observations = num_single_obs * frame_stack
            num_actions = 21
            action_scale = 0.25

    print(f'[INFO] Bumi URDF source: {BUMI_URDF_PATH}')
    print(f'[INFO] MuJoCo model: {BUMI_MJCF_PATH}')
    policy = torch.jit.load(load_model)
    run_mujoco(policy, Sim2simCfg(), args.headless)
