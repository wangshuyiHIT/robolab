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


import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robolab.assets import ISAAC_DATA_DIR

# Single source of truth for every Bumi train / play / conversion path.
BUMI_URDF_PATH = os.path.join(
    ISAAC_DATA_DIR,
    "robots",
    "roboparty",
    "bumi",
    "urdf",
    "bumi_edu_pro_collision.urdf",
)
BUMI_USD_DIR = os.path.join(
    ISAAC_DATA_DIR,
    "robots",
    "roboparty",
    "bumi",
    "usd",
)
# MuJoCo companion of the same robot (floating-base MJCF built from the URDF).
BUMI_MJCF_PATH = os.path.join(
    ISAAC_DATA_DIR,
    "robots",
    "roboparty",
    "bumi",
    "mjcf",
    "bumi.xml",
)

RPO_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{ISAAC_DATA_DIR}/robots/roboparty/rpo/urdf/rpo.urdf",
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
        articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            "left_thigh_pitch_joint": -0.1,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_arm_pitch_joint": 0.18,
            "left_arm_roll_joint": 0.06,
            "left_elbow_pitch_joint": 0.78,
            "right_thigh_pitch_joint": -0.1,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_arm_pitch_joint": 0.18,
            "right_arm_roll_joint": -0.06,
            "right_elbow_pitch_joint": 0.78,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_thigh_yaw_joint",
                ".*_thigh_roll_joint",
                ".*_thigh_pitch_joint",
                ".*_knee_joint",
                ".*torso.*",
            ],
            effort_limit_sim=120.0,
            velocity_limit_sim=25.0,
            stiffness={
                ".*_thigh_yaw_joint": 100.0,
                ".*_thigh_roll_joint": 100.0,
                ".*_thigh_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
                ".*torso.*": 150.0,
            },
            damping={
                ".*_thigh_yaw_joint": 3.3,
                ".*_thigh_roll_joint": 3.3,
                ".*_thigh_pitch_joint": 3.3,
                ".*_knee_joint": 5.0,
                ".*torso.*": 5.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "shoulders": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_arm_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
            ],
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            stiffness={
                ".*_elbow_pitch_joint": 30.0,
                ".*_elbow_yaw_joint": 20.0,
            },
            damping={
                ".*_elbow_pitch_joint": 1.5,
                ".*_elbow_yaw_joint": 1.0,
            },
            effort_limit_sim=27.0,
            velocity_limit_sim=8.0,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
    },
)


# The sole is at z=-0.4733 m relative to base_link for this leg pose (measured
# from the active URDF foot collision boxes).  A 0.48 m spawn height therefore
# gives roughly 6 mm of clearance instead of dropping the robot from 0.65 m.
BUMI_STANDING_ROOT_HEIGHT = 0.48
# Bumi and RPO both use arm_pitch axis +Y: positive pitch swings the elbows
# BACKWARD (negative world X).  RPO's standing (+0.18, +0.18) is a slight rear
# tuck.  Bumi needs a left/right-symmetric, slightly forward stance, so pitch
# must be the SAME negative value on both arms — never opposite signs (that
# would put one hand forward and the other back).
# Roll/yaw are mirrored (±).  Elbow bend is negative on Bumi (limit [-2.26, 0]),
# opposite in sign to RPO's positive elbow bend.
BUMI_STANDING_JOINT_POS = {
    "l_leg_yaw_joint": 0.0,
    "l_leg_roll_joint": 0.0,
    "l_leg_pitch_joint": -0.1495,
    "l_knee_pitch_joint": 0.3215,
    "l_ankle_pitch_joint": -0.1720,
    "l_ankle_roll_joint": 0.0,
    "r_leg_yaw_joint": 0.0,
    "r_leg_roll_joint": 0.0,
    "r_leg_pitch_joint": -0.1495,
    "r_knee_pitch_joint": 0.3215,
    "r_ankle_pitch_joint": -0.1720,
    "r_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "l_arm_pitch_joint": -0.20,
    "l_arm_roll_joint": 0.15,
    "l_arm_yaw_joint": 0.0,
    "l_elbow_pitch_joint": -0.35,
    "r_arm_pitch_joint": -0.20,
    "r_arm_roll_joint": -0.15,
    "r_arm_yaw_joint": 0.0,
    "r_elbow_pitch_joint": -0.35,
}


BUMI_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=BUMI_URDF_PATH,
        usd_dir=BUMI_USD_DIR,
        usd_file_name="bumi_edu_pro_collision.usd",
        # Always rebuild USD from the pinned URDF so play/train cannot reuse a
        # stale /tmp conversion of another Bumi file.
        force_usd_conversion=True,
        fix_base=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0,
                damping=0,
            )
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, BUMI_STANDING_ROOT_HEIGHT),
        joint_pos=BUMI_STANDING_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.90,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_leg_yaw_joint",
                ".*_leg_roll_joint",
                ".*_leg_pitch_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit_sim={
                ".*_leg_yaw_joint": 27.0,
                ".*_leg_roll_joint": 27.0,
                ".*_leg_pitch_joint": 60.0,
                ".*_knee_pitch_joint": 60.0,
            },
            velocity_limit_sim={
                ".*_leg_yaw_joint": 9.0,
                ".*_leg_roll_joint": 12.0,
                ".*_leg_pitch_joint": 12.0,
                ".*_knee_pitch_joint": 12.0,
            },
            stiffness={
                ".*_leg_yaw_joint": 60.0,
                ".*_leg_roll_joint": 60.0,
                ".*_leg_pitch_joint": 60.0,
                ".*_knee_pitch_joint": 45.0,
            },
            damping={
                ".*_leg_yaw_joint": 2.5,
                ".*_leg_roll_joint": 3.0,
                ".*_leg_pitch_joint": 3.0,
                ".*_knee_pitch_joint": 2.0,
            },
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "waist": DelayedPDActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=27.0,
            velocity_limit_sim=9.0,
            stiffness=53.0,
            damping=3.4,
            armature=0.01,
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness={
                ".*_ankle_pitch_joint": 10.0,
                ".*_ankle_roll_joint": 10.0,
            },
            damping={
                ".*_ankle_pitch_joint": 0.5,
                ".*_ankle_roll_joint": 0.5,
            },
            armature={
                ".*_ankle_pitch_joint": 0.01,
                ".*_ankle_roll_joint": 0.01,
            },
            min_delay=0,
            max_delay=2,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_arm_pitch_joint",
                ".*_arm_roll_joint",
                ".*_arm_yaw_joint",
                ".*_elbow_pitch_joint",
            ],
            effort_limit_sim=5.5,
            velocity_limit_sim=50.0,
            stiffness=12.0,
            damping=0.4,
            armature=0.001,
            min_delay=0,
            max_delay=2,
        ),
    },
)


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


BUMI_LINKS = [
    "base_link",
    "waist_yaw_link",
    "l_arm_pitch_link",
    "l_arm_roll_link",
    "l_arm_yaw_link",
    "l_elbow_pitch_link",
    "r_arm_pitch_link",
    "r_arm_roll_link",
    "r_arm_yaw_link",
    "r_elbow_pitch_link",
    "l_leg_pitch_link",
    "l_leg_roll_link",
    "l_leg_yaw_link",
    "l_knee_pitch_link",
    "l_ankle_pitch_link",
    "l_ankle_roll_link",
    "r_leg_pitch_link",
    "r_leg_roll_link",
    "r_leg_yaw_link",
    "r_knee_pitch_link",
    "r_ankle_pitch_link",
    "r_ankle_roll_link",
]
