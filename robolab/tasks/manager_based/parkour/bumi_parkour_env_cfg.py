import copy
import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from robolab import ROBOLAB_ROOT_DIR
from robolab.assets.robots.roboparty import BUMI_CFG, BUMI_LINKS
from robolab.sensors import get_link_prim_targets
from robolab.tasks.manager_based.parkour.rpo_parkour_env_cfg import (
    AMP_NUM_STEPS,
    ROUGH_TERRAINS_CFG,
    ROUGH_TERRAINS_CFG_PLAY,
    RPOParkourRoughEnvCfg,
    RPOParkourRoughEnvCfg_PLAY,
)


BUMI_PARKOUR_KEY_BODY_NAMES = [
    "l_ankle_roll_link",
    "r_ankle_roll_link",
    "l_knee_pitch_link",
    "r_knee_pitch_link",
    "l_elbow_pitch_link",
    "r_elbow_pitch_link",
]

BUMI_ROUGH_TERRAINS_CFG = copy.deepcopy(ROUGH_TERRAINS_CFG)
BUMI_ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG_PLAY)


def _apply_bumi_parkour_overrides(cfg):
    cfg.scene.robot = BUMI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.left_height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/l_ankle_roll_link"
    cfg.scene.right_height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/r_ankle_roll_link"
    cfg.scene.leg_volume_points.prim_path = "{ENV_REGEX_NS}/Robot/.*_ankle_roll_link"
    cfg.scene.knee_volume_points.prim_path = "{ENV_REGEX_NS}/Robot/.*_knee_pitch_link"
    cfg.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/waist_yaw_link"
    cfg.scene.camera.mesh_prim_paths = ["/World/ground", *get_link_prim_targets(BUMI_LINKS)]
    cfg.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/waist_yaw_link"

    cfg.motion_data.motion_dataset.motion_data_dir = os.path.join(
        ROBOLAB_ROOT_DIR, "data", "motions", "bumi_lab"
    )
    cfg.animation.animation.num_steps_to_use = AMP_NUM_STEPS
    cfg.observations.disc.key_body_pos_b.params = {
        "asset_cfg": SceneEntityCfg(
            name="robot",
            body_names=BUMI_PARKOUR_KEY_BODY_NAMES,
            preserve_order=True,
        )
    }
    cfg.observations.disc.history_length = AMP_NUM_STEPS

    rewards = cfg.rewards.rewards
    rewards.rpo_thigh_yaw_joint_sign_penalty = None
    rewards.joint_deviation_upper_body.params["asset_cfg"] = SceneEntityCfg(
        "robot",
        joint_names=[".*_arm_.*_joint", ".*_elbow_pitch_joint", "waist_yaw_joint"],
    )
    rewards.freeze_upper_torso.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])
    rewards.pelvis_orientation_l2.params["asset_cfg"] = SceneEntityCfg("robot", body_names="waist_yaw_link")
    rewards.feet_stumble.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces",
        body_names=[".*_ankle_roll_link", ".*_knee_pitch_link"],
    )

    cfg.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
        "contact_forces",
        body_names="waist_yaw_link",
    )
    cfg.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["base_link", "waist_yaw_link"]
    cfg.events.scale_link_mass.params["asset_cfg"].body_names = ["l_.*_link", "r_.*_link"]


@configclass
class BumiParkourRoughEnvCfg(RPOParkourRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = BUMI_ROUGH_TERRAINS_CFG
        _apply_bumi_parkour_overrides(self)


@configclass
class BumiParkourRoughEnvCfg_PLAY(RPOParkourRoughEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = BUMI_ROUGH_TERRAINS_CFG_PLAY
        _apply_bumi_parkour_overrides(self)


@configclass
class BumiParkourEnvCfg(BumiParkourRoughEnvCfg):
    pass


@configclass
class BumiParkourEnvCfg_PLAY(BumiParkourRoughEnvCfg_PLAY):
    pass
