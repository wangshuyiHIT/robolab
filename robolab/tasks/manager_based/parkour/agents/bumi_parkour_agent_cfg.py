from isaaclab.utils import configclass

from robolab.tasks.manager_based.parkour.agents.rpo_parkour_agent_cfg import RPOParkourAmpRunnerCfg


@configclass
class BumiParkourAmpRunnerCfg(RPOParkourAmpRunnerCfg):
    def __post_init__(self):
        self.experiment_name = "bumi_parkour"
        self.wandb_project = "bumi_parkour"
        self.algorithm.symmetry_cfg = None
        self.policy.encoder_cfg = dict(self.policy.encoder_cfg)
        self.policy.encoder_cfg.pop("last_activation", None)
