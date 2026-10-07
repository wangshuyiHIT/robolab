import gymnasium as gym

from . import agents

gym.register(
    id="RPO-Parkour",
    entry_point="robolab.tasks.manager_based.parkour.parkour_env:ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_parkour_env_cfg:RPOParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rpo_parkour_agent_cfg:RPOParkourAmpRunnerCfg",
    },
)

gym.register(
    id="RPO-Parkour-Play",
    entry_point="robolab.tasks.manager_based.parkour.parkour_env:ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rpo_parkour_env_cfg:RPOParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rpo_parkour_agent_cfg:RPOParkourAmpRunnerCfg",
    },
)

gym.register(
    id="Bumi-Parkour",
    entry_point="robolab.tasks.manager_based.parkour.parkour_env:ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bumi_parkour_env_cfg:BumiParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.bumi_parkour_agent_cfg:BumiParkourAmpRunnerCfg",
    },
)

gym.register(
    id="Bumi-Parkour-Play",
    entry_point="robolab.tasks.manager_based.parkour.parkour_env:ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bumi_parkour_env_cfg:BumiParkourEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.bumi_parkour_agent_cfg:BumiParkourAmpRunnerCfg",
    },
)
