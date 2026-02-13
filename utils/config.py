"""
Global configuration for MAPPO experiments
"""

CONFIG = {
    # === Training ===
    "episodes": 50,
    "learning_rate": 1e-3,

    # === Observation & Model ===
    "obs_dim": 3,
    "hidden_dim": 64,

    # === Action dimensions per agent ===
    "scheduler_action_dim": 10,
    "dispatcher_action_dim": 5,
    "plant_action_dim": 3,

    # === Environment ===
    "domain_randomization": True,

    # === Paths ===
    "model_dir": "trained_models",
    "log_dir": "logs",

    # === Evaluation ===
    "baseline_episodes": 20
}
