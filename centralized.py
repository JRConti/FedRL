"""Centralized Training of a DQN with Stable-Baselines3 and W&B."""

from pathlib import Path

import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from config_dqn import DQNConfig


# Experiment settings.
ENV_ID = "CartPole-v1"
TOTAL_TIMESTEPS = 50_000
EVAL_FREQ = 1_000
EVAL_EPISODES = 20
OUTPUT_DIR = Path("runs/centralized")
WANDB_PROJECT = "SB3"

# All DQN parameters are defined here. Edit this instance to try a different
# configuration.
DQN_CONFIG = DQNConfig(
    policy="MlpPolicy",
    learning_rate=1e-3, # 1e-4
    buffer_size=1_000_000,
    learning_starts=100,
    batch_size=32,
    tau=1.0,
    gamma=0.99, # 0.99
    train_freq=4,
    gradient_steps=1,
    replay_buffer_class=None,
    replay_buffer_kwargs=None,
    optimize_memory_usage=False,
    n_steps=1,
    target_update_interval=10, # 10_000
    exploration_fraction=0.1, # 0.1
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05, # 0.05
    max_grad_norm=10.0,
    stats_window_size=100,
    tensorboard_log=None,
    policy_kwargs=None,
    verbose=1,
    seed=0,
    device="auto",
    _init_setup_model=True,
    net_arch=[256, 256], # None
    activation_fn=None,
    features_extractor_class=None,
    features_extractor_kwargs=None,
    normalize_images=None,
    optimizer_class=None,
    optimizer_kwargs=None,
)


DQN_CONFIG = DQNConfig(
    policy="MlpPolicy",
    learning_rate=2.3e-3,
    batch_size=64,
    buffer_size=100_000,
    learning_starts=1_000,
    gamma=0.99,
    target_update_interval=10,
    train_freq=256,
    gradient_steps=128,
    exploration_fraction=0.16,
    exploration_final_eps=0.04,
    net_arch=[256, 256],
    seed=0,
    device="cpu",
    verbose=1,
)


def main() -> None:
    wandb_config = DQN_CONFIG.to_wandb_config()
    wandb_config.update(
        {
            "env_id": ENV_ID,
            "total_timesteps": TOTAL_TIMESTEPS,
            "eval_freq": EVAL_FREQ,
            "eval_episodes": EVAL_EPISODES,
        }
    )

    with wandb.init(
        project=WANDB_PROJECT,
        group="centralized-dqn",
        job_type="centralized-training",
        tags=["centralized", "DQN", "SB3"],
        config=wandb_config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    ) as run:
        # SB3 logs metrics against the number of environment timesteps. Tell
        # W&B to use that step as the default x-axis instead of W&B's internal
        # logging event counter.
        run.define_metric("global_step")
        run.define_metric("*", step_metric="global_step")

        run_dir = OUTPUT_DIR / run.id
        tensorboard_dir = run_dir / "tensorboard"
        eval_dir = run_dir / "eval"
        for directory in (tensorboard_dir, eval_dir):
            directory.mkdir(parents=True, exist_ok=True)

        seed = DQN_CONFIG.seed if DQN_CONFIG.seed is not None else 0
        train_env = make_vec_env(ENV_ID, n_envs=1, seed=seed)
        eval_env = make_vec_env(ENV_ID, n_envs=1, seed=seed + 10_000)

        try:
            model = DQN(
                DQN_CONFIG.policy,
                train_env,
                **DQN_CONFIG.to_kwargs(tensorboard_log=str(tensorboard_dir)),
            )
            callbacks = CallbackList(
                [
                    EvalCallback(
                        eval_env,
                        log_path=str(eval_dir),
                        eval_freq=EVAL_FREQ,
                        n_eval_episodes=EVAL_EPISODES,
                        deterministic=True,
                    ),
                    WandbCallback(
                        verbose=DQN_CONFIG.verbose,
                    ),
                ]
            )

            model.learn(
                total_timesteps=TOTAL_TIMESTEPS,
                callback=callbacks,
                tb_log_name="DQN",
                progress_bar=False,
            )
            run.summary["trained_timesteps"] = model.num_timesteps
        finally:
            train_env.close()
            eval_env.close()


if __name__ == "__main__":
    main()
