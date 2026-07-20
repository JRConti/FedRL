"""Centralized Training of a DQN with Stable-Baselines3 and W&B."""

from pathlib import Path

import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from wandb.integration.sb3 import WandbCallback

from config_dqn import CARTPOLE_DQN_CONFIG as DQN_CONFIG


# Experiment settings.
ENV_ID = "CartPole-v1"
TOTAL_TIMESTEPS = 50_000
EVAL_FREQ = 1_000
EVAL_EPISODES = 20
OUTPUT_DIR = Path("runs/centralized")
WANDB_PROJECT = "SB3"

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
        set_random_seed(seed) # SB3 fixing seed for reproducibility. See https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#fixing-random-seeds-for-reproducibility

        eval_env = make_vec_env(ENV_ID, n_envs=1, seed=seed) # as in RL Zoo
        train_env = make_vec_env(ENV_ID, n_envs=1, seed=seed)

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
