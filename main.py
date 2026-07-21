import argparse
import time
from datetime import datetime
from pathlib import Path

from config_dqn import DQNConfig, get_rb3_dqn_config
from dqn_experiment import (
    build_federated_server,
    close_federated_server,
    load_q_network_parameters,
    make_dqn_model,
    record_global_model,
    save_global_model,
    write_history,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from wandb_logging import (
    finish_wandb_runs,
    initialize_training_wandb,
    log_centralized_metrics,
    log_round_metrics,
)


BROADCAST = False # whether to broadcast the global model to clients after each federated round

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a federated DQN with FedAvg.")
    parser.add_argument("--env-id", default="CartPole-v1", help="Gymnasium environment id.")
    parser.add_argument("--num-clients", type=int, default=1, help="Number of federated clients.")
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=10,
        help="Number of federated rounds.",
    )
    parser.add_argument(
        "--timesteps-per-round",
        type=int,
        default=5_000,
        help="Local DQN timesteps per client and round.",
    )
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes.")
    parser.add_argument(
        "--video-step-length",
        "--video_step_length",
        dest="video_step_length",
        type=int,
        default=0,
        help="Number of timesteps in saved video. 0 means no video is saved.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional path where the final global DQN model is saved.",
    )
    parser.add_argument(
        "--history-json",
        type=Path,
        default=None,
        help="Optional path where round metrics are written as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Choice 1: SB3 default parameters
    # dqn_config = DQNConfig()

    # Choice 2: tuned RL-Zoo parameters (if available for the environment, in hyperparams_rl_zoo folder)
    dqn_config = get_rb3_dqn_config(args.env_id)

    # # Choice 3: custom parameters
    # dqn_config = DQNConfig(
    #     policy="MlpPolicy",
    #     learning_rate=2.3e-3,
    #     batch_size=64,
    #     buffer_size=100_000,
    #     learning_starts=1_000,
    #     gamma=0.99,
    #     target_update_interval=10,
    #     train_freq=256,
    #     gradient_steps=128,
    #     exploration_fraction=0.16,
    #     exploration_final_eps=0.04,
    #     net_arch=[256, 256],
    # )

    seed = dqn_config.seed if dqn_config.seed is not None else 2
    set_random_seed(seed) # SB3 fixing seed for reproducibility. See https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#fixing-random-seeds-for-reproducibility


    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_id = f"DQN-fedavg-{args.env_id}-{timestamp}"
    client_id_list = [f"client-{i}" for i in range(args.num_clients)]

    wandb_runs = None
    server = None
    global_eval_model = None
    centralized_model = None
    centralized_eval_env = None

    try:
        wandb_runs = initialize_training_wandb(
            args,
            experiment_id,
            client_id_list,
            dqn_config=dqn_config,
            seed=seed,
        )

        # define clients and server
        server = build_federated_server(
            env_id=args.env_id,
            base_seed=seed,
            client_ids=client_id_list,
            dqn_config=dqn_config,
            eval_episodes=args.eval_episodes,
            client_sb3_runs=wandb_runs["client_sb3"],
        )

        # define a global model for evaluation of the aggregated parameters
        global_eval_model = make_dqn_model(
            args.env_id,
            seed,
            dqn_config=dqn_config,
        )

        # define a centralized model for comparison with federated training
        centralized_model = make_dqn_model(
            args.env_id,
            seed,
            dqn_config=dqn_config,
        )
        # eval env for centralized model
        centralized_eval_env = make_vec_env(args.env_id, n_envs=1, seed=seed)

        history = []
        final_eval = {"mean_reward": 0.0, "std_reward": 0.0}
        t1 = time.perf_counter()
        for i in range(args.num_rounds):

            # Federated training round (clients train/eval locally and send parameters to server for aggregation)
            round_metrics = server.train_round(args.timesteps_per_round, broadcast=BROADCAST)
            history.append(round_metrics)
            print(
                f"Round {round_metrics['round']}/{args.num_rounds} "
                f"completed with {round_metrics['num_clients']} clients."
            )

            # Global evaluation of the aggregated model
            load_q_network_parameters(
                global_eval_model,
                server.global_parameters,
            )
            mean_reward, std_reward = evaluate_policy(
                global_eval_model,
                global_eval_model.get_env(),
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
            )
            mean_reward, std_reward = float(mean_reward), float(std_reward)
            final_eval = {"mean_reward": mean_reward, "std_reward": std_reward}

            # Log clients/server metrics to WandB and print to console
            log_round_metrics(
                round_index=i,
                round_metrics=round_metrics,
                mean_reward=mean_reward,
                std_reward=std_reward,
                elapsed_seconds=time.perf_counter() - t1,
                args=args,
                runs=wandb_runs["summary"],
                client_runs=wandb_runs["client_summary"],
            )
            print(
                f"Global evaluation after {i + 1} rounds: "
                f"mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}"
            )




            # Centralized training and evaluation for comparison
            centralized_model.learn(
                total_timesteps=args.timesteps_per_round,
                reset_num_timesteps=False,
            )
            centralized_mean_reward, centralized_std_reward = evaluate_policy(
                centralized_model,
                centralized_eval_env,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
            )
            centralized_mean_reward = float(centralized_mean_reward)
            centralized_std_reward = float(centralized_std_reward)
            
            # Log centralized metrics to WandB and print to console
            log_centralized_metrics(
                round_index=i,
                mean_reward=centralized_mean_reward,
                std_reward=centralized_std_reward,
                total_timesteps=centralized_model.num_timesteps,
                args=args,
                runs=wandb_runs["summary"],
            )
            print(
                f"Centralized evaluation after {i + 1} rounds: "
                f"timesteps={centralized_model.num_timesteps}, "
                f"mean_reward={centralized_mean_reward:.2f}, "
                f"std_reward={centralized_std_reward:.2f}"
            )
    finally:
        if global_eval_model is not None:
            global_eval_model.get_env().close()
        if centralized_eval_env is not None:
            centralized_eval_env.close()
        if centralized_model is not None:
            centralized_model.get_env().close()
        if server is not None:
            close_federated_server(server)
        if wandb_runs is not None:
            finish_wandb_runs(wandb_runs)

    t2 = time.perf_counter()
    print("Calculation TIME")
    print(t2 - t1)
    if args.history_json is not None:
        write_history(args.history_json, history, final_eval)
        print(f"Wrote history to {args.history_json}")

    if args.save_model is not None:
        save_global_model(
            args.save_model,
            server.global_parameters,
            args.env_id,
            seed + 20_000,
            dqn_config=dqn_config,
        )
        print(f"Saved final global model to {args.save_model}")

    if args.video_step_length > 0:
        record_global_model(
            path=Path("videos") / experiment_id,
            parameters=server.global_parameters,
            env_id=args.env_id,
            seed=seed,
            video_length=args.video_step_length,
            dqn_config=dqn_config,
        )


if __name__ == "__main__":
    main()
