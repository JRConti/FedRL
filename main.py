import argparse
import time
from datetime import datetime
from pathlib import Path

from config_dqn import CARTPOLE_DQN_CONFIG
from dqn_experiment import (
    build_federated_server,
    close_federated_server,
    evaluate_dqn,
    load_q_network_parameters,
    make_dqn_env,
    make_dqn_model,
    record_global_model,
    save_global_model,
    write_history,
)
from wandb_logging import (
    finish_wandb_runs,
    initialize_training_wandb,
    log_centralized_metrics,
    log_round_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a federated DQN with FedAvg.")
    parser.add_argument("--env-id", default="CartPole-v1", help="Gymnasium environment id.")
    parser.add_argument("--num-clients", type=int, default=2, help="Number of federated clients.")
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
    args = parser.parse_args()
    args.seed = CARTPOLE_DQN_CONFIG.seed or 0
    return args


def main() -> None:
    args = parse_args()
    args.dqn_config = CARTPOLE_DQN_CONFIG

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_id = f"DQN-fedavg-{args.env_id}-{timestamp}"
    client_id_list = [f"client-{i}" for i in range(args.num_clients)]

    wandb_runs = initialize_training_wandb(args, experiment_id, client_id_list)

    server = build_federated_server(
        env_id=args.env_id,
        base_seed=args.seed,
        client_ids=client_id_list,
        experiment_id=experiment_id,
        eval_episodes=args.eval_episodes,
        client_sb3_runs=wandb_runs["client_sb3"],
    )
    centralized_model = make_dqn_model(args.env_id, args.seed)
    centralized_eval_env = make_dqn_env(args.env_id, args.seed)
    global_eval_model = make_dqn_model(args.env_id, args.seed)

    history = []
    final_eval = {"mean_reward": 0.0, "std_reward": 0.0}
    t1 = time.perf_counter()
    try:
        for i in range(args.num_rounds):

            # Federated training round
            round_metrics = server.train_round(args.timesteps_per_round)
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
            mean_reward, std_reward = evaluate_dqn(
                global_eval_model,
                global_eval_model.get_env(),
                args.eval_episodes,
            )
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
            centralized_mean_reward, centralized_std_reward = evaluate_dqn(
                centralized_model,
                centralized_eval_env,
                args.eval_episodes,
            )
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
        global_eval_model.get_env().close()
        centralized_eval_env.close()
        centralized_model.get_env().close()
        close_federated_server(server)
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
            args.seed + 20_000,
        )
        print(f"Saved final global model to {args.save_model}")

    if args.video_step_length > 0:
        record_global_model(
            path=Path("videos") / experiment_id,
            parameters=server.global_parameters,
            env_id=args.env_id,
            seed=args.seed,
            video_length=args.video_step_length,
        )


if __name__ == "__main__":
    main()
