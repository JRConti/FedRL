import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th

from client import FederatedDQNClient
from config_dqn import DQNConfig
from server import FederatedDQNServer
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from wandb_logging import (
    finish_wandb_runs,
    initialize_training_wandb,
    log_centralized_metrics,
    log_round_metrics,
)


ALGORITHMS = {
    "DQN": DQN,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a federated DQN with FedAvg.")
    parser.add_argument(
        "--algo",
        choices=tuple(ALGORITHMS),
        default="DQN",
        help="RL algorithm to use.",
    )
    parser.add_argument("--env-id", default="CartPole-v1", help="Gymnasium environment id.")
    parser.add_argument("--num-clients", type=int, default=1, help="Number of federated clients.")
    # 2
    parser.add_argument("--num-rounds", type=int, default=1_000, help="Number of federated rounds.") # 10
    parser.add_argument(
        "--timesteps-per-round",
        type=int,
        default=1_000, # 
        help="Local DQN timesteps per client and round.",
    )
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--learning-starts", type=int, default=1, help="DQN learning_starts.") 
    # 100

    parser.add_argument("--buffer-size", type=int, default=1_000_000, help="DQN replay buffer size.")
    parser.add_argument("--batch-size", type=int, default=32, help="DQN batch size.")
    parser.add_argument("--verbose", type=int, default=0, help="Stable-Baselines3 verbosity.")
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


def make_env(env_id: str, seed: int) -> gym.Env:
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return Monitor(env)


def make_video_env(env_id: str) -> gym.Env:
    env = gym.make(env_id, render_mode="rgb_array")
    return env


def model_class(args: argparse.Namespace) -> type[DQN]:
    return ALGORITHMS[getattr(args, "algo", "DQN")]


def make_model(
    env_id: str,
    seed: int,
    args: argparse.Namespace,
    tensorboard_log: str | None = None,
) -> DQN:
    env = make_env(env_id, seed)
    config = DQNConfig.from_args(args)
    return model_class(args)(
        config.policy,
        env,
        **config.to_kwargs(seed=seed, tensorboard_log=tensorboard_log),
    )


def build_server(
    args: argparse.Namespace,
    experiment_id: str,
    client_id_list: list[str],
    client_sb3_runs: dict[str, Any] | None = None,
) -> FederatedDQNServer:
    clients = []
    for i in range(args.num_clients):
        client_seed = args.seed + i
        client_id = client_id_list[i]
        model = make_model(
            args.env_id,
            client_seed,
            args,
        )
        client = FederatedDQNClient(
            client_id=client_id,
            project_id=experiment_id,
            model=model,
            eval_episodes=args.eval_episodes,
            native_wandb_run=(
                None if client_sb3_runs is None else client_sb3_runs.get(client_id)
            ),
        )
        clients.append(client)
    return FederatedDQNServer(clients)


def load_global_parameters(model: DQN, parameters: list[np.ndarray | th.Tensor]) -> None:
    state_dict = model.q_net.state_dict()
    if len(state_dict) != len(parameters):
        raise ValueError("Global parameters do not match the DQN Q-network.")

    for (name, old_tensor), new_value in zip(state_dict.items(), parameters):
        if tuple(new_value.shape) != tuple(old_tensor.shape):
            raise ValueError(
                f"Global parameter '{name}' has shape {tuple(new_value.shape)}, "
                f"expected {tuple(old_tensor.shape)}."
            )
        state_dict[name] = th.as_tensor(
            new_value,
            dtype=old_tensor.dtype,
            device=old_tensor.device,
        )

    model.q_net.load_state_dict(state_dict)
    model.q_net_target.load_state_dict(state_dict)


def evaluate_global_model(
    server: FederatedDQNServer,
    args: argparse.Namespace,
    eval_model: DQN,
) -> tuple[float, float]:
    load_global_parameters(eval_model, server.global_parameters)
    mean_reward, std_reward = evaluate_policy(
        eval_model,
        eval_model.get_env(),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )
    return float(mean_reward), float(std_reward)


def write_history(path: Path, history: list[dict[str, Any]], final_eval: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "history": history,
        "final_eval": final_eval,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_global_model(path: Path, server: FederatedDQNServer, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = make_model(args.env_id, args.seed + 20_000, args)
    try:
        load_global_parameters(model, server.global_parameters)
        model.save(path)
    finally:
        model.get_env().close()


def close_server_clients(server: FederatedDQNServer) -> None:
    for client in server.clients:
        client.model.get_env().close()


def save_video(path: Path, parameters: list[np.ndarray | th.Tensor], args: argparse.Namespace) -> None:
    path.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv([lambda: make_video_env(args.env_id)])
    env = VecVideoRecorder(
        env,
        path,
        record_video_trigger=lambda x: x == 0,
        video_length=args.video_step_length,
    )
    try:
        model = model_class(args)(
            "MlpPolicy",
            env,
            **DQNConfig.from_args(args).to_kwargs(seed=args.seed),
        )
        load_global_parameters(model, parameters)

        obs = env.reset()
        for _ in range(args.video_step_length):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    args.dqn_config = DQNConfig.from_args(args)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_id = f"{args.algo}-fedavg-{args.env_id}-{timestamp}"
    client_id_list = [f"client-{i}" for i in range(args.num_clients)]

    wandb_runs = initialize_training_wandb(args, experiment_id, client_id_list)

    server = build_server(args, experiment_id, client_id_list, wandb_runs["client_sb3"])
    centralized_model = make_model(args.env_id, args.seed + 30_000, args)
    centralized_eval_env = make_env(args.env_id, args.seed + 10_000)
    global_eval_model = make_model(args.env_id, args.seed + 10_000, args)

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
            mean_reward, std_reward = evaluate_global_model(
                server,
                args,
                global_eval_model,
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
            centralized_mean_reward, centralized_std_reward = evaluate_policy(
                centralized_model,
                centralized_eval_env,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
            )
            centralized_mean_reward = float(centralized_mean_reward)
            centralized_std_reward = float(centralized_std_reward)
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
        close_server_clients(server)
        finish_wandb_runs(wandb_runs)

    t2 = time.perf_counter()
    print("Calculation TIME")
    print(t2 - t1)
    if args.history_json is not None:
        write_history(args.history_json, history, final_eval)
        print(f"Wrote history to {args.history_json}")

    if args.save_model is not None:
        save_global_model(args.save_model, server, args)
        print(f"Saved final global model to {args.save_model}")

    if args.video_step_length > 0:
        parameters = server.global_parameters
        save_video(
            path=Path("videos") / experiment_id,
            parameters=parameters,
            args=args,
        )

if __name__ == "__main__":
    main()
