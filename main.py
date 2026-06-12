import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import gymnasium as gym
from datetime import datetime

from client import FederatedDQNClient
from server import FederatedDQNServer
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a federated DQN with FedAvg.")
    parser.add_argument("--env-id", default="CartPole-v1", help="Gymnasium environment id.")
    parser.add_argument("--num-clients", type=int, default=2, help="Number of federated clients.")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of federated rounds.")
    parser.add_argument(
        "--timesteps-per-round",
        type=int,
        default=1_000,
        help="Local DQN timesteps per client and round.",
    )
    parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--learning-starts", type=int, default=100, help="DQN learning_starts.")
    parser.add_argument("--buffer-size", type=int, default=50_000, help="DQN replay buffer size.")
    parser.add_argument("--batch-size", type=int, default=32, help="DQN batch size.")
    parser.add_argument("--verbose", type=int, default=0, help="Stable-Baselines3 verbosity.")
    parser.add_argument("--parallel", type=bool, default=False, help="Calculate experiements in series or parallel.")
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


def make_model(env_id: str,
               seed: int,
               args: argparse.Namespace,
               wand_run: wandb.sdk.wandb_run.Run | None) -> DQN:
    env = make_env(env_id, seed)
    if wand_run is None:
        return DQN(
            "MlpPolicy",
            env,
            seed=seed,
            learning_starts=args.learning_starts,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            verbose=args.verbose,
        )

    else:
        return DQN(
            "MlpPolicy",
            env,
            seed=seed,
            learning_starts=args.learning_starts,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            verbose=args.verbose,
            tensorboard_log=f"runs/{wand_run.id}",
        )


def build_server(args: argparse.Namespace, project_id: str, client_id_list: list[str]) -> FederatedDQNServer:
    clients = []
    config = {
        "policy_type": "DQN",
        "env_name": args.env_id,
        "total_timesteps": (args.timesteps_per_round)
    }
    for i in range(args.num_clients):
        client_seed = args.seed + i
        if args.parallel and (i > 0):
            model = make_model(args.env_id, client_seed, args, None)
            clients.append(FederatedDQNClient(client_id=client_id_list[i],
                                              model=model,
                                              env_id=args.env_id))
        else:
            run = wandb.init(
                project=project_id,
                id = client_id_list[i],
                config=config,
                sync_tensorboard=True,
            )
            model = make_model(args.env_id, client_seed, args, run)
            clients.append(FederatedDQNClient(client_id=client_id_list[i],
                                              model=model,
                                              env_id=args.env_id))
            run.finish()
        
    return FederatedDQNServer(clients)


def load_global_parameters(model: DQN, parameters: list[np.ndarray]) -> None:
    state_dict = model.q_net.state_dict()
    if len(state_dict) != len(parameters):
        raise ValueError("Global parameters do not match the DQN Q-network.")

    for (name, old_tensor), new_value in zip(state_dict.items(), parameters):
        state_dict[name] = old_tensor.new_tensor(new_value)

    model.q_net.load_state_dict(state_dict)
    model.q_net_target.load_state_dict(state_dict)


def evaluate_global_model(
    server: FederatedDQNServer,
    args: argparse.Namespace,
    wand_run: wandb.sdk.wandb_run.Run | None,
) -> tuple[float, float]:
    eval_model = make_model(args.env_id,
                            args.seed + 10_000,
                            args,
                            wand_run)
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
    load_global_parameters(model, server.global_parameters)
    model.save(path)


def main() -> None:
    args = parse_args()
    project_id = str(datetime.now()).split(".")[0].replace(" ","-").replace(":","")
    if args.parallel and (args.num_clients > 1):
        client_id_list = [project_id + "-" + str(0)] + [None]*(args.num_clients-1)
    else:
        client_id_list = [project_id + "-" + str(i) for i in range(args.num_clients)]
    server = build_server(args, project_id, client_id_list)
    history = []
    for i in range(args.num_rounds):
        round_metrics = server.train_round(args.timesteps_per_round,
                                           client_id_list)
        history.append(round_metrics)
        print(
            f"Round {round_metrics['round']}/{args.num_rounds} "
            f"completed with {round_metrics['num_clients']} clients."
        )
    print(history)
    #Add custom table with aggregated values.
    column_list = ["round", "agent","total_timesteps", "mean_reward"]

    data = [[] for j in range(args.num_clients)]
    for i in range(len(history)):
        for j in range(len(history[i]["client_metrics"])):
            data[j].append([i,
                         j,
                         history[i]["client_metrics"][j]["total_timesteps"],
                         history[i]["client_metrics"][j]["mean_reward"]])
    print(data)
    for j in range(args.num_clients):
        config = {
            "policy_type": "DQN",
            "env_name": args.env_id,
            "total_timesteps": (args.timesteps_per_round)
        }
        if j ==0:
            run = wandb.init(
                project=project_id,
                id = project_id + "-" + str(j),
                config=config,
                sync_tensorboard=True,
                resume = "must",
            )
        else:
            run = wandb.init(
                project=project_id,
                id = project_id + "-" + str(j),
                config=config,
                sync_tensorboard=True,
            )
        table = wandb.Table(data=data[j], columns = column_list)
        run.log({"my_lineplot_id" : wandb.plot.line(table, "total_timesteps", 
                   "mean_reward", title="Mean Reward by Timestep")})
        run.finish()
    
    config = {
        "policy_type": "DQN",
        "env_name": args.env_id,
        "total_timesteps": (args.num_rounds)
    }
    server_run = wandb.init(
        project=project_id,
        id = project_id + "-server",
        config=config,
        sync_tensorboard=True,
        )
    mean_reward, std_reward = evaluate_global_model(server, args, server_run)
    final_eval = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "eval_episodes": args.eval_episodes,
    }
    print(
        "Final global evaluation: "
        f"mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}"
    )

    if args.history_json is not None:
        write_history(args.history_json, history, final_eval)
        print(f"Wrote history to {args.history_json}")

    if args.save_model is not None:
        save_global_model(args.save_model, server, args)
        print(f"Saved final global model to {args.save_model}")
    '''
    #Add custom table with aggregated values.
    column_list = ["round", "agent","total_timesteps", "mean_reward"]
    data = []
    for i in range(len(history)):
        for j in range(len(history[i]["client_metrics"])):
            data.append([i,j, history[i]["client_metrics"][j]["total_timesteps"], history[i]["client_metrics"][j]["mean_reward"]])
    table = wandb.Table(data=data, columns = column_list)
    server_run.log({"my_lineplot_id" : wandb.plot.line(table, "total_timesteps", 
               "mean_reward", stroke=None, title="Mean Reward by Timestep")})
    '''
    server_run.finish()

if __name__ == "__main__":
    main()
