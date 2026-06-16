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
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

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
    parser.add_argument("--real_time_communication", type=bool, default=False, help="Communicate wandb results in real time.")
    parser.add_argument("--video_step_length", type=int, default=0, help="Number of timesteps in saved video.  0 means no video is saved.")
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

def make_video_env() -> gym.Env:
    env_id = 'CartPole-v1'
    env = gym.make(env_id,  render_mode = 'rgb_array')
    return env
    
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

def initialize_wandb(args: argparse.Namespace, project_id: str | None, client_id: str | None, resume: bool, total_timesteps: int):
    #here is a basic function that initializes a wandb connection.
    #If this is a reinitialization of an existing connection, we must set resume=True.
    #The reason is that wandb throws an error if you set resume="must" for the first connection.
    #But it is obligatory for all subsequent connections.
    config = {
        "policy_type": "DQN",
        "env_name": args.env_id,
        "total_timesteps": total_timesteps
    }
    #If we are reopening an exiting connection, we must specify this in the initialization parameters
    if resume:
        resume_val = "must"
    else:
        resume_val = None
    run = wandb.init(
            project=project_id,
            id = client_id,
            config=config,
            sync_tensorboard=True,
            resume=resume_val,
        )
    return(run)

def build_server(args: argparse.Namespace, project_id: str, client_id_list: list[str]) -> FederatedDQNServer:
    clients = []
    #We build our server.
    #The key addition step is that each client must have its wandb run associated with it from its creation.
    #Thus, we intitialize a run for each of these clients and close it as well.
    #This is because having multiple open connections for the same project results in errors.
    for i in range(args.num_clients):
        client_seed = args.seed + i
        run = initialize_wandb(args, project_id, client_id_list[i], False, args.timesteps_per_round)
        model = make_model(args.env_id, client_seed, args, run)
        client = FederatedDQNClient(client_id=client_id_list[i],
                                          project_id=project_id,
                                          model=model,
                                          env_id=args.env_id)
        clients.append(client)
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
    model = make_model(args.env_id, args.seed + 20_000, args, None)
    load_global_parameters(model, server.global_parameters)
    model.save(path)

def save_video(path: Path, parameters: list[np.ndarray], args: argparse.Namespace) -> None:
    #Here we save our server video.  Can be adapted to clients if necessary.
    env = DummyVecEnv([make_video_env])
    env = VecVideoRecorder(
        env,
        path,
        record_video_trigger=lambda x: x == 0,
        video_length=args.video_step_length,
    )
    model = DQN("MlpPolicy", env, verbose=1)
    load_global_parameters(model, parameters)

    obs = env.reset()
    for _ in range(args.video_step_length):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

def push_timestep_data(run: wandb.sdk.wandb_run.Run,
                       data: list[list],
                       column_list: list, 
                       title: str,
                       x_var: str,
                       y_var: str):
    
    table = wandb.Table(data=data, columns=column_list)
    run.log({title : wandb.plot.line(table, x_var, y_var, title=title)})

def main() -> None:
    args = parse_args()
    #Here we create unique keys for the project based on the current datetime.
    #We do the same for each client, then build our sever.
    project_id = str(datetime.now()).split(".")[0].replace(" ","-").replace(":","")
    client_id_list = [project_id + "-" + str(i) for i in range(args.num_clients)]
    server = build_server(args, project_id, client_id_list)

    #We now run our main loop.
    #We save the metrics for loggin purposes and also extract them as lists
    #in our data object which will be saved as a wandb table.
    #After each training round, we push our tables so that the reward
    #progress can be monitored as close to real time as possible.
    #This could probably be done incrementally.  To improve.
    column_list = ["round", "agent","total_timesteps", "mean_reward"]
    title = "Mean Reward by Timestep"
    history = []
    server_data = []
    data = [[] for i in range(args.num_rounds)]
    for i in range(args.num_rounds):
        round_metrics = server.train_round(args.timesteps_per_round,
                                           args.real_time_communication)
        history.append(round_metrics)
        print(
            f"Round {round_metrics['round']}/{args.num_rounds} "
            f"completed with {round_metrics['num_clients']} clients."
        )
        #Now we post the server model statistics.
        resume_val = (i>0)
        server_run = initialize_wandb(args=args,
                                     project_id=project_id,
                                     client_id=project_id + "-server",
                                     resume=resume_val,
                                     total_timesteps=(i+1)*args.timesteps_per_round)
        mean_reward, std_reward = evaluate_global_model(server, args, server_run)
        server_data.append([i,
                             "server",
                             (i+1)*args.timesteps_per_round,
                             mean_reward])

        push_timestep_data(server_run,
                                   data=server_data,
                                   column_list=column_list,
                                   title=title,
                                   x_var="total_timesteps",
                                   y_var="mean_reward")
        server_run.finish()
        for j in range(len(round_metrics["client_metrics"])):
            data[j].append([i,
                         j,
                         round_metrics["client_metrics"][j]["total_timesteps"],
                         round_metrics["client_metrics"][j]["mean_reward"]])
            #Pushing this rounds data.
            run = initialize_wandb(args,
                                   project_id,
                                   client_id_list[j],
                                   True,
                                   args.timesteps_per_round)
            #Push our mean reward table
            #We can also push some other data now that the connection is open.
            push_timestep_data(run,
                           data=data[j],
                           column_list=column_list,
                           title=title,
                           x_var="total_timesteps",
                           y_var="mean_reward")
            run.finish()
            
    #We now intialize a server wandb connection.
    #The current status is we are saving the global metrics but will expand on this.
    server_run = initialize_wandb(args=args,
                                 project_id=project_id,
                                 client_id=project_id + "-server",
                                 resume=True,
                                 total_timesteps=args.num_rounds)
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

    if args.video_step_length > 0:
        parameters = server.global_parameters
        save_video(path=f"videos/{project_id}",
                            parameters=parameters,
                            args=args)
    server_run.finish()

if __name__ == "__main__":
    main()
