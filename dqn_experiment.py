import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, VecVideoRecorder

from client import FederatedDQNClient
from config_dqn import CARTPOLE_DQN_CONFIG
from server import FederatedDQNServer


Parameters = Iterable[np.ndarray | th.Tensor]


def make_dqn_env(
    env_id: str,
    seed: int,
    *,
    render_mode: str | None = None,
) -> VecEnv:
    env_kwargs = None if render_mode is None else {"render_mode": render_mode}
    return make_vec_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs=env_kwargs,
    )


def make_dqn_model(
    env_id: str,
    seed: int,
    *,
    tensorboard_log: str | None = None,
) -> DQN:
    env = make_dqn_env(env_id, seed)
    try:
        return DQN(
            CARTPOLE_DQN_CONFIG.policy,
            env,
            **CARTPOLE_DQN_CONFIG.to_kwargs(
                seed=seed,
                tensorboard_log=tensorboard_log,
            ),
        )
    except Exception:
        env.close()
        raise


def load_q_network_parameters(model: DQN, parameters: Parameters) -> None:
    state_dict = model.q_net.state_dict()
    new_parameters = list(parameters)
    if len(state_dict) != len(new_parameters):
        raise ValueError("Global parameters do not match the DQN Q-network.")

    for (name, old_tensor), new_value in zip(state_dict.items(), new_parameters):
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

    model.set_parameters(
        {
            "q_net": state_dict,
            "q_net_target": state_dict,
        },
        exact_match=False,
    )


def evaluate_dqn(
    model: DQN,
    env: VecEnv,
    eval_episodes: int,
) -> tuple[float, float]:
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=eval_episodes,
        deterministic=True,
    )
    return float(mean_reward), float(std_reward)


def build_federated_server(
    env_id: str,
    base_seed: int,
    client_ids: list[str],
    experiment_id: str,
    eval_episodes: int,
    client_sb3_runs: dict[str, Any] | None = None,
) -> FederatedDQNServer:
    clients = []
    try:
        for index, client_id in enumerate(client_ids):
            model = make_dqn_model(env_id, base_seed + index)
            clients.append(
                FederatedDQNClient(
                    client_id=client_id,
                    project_id=experiment_id,
                    model=model,
                    eval_episodes=eval_episodes,
                    native_wandb_run=(
                        None
                        if client_sb3_runs is None
                        else client_sb3_runs.get(client_id)
                    ),
                )
            )
    except Exception:
        for client in clients:
            client.model.get_env().close()
        raise
    return FederatedDQNServer(clients)


def close_federated_server(server: FederatedDQNServer) -> None:
    for client in server.clients:
        client.model.get_env().close()


def save_global_model(
    path: Path,
    parameters: Parameters,
    env_id: str,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = make_dqn_model(env_id, seed)
    try:
        load_q_network_parameters(model, parameters)
        model.save(path)
    finally:
        model.get_env().close()


def record_global_model(
    path: Path,
    parameters: Parameters,
    env_id: str,
    seed: int,
    video_length: int,
) -> None:
    env = VecVideoRecorder(
        make_dqn_env(env_id, seed, render_mode="rgb_array"),
        str(path),
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
    )
    try:
        model = DQN(
            CARTPOLE_DQN_CONFIG.policy,
            env,
            **CARTPOLE_DQN_CONFIG.to_kwargs(seed=seed),
        )
        load_q_network_parameters(model, parameters)

        observation = env.reset()
        for _ in range(video_length):
            action, _ = model.predict(observation, deterministic=True)
            observation, _, _, _ = env.step(action)
    finally:
        env.close()


def write_history(
    path: Path,
    history: list[dict[str, Any]],
    final_eval: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"history": history, "final_eval": final_eval}, indent=2),
        encoding="utf-8",
    )
