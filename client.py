from typing import Any, Iterable

import numpy as np
import torch as th

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
from wandb.integration.sb3 import WandbCallback


class FederatedDQNClient:
    """
    Simple federated client for Stable-Baselines3 DQN.

    The client sends and receives only the online Q-network weights. Those are
    the trainable DQN parameters that can be averaged by a federated server.
    """

    def __init__(
        self,
        client_id: str | int,
        project_id: str | int,
        model: DQN,
        env_id: str,
        aggregation_weight: float = 1.0,
        sync_target_network: bool = True,
    ) -> None:
        self.client_id = client_id
        self.project_id = project_id
        self.env_id = env_id
        self.model = model
        self.aggregation_weight = aggregation_weight
        self.sync_target_network = sync_target_network

    def train(self,
              total_timesteps: int,
              wand_run_id: str | None,
              **learn_kwargs: Any) -> dict[str, Any]:
        """Train the local DQN model for one federated round."""
        learn_kwargs.setdefault("reset_num_timesteps", False)
        timesteps_before = self.model.num_timesteps

        if wand_run_id is None:
            self.model.learn(total_timesteps=total_timesteps,
                             **learn_kwargs)
        else:
            config = {
                "policy_type": "DQN",
                "env_name": self.env_id,
                "total_timesteps": (total_timesteps)
            }
            run = self.reinitialize_wandb(wand_run_id, config)
            wandCallBack=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{wand_run_id}",
                verbose=2,
            )
            self.model.learn(total_timesteps=total_timesteps,
                             callback=wandCallBack,
                             **learn_kwargs)
            run.finish()
        trained_timesteps = self.model.num_timesteps - timesteps_before
        mean_reward, std_reward = evaluate_policy(self.model,
                        self.model.get_env(),
                        n_eval_episodes=10,
                        deterministic=True,
                        )

        return {
            "client_id": self.client_id,
            "timesteps": trained_timesteps,
            "total_timesteps": self.model.num_timesteps,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        }

    def get_parameters(self) -> list[np.ndarray]:
        """Return the online Q-network weights as a list of NumPy arrays."""
        return [
            tensor.detach().cpu().numpy().copy()
            for tensor in self.model.q_net.state_dict().values()
        ]

    def set_parameters(self, parameters: Iterable[np.ndarray]) -> None:
        """Load global Q-network weights received from the federated server."""
        state_dict = self.model.q_net.state_dict()

        for (name, old_tensor), new_value in zip(state_dict.items(), parameters):
            state_dict[name] = th.as_tensor(
                new_value,
                dtype=old_tensor.dtype,
                device=old_tensor.device,
            )

        self.model.q_net.load_state_dict(state_dict)

        # DQN uses a target network for TD targets. Sync it after receiving a
        # global model so the next local round starts from a consistent state.
        if self.sync_target_network:
            self.model.q_net_target.load_state_dict(state_dict)

    def reinitialize_wandb(self, wand_run_id: str, config: dict):
        run = wandb.init(
                        project=self.project_id,
                        id = wand_run_id,
                        config=config,
                        sync_tensorboard=True,
                        resume = "must")
        return run
        
    def fit(
        self,
        parameters: Iterable[np.ndarray] | None,
        total_timesteps: int,
        wand_run_id: str | None,
        **learn_kwargs: Any,
    ) -> tuple[list[np.ndarray], float, dict[str, Any]]:
        """
        Run one federated round: load global weights, train locally, return
        updated weights and the client aggregation weight.
        """
        if parameters is not None:
            self.set_parameters(parameters)

        if wand_run_id is None:
            metrics = self.train(total_timesteps, None, **learn_kwargs)
        else:
            metrics = self.train(total_timesteps, wand_run_id, **learn_kwargs)
        return self.get_parameters(), self.aggregation_weight, metrics


