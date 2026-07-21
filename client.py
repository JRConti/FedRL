from typing import Any, Iterable

import numpy as np
import torch as th

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Logger

from wandb_logging import WandbSB3OutputFormat


class FederatedDQNClient:
    """
    Simple federated client for Stable-Baselines3 DQN.

    The client sends and receives only the online Q-network weights. Those are
    the trainable DQN parameters that can be averaged by a federated server.
    """

    def __init__(
        self,
        client_id: str | int,
        model: DQN,
        eval_env: Any,
        aggregation_weight: float = 1.0,
        sync_target_network: bool = True,
        eval_episodes: int = 10,
        native_wandb_run: Any | None = None,
    ) -> None:
        self.client_id = client_id
        self.model = model
        self.aggregation_weight = aggregation_weight
        self.sync_target_network = sync_target_network
        self.eval_env = eval_env
        self.eval_episodes = eval_episodes
        self.native_wandb_run = native_wandb_run
        self._configure_logger()

    def train(self, total_timesteps: int, **learn_kwargs: Any) -> dict[str, Any]:
        """Train the local DQN model for one federated round."""
        learn_kwargs.setdefault("reset_num_timesteps", False)
        learn_kwargs.setdefault("tb_log_name", f"client_{self.client_id}")
        timesteps_before = self.model.num_timesteps

        self.model.learn(total_timesteps=total_timesteps, **learn_kwargs)

        trained_timesteps = self.model.num_timesteps - timesteps_before
        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.eval_episodes,
            deterministic=True,
        )

        return {
            "client_id": self.client_id,
            "timesteps": trained_timesteps,
            "total_timesteps": self.model.num_timesteps,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        }

    def _configure_logger(self) -> None:
        if self.native_wandb_run is None:
            return

        self.model.set_logger(
            Logger(
                folder=None,
                output_formats=[
                    WandbSB3OutputFormat(
                        self.native_wandb_run,
                        metric_prefix=str(self.client_id),
                    )
                ],
            )
        )

    def get_parameters(self) -> list[np.ndarray | th.Tensor]:
        """Return CPU copies of the online Q-network weights."""
        return [
            tensor.detach().cpu().clone()
            for tensor in self.model.q_net.state_dict().values()
        ]

    def set_parameters(self, parameters: Iterable[np.ndarray | th.Tensor]) -> None:
        """Load global Q-network weights received from the federated server."""
        state_dict = self.model.q_net.state_dict()
        new_parameters = list(parameters)

        if len(new_parameters) != len(state_dict):
            raise ValueError(
                "Received parameters do not match the DQN Q-network tensor count."
            )

        for (name, old_tensor), new_value in zip(state_dict.items(), new_parameters):
            if new_value.shape != tuple(old_tensor.shape):
                raise ValueError(
                    f"Received parameter '{name}' has shape {new_value.shape}, "
                    f"expected {tuple(old_tensor.shape)}."
                )
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

    def fit(
        self,
        parameters: Iterable[np.ndarray | th.Tensor] | None,
        total_timesteps: int,
        **learn_kwargs: Any,
    ) -> tuple[list[np.ndarray | th.Tensor], float, dict[str, Any]]:
        """
        Run one federated round: load global weights, train locally, return
        updated weights and the client aggregation weight.
        """
        if parameters is not None:
            self.set_parameters(parameters)

        metrics = self.train(total_timesteps, **learn_kwargs)
        return self.get_parameters(), self.aggregation_weight, metrics
