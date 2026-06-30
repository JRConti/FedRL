from typing import Any, Iterable

import numpy as np
import torch as th

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import KVWriter, Logger


class WandbSB3OutputFormat(KVWriter):
    """Write Stable-Baselines3 scalar logger values to one W&B run."""

    def __init__(
        self,
        run: Any,
        step_metric: str = "global_step",
        metric_prefix: str | None = None,
        include_unprefixed_metrics: bool = True,
    ) -> None:
        self.run = run
        self.step_metric = step_metric
        self.metric_prefix = metric_prefix.strip("/") if metric_prefix else None
        self.include_unprefixed_metrics = include_unprefixed_metrics

    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        payload: dict[str, float | int] = {
            self.step_metric: step,
            "local step": step,
        }
        for key, value in key_values.items():
            excluded = key_excluded.get(key, ())
            if "wandb" in excluded:
                continue
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, bool):
                self._add_metric(payload, key, int(value))
            elif isinstance(value, int | float):
                self._add_metric(payload, key, value)

        if len(payload) > 1:
            self.run.log(payload, step=step)

    def _add_metric(self, payload: dict[str, float | int], key: str, value: float | int) -> None:
        if self.include_unprefixed_metrics:
            payload[key] = value
        payload[self._metric_name(key)] = value

    def _metric_name(self, key: str) -> str:
        if self.metric_prefix is None:
            return key
        return f"{self.metric_prefix}/{key}"

    def close(self) -> None:
        pass


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
        aggregation_weight: float = 1.0,
        sync_target_network: bool = True,
        native_wandb_run: Any | None = None,
    ) -> None:
        self.client_id = client_id
        self.project_id = project_id
        self.model = model
        self.aggregation_weight = aggregation_weight
        self.sync_target_network = sync_target_network
        self.native_wandb_run = native_wandb_run

    def train(self, total_timesteps: int, **learn_kwargs: Any) -> dict[str, Any]:
        """Train the local DQN model for one federated round."""
        learn_kwargs.setdefault("reset_num_timesteps", False)
        learn_kwargs.setdefault("tb_log_name", f"client_{self.client_id}")
        timesteps_before = self.model.num_timesteps

        if self.native_wandb_run is None:
            self.model.learn(total_timesteps=total_timesteps, **learn_kwargs)
        else:
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
            self.model.learn(total_timesteps=total_timesteps, **learn_kwargs)

        trained_timesteps = self.model.num_timesteps - timesteps_before
        mean_reward, std_reward = evaluate_policy(self.model,
                        self.model.get_env(),
                        n_eval_episodes=10,
                        deterministic=True,
                        )

        return {
            "client_id": self.client_id,
            "project_id": self.project_id,
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

    def fit(
        self,
        parameters: Iterable[np.ndarray] | None,
        total_timesteps: int,
        **learn_kwargs: Any,
    ) -> tuple[list[np.ndarray], float, dict[str, Any]]:
        """
        Run one federated round: load global weights, train locally, return
        updated weights and the client aggregation weight.
        """
        if parameters is not None:
            self.set_parameters(parameters)

        metrics = self.train(total_timesteps, **learn_kwargs)
        return self.get_parameters(), self.aggregation_weight, metrics
