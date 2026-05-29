from typing import Any, Iterable

import numpy as np

from client import FederatedDQNClient


class FederatedDQNServer:
    """
    Simple FedAvg server for federated DQN clients.

    The server assumes all clients use the same DQN architecture and therefore
    return parameter lists with the same length and shapes.
    """

    def __init__(self, clients: Iterable[FederatedDQNClient]) -> None:
        self.clients = list(clients)
        if not self.clients:
            raise ValueError("FederatedDQNServer needs at least one client.")

        self.global_parameters = self.clients[0].get_parameters()
        self.round = 0

    def train_round(self, total_timesteps: int, **learn_kwargs: Any) -> dict[str, Any]:
        """Run one federated round over all clients."""
        client_results = []

        for client in self.clients:
            parameters, weight, metrics = client.fit(
                self.global_parameters,
                total_timesteps,
                **learn_kwargs,
            )
            client_results.append((parameters, weight, metrics))

        self.global_parameters = self.aggregate(
            [parameters for parameters, _, _ in client_results],
            [weight for _, weight, _ in client_results],
        )
        self.broadcast(self.global_parameters)
        self.round += 1

        return {
            "round": self.round,
            "num_clients": len(self.clients),
            "client_metrics": [metrics for _, _, metrics in client_results],
        }

    def train(
        self,
        num_rounds: int,
        total_timesteps_per_round: int,
        **learn_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run several federated rounds."""
        history = []
        for _ in range(num_rounds):
            history.append(self.train_round(total_timesteps_per_round, **learn_kwargs))
        return history

    def broadcast(self, parameters: list[np.ndarray]) -> None:
        """Send global parameters to every client."""
        for client in self.clients:
            client.set_parameters(parameters)

    @staticmethod
    def aggregate(
        client_parameters: list[list[np.ndarray]],
        client_weights: list[float],
    ) -> list[np.ndarray]:
        """Compute a weighted average of client parameters."""
        if not client_parameters:
            raise ValueError("No client parameters to aggregate.")
        if len(client_parameters) != len(client_weights):
            raise ValueError("client_parameters and client_weights must have the same length.")

        total_weight = float(sum(client_weights))
        if total_weight <= 0:
            raise ValueError("The sum of client weights must be positive.")

        num_tensors = len(client_parameters[0])
        for parameters in client_parameters:
            if len(parameters) != num_tensors:
                raise ValueError("All clients must return the same number of tensors.")

        averaged_parameters = []
        for tensor_index in range(num_tensors):
            weighted_sum = np.zeros_like(client_parameters[0][tensor_index])

            for parameters, weight in zip(client_parameters, client_weights):
                if parameters[tensor_index].shape != weighted_sum.shape:
                    raise ValueError("All matching tensors must have the same shape.")
                weighted_sum += parameters[tensor_index] * (weight / total_weight)

            averaged_parameters.append(weighted_sum)

        return averaged_parameters


