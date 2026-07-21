from typing import Any, Iterable

import numpy as np
import torch as th

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

    def train_round(self, total_timesteps: int, broadcast: bool = True, **learn_kwargs: Any) -> dict[str, Any]:
        """
        Run one federated round and update global parameters.

        Clients receive the current global parameters at the start of their
        local fit. The newly aggregated global parameters are not broadcast
        back to clients after the round.
        """
        client_results = []

        # Distribute the current global parameters to all clients, train locally, and collect their updated parameters and metrics.
        for client in self.clients:
            # If broadcast is True, send the current global parameters to the client. Otherwise, the client will use its own parameters for local training.
            if broadcast:
                broadcast_parameters = self.global_parameters
            else:
                broadcast_parameters = None  # Clients will use their own parameters if not broadcasting.
            
            parameters, weight, metrics = client.fit(
                broadcast_parameters,
                total_timesteps,
                **learn_kwargs,
            )
            client_results.append((parameters, weight, metrics))

        # Aggregate the client parameters using their respective weights to update the global model.
        client_parameters = [parameters for parameters, _, _ in client_results]
        client_weights = [weight for _, weight, _ in client_results]
        self.global_parameters = self.aggregate(
            client_parameters,
            client_weights,
        )

        self.round += 1

        return {
            "round": self.round,
            "num_clients": len(self.clients),
            "client_metrics": [metrics for _, _, metrics in client_results],
        }

    def broadcast(self, parameters: list[np.ndarray | th.Tensor]) -> None:
        """Send global parameters to every client.
        Not used for now."""
        for client in self.clients:
            client.set_parameters(parameters)

    @staticmethod
    def aggregate(
        client_parameters: list[list[np.ndarray | th.Tensor]],
        client_weights: list[float],
    ) -> list[np.ndarray | th.Tensor]:
        """Compute a weighted average of client parameters."""
        if not client_parameters:
            raise ValueError("No client parameters to aggregate.")
        if len(client_parameters) != len(client_weights):
            raise ValueError("client_parameters and client_weights must have the same length.")

        total_weight = float(sum(client_weights))
        if total_weight <= 0:
            raise ValueError("The sum of client weights must be positive.")
        if any(weight < 0 for weight in client_weights):
            raise ValueError("Client weights must be non-negative.")

        num_tensors = len(client_parameters[0])
        for parameters in client_parameters:
            if len(parameters) != num_tensors:
                raise ValueError("All clients must return the same number of tensors.")

        averaged_parameters = []
        normalized_weights = [float(weight / total_weight) for weight in client_weights]
        for tensor_index in range(num_tensors):
            first_tensor = client_parameters[0][tensor_index]
            weighted_sum = (
                th.zeros_like(first_tensor)
                if isinstance(first_tensor, th.Tensor)
                else np.zeros_like(first_tensor)
            )

            for client_index, (parameters, weight) in enumerate(
                zip(client_parameters, normalized_weights)
            ):
                if tuple(parameters[tensor_index].shape) != tuple(weighted_sum.shape):
                    raise ValueError(
                        f"Tensor {tensor_index} from client {client_index} has shape "
                        f"{parameters[tensor_index].shape}, expected {weighted_sum.shape}."
                    )
                if isinstance(weighted_sum, th.Tensor):
                    weighted_sum.add_(
                        th.as_tensor(
                            parameters[tensor_index],
                            dtype=weighted_sum.dtype,
                            device=weighted_sum.device,
                        ),
                        alpha=weight,
                    )
                else:
                    weighted_sum += np.asarray(parameters[tensor_index]) * weight

            averaged_parameters.append(weighted_sum)

        return averaged_parameters
