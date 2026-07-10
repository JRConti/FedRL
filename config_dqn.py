from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


@dataclass
class DQNConfig:
    POLICY_KWARG_FIELDS: ClassVar[tuple[str, ...]] = (
        "net_arch",
        "activation_fn",
        "features_extractor_class",
        "features_extractor_kwargs",
        "normalize_images",
        "optimizer_class",
        "optimizer_kwargs",
    )

    policy: str | type[Any] = "MlpPolicy"
    learning_rate: float | Any = 1e-4
    buffer_size: int = 1_000_000
    learning_starts: int = 100
    batch_size: int = 32
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int | tuple[int, str] = 4
    gradient_steps: int = 1
    replay_buffer_class: type[Any] | None = None
    replay_buffer_kwargs: dict[str, Any] | None = None
    optimize_memory_usage: bool = False
    n_steps: int = 1
    target_update_interval: int = 10_000
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    max_grad_norm: float = 10.0
    stats_window_size: int = 100
    tensorboard_log: str | None = None
    policy_kwargs: dict[str, Any] | None = None
    verbose: int = 0
    seed: int | None = None
    device: Any = "auto"
    _init_setup_model: bool = True

    net_arch: list[int] | None = None
    activation_fn: type[nn.Module] | None = None
    features_extractor_class: type[BaseFeaturesExtractor] | None = None
    features_extractor_kwargs: dict[str, Any] | None = None
    normalize_images: bool | None = None
    optimizer_class: type[th.optim.Optimizer] | None = None
    optimizer_kwargs: dict[str, Any] | None = None

    def to_kwargs(
        self,
        seed: int | None = None,
        tensorboard_log: str | None = None,
    ) -> dict[str, Any]:
        kwargs = asdict(self)
        kwargs.pop("policy")

        policy_kwargs = dict(kwargs["policy_kwargs"] or {})
        for field_name in self.POLICY_KWARG_FIELDS:
            value = kwargs.pop(field_name)
            if value is not None:
                policy_kwargs[field_name] = value
        kwargs["policy_kwargs"] = policy_kwargs or None

        if tensorboard_log is not None:
            kwargs["tensorboard_log"] = tensorboard_log
        if seed is not None:
            kwargs["seed"] = seed

        return kwargs

    def to_wandb_config(self) -> dict[str, Any]:
        config = asdict(self)
        if isinstance(self.policy, type):
            config["policy"] = self.policy.__name__
        if self.replay_buffer_class is not None:
            config["replay_buffer_class"] = self.replay_buffer_class.__name__
        if self.activation_fn is not None:
            config["activation_fn"] = self.activation_fn.__name__
        if self.features_extractor_class is not None:
            config["features_extractor_class"] = self.features_extractor_class.__name__
        if self.optimizer_class is not None:
            config["optimizer_class"] = self.optimizer_class.__name__
        return config

    @classmethod
    def from_args(cls, args: Any) -> "DQNConfig":
        config = cls()
        config.learning_starts = getattr(args, "learning_starts", config.learning_starts)
        config.buffer_size = getattr(args, "buffer_size", config.buffer_size)
        config.batch_size = getattr(args, "batch_size", config.batch_size)
        config.verbose = getattr(args, "verbose", config.verbose)
        return config
