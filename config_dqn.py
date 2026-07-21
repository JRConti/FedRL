import ast
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

import torch as th
import yaml
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

DQN_HYPERPARAMS_PATH = (
    Path(__file__).resolve().parent / "hyperparams_rl_zoo" / "dqn.yml"
)


def _parse_dict_kwargs(name: str, value: Any) -> dict[str, Any] | None:
    """Parse a string expression of a dictionary into a Python dictionary."""
    if value is None or isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a mapping or a string")

    expression = ast.parse(value, mode="eval").body
    if (
        not isinstance(expression, ast.Call)
        or not isinstance(expression.func, ast.Name)
        or expression.func.id != "dict"
        or expression.args
    ):
        raise ValueError(f"Unsupported {name} expression: {value!r}")

    return {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in expression.keywords
        if keyword.arg is not None
    }


def read_hyperparameters(env_id: str) -> dict[str, Any]:
    """Read the exact environment entry, falling back to the YAML default."""
    with DQN_HYPERPARAMS_PATH.open(encoding="utf-8") as hyperparams_file:
        all_hyperparams = yaml.safe_load(hyperparams_file)

    if not isinstance(all_hyperparams, dict):
        raise ValueError(f"Invalid DQN hyperparameters file: {DQN_HYPERPARAMS_PATH}")

    if env_id in all_hyperparams:
        selected_hyperparams = all_hyperparams[env_id]
    elif "default" in all_hyperparams:
        selected_hyperparams = all_hyperparams["default"]
    else:
        raise ValueError(
            f"DQN hyperparameters not found for {env_id!r} "
            f"in {DQN_HYPERPARAMS_PATH}"
        )

    if not isinstance(selected_hyperparams, dict):
        raise ValueError(f"Invalid DQN hyperparameters for environment {env_id!r}")

    return dict(selected_hyperparams)


def _preprocess_hyperparams(
    hyperparams: dict[str, Any],
    env_id: str,
) -> dict[str, Any]:
    """Convert YAML values to SB3 values and remove training-only settings."""
    hyperparams = hyperparams.copy()

    if isinstance(hyperparams.get("train_freq"), list):
        hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

    for kwargs_name in ("policy_kwargs", "replay_buffer_kwargs"):
        if kwargs_name in hyperparams:
            hyperparams[kwargs_name] = _parse_dict_kwargs(
                kwargs_name,
                hyperparams[kwargs_name],
            )

    hyperparams.pop("n_timesteps", None)

    config_fields = {field.name for field in fields(DQNConfig)}
    unsupported = hyperparams.keys() - config_fields
    if unsupported:
        unsupported_names = ", ".join(sorted(unsupported))
        raise ValueError(
            f"Unsupported DQN hyperparameters for {env_id!r}: {unsupported_names}"
        )

    return hyperparams


def get_rb3_dqn_config(env_id: str) -> DQNConfig:
    """Load and preprocess the RL Zoo DQN config for ``env_id``."""
    hyperparams = read_hyperparameters(env_id)
    return DQNConfig(**_preprocess_hyperparams(hyperparams, env_id))
