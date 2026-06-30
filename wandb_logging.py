import os
from typing import Any

import wandb

WANDB_PROJECT = "FedRL"

# W&B has a built-in X axis named "Step". We still log explicit axes so the UI
# can distinguish federated rounds from local SB3 timesteps.
ROUND_AXIS = "round"
LOCAL_STEPS_AXIS = "local step"
GLOBAL_STEP_AXIS = "global_step"

# Online W&B can be slow to start when the network is busy. Keep this modest so
# a bad connection does not block the whole script for too long.
WANDB_INIT_TIMEOUT_SECONDS = int(os.getenv("WANDB_INIT_TIMEOUT_SECONDS", "120"))

# These prefixes define the custom panels used for the non-SB3 summaries.
SUMMARY_PREFIXES = (
    "server_by_round",
    "server_by_local_steps",
    "clients_by_round",
    "clients_by_local_steps",
    "centralized_by_round",
    "centralized_by_local_steps",
)
SB3_PREFIXES = ("rollout", "train", "time")


def wandb_config(args: Any, experiment_id: str) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "policy_type": "DQN",
        "env_name": args.env_id,
        "num_clients": args.num_clients,
        "num_rounds": args.num_rounds,
        "timesteps_per_round": args.timesteps_per_round,
        "total_client_timesteps": args.num_rounds * args.timesteps_per_round,
        "total_env_steps": args.num_rounds * args.timesteps_per_round * args.num_clients,
        "eval_episodes": args.eval_episodes,
        "seed": args.seed,
        "learning_starts": args.learning_starts,
        "buffer_size": args.buffer_size,
        "batch_size": args.batch_size,
    }


def make_wandb_run(
    args: Any,
    experiment_id: str,
    name: str,
    job_type: str,
    tags: list[str],
    reinit: bool = True,
) -> wandb.sdk.wandb_run.Run:
    """Create one W&B run with the naming/grouping conventions of this project."""
    init_kwargs = {
        "project": WANDB_PROJECT,
        "name": f"{experiment_id}/{name}",
        "group": experiment_id,
        "job_type": job_type,
        "tags": [experiment_id, *tags],
        "config": wandb_config(args, experiment_id),
    }
    if reinit:
        init_kwargs["reinit"] = "create_new"
    if hasattr(wandb, "Settings"):
        init_kwargs["settings"] = wandb.Settings(
            init_timeout=WANDB_INIT_TIMEOUT_SECONDS,
        )
    return wandb.init(**init_kwargs)


def define_metric(run: wandb.sdk.wandb_run.Run, name: str, **kwargs: Any) -> None:
    """Register how W&B should plot a metric, including its X axis."""
    if hasattr(run, "define_metric"):
        run.define_metric(name, **kwargs)
    else:
        wandb.define_metric(name, **kwargs)


def configure_summary_run(run: wandb.sdk.wandb_run.Run) -> None:
    """Configure round/local-step axes for server, centralized, and client summaries."""
    define_metric(run, ROUND_AXIS, hidden=True)
    define_metric(run, LOCAL_STEPS_AXIS, hidden=True)
    for prefix in SUMMARY_PREFIXES:
        axis = ROUND_AXIS if prefix.endswith("_by_round") else LOCAL_STEPS_AXIS
        define_metric(run, f"{prefix}/*", step_metric=axis, step_sync=True)


def configure_sb3_run(run: wandb.sdk.wandb_run.Run, client_id: str) -> None:
    """Configure a per-client run that receives native SB3 logger metrics."""
    define_metric(run, LOCAL_STEPS_AXIS, hidden=True)
    define_metric(run, GLOBAL_STEP_AXIS, hidden=True)
    for prefix in (client_id, *SB3_PREFIXES):
        define_metric(run, f"{prefix}/*", step_metric=GLOBAL_STEP_AXIS, step_sync=True)


def initialize_training_wandb(
    args: Any,
    experiment_id: str,
    client_ids: list[str],
) -> dict[str, Any]:
    """Open all W&B runs needed by one training experiment.

    Summary runs hold server/centralized curves. Client summary runs are kept
    separate so W&B can superpose clients as different runs in the same panel.
    SB3 runs are also per-client because they represent individual local models.
    """
    summary_runs = {
        ROUND_AXIS: make_wandb_run(
            args,
            experiment_id,
            "summary/rounds",
            job_type="summary",
            tags=["summary", "rounds"],
            reinit=False,
        ),
        LOCAL_STEPS_AXIS: make_wandb_run(
            args,
            experiment_id,
            "summary/local-steps",
            job_type="summary",
            tags=["summary", "local-steps"],
        ),
    }
    for run in summary_runs.values():
        configure_summary_run(run)

    client_sb3_runs = {}
    client_summary_runs = {}
    for client_id in client_ids:
        # Two runs per client are needed here: one with "round" as Step, one
        # with "local step" as Step. This prevents W&B from mixing both axes.
        client_summary_runs[client_id] = {
            ROUND_AXIS: make_wandb_run(
                args,
                experiment_id,
                f"{client_id}/rounds",
                job_type="client",
                tags=["client", client_id, "rounds"],
            ),
            LOCAL_STEPS_AXIS: make_wandb_run(
                args,
                experiment_id,
                f"{client_id}/local-steps",
                job_type="client",
                tags=["client", client_id, "local-steps"],
            ),
        }
        for summary_run in client_summary_runs[client_id].values():
            configure_summary_run(summary_run)

        run = make_wandb_run(
            args,
            experiment_id,
            f"{client_id}/sb3",
            job_type="client",
            tags=["client", client_id, "sb3"],
        )
        configure_sb3_run(run, client_id)
        client_sb3_runs[client_id] = run

    return {
        "summary": summary_runs,
        "client_summary": client_summary_runs,
        "client_sb3": client_sb3_runs,
    }


def finish_wandb_runs(runs: Any) -> None:
    """Finish every run in a nested dict of W&B runs."""
    if isinstance(runs, dict):
        for value in runs.values():
            finish_wandb_runs(value)
    elif runs is not None:
        runs.finish()


def axis_payloads(
    category: str,
    metrics: dict[str, Any],
    round_number: int,
    local_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the same metrics twice, once for each custom X axis."""
    by_round = {ROUND_AXIS: round_number}
    by_local_steps = {LOCAL_STEPS_AXIS: local_steps}
    for name, value in metrics.items():
        by_round[f"{category}_by_round/{name}"] = value
        by_local_steps[f"{category}_by_local_steps/{name}"] = value
    return by_round, by_local_steps


def log_to_axis_runs(
    runs: dict[str, wandb.sdk.wandb_run.Run] | None,
    by_round: dict[str, Any],
    by_local_steps: dict[str, Any],
    round_number: int | None = None,
    local_steps: int | None = None,
) -> None:
    """Log a pair of payloads to the matching round/local-step runs."""
    if runs is None:
        wandb.log(by_round)
        wandb.log(by_local_steps)
        return
    runs[ROUND_AXIS].log(by_round, step=round_number)
    runs[LOCAL_STEPS_AXIS].log(by_local_steps, step=local_steps)


def log_round_metrics(
    round_index: int,
    round_metrics: dict[str, Any],
    mean_reward: float,
    std_reward: float,
    elapsed_seconds: float,
    args: Any,
    runs: dict[str, wandb.sdk.wandb_run.Run] | None = None,
    client_runs: dict[str, dict[str, wandb.sdk.wandb_run.Run]] | None = None,
) -> None:
    """Log global server metrics and per-client reward summaries for one round."""
    round_number = round_index + 1
    local_steps = round_number * args.timesteps_per_round
    total_env_steps = local_steps * args.num_clients

    log_to_axis_runs(
        runs,
        *axis_payloads(
            "server",
            {
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "total_env_steps": total_env_steps,
                "round_env_steps": args.timesteps_per_round * args.num_clients,
                "elapsed_seconds": elapsed_seconds,
            },
            round_number,
            local_steps,
        ),
        round_number=round_number,
        local_steps=local_steps,
    )

    if client_runs is None:
        return

    for client_metrics in round_metrics["client_metrics"]:
        client_id = client_metrics["client_id"]
        client_run_pair = client_runs.get(client_id)
        if client_run_pair is None:
            continue
        # Same metric names across client runs make W&B overlay clients in a
        # single clients_by_* panel instead of creating one panel per client.
        log_to_axis_runs(
            client_run_pair,
            {
                ROUND_AXIS: round_number,
                "clients_by_round/mean_reward": client_metrics["mean_reward"],
                "clients_by_round/std_reward": client_metrics["std_reward"],
            },
            {
                LOCAL_STEPS_AXIS: local_steps,
                "clients_by_local_steps/mean_reward": client_metrics["mean_reward"],
                "clients_by_local_steps/std_reward": client_metrics["std_reward"],
            },
            round_number=round_number,
            local_steps=local_steps,
        )


def log_centralized_metrics(
    round_index: int,
    mean_reward: float,
    std_reward: float,
    total_timesteps: int,
    args: Any,
    runs: dict[str, wandb.sdk.wandb_run.Run] | None = None,
) -> None:
    """Log the centralized baseline on the same axes as the federated server."""
    round_number = round_index + 1
    local_steps = round_number * args.timesteps_per_round
    log_to_axis_runs(
        runs,
        *axis_payloads(
            "centralized",
            {
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "total_timesteps": total_timesteps,
            },
            round_number,
            local_steps,
        ),
        round_number=round_number,
        local_steps=local_steps,
    )
