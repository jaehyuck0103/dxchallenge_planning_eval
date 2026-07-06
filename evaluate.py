"""DX challenge — motion planning evaluation script.

Evaluates a participant submission (a ``WaymaxActorCore`` planner) on a rideflux
dataset with pure Waymax and reports the rideflux score.

Setup (mirrors the V-Max baseline evaluation):
  - ``PlanningAgentEnvironment``: the SDC is driven by the participant actor
    through an ``InvertibleBicycleModel(normalize_actions=True)``; every other
    object replays its logged trajectory.
  - No SDC paths are built (roadgraph-free metrics only).
  - Episodes terminate early on collision (``overlap``) or offroad
    (``offroad_in_box``), like the baseline.
  - The state handed to the actor is scrubbed of future information: for
    timesteps after the current one, ``log_trajectory`` and
    ``log_traffic_light`` are invalidated and zeroed — except the ego's final
    logged state, which is kept as the goal.

Scenarios are evaluated in lockstep batches: the actor's ``init`` /
``select_action`` are vmapped across the batch (each traced call still sees an
unbatched state, so per-scenario semantics are unchanged). This requires the
submission to be JAX-traceable — there is no sequential fallback.

Usage:
  uv run evaluate.py \
      --path_dataset /path/to/rideflux_validation.tfrecord@495 \
      --submission submission_example \
      --batch_size 16 --max_scenarios 100
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import csv
import importlib.util
import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from waymax import config as _config
from waymax import dataloader, datatypes, dynamics
from waymax import env as _env
from waymax.agents import actor_core
from waymax.dataloader.dataloader_utils import generate_sharded_filenames
from waymax.metrics.imitation import LogDivergenceMetric
from waymax.metrics.overlap import OverlapMetric

import challenge_metrics

# Metrics that end the episode when they fire (same as the V-Max baseline
# evaluation on rideflux).
TERMINATION_KEYS = ("overlap", "offroad_in_box")


def parse_args():
    """Parse command-line arguments for the challenge evaluation."""
    parser = argparse.ArgumentParser(description="DX challenge planning evaluation")
    parser.add_argument(
        "--path_dataset",
        "-pd",
        type=str,
        required=True,
        help="Path to the evaluation dataset (tfrecord, '@N' sharding supported)",
    )
    parser.add_argument(
        "--submission",
        "-s",
        type=str,
        required=True,
        help="Path to the submission directory (must contain actor.py with create_actor())",
    )
    parser.add_argument(
        "--max_num_objects",
        "-o",
        type=int,
        default=64,
        help="Maximum number of objects in the scene (default: 64)",
    )
    parser.add_argument(
        "--batch_size",
        "-bs",
        type=int,
        default=16,
        help="Scenarios evaluated in lockstep per batch (default: 16). The "
        "actor's init/select_action are vmapped across the batch, so the "
        "submission must be JAX-traceable.",
    )
    parser.add_argument(
        "--max_scenarios",
        "-n",
        type=int,
        default=None,
        help="Evaluate only the first N scenarios (default: all)",
    )
    parser.add_argument(
        "--output_dir",
        "-od",
        type=str,
        default=None,
        help="Directory for result files (default: results/<submission name>)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed passed to the actor (default: 0)",
    )
    parser.add_argument(
        "--disable_future_masking",
        action="store_true",
        help="DEBUG ONLY: hand the actor the raw state including logged future. "
        "Never use for actual grading.",
    )
    return parser.parse_args()


def load_actor(submission_dir: str) -> actor_core.WaymaxActorCore:
    """Load the participant actor from ``<submission_dir>/actor.py``.

    The module must expose ``create_actor(submission_dir) -> WaymaxActorCore``.
    """
    submission_dir = os.path.abspath(submission_dir)
    actor_file = os.path.join(submission_dir, "actor.py")
    if not os.path.isfile(actor_file):
        raise FileNotFoundError(f"actor.py not found in submission: {actor_file}")

    # Allow the submission to import its own sibling modules / load its weights.
    sys.path.insert(0, submission_dir)
    spec = importlib.util.spec_from_file_location("participant_actor", actor_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "create_actor"):
        raise AttributeError("actor.py must define create_actor(submission_dir)")

    actor = module.create_actor(submission_dir)

    if not isinstance(actor, actor_core.WaymaxActorCore):
        raise TypeError(
            f"create_actor() must return a WaymaxActorCore subclass, got {type(actor)}"
        )

    return actor


def mask_future_information(
    state: datatypes.SimulatorState,
) -> datatypes.SimulatorState:
    """Scrub future information from the state handed to the participant actor.

    For timesteps strictly after ``state.timestep``, the logged trajectory of
    every object and the logged traffic-light states are invalidated AND their
    values zeroed (so ignoring the valid flags leaks nothing). The single
    exception is the ego's last valid logged state, which stays visible as the
    goal of the scenario.
    """
    log = state.log_trajectory
    num_steps = log.num_timesteps
    time_idx = jnp.arange(num_steps)
    visible = time_idx <= state.timestep  # history + current timestep

    is_sdc = state.object_metadata.is_sdc
    sdc_valid = log.valid[jnp.argmax(is_sdc)]
    # Last valid logged index of the ego = the goal point.
    goal_idx = (num_steps - 1) - jnp.argmax(sdc_valid[::-1])

    keep = visible[None, :] | (is_sdc[:, None] & (time_idx == goal_idx)[None, :])
    new_valid = log.valid & keep

    def scrub(x):
        # Zero only outside `keep` (i.e. the hidden future): visible history is
        # left bit-identical, including its invalid-entry padding values.
        return jnp.where(keep, x, jnp.zeros_like(x))

    new_log = log.replace(
        x=scrub(log.x),
        y=scrub(log.y),
        z=scrub(log.z),
        vel_x=scrub(log.vel_x),
        vel_y=scrub(log.vel_y),
        yaw=scrub(log.yaw),
        timestamp_micros=scrub(log.timestamp_micros),
        length=scrub(log.length),
        width=scrub(log.width),
        height=scrub(log.height),
        valid=new_valid,
    )

    tl = state.log_traffic_light
    tl_visible = jnp.broadcast_to(visible[None, :], tl.valid.shape)
    tl_valid = tl.valid & tl_visible

    def scrub_tl(x):
        return jnp.where(tl_visible, x, jnp.zeros_like(x))

    new_tl = tl.replace(
        x=scrub_tl(tl.x),
        y=scrub_tl(tl.y),
        z=scrub_tl(tl.z),
        state=scrub_tl(tl.state),
        lane_ids=scrub_tl(tl.lane_ids),
        valid=tl_valid,
    )

    return state.replace(log_trajectory=new_log, log_traffic_light=new_tl)


def iter_scenarios(path: str, max_num_objects: int):
    """Yield scenarios one by one in canonical order (file by file, record by record).

    A fresh single-file pipeline per shard keeps the record order independent
    of batching and of tf.data's parallel interleave: waymax's sharded '@N'
    pipeline (num_shards sub-streams + AUTOTUNE interleave) yields records in a
    different order for batched vs unbatched runs, which would break
    per-scenario comparability across batch sizes.
    """
    if "@" in os.path.basename(path):
        files = generate_sharded_filenames(path)
    else:
        files = [path]

    for file_path in files:
        config = _config.DatasetConfig(
            path=file_path,
            max_num_objects=max_num_objects,
            repeat=1,
            shuffle_seed=None,
            num_shards=1,
        )
        yield from dataloader.simulator_state_generator(config=config)


def iter_batches(scenario_iter, batch_size: int):
    """Stack scenarios into batches of `batch_size` (the final one may be smaller)."""
    buffer = []
    for scenario in scenario_iter:
        buffer.append(scenario)
        if len(buffer) == batch_size:
            yield jax.tree.map(lambda *xs: jnp.stack(xs), *buffer)
            buffer = []
    if buffer:
        yield jax.tree.map(lambda *xs: jnp.stack(xs), *buffer)


def make_env(max_num_objects: int) -> _env.PlanningAgentEnvironment:
    """Build the planning environment: bicycle-driven SDC, logged others."""
    env_config = _config.EnvironmentConfig(
        max_num_objects=max_num_objects,
        controlled_object=_config.ObjectType.SDC,
        compute_reward=False,
        metrics=_config.MetricsConfig(metrics_to_run=()),
        init_steps=11,
    )
    dynamics_model = dynamics.InvertibleBicycleModel(normalize_actions=True)

    # No sim_agent_actors: every non-SDC object replays its logged trajectory.
    return _env.PlanningAgentEnvironment(dynamics_model, env_config)


def compute_step_metrics(state: datatypes.SimulatorState) -> dict[str, jax.Array]:
    """Compute the per-step challenge metrics on the (true) simulator state."""
    sdc_index = jnp.argmax(state.object_metadata.is_sdc)

    return {
        "overlap": OverlapMetric().compute(state).value[sdc_index],
        "offroad_in_box": challenge_metrics.is_sdc_offroad_in_box(state),
        "comfort": challenge_metrics.compute_comfort(state),
        "progress_ratio": challenge_metrics.compute_progress_ratio(state),
        "log_divergence": LogDivergenceMetric().compute(state).value[sdc_index],
    }


def run_episode_batch(
    scenario_batch, jit_reset, jit_batch_init, jit_batch_select, jit_step_and_measure, rng
):
    """Roll out a batch of scenarios in lockstep and return per-scenario records.

    All scenarios in the batch run the full horizon (the sim keeps stepping
    already-terminated ones); each scenario's records are truncated at its
    first termination step afterwards. This is equivalent to the sequential
    early break, because a step's metrics only depend on the trajectory up to
    that step.
    """
    state = jit_reset(scenario_batch)
    batch_size = int(state.shape[0])
    num_steps = int(np.max(np.asarray(state.remaining_timesteps)))

    rng, init_rng = jax.random.split(rng)
    actor_state = jit_batch_init(jax.random.split(init_rng, batch_size), state)

    alive = np.ones(batch_size, dtype=bool)
    episode_length = np.zeros(batch_size, dtype=int)
    step_records = []

    for _ in range(num_steps):
        rng, step_rng = jax.random.split(rng)
        output = jit_batch_select(state, actor_state, jax.random.split(step_rng, batch_size))
        actor_state = output.actor_state

        state, step_metrics = jit_step_and_measure(state, output.action)
        step_metrics = {key: np.asarray(value) for key, value in step_metrics.items()}
        step_records.append(step_metrics)

        fired = np.zeros(batch_size, dtype=bool)
        for key in TERMINATION_KEYS:
            fired |= step_metrics[key] > 0
        episode_length[alive & fired] = len(step_records)
        alive &= ~fired
        if not alive.any():
            break

    # Scenarios that never fired a termination metric ran to the end.
    episode_length[alive] = len(step_records)

    return [
        {
            key: np.array([record[key][i] for record in step_records[: episode_length[i]]])
            for key in step_records[0]
        }
        for i in range(batch_size)
    ]


def write_results(output_dir: str, rows: list[dict], summary: dict) -> None:
    """Write per-scenario CSV and a summary text file."""
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "evaluation_episodes.csv")
    fieldnames = ["scenario_index"] + [k for k in rows[0] if k != "scenario_index"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    txt_path = os.path.join(output_dir, "evaluation_results.txt")
    key_width = max(len(key) for key in summary)
    with open(txt_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write(f"DX challenge evaluation - {len(rows)} scenarios\n")
        f.write("=" * 50 + "\n")
        for key, value in summary.items():
            f.write(f"{key:<{key_width}} {value:>12.5f}\n")

    print(f"-> Results written to {csv_path} and {txt_path}")


def main():
    args = parse_args()

    print(f"-> Loading submission from {args.submission} ...")
    actor = load_actor(args.submission)
    print(f"-> Loaded actor: {actor.name}")

    env = make_env(args.max_num_objects)
    batch_size = args.batch_size

    scenario_iter = iter_scenarios(args.path_dataset, args.max_num_objects)

    if args.disable_future_masking:
        print("-> WARNING: future masking DISABLED (debug mode, not valid for grading)")
        mask_fn = lambda state: state  # noqa: E731
    else:
        mask_fn = mask_future_information

    def step_and_measure(state, action):
        new_state = env.step(state, action)
        return new_state, compute_step_metrics(new_state)

    # Lockstep batch evaluation: the actor's init/select_action are vmapped
    # over the batch — each traced call still sees an UNBATCHED state, so
    # per-scenario semantics are unchanged, but the actor must be
    # JAX-traceable. Future masking runs inside the same jit, upstream of the
    # actor.
    print(f"-> Lockstep batch evaluation (batch_size={batch_size}, JAX-traceable actor required)")
    jit_reset = jax.jit(jax.vmap(env.reset))
    jit_step_and_measure = jax.jit(jax.vmap(step_and_measure))
    jit_batch_init = jax.jit(jax.vmap(lambda key, state: actor.init(key, mask_fn(state))))
    jit_batch_select = jax.jit(
        jax.vmap(
            lambda state, actor_state, key: actor.select_action(
                None, mask_fn(state), actor_state, key
            )
        )
    )

    def error_episode():
        return {
            "episode_length": 0,
            "progress_ratio": 0.0,
            "comfort": 0.0,
            "overlap": 0.0,
            "offroad_in_box": 0.0,
            "log_divergence": 0.0,
            "accuracy": 0.0,
            "rideflux_score": 0.0,
            "error": 1,
        }

    rng = jax.random.PRNGKey(args.seed)
    rows = []
    num_errors = 0
    start_time = time.time()

    def append_row(episode):
        episode["scenario_index"] = len(rows)
        rows.append(episode)
        if len(rows) % 50 == 0:
            elapsed = time.time() - start_time
            mean_score = np.mean([r["rideflux_score"] for r in rows])
            print(
                f"-> {len(rows)} scenarios | rideflux_score {mean_score:.4f} "
                f"| {elapsed / len(rows):.2f}s per scenario"
            )

    for scenario_batch in iter_batches(scenario_iter, batch_size):
        if args.max_scenarios is not None and len(rows) >= args.max_scenarios:
            break

        rng, batch_rng = jax.random.split(rng)
        n_in_batch = int(scenario_batch.shape[0])
        try:
            per_scenario_arrays = run_episode_batch(
                scenario_batch,
                jit_reset,
                jit_batch_init,
                jit_batch_select,
                jit_step_and_measure,
                batch_rng,
            )
            episodes = []
            for step_arrays in per_scenario_arrays:
                episode = challenge_metrics.episode_scores(step_arrays, TERMINATION_KEYS)
                episode["error"] = 0
                episodes.append(episode)
        except Exception:
            print(f"-> ERROR in batch starting at scenario {len(rows)} (all scored 0):")
            traceback.print_exc()
            num_errors += n_in_batch
            episodes = [error_episode() for _ in range(n_in_batch)]

        for episode in episodes:
            if args.max_scenarios is not None and len(rows) >= args.max_scenarios:
                break
            append_row(episode)

    if not rows:
        raise RuntimeError("No scenarios were evaluated (empty dataset?)")

    total_time = time.time() - start_time

    summary_keys = [
        "rideflux_score",
        "progress_ratio",
        "comfort",
        "overlap",
        "offroad_in_box",
        "accuracy",
        "log_divergence",
        "episode_length",
        "error",
    ]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in summary_keys}

    output_dir = args.output_dir or os.path.join(
        "results", os.path.basename(os.path.abspath(args.submission))
    )
    write_results(output_dir, rows, summary)

    print(
        f"\n-> Evaluation completed: {len(rows)} scenarios in {total_time:.1f}s "
        f"(avg {total_time / len(rows):.2f}s per scenario, {num_errors} errors)"
    )
    print("\n===== DX challenge result =====")
    for key, value in summary.items():
        print(f"{key:<20} {value:>10.5f}")
    print("===============================")
    print(f"RIDEFLUX SCORE: {summary['rideflux_score']:.5f}")


if __name__ == "__main__":
    main()
