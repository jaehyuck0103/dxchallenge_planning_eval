"""Per-episode aggregation and rideflux score.

Mirrors the V-Max baseline evaluation exactly
(vmax/scripts/evaluate/utils.py::append_episode_metrics +
vmax/simulator/metrics/collector.py::_metrics_operands +
vmax/simulator/metrics/aggregators.py::rideflux_aggregate_score):

per episode, over the executed steps only (up to termination):
    progress_ratio  -> value at the final executed step
    comfort         -> mean over steps (fraction of comfortable steps)
    overlap         -> max (1.0 if any collision)
    offroad_in_box  -> max (1.0 if ever offroad)
    log_divergence  -> mean (diagnostic only, not part of the score)

score per episode:
    rideflux_score = (7 * clip(progress_ratio, 0, 1) + 3 * comfort) / 10
                     * (1 - overlap) * (1 - offroad_in_box)

The challenge score is the mean of the per-episode rideflux_score over all
evaluated scenarios.
"""

import numpy as np


def rideflux_aggregate_score(metrics_dict: dict) -> float:
    """Rideflux aggregate score from per-episode metric values."""
    avg_score = 7 * np.clip(metrics_dict["progress_ratio"], 0, 1) + 3 * metrics_dict["comfort"]
    avg_score /= 10

    mul_score = (1 - metrics_dict["overlap"]) * (1 - metrics_dict["offroad_in_box"])

    return float(avg_score * mul_score)


def episode_scores(step_metrics: dict[str, np.ndarray], termination_keys: tuple[str, ...]) -> dict:
    """Aggregate per-step metric arrays of one episode into episode-level values.

    Args:
        step_metrics: Mapping metric name -> array of per-step values for the
            executed steps of the episode (already truncated at termination).
        termination_keys: Metric names whose activation ends an episode; used
            for the ``accuracy`` (episode success) flag.

    Returns:
        Episode-level metrics including ``rideflux_score``.

    """
    episode = {
        "episode_length": int(len(step_metrics["progress_ratio"])),
        "progress_ratio": float(step_metrics["progress_ratio"][-1]),
        "comfort": float(np.mean(step_metrics["comfort"])),
        "overlap": float(np.max(step_metrics["overlap"])),
        "offroad_in_box": float(np.max(step_metrics["offroad_in_box"])),
        "log_divergence": float(np.mean(step_metrics["log_divergence"])),
    }

    # Episode success: no termination-triggering metric ever fired.
    failed = sum(np.sum(step_metrics[key]) for key in termination_keys)
    episode["accuracy"] = float(1 - (failed > 0))

    episode["rideflux_score"] = rideflux_aggregate_score(episode)

    return episode
