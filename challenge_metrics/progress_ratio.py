"""Progress ratio metric, copied from V-Max (vmax/simulator/metrics/progress_ratio.py).

Roadgraph-free: measures arc-length progress of the simulated SDC along the
expert (logged) trajectory, as a ratio of the expert's total traveled distance.
"""

import jax
import jax.numpy as jnp
from waymax import datatypes


def compute_progress_ratio(simulator_state: datatypes.SimulatorState) -> jax.Array:
    """Compute the progress ratio of the SDC against the logged expert trajectory.

    Args:
        simulator_state: Current simulator state (unbatched).

    Returns:
        Scalar float32 progress ratio.

    """
    sdc_index = jnp.argmax(simulator_state.object_metadata.is_sdc)
    sdc_traj = jax.tree.map(lambda x: x[sdc_index], simulator_state.sim_trajectory)
    expert_traj = jax.tree.map(lambda x: x[sdc_index], simulator_state.log_trajectory)

    return progress_ratio(sdc_traj, expert_traj)


def progress_ratio(sdc_traj: datatypes.Trajectory, expert_traj: datatypes.Trajectory):
    """Calculate the progress ratio based on the distance traveled along the expert trajectory.

    Args:
        sdc_traj: Ego vehicle trajectory.
        expert_traj: Expert (logged) trajectory.

    Returns:
        The progress ratio as a normalized value.

    """
    # Ignore the first 10 timesteps
    sdc_traj = jax.tree.map(lambda x: x[10:], sdc_traj)
    expert_traj = jax.tree.map(lambda x: x[10:], expert_traj)

    # Distances traveled by the expert:
    expert_displacement = jnp.diff(expert_traj.stack_fields(["x", "y"]), axis=0)
    expert_dist = jnp.linalg.norm(expert_displacement, axis=-1)
    expert_cum_dist = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(expert_dist)])

    # Find closest points to expert traj
    def closest_point_dist(sdc_xy):
        dists = jnp.linalg.norm(sdc_xy - expert_traj.xy, axis=-1)
        min_idx = jnp.argmin(dists)
        return expert_cum_dist[min_idx]

    sdc_progress = jax.vmap(closest_point_dist)(sdc_traj.xy)
    sdc_progress = jnp.where(sdc_traj.valid, sdc_progress, 0.0)

    progress_ratio = jnp.max(sdc_progress) / expert_cum_dist[-1]

    # Little safety for cases where the expert didn't move
    progress_ratio = jnp.where(expert_cum_dist[-1] < 0.5, 1.0, progress_ratio)

    return progress_ratio
