"""nuPlan-style comfort metric, copied from V-Max (vmax/simulator/metrics/comfort.py).

Per-step binary value: 1.0 if all six kinematic thresholds hold over the last
1 second (10 steps) of the simulated SDC trajectory, else 0.0. The per-episode
comfort is the mean of the per-step values (fraction of comfortable steps),
matching the V-Max baseline aggregation.
"""

import jax
import jax.numpy as jnp
from waymax import datatypes

from .savgol import savgol_filter_jax

TIME_DELTA = 0.1  # simulator step length (s), same as V-Max constants.TIME_DELTA


def compute_comfort(simulator_state: datatypes.SimulatorState) -> jax.Array:
    """Compute the per-step comfort value (1.0 comfortable / 0.0 not) for the SDC.

    Args:
        simulator_state: The current simulator state (unbatched).

    Returns:
        Scalar float32 comfort value at the current timestep.

    """
    past_traj = datatypes.dynamic_slice(
        simulator_state.sim_trajectory,
        simulator_state.timestep - 9,
        10,
        -1,
    )
    sdc_index = jnp.argmax(simulator_state.object_metadata.is_sdc)

    sdc_traj = jax.tree.map(lambda x: x[sdc_index], past_traj)

    lateral_accel = _compute_lateral_acceleration(sdc_traj, TIME_DELTA)
    long_accel = _compute_longitudinal_acceleration(sdc_traj, TIME_DELTA)
    long_jerk = _compute_longitudinal_jerk(sdc_traj, TIME_DELTA)
    yaw_rate = _compute_ego_yaw_rate(sdc_traj, TIME_DELTA)
    yaw_accel = _compute_ego_yaw_acceleration(sdc_traj, TIME_DELTA)

    value = (jnp.max(jnp.abs(lateral_accel)) <= 2.0).astype(jnp.float32)
    value *= jnp.min(long_accel) >= -4.05
    value *= jnp.max(long_accel) <= 2.40
    value *= jnp.max(jnp.abs(long_jerk)) <= 8.3
    value *= jnp.max(jnp.abs(yaw_accel)) <= 2.2
    value *= jnp.max(jnp.abs(yaw_rate)) <= 0.95

    return value


def _compute_lateral_acceleration(sdc_traj: datatypes.Trajectory, dt: float):
    """Compute lateral acceleration using the vehicle trajectory."""
    yaw = sdc_traj.yaw
    vel = sdc_traj.stack_fields(["vel_x", "vel_y"])

    lateral_direction = jnp.stack([-jnp.sin(yaw), jnp.cos(yaw)], axis=-1)
    lateral_velocity = jnp.sum(vel * lateral_direction, axis=-1)

    lateral_acceleration = savgol_filter_jax(
        lateral_velocity,
        window_length=5,
        polyorder=2,
        deriv=1,
        delta=dt,
    )

    return lateral_acceleration


def _compute_longitudinal_acceleration(sdc_traj: datatypes.Trajectory, dt: float):
    """Compute the longitudinal acceleration along the vehicle trajectory."""
    yaw = sdc_traj.yaw
    vel = sdc_traj.stack_fields(["vel_x", "vel_y"])

    longitudinal_direction = jnp.stack([jnp.cos(yaw), jnp.sin(yaw)], axis=-1)
    longitudinal_velocity = jnp.sum(vel * longitudinal_direction, axis=-1)

    longitudinal_acceleration = savgol_filter_jax(
        longitudinal_velocity,
        window_length=5,
        polyorder=2,
        deriv=1,
        delta=dt,
    )

    return longitudinal_acceleration


def _compute_longitudinal_jerk(sdc_traj: datatypes.Trajectory, dt: float):
    """Compute the longitudinal jerk over the vehicle trajectory."""
    yaw = sdc_traj.yaw
    vel = sdc_traj.stack_fields(["vel_x", "vel_y"])

    longitudinal_direction = jnp.stack([jnp.cos(yaw), jnp.sin(yaw)], axis=-1)
    longitudinal_velocity = jnp.sum(vel * longitudinal_direction, axis=-1)

    longitudinal_jerk = savgol_filter_jax(
        longitudinal_velocity,
        window_length=5,
        polyorder=3,
        deriv=2,
        delta=dt,
    )

    return longitudinal_jerk


def _compute_ego_yaw_rate(sdc_traj: datatypes.Trajectory, dt: float):
    """Compute the yaw rate of the ego vehicle."""
    yaw = phase_unwrap(sdc_traj.yaw)
    yaw_rate = savgol_filter_jax(yaw, window_length=5, polyorder=2, deriv=1, delta=dt)
    return yaw_rate


def _compute_ego_yaw_acceleration(sdc_traj: datatypes.Trajectory, dt: float):
    """Compute the yaw acceleration of the ego vehicle."""
    yaw = phase_unwrap(sdc_traj.yaw)
    yaw_accel = savgol_filter_jax(yaw, window_length=5, polyorder=3, deriv=2, delta=dt)
    return yaw_accel


def phase_unwrap(headings):
    """Unwrap the heading angles to avoid discontinuities.

    There are some jumps in the heading (e.g. from -pi to +pi) which cause the
    yaw derivative approximations to blow up; remove 2*pi jumps.
    """
    two_pi = 2.0 * jnp.pi
    adjustments = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.float32), jnp.cumsum(jnp.round(jnp.diff(headings) / two_pi))]
    )
    unwrapped = headings - two_pi * adjustments
    return unwrapped
