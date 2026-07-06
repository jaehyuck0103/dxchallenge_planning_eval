"""Direction-free, 2D offroad metric, copied from V-Max (vmax/simulator/metrics/offroad_in_box.py).

Waymax's native ``OffroadMetric`` decides the on/off-road *side* from the
road-edge tangent direction (``dir_xyz``) and polyline ``ids``. On datasets
where those are unreliable (e.g. rideflux: fragmented/interleaved ids, noisy
directions) the sign of that signed-distance can flip, giving spurious
offroad/onroad results.

This metric avoids ``dir_xyz``/``ids`` entirely and uses only road-edge point
*positions* (``xyz``): the SDC is flagged offroad when any road-edge point lies
inside its 2D bounding-box footprint (expanded/shrunk by ``margin``). z is
intentionally NOT considered: the simulated z is frozen at the episode's
initial value (bicycle dynamics only update x/y/yaw/velocity).
"""

import jax.numpy as jnp
from waymax import datatypes


def is_sdc_offroad_in_box(
    state: datatypes.SimulatorState,
    margin: float = -0.3,
) -> jnp.ndarray:
    """Return 1.0 if any road-edge point lies inside the SDC 2D bounding box.

    Uses road-edge point xy positions only (no direction / ids / z).

    Args:
        state: Current simulator state (unbatched).
        margin: Footprint half-extent expansion (meters); negative shrinks the box.
            The default matches the V-Max baseline evaluation.

    Returns:
        Scalar float (1.0 offroad, 0.0 otherwise).

    """
    sdc_index = jnp.argmax(state.object_metadata.is_sdc)

    # SDC footprint at the current timestep: [x, y, length, width, yaw].
    traj_5dof = state.current_sim_trajectory.stack_fields(
        ["x", "y", "length", "width", "yaw"]
    ).squeeze()
    cx, cy, length, width, yaw = traj_5dof[sdc_index]

    rg = state.roadgraph_points
    is_edge = datatypes.is_road_edge(rg.types) & rg.valid

    # Express edge points in the SDC-local (axis-aligned) frame, then 2D box test.
    dx = rg.x - cx
    dy = rg.y - cy
    cos_y = jnp.cos(yaw)
    sin_y = jnp.sin(yaw)
    local_x = dx * cos_y + dy * sin_y
    local_y = -dx * sin_y + dy * cos_y
    inside = (jnp.abs(local_x) <= length / 2.0 + margin) & (
        jnp.abs(local_y) <= width / 2.0 + margin
    )

    return jnp.any(inside & is_edge).astype(jnp.float32)
