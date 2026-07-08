"""Vec feature extractor ported from V-Max for the SAC baseline submission.

Trimmed copy of ``vmax/simulator/features/extractor/vec_extractor.py`` and its
dependencies (SDC observation builder override, box roadgraph filter,
normalization utils, feature dataclasses). Behavior-preserving for this run's
configuration; removed parts are the roadgraph-based ``sdc_paths`` machinery
(this model's path target is the ego's logged goal endpoint — visible in the
challenge state), plotting, and observation noise/masking.
"""

from collections.abc import Sequence
from dataclasses import field
from typing import Any

import chex
import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax.datatypes.observation import (
    ObjectPose2D,
    combine_two_object_pose_2d,
    global_observation_from_state,
    transform_roadgraph_points,
    transform_traffic_lights,
    transform_trajectory,
)
from waymax.datatypes.roadgraph import RoadgraphPoints
from waymax.utils import geometry

# ---------------------------------------------------------------------------
# vmax/simulator/features/extractor/utils.py
# ---------------------------------------------------------------------------

MAX_SPEED = 30  # m/s

RG_MAPPING = (0, 1, 2, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 5, 5, 6, 7, 8, 0)
TL_MAPPING = tuple(range(9))
OBJECT_MAPPING = tuple(range(5))


def get_feature_size(feature_key: str, dict_mapping: dict) -> int:
    """Get the size of the feature."""
    if feature_key in ["xy", "vel_xy", "dir_xy"]:
        return 2
    elif feature_key in ["speed", "length", "width", "height", "valid", "yaw", "arc_length"]:
        return 1
    elif feature_key == "state":
        return max(dict_mapping["state"])
    elif feature_key == "types":
        return max(dict_mapping["types"])
    elif feature_key == "object_types":
        return max(dict_mapping["object_types"])
    else:
        raise ValueError(f"Feature {feature_key} not supported")


def normalize_path(x: jax.Array, meters: int) -> jax.Array:
    """Normalize the path by the meters."""
    x = jnp.clip(x, min=-5 * meters, max=5 * meters)
    x = x / meters

    return x


def normalize_by_feature(
    data: jax.Array, feature_key: str, meters: int, dict_mapping: dict
) -> jax.Array:
    """Normalize the data by the feature key."""
    if feature_key == "xy":
        data = normalize_path(data, meters)
    elif feature_key == "state":  # trafficlight
        data = onehot_encoder(data, dict_mapping["state"])
    elif feature_key == "types":  # roadgraph
        data = onehot_encoder(data, dict_mapping["types"])
    elif feature_key == "object_types":  # objects
        data = onehot_encoder(data, dict_mapping["object_types"])
    elif feature_key == "vel_xy":
        data = jnp.clip(data, min=0, max=MAX_SPEED)
        data = data / MAX_SPEED  # m/s
    elif feature_key in ["length", "width", "height"]:
        data = data / meters  # m
    elif feature_key in ["valid", "yaw", "arc_length", "dir_xy", "ids"]:
        pass
    else:
        raise ValueError(f"Feature {feature_key} not supported")

    return data


def onehot_encoder(types: jax.Array, mapping: tuple[int]) -> jax.Array:
    """One-hot encoder for the type of objects."""
    mapped = jnp.take(jnp.array(mapping), types, axis=-1)
    onehot = jax.nn.one_hot(mapped, max(mapping) + 1, axis=-1)
    # Drop the first "unknown" column. The result will have a size of max_val
    return onehot[..., 1:]


# ---------------------------------------------------------------------------
# vmax/simulator/operations.py
# ---------------------------------------------------------------------------


def get_index(x: jnp.ndarray, k: int = 1, squeeze: bool = True) -> jnp.ndarray:
    """Get the index of the maximum value (or top-k indices) in an array."""
    if k == 1:
        idx = jnp.argmax(x, keepdims=not squeeze)
    else:
        idx = jax.lax.top_k(x, k)[1]

        if squeeze:
            return idx.squeeze()

    return idx


# ---------------------------------------------------------------------------
# vmax/simulator/overrides/datatypes/roadgraph.py
# ---------------------------------------------------------------------------


def rotate_rectangle(rectangle: jnp.ndarray, yaw: jnp.ndarray) -> jnp.ndarray:
    """Rotate a rectangle by a specified yaw angle."""
    rot_matrix = jnp.array(
        [
            [jnp.cos(yaw), -jnp.sin(yaw)],
            [jnp.sin(yaw), jnp.cos(yaw)],
        ],
    )
    return jnp.dot(rectangle, rot_matrix)


def points_in_rectangle(points: jnp.ndarray, rectangle: jnp.ndarray) -> jnp.ndarray:
    """Determine which points lie inside a rectangle."""
    edge1 = rectangle[1] - rectangle[0]  # Top edge
    edge2 = rectangle[3] - rectangle[0]  # Right edge

    edge1_normalized = edge1 / jnp.linalg.norm(edge1)
    edge2_normalized = edge2 / jnp.linalg.norm(edge2)

    points_local = points - rectangle[0]  # Translate to origin

    proj1 = jnp.dot(points_local, edge1_normalized)
    proj2 = jnp.dot(points_local, edge2_normalized)

    in_bounds = jnp.logical_and(
        jnp.logical_and(proj1 >= 0, proj1 <= jnp.linalg.norm(edge1)),
        jnp.logical_and(proj2 >= 0, proj2 <= jnp.linalg.norm(edge2)),
    )

    return in_bounds


def filter_box_roadgraph_points(
    roadgraph: RoadgraphPoints,
    reference_points: jax.Array,
    reference_yaw: jax.Array,
    meters_box: dict,
    topk: int,
) -> RoadgraphPoints:
    """Filter roadgraph points within a bounding box and return the top-k closest."""
    chex.assert_equal_shape_prefix([roadgraph, reference_points], reference_points.ndim - 1)
    chex.assert_equal(len(roadgraph.shape), reference_points.ndim)
    chex.assert_equal(reference_points.shape[-1], 2)

    reference_box = jnp.array(
        [
            [-meters_box["back"], -meters_box["right"]],
            [meters_box["front"], -meters_box["right"]],
            [meters_box["front"], meters_box["left"]],
            [-meters_box["back"], meters_box["left"]],
        ],
    )

    rotated_box = rotate_rectangle(reference_box, reference_yaw).squeeze()
    translated_box = rotated_box + reference_points

    distances = jnp.linalg.norm(reference_points[..., jnp.newaxis, :] - roadgraph.xy, axis=-1)

    roadgraph.valid = points_in_rectangle(roadgraph.xy, translated_box)
    valid_distances = jnp.where(roadgraph.valid, distances, float("inf"))
    _, top_idx = jax.lax.top_k(-valid_distances, topk)

    # Rearrange the idx to respect the original order
    _idx = jnp.argsort(top_idx, axis=-1)
    top_idx = jnp.take_along_axis(top_idx, _idx, axis=-1)

    stacked = jnp.stack(
        [
            roadgraph.x,
            roadgraph.y,
            roadgraph.z,
            roadgraph.dir_x,
            roadgraph.dir_y,
            roadgraph.dir_z,
            roadgraph.types,
            roadgraph.ids,
            roadgraph.valid,
        ],
        axis=-1,
        dtype=jnp.float32,
    )
    filtered = jnp.take_along_axis(stacked, top_idx[..., None], axis=-2)

    return RoadgraphPoints(
        x=filtered[..., 0],
        y=filtered[..., 1],
        z=filtered[..., 2],
        dir_x=filtered[..., 3],
        dir_y=filtered[..., 4],
        dir_z=filtered[..., 5],
        types=filtered[..., 6].astype(jnp.int32),
        ids=filtered[..., 7].astype(jnp.int32),
        valid=filtered[..., 8].astype(jnp.bool_),
    )


# ---------------------------------------------------------------------------
# vmax/simulator/overrides/datatypes/observation.py (sdc_paths handling removed)
# ---------------------------------------------------------------------------


def transform_observation(observation, pose2d: ObjectPose2D):
    """Transform an Observation into coordinates specified by pose2d."""
    chex.assert_equal_shape([observation, pose2d])

    pose = combine_two_object_pose_2d(src_pose=observation.pose2d, dst_pose=pose2d)
    transf_traj = transform_trajectory(observation.trajectory, pose)
    transf_rg = transform_roadgraph_points(observation.roadgraph_static_points, pose)
    transf_tls = transform_traffic_lights(observation.traffic_lights, pose)

    obs = observation.replace(
        trajectory=transf_traj,
        roadgraph_static_points=transf_rg,
        traffic_lights=transf_tls,
        pose2d=pose2d,
    )
    obs.validate()
    return obs


def sdc_observation_from_state(
    state: datatypes.SimulatorState,
    obs_num_steps: int = 1,
    roadgraph_top_k: int | None = 1000,
    meters_box: dict | None = None,
):
    """Construct the SDC-frame Observation from a SimulatorState (jit-able)."""
    obj_xy = state.current_sim_trajectory.xy[..., 0, :]
    obj_yaw = state.current_sim_trajectory.yaw[..., 0]
    obj_valid = state.current_sim_trajectory.valid[..., 0]

    _, sdc_idx = jax.lax.top_k(state.object_metadata.is_sdc, k=1)
    sdc_xy = jnp.take_along_axis(obj_xy, sdc_idx[..., jnp.newaxis], axis=-2)
    sdc_yaw = jnp.take_along_axis(obj_yaw, sdc_idx, axis=-1)
    sdc_valid = jnp.take_along_axis(obj_valid, sdc_idx, axis=-1)

    if meters_box is None:
        raise ValueError("This port only supports the meters_box roadgraph filter.")

    num_obj = 1
    global_obs = global_observation_from_state(state, obs_num_steps, num_obj=num_obj)
    is_ego = state.object_metadata.is_sdc[..., jnp.newaxis, :]
    global_obs_filter = global_obs.replace(
        is_ego=is_ego,
        roadgraph_static_points=filter_box_roadgraph_points(
            global_obs.roadgraph_static_points,
            sdc_xy,
            sdc_yaw,
            meters_box,
            roadgraph_top_k,
        ),
    )

    pose2d = ObjectPose2D.from_center_and_yaw(xy=sdc_xy, yaw=sdc_yaw, valid=sdc_valid)
    chex.assert_equal(pose2d.shape, state.shape + (1,))
    return transform_observation(global_obs_filter, pose2d)


# ---------------------------------------------------------------------------
# vmax/simulator/features/features_datatypes.py (plotting removed)
# ---------------------------------------------------------------------------


@chex.dataclass
class ObjectFeatures:
    """Features of dynamic objects."""

    field_names: Sequence[str]
    xy: jax.Array = field(default_factory=lambda: jnp.array(()))
    vel_xy: jax.Array = field(default_factory=lambda: jnp.array(()))
    yaw: jax.Array = field(default_factory=lambda: jnp.array(()))
    length: jax.Array = field(default_factory=lambda: jnp.array(()))
    width: jax.Array = field(default_factory=lambda: jnp.array(()))
    object_types: jax.Array = field(default_factory=lambda: jnp.array(()))
    valid: jax.Array = field(default_factory=lambda: jnp.array(()))

    def stack_fields(self) -> jax.Array:
        if len(self.field_names) == 0:
            return jnp.array(())
        return jnp.concatenate(
            [getattr(self, field_name) for field_name in self.field_names], axis=-1
        )


@chex.dataclass
class RoadgraphFeatures:
    """Features of the road graph."""

    field_names: Sequence[str]
    xy: jax.Array = field(default_factory=lambda: jnp.array(()))
    dir_xy: jax.Array = field(default_factory=lambda: jnp.array(()))
    types: jax.Array = field(default_factory=lambda: jnp.array(()))
    ids: jax.Array = field(default_factory=lambda: jnp.array(()))
    valid: jax.Array = field(default_factory=lambda: jnp.array(()))

    def stack_fields(self) -> jax.Array:
        if len(self.field_names) == 0:
            return jnp.array(())
        return jnp.concatenate(
            [getattr(self, field_name) for field_name in self.field_names], axis=-1
        )


@chex.dataclass
class TrafficLightFeatures:
    """Features of traffic lights."""

    field_names: Sequence[str]
    xy: jax.Array = field(default_factory=lambda: jnp.array(()))
    state: jax.Array = field(default_factory=lambda: jnp.array(()))
    ids: jax.Array = field(default_factory=lambda: jnp.array(()))
    valid: jax.Array = field(default_factory=lambda: jnp.array(()))

    def stack_fields(self) -> jax.Array:
        if len(self.field_names) == 0:
            return jnp.array(())
        return jnp.concatenate(
            [getattr(self, field_name) for field_name in self.field_names], axis=-1
        )


@chex.dataclass
class PathTargetFeatures:
    """Features of path targets."""

    xy: jax.Array = field(default_factory=lambda: jnp.array(()))
    valid: jax.Array = field(default_factory=lambda: jnp.array(()))

    @property
    def data(self) -> jax.Array:
        return self.xy


# ---------------------------------------------------------------------------
# vmax/simulator/features/extractor/vec_extractor.py (vec path only)
# ---------------------------------------------------------------------------

FEATURE_MAP = {
    "waypoints": ("xy",),
    "velocity": ("vel_xy",),
    "speed": ("speed",),
    "yaw": ("yaw",),
    "size": ("length", "width"),
    "valid": ("valid",),
    "direction": ("dir_xy",),
    "types": ("types",),
    "state": ("state",),
    "object_types": ("object_types",),
}


class VecFeaturesExtractor:
    """Vectorized feature extractor (V-Max ``observation_type=vec``)."""

    def __init__(
        self,
        obs_past_num_steps: int | None = None,
        objects_config: dict[str, Any] | None = None,
        roadgraphs_config: dict[str, Any] | None = None,
        traffic_lights_config: dict[str, Any] | None = None,
        path_target_config: dict[str, Any] | None = None,
    ) -> None:
        self._obs_past_num_steps = obs_past_num_steps or 1

        self._objects_config = objects_config or {"features": []}
        self._roadgraphs_config = roadgraphs_config or {"features": []}
        self._traffic_light_config = traffic_lights_config or {"features": []}
        self._path_target_config = path_target_config or {"features": []}

        # Only direct integer element types are supported in this port.
        self._roadgraph_element_types = self._roadgraphs_config.get("element_types", None)
        if self._roadgraph_element_types is not None:
            assert all(isinstance(t, int) for t in self._roadgraph_element_types)

        self._num_closest_objects = self._objects_config.get("num_closest_objects", 8)

        self._meters_box = self._roadgraphs_config.get("meters_box")
        self._roadgraph_top_k = self._roadgraphs_config.get("roadgraph_top_k", 1000)
        self._roadgraph_interval = self._roadgraphs_config.get("interval", 1)
        self._max_meters = self._roadgraphs_config.get("max_meters", 50)

        if self._meters_box is None:
            self._roadgraph_top_k_prefilter = max(self._roadgraph_top_k, 2000)
        else:
            self._roadgraph_top_k_prefilter = (
                self._meters_box["front"] + self._meters_box["back"]
            ) * (self._meters_box["left"] + self._meters_box["right"])

        self._num_closest_traffic_lights = self._traffic_light_config.get(
            "num_closest_traffic_lights", 16
        )

        self._num_target_path_points = self._path_target_config.get("num_points", 10)
        self._points_gap = self._path_target_config.get("points_gap", 5)

        self._dict_mapping = {
            "types": RG_MAPPING,
            "state": TL_MAPPING,
            "object_types": OBJECT_MAPPING,
        }

        self._object_features_key = self._extract_feature_keys(self._objects_config["features"])
        self._roadgraph_features_key = self._extract_feature_keys(
            self._roadgraphs_config["features"]
        )
        self._traffic_lights_features_key = self._extract_feature_keys(
            self._traffic_light_config["features"]
        )
        self._path_target_features_key = self._extract_feature_keys(
            self._path_target_config["features"]
        )

    def _extract_feature_keys(self, feature_names: list[str]) -> list[str]:
        result = []
        for key in feature_names:
            if key not in FEATURE_MAP:
                raise ValueError(f"Unknown feature name '{key}'")
            result.extend(FEATURE_MAP[key])
        return result

    def get_features_size(self, feature_keys) -> int:
        return sum([get_feature_size(key, self._dict_mapping) for key in feature_keys])

    def _get_sdc_observation(self, state: datatypes.SimulatorState):
        sdc_observation = sdc_observation_from_state(
            state,
            self._obs_past_num_steps,
            self._roadgraph_top_k_prefilter,
            self._meters_box,
        )

        return jax.tree.map(lambda x: x[0], sdc_observation)

    def extract_features(self, state: datatypes.SimulatorState):
        """Extract the five feature groups from the simulator state."""
        sdc_observation = self._get_sdc_observation(state)

        objects_features = self._build_objects_features(sdc_observation)
        roadgraphs_features = self._build_roadgraph_features(sdc_observation)
        traffic_lights_features = self._build_traffic_lights_features(sdc_observation)
        path_target_features = self._build_expert_target_features(sdc_observation, state)

        # (num_agents + 1, obs_past_num_steps, num_trajectories_features)
        stack_object_features = objects_features.stack_fields()
        # (obs_past_num_steps, F), the SDC is always the first (closest) object
        sdc_object_features = stack_object_features[0, :, :]
        # (num_closest_agents, obs_past_num_steps, F)
        other_objects_features = stack_object_features[1:, :, :]

        return (
            sdc_object_features,
            other_objects_features,
            roadgraphs_features.stack_fields(),
            traffic_lights_features.stack_fields(),
            path_target_features.data,
        )

    def observe(self, state: datatypes.SimulatorState) -> jax.Array:
        """Extract features and flatten into the policy's observation vector."""
        features = self.extract_features(state)

        return jnp.concatenate([jnp.reshape(f, (-1,)) for f in features], axis=0)

    def unflatten_features(self, vectorized_obs: jax.Array):
        """Unflatten a vectorized observation into features and masks."""
        batch_dims = vectorized_obs.shape[-3:-1]
        flatten_size = vectorized_obs.shape[-1]
        unflatten_size = 0

        object_features_size = self.get_features_size(self._object_features_key)
        roadgraph_features_size = self.get_features_size(self._roadgraph_features_key)
        traffic_lights_features_size = self.get_features_size(self._traffic_lights_features_key)
        path_target_feature_size = self.get_features_size(self._path_target_features_key)

        sdc_object_size = 1 * self._obs_past_num_steps * object_features_size
        sdc_object_features = vectorized_obs[
            ..., unflatten_size : unflatten_size + sdc_object_size
        ]
        sdc_object_features = sdc_object_features.reshape(
            *batch_dims,
            1,
            self._obs_past_num_steps,
            object_features_size,
        )
        unflatten_size += sdc_object_size

        other_objects_size = (
            self._num_closest_objects * self._obs_past_num_steps * object_features_size
        )
        other_objects_features = vectorized_obs[
            ..., unflatten_size : unflatten_size + other_objects_size
        ]
        other_objects_features = other_objects_features.reshape(
            *batch_dims,
            self._num_closest_objects,
            self._obs_past_num_steps,
            object_features_size,
        )
        unflatten_size += other_objects_size

        roadgraph_size = self._roadgraph_top_k * roadgraph_features_size
        roadgraphs_features = vectorized_obs[..., unflatten_size : unflatten_size + roadgraph_size]
        roadgraphs_features = roadgraphs_features.reshape(
            *batch_dims,
            self._roadgraph_top_k,
            roadgraph_features_size,
        )
        unflatten_size += roadgraph_size

        traffic_lights_size = (
            self._num_closest_traffic_lights
            * self._obs_past_num_steps
            * traffic_lights_features_size
        )
        traffic_lights_features = vectorized_obs[
            ..., unflatten_size : unflatten_size + traffic_lights_size
        ]
        traffic_lights_features = traffic_lights_features.reshape(
            *batch_dims,
            self._num_closest_traffic_lights,
            self._obs_past_num_steps,
            traffic_lights_features_size,
        )
        unflatten_size += traffic_lights_size

        path_target_size = self._num_target_path_points * path_target_feature_size
        path_target_features = vectorized_obs[
            ..., unflatten_size : unflatten_size + path_target_size
        ]
        path_target_features = path_target_features.reshape(
            *batch_dims,
            self._num_target_path_points,
            path_target_feature_size,
        )
        unflatten_size += path_target_size

        assert (
            flatten_size == unflatten_size
        ), f"Unflatten size {unflatten_size} does not match {flatten_size}"

        features = (
            sdc_object_features[..., :-1],
            other_objects_features[..., :-1],
            roadgraphs_features[..., :-1],
            traffic_lights_features[..., :-1],
            path_target_features,
        )
        masks = (
            sdc_object_features[..., -1].astype(bool),
            other_objects_features[..., -1].astype(bool),
            roadgraphs_features[..., -1].astype(bool),
            traffic_lights_features[..., -1].astype(bool),
        )

        return features, masks

    def _build_objects_features(self, sdc_obs) -> ObjectFeatures:
        object_features = ObjectFeatures(field_names=self._object_features_key)

        if not self._object_features_key:
            return object_features

        distances_ego_objects = jnp.linalg.norm(sdc_obs.trajectory.xy[:, -1, :], axis=-1)
        distances_ego_valid_objects = jnp.where(
            sdc_obs.trajectory.valid[:, -1], distances_ego_objects, jnp.inf
        )

        closest_object_idxs = get_index(
            -distances_ego_valid_objects,
            k=self._num_closest_objects + 1,
            squeeze=False,
        )

        object_features = ObjectFeatures(field_names=self._object_features_key)
        for key in self._object_features_key:
            feature = (
                getattr(sdc_obs.metadata, key)
                if key == "object_types"
                else getattr(sdc_obs.trajectory, key)
            )
            feature = feature[closest_object_idxs]
            feature = normalize_by_feature(feature, key, self._max_meters, self._dict_mapping)

            if feature.ndim == 2:
                feature = jnp.expand_dims(feature, axis=-1)

            setattr(object_features, key, feature)

        return object_features

    def _build_roadgraph_features(self, sdc_obs) -> RoadgraphFeatures:
        roadgraph_features = RoadgraphFeatures(field_names=self._roadgraph_features_key)

        if len(self._roadgraph_features_key) == 0:
            return roadgraph_features

        roadgraph_points = self._reduce_and_filter_roadgraph_points(
            sdc_obs.roadgraph_static_points
        )

        for key in self._roadgraph_features_key:
            feature = getattr(roadgraph_points, key)
            feature = normalize_by_feature(feature, key, self._max_meters, self._dict_mapping)

            if feature.ndim == 1:
                feature = jnp.expand_dims(feature, axis=-1)

            setattr(roadgraph_features, key, feature)

        return roadgraph_features

    def _build_traffic_lights_features(self, sdc_obs) -> TrafficLightFeatures:
        traffic_light_features = TrafficLightFeatures(
            field_names=self._traffic_lights_features_key
        )

        if len(self._traffic_lights_features_key) == 0:
            return traffic_light_features

        distances_traffic_lights = jnp.linalg.norm(sdc_obs.traffic_lights.xy[:, -1], axis=-1)
        distances_traffic_lights_valid = jnp.where(
            sdc_obs.traffic_lights.valid[:, -1],
            distances_traffic_lights,
            jnp.inf,
        )
        closest_tl_idxs = get_index(
            -distances_traffic_lights_valid,
            k=self._num_closest_traffic_lights,
            squeeze=False,
        )

        for key in self._traffic_lights_features_key:
            feature = getattr(sdc_obs.traffic_lights, key)[closest_tl_idxs]
            feature = normalize_by_feature(feature, key, self._max_meters, self._dict_mapping)

            if feature.ndim == 2:
                feature = jnp.expand_dims(feature, axis=-1)

            setattr(traffic_light_features, key, feature)

        return traffic_light_features

    def _build_expert_target_features(self, sdc_obs, state) -> PathTargetFeatures:
        """Path target = the SDC's last valid logged point (the challenge goal)."""
        if not self._path_target_features_key:
            return PathTargetFeatures()

        sdc_index = get_index(state.object_metadata.is_sdc)
        sdc_log = jax.tree.map(lambda x: x[sdc_index], state.log_trajectory)
        steps = jnp.arange(sdc_log.valid.shape[-1])
        last_valid_idx = jnp.max(jnp.where(sdc_log.valid, steps, 0))
        endpoint_global = sdc_log.xy[last_valid_idx]  # (2,) world frame

        endpoint_local = geometry.transform_points(sdc_obs.pose2d.matrix, endpoint_global)

        path_target = jnp.broadcast_to(endpoint_local, (self._num_target_path_points, 2))
        path_target = normalize_path(path_target, self._max_meters)

        return PathTargetFeatures(xy=path_target)

    def _reduce_and_filter_roadgraph_points(self, roadgraph) -> datatypes.RoadgraphPoints:
        valid_filter = self._filter_roadgraph_points(roadgraph)
        roadgraph.valid = roadgraph.valid & valid_filter

        xy = roadgraph.xy
        dist = jnp.linalg.norm(xy, axis=-1)
        dist = jnp.where(roadgraph.valid, dist, jnp.inf)

        idx_to_keep = jnp.arange(0, len(dist), self._roadgraph_interval)
        mask = jnp.zeros_like(dist, dtype=bool).at[idx_to_keep].set(True)

        masked_dist = jnp.where(mask, dist, jnp.inf)

        _, idx = jax.lax.top_k(-masked_dist, min(self._roadgraph_top_k, masked_dist.size))

        return jax.tree.map(lambda x: x[idx], roadgraph)

    def _filter_roadgraph_points(self, roadgraph) -> jax.Array:
        if self._roadgraph_element_types is None:
            return jnp.ones_like(roadgraph.valid, dtype=bool)

        result = jnp.zeros_like(roadgraph.valid, dtype=bool)
        for element_type in self._roadgraph_element_types:
            result = result | (roadgraph.types == element_type)

        return result
