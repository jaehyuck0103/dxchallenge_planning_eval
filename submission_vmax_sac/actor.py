"""V-Max SAC baseline (run 260623_1_rideflux) as a challenge submission.

Self-contained port of the V-Max inference path: vec feature extraction
(`feature_extractor.py`), LQ-encoder policy network (`network.py`) and the
exported policy weights (`weights.pkl`). Deterministic SAC action =
tanh(loc of the gaussian head), already in the [-1, 1] bicycle action range.
"""

import os
import pickle

import jax.numpy as jnp
from waymax import datatypes
from waymax.agents import actor_core

from feature_extractor import VecFeaturesExtractor
from network import build_policy_network, deterministic_action

# observation_config of the training run (runs/260623_1_rideflux/.hydra/config.yaml).
OBSERVATION_CONFIG = {
    "obs_past_num_steps": 5,
    "objects_config": {
        "features": ["waypoints", "velocity", "yaw", "size", "valid"],
        "num_closest_objects": 16,
    },
    "roadgraphs_config": {
        "features": ["waypoints", "direction", "valid"],
        "element_types": [15, 16],  # road edges
        "interval": 2,
        "max_meters": 70,
        "roadgraph_top_k": 200,
        "meters_box": {"front": 70, "back": 5, "left": 20, "right": 20},
    },
    "traffic_lights_config": {
        "features": ["waypoints", "state", "valid"],
        "num_closest_traffic_lights": 5,
    },
    "path_target_config": {
        "features": ["waypoints"],
        "num_points": 1,
        "points_gap": 50,
    },
}


class VmaxSacPlanner(actor_core.WaymaxActorCore):
    """SAC policy (LQ encoder) trained with the V-Max baseline on rideflux."""

    def __init__(self, params):
        self._params = params
        self._extractor = VecFeaturesExtractor(**OBSERVATION_CONFIG)
        self._policy = build_policy_network(self._extractor.unflatten_features)

    def init(self, rng, state):
        """No recurrent state."""
        return None

    def select_action(self, params, state, actor_state, rng):
        del params, actor_state, rng  # the planner owns its weights

        obs = self._extractor.observe(state)  # (obs_dim,)
        logits = self._policy.apply(self._params, obs[None])[0]  # (4,)
        action_data = deterministic_action(logits)  # (2,) in [-1, 1]

        action = datatypes.Action(data=action_data, valid=jnp.ones((1,), dtype=jnp.bool_))
        return actor_core.WaymaxActorOutput(
            actor_state=None,
            action=action,
            is_controlled=state.object_metadata.is_sdc,
        )

    @property
    def name(self) -> str:
        return "vmax_sac_lq_260623_1_rideflux"


def create_actor(submission_dir: str) -> actor_core.WaymaxActorCore:
    """Entry point called by the evaluator."""
    with open(os.path.join(submission_dir, "weights.pkl"), "rb") as f:
        params = pickle.load(f)

    return VmaxSacPlanner(params)
