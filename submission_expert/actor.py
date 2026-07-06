"""Organizer-side sanity submission: expert log replay via inverse bicycle dynamics.

Requires running evaluate.py with --disable_future_masking (the expert reads the
logged future to invert actions). Expected to score high (~0.96 on the filtered
validation split); validates the evaluation pipeline end to end.

NOT a valid participant submission — future masking is off.
"""

from waymax import dynamics
from waymax.agents import actor_core, create_expert_actor
from waymax.env.planning_agent_environment import PlanningAgentDynamics


def create_actor(submission_dir: str) -> actor_core.WaymaxActorCore:
    del submission_dir
    # No jit here: the evaluator wraps select_action in jit(vmap(...)) itself.
    return create_expert_actor(
        dynamics_model=PlanningAgentDynamics(
            dynamics.InvertibleBicycleModel(normalize_actions=True)
        ),
    )
