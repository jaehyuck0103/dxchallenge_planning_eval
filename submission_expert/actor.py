"""Organizer-side sanity submission: expert log replay via inverse bicycle dynamics.

Requires running evaluate.py with --disable_future_masking (the expert reads the
logged future to invert actions). Expected to score high (~0.96 on the filtered
validation split); validates the evaluation pipeline end to end.

NOT a valid participant submission — future masking is off.
"""

import jax
from waymax import dynamics
from waymax.agents import actor_core, create_expert_actor
from waymax.env.planning_agent_environment import PlanningAgentDynamics


def create_actor(submission_dir: str) -> actor_core.WaymaxActorCore:
    del submission_dir
    expert = create_expert_actor(
        dynamics_model=PlanningAgentDynamics(
            dynamics.InvertibleBicycleModel(normalize_actions=True)
        ),
    )
    # jit is the submission's own responsibility: the evaluator never jits
    # participant code. select_action is pure, so wrapping it is enough.
    return actor_core.actor_core_factory(
        init=expert.init,
        select_action=jax.jit(expert.select_action),
        name=expert.name,
    )
