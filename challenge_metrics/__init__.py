"""Challenge metrics.

Self-contained copies of the V-Max metrics needed for the rideflux score
(progress_ratio, comfort, offroad_in_box) plus the score aggregation.
No dependency on the V-Max package — only waymax / jax / numpy.
"""

from .aggregate import episode_scores, rideflux_aggregate_score
from .comfort import compute_comfort
from .offroad_in_box import is_sdc_offroad_in_box
from .progress_ratio import compute_progress_ratio

__all__ = [
    "compute_comfort",
    "compute_progress_ratio",
    "episode_scores",
    "is_sdc_offroad_in_box",
    "rideflux_aggregate_score",
]
