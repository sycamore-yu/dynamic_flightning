from .env_base import rollout, rollout_recurrent

__all__ = [
    "rollout",
    "rollout_recurrent",
    "HoveringStateEnv",
    "HoveringFeaturesEnv",
    "DynamicAvoidanceEnv",
]


def __getattr__(name):
    if name == "HoveringStateEnv":
        from .hovering_state_env import HoveringStateEnv

        return HoveringStateEnv
    if name == "HoveringFeaturesEnv":
        from .hovering_features_env import HoveringFeaturesEnv

        return HoveringFeaturesEnv
    if name == "DynamicAvoidanceEnv":
        from .dynamic_avoidance_env import DynamicAvoidanceEnv

        return DynamicAvoidanceEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
