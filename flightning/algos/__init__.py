"""RL algorithms shipped with ``flightning``.

Three algorithms are currently available:

* :mod:`flightning.algos.bptt` — Back-Propagation Through Time
* :mod:`flightning.algos.ppo`  — Proximal Policy Optimization
* :mod:`flightning.algos.shac` — Short-Horizon Actor Critic

All three expose a ``train(env, train_state, ..., config=Config())`` API
and return a ``{"runner_state": ..., "metrics": ...}`` dict. For one-line
imports use:

>>> from flightning.algos import train_shac, SHACConfig
"""

from flightning.algos import bptt, ppo, shac, dva
from flightning.algos._common import (
    clip_grads,
    ema_update,
    get_rollouts,
    td_lambda_targets,
)
from flightning.algos.bptt import Config as BPTTConfig
from flightning.algos.bptt import train as train_bptt
from flightning.algos.ppo import Config as PPOConfig
from flightning.algos.ppo import train as train_ppo
from flightning.algos.shac import Config as SHACConfig
from flightning.algos.shac import train as train_shac
from flightning.algos.dva import DVAConfig
from flightning.algos.dva import train as train_dva

__all__ = [
    "bptt",
    "ppo",
    "shac",
    "dva",
    "train_bptt",
    "train_ppo",
    "train_shac",
    "train_dva",
    "BPTTConfig",
    "PPOConfig",
    "SHACConfig",
    "DVAConfig",
    "clip_grads",
    "ema_update",
    "get_rollouts",
    "td_lambda_targets",
]
