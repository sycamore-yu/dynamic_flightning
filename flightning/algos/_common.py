"""Shared building blocks for the RL algorithms in ``flightning.algos``.

The functions here are algorithm-agnostic (gradient clipping, TD-lambda
targets, EMA parameter updates, rollout collection) and are re-used by
``shac``, ``ppo`` and ``bptt``.
"""

from functools import partial
from typing import Callable, Optional

import jax
import jax.numpy as jnp

from flightning.envs.env_base import Env, EnvState, EnvTransition


def clip_grads(grads, max_norm: float):
    """Clip pytree of gradients to ``max_norm`` (L2, global)."""
    flat, _ = jax.tree_util.tree_flatten(grads)
    total = sum(jnp.sum(g ** 2) for g in flat if g is not None)
    norm = jnp.sqrt(total)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def td_lambda_targets(
    rewards: jax.Array,
    dones: jax.Array,
    next_values: jax.Array,
    gamma: float,
    lam: float,
) -> jax.Array:
    """Compute TD-lambda targets using DiffRL's backward Ai/Bi trace.

    Args:
        rewards: (T, N) per-step rewards.
        dones: (T, N) float done flags (1.0 when the episode ended at t).
        next_values: (T+1, N) bootstrap values, ``next_values[t+1]`` being
            V(s_{t+1}) with ``next_values[T]`` the terminal bootstrap.
        gamma: discount.
        lam: trace decay (0 < lam <= 1).

    Returns:
        targets: (T, N) such that ``targets[t]`` is the TD-lambda target
        for step t.
    """

    def _scan_fn(carry, step):
        Ai_prev, Bi_prev, lam_prev = carry
        r, d, v_next = step
        lam_t = lam_prev * lam * (1.0 - d) + d
        Ai = (1.0 - d) * (
            lam * gamma * Ai_prev
            + gamma * v_next
            + (1.0 - lam_t) / (1.0 - lam) * r
        )
        Bi = gamma * (v_next * d + Bi_prev * (1.0 - d)) + r
        target = (1.0 - lam) * Ai + lam_t * Bi
        return (Ai, Bi, lam_t), target

    bootstrap_values = next_values[1:]
    final_dones = dones.at[-1].set(1.0)
    init = (
        jnp.zeros_like(next_values[0]),
        jnp.zeros_like(next_values[0]),
        jnp.ones_like(next_values[0]),
    )
    _, targets_rev = jax.lax.scan(
        _scan_fn,
        init,
        (rewards[::-1], final_dones[::-1], bootstrap_values[::-1]),
    )
    return targets_rev[::-1]


def ema_update(params, target_params, alpha: float):
    """Polyak/EMA update: ``target = alpha * target + (1 - alpha) * params``.

    Used for target networks in SHAC, SAC, DQN-style algorithms.
    """
    return jax.tree_util.tree_map(
        lambda p, pt: alpha * pt + (1.0 - alpha) * p, params, target_params
    )


def get_rollouts(
    env: Env,
    policy: Callable,
    num_rollouts: int,
    key: jax.Array,
):
    """Run ``num_rollouts`` rollouts of ``env`` under ``policy`` in parallel.

    ``policy`` is a callable ``(obs, key) -> action``. Returns the stacked
    ``EnvTransition`` pytree produced by :func:`flightning.envs.rollout`.
    """
    from flightning.envs import rollout  # local import avoids a cycle

    parallel_rollout = jax.vmap(rollout, in_axes=(None, 0, None))
    rollout_keys = jax.random.split(key, num_rollouts)
    return parallel_rollout(env, rollout_keys, policy)
