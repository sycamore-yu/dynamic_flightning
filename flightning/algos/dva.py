from functools import partial
from typing import Callable, NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from flightning.algos._common import (
    clip_grads,
    ema_update,
    td_lambda_targets,
)
from flightning.envs.env_base import Env, EnvState
from flightning.envs.wrappers import LogWrapper, VecEnv


class DVAConfig(NamedTuple):
    gamma: float = 0.99
    lam: float = 0.95
    critic_method: str = "td-lambda"
    target_critic_alpha: float = 0.4
    critic_iterations: int = 16
    num_batches: int = 4
    max_grad_norm: float = 1.0
    logging_freq: int = 10
    logging: bool = True


class DVAObservation(NamedTuple):
    actor_obs: jax.Array
    critic_obs: jax.Array


def default_observation_adapter(obs: jax.Array) -> DVAObservation:
    return DVAObservation(actor_obs=obs, critic_obs=obs)


class DVASample(NamedTuple):
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    done: jax.Array
    next_obs: jax.Array


class RunnerState(NamedTuple):
    actor_state: TrainState
    critic_state: TrainState
    target_critic_state: TrainState
    env_state: EnvState
    last_obs: jax.Array
    key: chex.PRNGKey
    epoch_idx: int


def _actor_sample_action(key, actor_state, obs, deterministic):
    """Sample actions for a *batch* of obs (one key per env in the batch)."""

    def _single(key_i, obs_i):
        return actor_state.apply_fn(
            actor_state.params,
            obs_i,
            key_i,
            deterministic,
            method="sample_action",
        )

    return jax.vmap(_single)(key, obs)


def train(
    env: Env,
    actor_state: TrainState,
    critic_state: Optional[TrainState] = None,
    *,
    observation_adapter: Callable[[jax.Array], DVAObservation] = default_observation_adapter,
    num_epochs: int = 100,
    num_steps_per_epoch: int = 50,
    num_envs: int = 64,
    key: chex.PRNGKey = jax.random.key(0),
    config: DVAConfig = DVAConfig(),
):
    """Train a policy with D.VA (Decoupled Visual-Based Analytical Policy Gradient)."""
    env = LogWrapper(env)
    env = VecEnv(env)
    use_shared_net = critic_state is None

    # ---- rollout ---------------------------------------------------------
    def _rollout(actor_state, env_state, last_obs, key):
        def _step(carry, key_step):
            actor_state, env_state, last_obs = carry
            key_action, key_step = jax.random.split(key_step)
            key_action = jax.random.split(key_action, num_envs)
            key_step = jax.random.split(key_step, num_envs)

            # Map observation using adapter
            dva_obs = observation_adapter(last_obs)
            # Stop gradient on actor observation inside the algorithm before forward pass
            actor_obs_stop = jax.lax.stop_gradient(dva_obs.actor_obs)

            sample_out = _actor_sample_action(
                key_action, actor_state, actor_obs_stop, False
            )
            action = sample_out.action
            trans = env.step(env_state, action, key_step)
            done = jnp.logical_or(trans.terminated, trans.truncated).astype(
                jnp.float32
            )
            sample = DVASample(
                obs=last_obs,
                action=action,
                reward=trans.reward,
                done=done,
                next_obs=trans.obs,
            )
            return (
                (actor_state, trans.state, trans.obs),
                sample,
            )

        keys = jax.random.split(key, num_steps_per_epoch)
        (_, final_env_state, final_obs), samples = jax.lax.scan(
            _step,
            (actor_state, env_state, last_obs),
            keys,
        )
        return samples, final_env_state, final_obs

    # ---- actor loss ------------------------------------------------------
    def _actor_loss_fn(
        actor_params, critic_params, env_state, last_obs, key
    ):
        view_actor = actor_state.replace(params=actor_params)
        samples, final_env_state, final_obs = _rollout(
            view_actor, env_state, last_obs, key
        )

        # Bootstrap values with the (stop-grad) target critic
        dva_obs_initial = observation_adapter(last_obs)
        dva_obs_next = observation_adapter(samples.next_obs)

        v_initial = _eval_critic(critic_params, dva_obs_initial.critic_obs)
        v_next = _eval_critic(critic_params, dva_obs_next.critic_obs)
        # Shape: (T+1, N); next_values[t + 1] is V(s_{t+1}).
        next_values = jnp.concatenate(
            [v_initial[None, ...], v_next], axis=0
        )  # (T+1, N)

        rewards = samples.reward                 # (T, N)
        dones = samples.done                     # (T, N)
        gamma = config.gamma

        def _return_scan(carry, step):
            acc, g = carry
            r, d, v_tp1, is_final = step
            acc_tp1 = acc + g * r
            bootstrap = gamma * g * v_tp1 * (1.0 - d)
            boundary = jnp.logical_or(d.astype(bool), is_final)
            loss_term = jnp.where(boundary, -(acc_tp1 + bootstrap), 0.0)
            next_acc = jnp.where(d.astype(bool), 0.0, acc_tp1)
            next_g = jnp.where(d.astype(bool), 1.0, g * gamma)
            return (next_acc, next_g), loss_term

        init = (
            jnp.zeros((num_envs,)),
            jnp.ones((num_envs,)),
        )
        is_final = jnp.arange(num_steps_per_epoch) == (num_steps_per_epoch - 1)
        steps = (rewards, dones, next_values[1:], is_final)
        _, loss_terms = jax.lax.scan(_return_scan, init, steps)
        actor_loss = loss_terms.sum() / (num_steps_per_epoch * num_envs)
        return actor_loss, (samples, next_values, final_env_state, final_obs)

    def _eval_critic(params, obs):
        if use_shared_net:
            return actor_state.apply_fn(params, obs)
        return critic_state.apply_fn(params, obs)

    # ---- epoch body ------------------------------------------------------
    def _epoch_fn(runner_state, _unused):
        (
            actor_state,
            critic_state,
            target_critic_state,
            env_state,
            last_obs,
            key,
            epoch_idx,
        ) = runner_state

        key, key_rollout, key_critic = jax.random.split(key, 3)

        # Actor gradient step
        actor_grad_fn = jax.value_and_grad(
            _actor_loss_fn, argnums=0, has_aux=True
        )
        (
            actor_loss,
            (samples, next_values, final_env_state, final_obs),
        ), actor_grads = actor_grad_fn(
            actor_state.params,
            target_critic_state.params,
            env_state,
            last_obs,
            key_rollout,
        )
        actor_grads_clipped = clip_grads(actor_grads, config.max_grad_norm)
        actor_state = actor_state.apply_gradients(grads=actor_grads_clipped)

        # Compute critic targets
        if config.critic_method == "td-lambda":
            target_values = td_lambda_targets(
                samples.reward,
                samples.done,
                next_values,
                config.gamma,
                config.lam,
            )  # (T, N)
        elif config.critic_method == "one-step":
            target_values = samples.reward + config.gamma * next_values[1:] * (1.0 - samples.done)
        else:
            raise ValueError(f"Unknown critic method: {config.critic_method}")

        # Map buffer obs using adapter to get critic_obs for dataset
        dva_obs_all = observation_adapter(samples.obs)
        flat_critic_obs = dva_obs_all.critic_obs.reshape((-1,) + dva_obs_all.critic_obs.shape[2:])
        flat_targets = target_values.reshape((-1,))

        total = flat_critic_obs.shape[0]
        batch_size = max(total // config.num_batches, 1)
        n_batches = total // batch_size
        flat_critic_obs = flat_critic_obs[: n_batches * batch_size]
        flat_targets = flat_targets[: n_batches * batch_size]
        batched_critic_obs = flat_critic_obs.reshape((n_batches, batch_size) + flat_critic_obs.shape[1:])
        batched_targets = flat_targets.reshape((n_batches, batch_size))

        def _critic_minibatch_step(critic_state, batch):
            batch_critic_obs, batch_tgt = batch

            def _loss(params):
                v = _eval_critic(params, batch_critic_obs)
                return jnp.mean((v - batch_tgt) ** 2)

            loss, grads = jax.value_and_grad(_loss)(critic_state.params)
            grads = jax.tree_util.tree_map(
                lambda g: jnp.nan_to_num(g, 0.0, 0.0, 0.0), grads
            )
            # Clip critic grads if max_grad_norm is set
            grads = clip_grads(grads, config.max_grad_norm)
            critic_state = critic_state.apply_gradients(grads=grads)
            return critic_state, loss

        def _critic_pass(critic_state, _):
            critic_state, losses = jax.lax.scan(
                _critic_minibatch_step,
                critic_state,
                (batched_critic_obs, batched_targets),
            )
            return critic_state, losses.mean()

        critic_state, critic_losses = jax.lax.scan(
            _critic_pass, critic_state, None, length=config.critic_iterations
        )
        value_loss = critic_losses[-1]

        # EMA update of target critic
        new_target_params = ema_update(
            critic_state.params,
            target_critic_state.params,
            config.target_critic_alpha,
        )
        target_critic_state = target_critic_state.replace(
            params=new_target_params
        )

        metric = {"actor_loss": actor_loss, "value_loss": value_loss}

        def _log(payload):
            ep_idx, actor_loss, value_loss = payload
            print(f"[dva] epoch {ep_idx}: actor_loss={actor_loss:.3f}, value_loss={value_loss:.3f}")

        jax.lax.cond(
            jnp.logical_and(
                config.logging, epoch_idx % config.logging_freq == 0
            ),
            lambda _: jax.debug.callback(_log, (epoch_idx, actor_loss, value_loss)),
            lambda _: None,
            None,
        )

        next_runner = RunnerState(
            actor_state=actor_state,
            critic_state=critic_state,
            target_critic_state=target_critic_state,
            env_state=final_env_state,
            last_obs=final_obs,
            key=key,
            epoch_idx=epoch_idx + 1,
        )
        return next_runner, metric

    # ---- driver ----------------------------------------------------------
    def _train(runner_state):
        final_state, metrics = jax.lax.scan(
            _epoch_fn, runner_state, None, length=num_epochs
        )
        return {"runner_state": final_state, "metrics": metrics}

    # Initialise environment + runner state.
    key, key_reset = jax.random.split(key)
    reset_keys = jax.random.split(key_reset, num_envs)
    env_state, obs = env.reset(reset_keys, None)

    if use_shared_net:
        critic_state = actor_state
        target_critic_state = actor_state
    else:
        target_critic_state = critic_state

    runner_state = RunnerState(
        actor_state=actor_state,
        critic_state=critic_state,
        target_critic_state=target_critic_state,
        env_state=env_state,
        last_obs=obs,
        key=key,
        epoch_idx=0,
    )
    return jax.jit(_train)(runner_state)
