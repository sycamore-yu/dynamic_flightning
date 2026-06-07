"""SHAC: Short-Horizon Actor Critic in JAX.

Based on "Is Model Ensemble Necessary? Model-Based RL via a Single Model
with Lipschitz Regularized Value Function" (Hu et al., 2024) and the
reference DiffRL implementation at https://github.com/NVlabs/DiffRL.

The key idea is to backpropagate gradients through short-horizon rollouts
of a (here differentiable) simulator to update the actor, while a separate
critic is trained by supervised regression on TD-lambda / one-step returns
and an EMA target critic stabilises the bootstrap.

This implementation follows the API conventions of
``flightning.algos.bptt`` and ``flightning.algos.ppo``:

* The caller provides a JAX ``Env``, an actor ``TrainState`` and an optional
  critic ``TrainState`` (if omitted, the actor network is reused as critic).
* ``train()`` returns ``{"runner_state": ..., "metrics": ...}``.
* Environment wrapping (``LogWrapper`` / ``VecEnv``) is applied inside
  ``train()``.
"""

from functools import partial
from typing import NamedTuple, Optional

import chex
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from flightning.envs.env_base import Env, EnvState
from flightning.envs.wrappers import LogWrapper, VecEnv


class Config(NamedTuple):
    gamma: float = 0.99
    lam: float = 0.95
    target_critic_alpha: float = 0.4  # EMA coeff for target critic
    critic_iterations: int = 16
    num_batches: int = 4
    max_grad_norm: float = 1.0
    logging_freq: int = 10
    logging: bool = True


class SHACSample(NamedTuple):
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


NUM_EPOCHS_PER_CALLBACK = 10


def _progress_callback_host(payload):
    epoch, actor_loss, mean_return = payload
    print(
        f"[shac] epoch {epoch}: actor_loss={actor_loss:.3f} "
        f"mean_return={mean_return:.3f}"
    )


def _progress_callback(epoch, actor_loss, mean_return):
    jax.lax.cond(
        pred=epoch % NUM_EPOCHS_PER_CALLBACK == 0,
        true_fun=lambda p: jax.debug.callback(_progress_callback_host, p),
        false_fun=lambda p: None,
        operand=(epoch, actor_loss, mean_return),
    )


def _actor_sample_action(key, actor_state, obs, deterministic):
    """Sample actions for a *batch* of obs (one key per env in the batch).

    Delegates to ``SHACActor.sample_action`` (which owns the learnable
    ``log_std`` parameter) via ``jax.vmap`` over the env batch axis.
    """

    def _single(key_i, obs_i):
        return actor_state.apply_fn(
            actor_state.params,
            obs_i,
            key_i,
            deterministic,
            method="sample_action",
        )

    return jax.vmap(_single)(key, obs)


def _td_lambda_targets(samples, next_values, gamma, lam):
    """Compute TD-lambda targets V_t = r_t + gamma * [(1-d) * V_{t+1} + d * 0]
    accumulated with the lambda trace.

    Follows DiffRL's ``SHAC.compute_target_values`` formulation.
    """
    rewards = samples.reward     # (T, N)
    dones = samples.done         # (T, N), float
    T = rewards.shape[0]

    def _scan_fn(carry, step):
        Ai_prev, Bi_prev = carry
        r, d, v_next = step
        lam_t = lam * (1.0 - d)
        Ai = (1.0 - d) * (
            lam * gamma * Ai_prev + gamma * v_next + r
        )
        Bi = gamma * (v_next * d + Bi_prev * (1.0 - d)) + r
        target = (1.0 - lam_t) * Ai + lam_t * Bi
        return (Ai, Bi), target

    init = (jnp.zeros_like(next_values[0]), jnp.zeros_like(next_values[0]))
    # Iterate from t=0 to t=T-1; Ai/Bi propagate along the lambda trace.
    steps = (rewards, dones, next_values[1:])  # next_values is length T+1
    _, targets_fwd = jax.lax.scan(_scan_fn, init, steps)

    # Fallback: one-step targets (used only if lam == 0 effectively).
    one_step = rewards + gamma * next_values[1:] * (1.0 - dones)
    return jax.lax.cond(
        lam > 0.0, lambda _: targets_fwd, lambda _: one_step, None
    )


def train(
    env: Env,
    actor_state: TrainState,
    critic_state: Optional[TrainState] = None,
    num_epochs: int = 100,
    num_steps_per_epoch: int = 50,
    num_envs: int = 64,
    key: chex.PRNGKey = jax.random.key(0),
    config: Config = Config(),
):
    """Train a policy with SHAC.

    Args:
        env: flightning environment (will be wrapped with LogWrapper + VecEnv).
        actor_state: TrainState whose ``apply_fn(params, obs)`` returns action
            means of shape ``(..., action_dim)``.
        critic_state: optional TrainState whose ``apply_fn(params, obs)``
            returns a scalar value. If ``None``, the actor network is reused
            as critic (its output must then be scalar-valued).
        num_epochs: number of outer training epochs.
        num_steps_per_epoch: rollout horizon H (per env) used for the actor
            gradient pass and critic dataset.
        num_envs: number of parallel envs.
        key: PRNG key.
        config: SHAC hyperparameters.
    """
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
            sample_out = _actor_sample_action(
                key_action, actor_state, last_obs, False
            )
            action = sample_out.action
            trans = env.step(env_state, action, key_step)
            done = jnp.logical_or(trans.terminated, trans.truncated).astype(
                jnp.float32
            )
            sample = SHACSample(
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

    # ---- actor loss (grad flows through env) -----------------------------
    def _actor_loss_fn(
        actor_params, critic_params, env_state, last_obs, key
    ):
        # Rebuild a "view" TrainState that carries these params for the
        # rollout helper. apply_fn is captured from actor_state.
        view_actor = actor_state.replace(params=actor_params)
        samples, final_env_state, final_obs = _rollout(
            view_actor, env_state, last_obs, key
        )

        # Bootstrap values with the (stop-grad) critic.
        v_final = _eval_critic(critic_params, final_obs)
        v_next = _eval_critic(critic_params, samples.next_obs)
        # Shape: (T+1, N); v_next covers t=0..T-1, v_final is t=T.
        next_values = jnp.concatenate(
            [v_next, v_final[None, ...]], axis=0
        )  # (T+1, N)

        # One-step / discounted-return baseline used as the actor loss
        # (matches DiffRL's rew_acc + gamma * V_{t+1} formulation).
        rewards = samples.reward                 # (T, N)
        dones = samples.done                     # (T, N)
        gamma = config.gamma

        def _return_scan(carry, step):
            acc, g = carry
            r, d, v_tp1 = step
            acc_tp1 = acc + g * r
            # terminal bootstrap: at done steps the next-step contribution
            # is zero (reset obs is unrelated to the terminal reward).
            term = gamma * g * v_tp1 * (1.0 - d)
            return (acc_tp1 * (1.0 - d), g * gamma * (1.0 - d) + d * gamma), (
                acc_tp1 + term
            )

        init = (
            jnp.zeros((num_envs,)),
            jnp.ones((num_envs,)),
        )
        # Iterate forward; v_next[t] is V(s_{t+1}) for t < T-1, v_final for T-1.
        steps = (rewards, dones, next_values[1:])
        _, returns_fwd = jax.lax.scan(_return_scan, init, steps)
        # Mean over horizon & envs of -return  (maximise return).
        actor_loss = -returns_fwd.mean()
        return actor_loss, (samples, next_values)

    def _eval_critic(params, obs):
        if use_shared_net:
            return actor_state.apply_fn(params, obs)
        return critic_state.apply_fn(params, obs)

    # ---- critic update ---------------------------------------------------
    def _eval_critic_with_state(cs, obs):
        if use_shared_net:
            return actor_state.apply_fn(cs.params, obs)
        return cs.apply_fn(cs.params, obs)

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

        # Actor gradient step (grads flow through env).
        actor_grad_fn = jax.value_and_grad(
            _actor_loss_fn, argnums=0, has_aux=True
        )
        (actor_loss, (samples, next_values)), actor_grads = actor_grad_fn(
            actor_state.params,
            target_critic_state.params,
            env_state,
            last_obs,
            key_rollout,
        )
        actor_grads_clipped = _clip_grads(actor_grads, config.max_grad_norm)
        actor_state = actor_state.apply_gradients(grads=actor_grads_clipped)

        # Compute critic targets (no grad).
        target_values = _td_lambda_targets(
            samples, next_values, config.gamma, config.lam
        )  # (T, N)

        # Reshape the rollout buffer into a flat dataset of size T*N and
        # partition it into a statically-sized minibatch sequence.
        flat_obs = samples.obs.reshape((-1,) + samples.obs.shape[2:])
        flat_targets = target_values.reshape((-1,))
        total = flat_obs.shape[0]
        batch_size = max(total // config.num_batches, 1)
        n_batches = total // batch_size
        flat_obs = flat_obs[: n_batches * batch_size]
        flat_targets = flat_targets[: n_batches * batch_size]
        batched_obs = flat_obs.reshape((n_batches, batch_size) + flat_obs.shape[1:])
        batched_targets = flat_targets.reshape((n_batches, batch_size))

        def _critic_minibatch_step(critic_state, batch):
            batch_obs, batch_tgt = batch

            def _loss(params):
                v = _eval_critic_with_state(
                    critic_state.replace(params=params), batch_obs
                )
                return jnp.mean((v - batch_tgt) ** 2)

            loss, grads = jax.value_and_grad(_loss)(critic_state.params)
            grads = jax.tree_util.tree_map(
                lambda g: jnp.nan_to_num(g, 0.0, 0.0, 0.0), grads
            )
            critic_state = critic_state.apply_gradients(grads=grads)
            return critic_state, loss

        def _critic_pass(critic_state, _):
            critic_state, losses = jax.lax.scan(
                _critic_minibatch_step,
                critic_state,
                (batched_obs, batched_targets),
            )
            return critic_state, losses.mean()

        critic_state, critic_losses = jax.lax.scan(
            _critic_pass, critic_state, None, length=config.critic_iterations
        )
        value_loss = critic_losses[-1]

        # EMA update of the target critic.
        alpha = config.target_critic_alpha
        new_target_params = jax.tree_util.tree_map(
            lambda p, pt: alpha * pt + (1.0 - alpha) * p,
            critic_state.params,
            target_critic_state.params,
        )
        target_critic_state = target_critic_state.replace(
            params=new_target_params
        )

        # Last obs / env state carry over from rollout.
        metric = {
            "actor_loss": actor_loss,
            "returned_episode_returns": samples.info.get(
                "returned_episode_returns", jnp.zeros(())
            )[-1].mean()
            if hasattr(samples, "info") and "info" in samples._fields
            else jnp.zeros(()),
        }

        def _log(payload):
            ep_idx, actor_loss, mean_ret = payload
            print(
                f"[shac] epoch {ep_idx}: actor_loss={actor_loss:.3f} "
                f"mean_return={mean_ret:.3f}"
            )

        jax.lax.cond(
            jnp.logical_and(
                config.logging, epoch_idx % config.logging_freq == 0
            ),
            lambda _: jax.debug.callback(
                _log,
                (epoch_idx, actor_loss, metric.get("returned_episode_returns", 0.0)),
            ),
            lambda _: None,
            None,
        )

        next_runner = RunnerState(
            actor_state=actor_state,
            critic_state=critic_state,
            target_critic_state=target_critic_state,
            env_state=env_state,
            last_obs=last_obs,
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


def _clip_grads(grads, max_norm):
    flat, _ = jax.tree_util.tree_flatten(grads)
    total = sum(jnp.sum(g ** 2) for g in flat if g is not None)
    norm = jnp.sqrt(total)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


if __name__ == "__main__":
    import optax

    from flightning.envs import HoveringStateEnv
    from flightning.envs.wrappers import NormalizeActionWrapper
    from flightning.modules.mlp import SHACActor, SHACCritic

    env = HoveringStateEnv()
    env = NormalizeActionWrapper(env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor_net = SHACActor(
        [obs_dim, 64, 64, action_dim], initial_scale=0.1, initial_log_std=0.0
    )
    critic_net = SHACCritic([obs_dim, 64, 64, 1], initial_scale=0.1)

    key = jax.random.key(0)
    key_a, key_c = jax.random.split(key)
    actor_params = actor_net.initialize(key_a)
    critic_params = critic_net.initialize(key_c)

    actor_state = TrainState.create(
        apply_fn=actor_net.apply,
        params=actor_params,
        tx=optax.adam(1e-3),
    )
    critic_state = TrainState.create(
        apply_fn=critic_net.apply,
        params=critic_params,
        tx=optax.adam(1e-3),
    )

    res = train(
        env,
        actor_state,
        critic_state,
        num_epochs=2,
        num_steps_per_epoch=8,
        num_envs=4,
        key=jax.random.key(1),
        config=Config(critic_iterations=2, num_batches=2),
    )
    print("final actor_loss =", res["metrics"]["actor_loss"][-1])
