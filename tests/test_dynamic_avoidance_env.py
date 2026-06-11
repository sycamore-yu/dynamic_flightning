import jax
import jax.numpy as jnp
import pytest
from flightning.envs.dynamic_avoidance_env import (
    DynamicAvoidanceEnv,
    DynamicAvoidanceEnvState,
    _reset_jit,
    _step_jit,
)


@pytest.fixture(scope="module")
def env():
    return DynamicAvoidanceEnv()


def test_env_basic(env):
    key = jax.random.PRNGKey(42)

    # 1. Reset
    state, obs = env.reset(key)
    assert isinstance(state, DynamicAvoidanceEnvState)
    assert obs.shape == (226,)

    # 2. Step
    action = env.hovering_action
    transition = env._step(state, action, key)
    assert transition.obs.shape == (226,)
    assert transition.reward.shape == ()
    assert transition.terminated.shape == ()
    assert transition.truncated.shape == ()
    assert jnp.isfinite(transition.reward)

def test_default_reset_starts_inside_training_bounds(env):
    keys = jax.random.split(jax.random.PRNGKey(0), 16)

    states, _ = jax.vmap(env.reset)(keys)
    max_abs_xy = jnp.max(jnp.abs(states.quadrotor_state.p[:, :2]), axis=1)
    assert bool(jnp.all(max_abs_xy < env.termination_xy_limit))

    actions = jnp.tile(env.hovering_action[None, :], (keys.shape[0], 1))
    transitions = jax.vmap(env._step)(states, actions, keys)
    done = transitions.terminated | transitions.truncated
    assert bool(jnp.mean(done.astype(jnp.float32)) < 1.0)

def test_jitted_functions_reuse_cache_across_default_env_instances(env):
    key = jax.random.PRNGKey(0)
    env.reset(key)
    reset_cache_size = _reset_jit._cache_size()
    state, _ = env.reset(key)
    env._step(state, env.hovering_action, key)
    step_cache_size = _step_jit._cache_size()

    another_env = DynamicAvoidanceEnv()
    another_state, _ = another_env.reset(key)
    another_env._step(another_state, another_env.hovering_action, key)

    assert _reset_jit._cache_size() == reset_cache_size
    assert _step_jit._cache_size() == step_cache_size


def test_env_jit(env):
    key = jax.random.PRNGKey(42)

    @jax.jit
    def run_reset_step(k):
        state, obs = env.reset(k)
        transition = env._step(state, env.hovering_action, k)
        return transition.reward

    reward = run_reset_step(key)
    assert jnp.isfinite(reward)

def test_env_vmap(env):
    num_envs = 4
    keys = jax.random.split(jax.random.PRNGKey(42), num_envs)

    # Vectorized reset
    v_reset = jax.vmap(env.reset)
    states, obses = v_reset(keys)
    assert obses.shape == (num_envs, 226)

    # Vectorized step
    v_step = jax.vmap(env._step)
    actions = jnp.tile(env.hovering_action[None, :], (num_envs, 1))
    transitions = v_step(states, actions, keys)
    assert transitions.obs.shape == (num_envs, 226)
    assert transitions.reward.shape == (num_envs,)

def test_env_scan(env):
    key = jax.random.PRNGKey(42)
    state, obs = env.reset(key)

    def step_fn(carry, x):
        state, k = carry
        action = env.hovering_action
        transition = env._step(state, action, k)
        next_k, _ = jax.random.split(k)
        return (transition.state, next_k), transition.reward

    keys = jax.random.split(key, 10)
    (final_state, final_k), rewards = jax.lax.scan(step_fn, (state, key), keys)
    assert rewards.shape == (10,)
    assert jnp.all(jnp.isfinite(rewards))

def test_env_action_clipping(env):
    key = jax.random.PRNGKey(42)
    state, _ = env.reset(key)

    # Action way beyond limits
    extreme_action = jnp.array([1000.0, 100.0, 100.0, 100.0])
    transition = env._step(state, extreme_action, key)
    # Check that clipped action is applied
    applied_action = transition.state.last_actions[-1]
    assert jnp.all(applied_action <= env.action_space.high)
    assert jnp.all(applied_action >= env.action_space.low)

def test_env_termination(env):
    key = jax.random.PRNGKey(42)
    state, _ = env.reset(key)

    # Set drone position to out of height bounds (z = 4.0)
    bad_quad_state = state.quadrotor_state.replace(p=jnp.array([0.0, 0.0, 4.0]))
    bad_state = state.replace(quadrotor_state=bad_quad_state)

    transition = env._step(bad_state, env.hovering_action, key)
    assert transition.terminated == True

    # Set drone position to out of bounds horizontally (x = 20.0)
    bad_quad_state2 = state.quadrotor_state.replace(p=jnp.array([20.0, 0.0, 2.0]))
    bad_state2 = state.replace(quadrotor_state=bad_quad_state2)
    transition2 = env._step(bad_state2, env.hovering_action, key)
    assert transition2.terminated == True
