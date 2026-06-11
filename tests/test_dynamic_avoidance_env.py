import jax
import jax.numpy as jnp
from flightning.envs.dynamic_avoidance_env import DynamicAvoidanceEnv, DynamicAvoidanceEnvState

def test_env_basic():
    env = DynamicAvoidanceEnv()
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

def test_env_jit():
    env = DynamicAvoidanceEnv()
    key = jax.random.PRNGKey(42)

    @jax.jit
    def run_reset_step(k):
        state, obs = env.reset(k)
        transition = env._step(state, env.hovering_action, k)
        return transition.reward

    reward = run_reset_step(key)
    assert jnp.isfinite(reward)

def test_env_vmap():
    env = DynamicAvoidanceEnv()
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

def test_env_scan():
    env = DynamicAvoidanceEnv()
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

def test_env_action_clipping():
    env = DynamicAvoidanceEnv()
    key = jax.random.PRNGKey(42)
    state, _ = env.reset(key)

    # Action way beyond limits
    extreme_action = jnp.array([1000.0, 100.0, 100.0, 100.0])
    transition = env._step(state, extreme_action, key)
    # Check that clipped action is applied
    applied_action = transition.state.last_actions[-1]
    assert jnp.all(applied_action <= env.action_space.high)
    assert jnp.all(applied_action >= env.action_space.low)

def test_env_termination():
    env = DynamicAvoidanceEnv()
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
