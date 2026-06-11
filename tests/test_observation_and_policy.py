import jax
import jax.numpy as jnp
from flightning.modules.observation_builder import ObservationBuilder
from flightning.modules.cnn_lidar_policy import CNNLidarActor, CNNLidarCritic
from flightning.utils import spaces

def test_observation_builder():
    lidar_scan = jnp.ones((1, 36, 6)) * 2.0
    drone_pos = jnp.array([1.0, 2.0, 3.0])
    target_pos = jnp.array([2.0, 3.0, 4.0])
    drone_vel = jnp.array([0.5, 0.5, 0.5])
    last_action = jnp.array([10.0, 0.1, -0.1, 0.0])

    obs = ObservationBuilder.get_observation(
        lidar_scan=lidar_scan,
        drone_pos=drone_pos,
        target_pos=target_pos,
        drone_vel=drone_vel,
        last_action=last_action
    )

    assert obs.shape == (226,)
    # check split values
    # lidar (216) should be all 2.0
    assert jnp.allclose(obs[:216], 2.0)
    # target_dir should be normalized [1.0, 1.0, 1.0]/sqrt(3)
    expected_dir = jnp.array([1.0, 1.0, 1.0]) / jnp.sqrt(3.0)
    assert jnp.allclose(obs[216:219], expected_dir)
    # velocity normalized (drone_vel / 5.0) -> [0.1, 0.1, 0.1]
    assert jnp.allclose(obs[219:222], 0.1)
    # last_action -> [10.0, 0.1, -0.1, 0.0]
    assert jnp.allclose(obs[222:], last_action)

def test_cnn_policy():
    key = jax.random.PRNGKey(42)
    key_obs, key_init, key_act = jax.random.split(key, 3)
    obs = jax.random.normal(key_obs, (226,))

    # Fusion input size is cnn_flat_size + 10.
    # CNN input: (36, 6, 1).
    # Conv 1: kernel (3,3), strides (2,1). Shape becomes (18, 6, 8)
    # Conv 2: kernel (3,3), strides (2,2). Shape becomes (9, 3, 16)
    # cnn_flat_size = 9 * 3 * 16 = 432.
    # Fusion size = 432 + 10 = 442.
    # Feature list: [442, 64, 64, 4] for actor
    feature_list = [442, 64, 64, 4]
    
    actor = CNNLidarActor(feature_list=feature_list)
    critic = CNNLidarCritic(feature_list=[442, 64, 64])

    # Initialize
    params_actor = actor.initialize(key_init)
    params_critic = critic.initialize(key_init)


    # 1. Output shapes (single & batched)
    action = actor.apply(params_actor, obs)
    assert action.shape == (4,)

    value = critic.apply(params_critic, obs)
    assert value.shape == ()  # scalar value

    # Batched inputs
    obs_batch = jnp.zeros((5, 226))
    action_batch = actor.apply(params_actor, obs_batch)
    assert action_batch.shape == (5, 4)

    bounded_actor = CNNLidarActor(
        feature_list=feature_list,
        action_bias=jnp.array([7.0, 0.0, 0.0, 0.0]),
        action_scale=jnp.array([2.0, 1.5, 1.5, 1.0]),
    )
    params_bounded = bounded_actor.initialize(key_init)
    bounded_action = bounded_actor.apply(params_bounded, obs)
    assert jnp.all(bounded_action <= jnp.array([9.0, 1.5, 1.5, 1.0]) + 1e-5)
    assert jnp.all(bounded_action >= jnp.array([5.0, -1.5, -1.5, -1.0]) - 1e-5)

    value_batch = critic.apply(params_critic, obs_batch)
    assert value_batch.shape == (5,)

    # 2. Gradient validation (BPTT/SHAC gradient flow smoke)
    # Validate that gradient w.r.t obs exists and is non-zero
    def loss_fn(o):
        a = actor.apply(params_actor, o)
        return jnp.sum(a ** 2)

    grad_fn = jax.grad(loss_fn)
    g = grad_fn(obs)
    # We should have non-zero gradients on active dimensions
    assert g.shape == (226,)
    assert not jnp.allclose(g, 0.0)
