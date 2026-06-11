import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from flightning.envs.dynamic_avoidance_env import DynamicAvoidanceEnv
from flightning.modules.cnn_lidar_policy import CNNLidarActor, CNNLidarCritic
from flightning.algos.bptt import train as train_bptt, Config as BPTTConfig
from flightning.algos.shac import train as train_shac, Config as SHACConfig

def test_bptt_smoke():
    env = DynamicAvoidanceEnv()
    key = jax.random.PRNGKey(42)

    # Actor network: CNNLidarActor
    feature_list = [442, 64, 64, 4]
    model = CNNLidarActor(feature_list=feature_list)

    key_init, key_train = jax.random.split(key)
    params = model.initialize(key_init)

    tx = optax.adam(1e-3)
    train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )

    # Run BPTT for 1 epoch, 10 steps, 4 envs
    config = BPTTConfig(logging=False)
    result = train_bptt(
        env=env,
        train_state=train_state,
        num_epochs=1,
        num_steps_per_epoch=10,
        num_envs=4,
        key=key_train,
        config=config
    )

    assert "runner_state" in result
    assert "metrics" in result
    assert jnp.isfinite(result["metrics"][-1])

def test_shac_smoke():
    env = DynamicAvoidanceEnv()
    key = jax.random.PRNGKey(42)

    # Actor & Critic networks
    feature_list_actor = [442, 64, 64, 4]
    feature_list_critic = [442, 64, 64]

    actor_model = CNNLidarActor(feature_list=feature_list_actor)
    critic_model = CNNLidarCritic(feature_list=feature_list_critic)

    key_init, key_train = jax.random.split(key)
    key_actor, key_critic = jax.random.split(key_init)

    actor_params = actor_model.initialize(key_actor)
    critic_params = critic_model.initialize(key_critic)

    tx = optax.adam(1e-3)
    actor_state = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_params,
        tx=tx
    )
    critic_state = TrainState.create(
        apply_fn=critic_model.apply,
        params=critic_params,
        tx=tx
    )

    # Run SHAC for 1 epoch, 10 steps, 4 envs
    config = SHACConfig(logging=False, critic_iterations=2, num_batches=1)
    result = train_shac(
        env=env,
        actor_state=actor_state,
        critic_state=critic_state,
        num_epochs=1,
        num_steps_per_epoch=10,
        num_envs=4,
        key=key_train,
        config=config
    )

    assert "runner_state" in result
    assert "metrics" in result
    assert jnp.isfinite(result["metrics"]["actor_loss"][-1])
