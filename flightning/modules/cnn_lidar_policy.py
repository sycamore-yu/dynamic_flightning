from typing import NamedTuple, Union, Callable
import chex
import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flightning.modules.mlp import SHACActionSample

class CNNLidarActor(nn.Module):
    """CNN-based Actor policy for dynamic avoidance task.

    Accepts a flat 226-dim observation vector. resizes the first 216 dimensions
    to a channel-first (1, 36, 6) image (transposed to 36x6x1 for Flax Conv),
    processes it with a CNN, and fuses it with the remaining 10 state dimensions
    before MLP layers project to actions.
    """
    feature_list: list  # e.g., [cnn_flat_size + 10, 64, 64, 4]
    nonlinearity: Callable = nn.relu
    initial_scale: float = 1.0
    action_bias: Union[float, jnp.ndarray] = 0.0
    action_scale: Union[None, float, jnp.ndarray] = None
    initial_log_std: float = 0.0
    min_std: float = 0.05

    def setup(self):
        self.log_std = self.param(
            "log_std",
            nn.initializers.constant(self.initial_log_std),
            (self.feature_list[-1],),
        )

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        # Get original batch shape prefix (e.g. () or (B,) or (T, B,))
        batch_shape = obs.shape[:-1]

        # 1. Split observation: LiDAR (216) and State (10)
        lidar_flat = obs[..., :216]
        state_feat = obs[..., 216:]

        # Flatten batch dimensions for Conv layer processing
        lidar_flat_merged = lidar_flat.reshape(-1, 216)
        lidar_image = lidar_flat_merged.reshape(-1, 1, 36, 6)
        lidar_image_tf = jnp.transpose(lidar_image, (0, 2, 3, 1))

        # 2. CNN Encoder
        h = nn.Conv(features=8, kernel_size=(3, 3), strides=(2, 1), padding="SAME")(lidar_image_tf)
        h = self.nonlinearity(h)
        h = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(h)
        h = self.nonlinearity(h)

        # Flatten CNN output: (N_total, 432)
        cnn_flat = h.reshape(h.shape[0], -1)

        # Restore original batch dimensions: batch_shape + (432,)
        cnn_flat_restored = cnn_flat.reshape(batch_shape + (432,))

        # 3. State Feature Fusion
        fused = jnp.concatenate([cnn_flat_restored, state_feat], axis=-1)

        # 4. MLP Trunk
        x = fused
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.variance_scaling(
                    self.initial_scale, mode="fan_avg", distribution="normal"
                ),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)

        x = nn.Dense(
            self.feature_list[-1],
            kernel_init=nn.initializers.variance_scaling(
                self.initial_scale, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
        )(x)

        if self.action_scale is not None:
            x = jnp.tanh(x) * self.action_scale

        return x + self.action_bias

    def sample_action(
        self, obs: jnp.ndarray, key: chex.PRNGKey, deterministic: bool = False
    ) -> SHACActionSample:
        mean = self(obs)
        std = jnp.maximum(jnp.exp(self.log_std), self.min_std)
        dist = distrax.MultivariateNormalDiag(mean, std)
        action = jax.lax.cond(
            deterministic,
            lambda _: mean,
            lambda _: dist.sample(seed=key),
            None,
        )
        return SHACActionSample(action=action, mean=mean, std=std)

    def initialize(self, key):
        x_rand = jax.random.normal(key, (216 + 10,))
        return self.init(key, x_rand)


class CNNLidarCritic(nn.Module):
    """CNN-based Critic value network for dynamic avoidance task."""
    feature_list: list  # e.g., [cnn_flat_size + 10, 64, 64]
    nonlinearity: Callable = nn.relu
    initial_scale: float = 1.0

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        # Get original batch shape prefix (e.g. () or (B,) or (T, B,))
        batch_shape = obs.shape[:-1]

        # 1. Split observation: LiDAR (216) and State (10)
        lidar_flat = obs[..., :216]
        state_feat = obs[..., 216:]

        # Flatten batch dimensions for Conv layer processing
        lidar_flat_merged = lidar_flat.reshape(-1, 216)
        lidar_image = lidar_flat_merged.reshape(-1, 1, 36, 6)
        lidar_image_tf = jnp.transpose(lidar_image, (0, 2, 3, 1))

        # 2. CNN Encoder
        h = nn.Conv(features=8, kernel_size=(3, 3), strides=(2, 1), padding="SAME")(lidar_image_tf)
        h = self.nonlinearity(h)
        h = nn.Conv(features=16, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(h)
        h = self.nonlinearity(h)

        # Flatten CNN output: (N_total, 432)
        cnn_flat = h.reshape(h.shape[0], -1)

        # Restore original batch dimensions: batch_shape + (432,)
        cnn_flat_restored = cnn_flat.reshape(batch_shape + (432,))

        # 3. State Feature Fusion
        fused = jnp.concatenate([cnn_flat_restored, state_feat], axis=-1)

        # 4. MLP Trunk
        x = fused
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.variance_scaling(
                    self.initial_scale, mode="fan_avg", distribution="normal"
                ),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)

        x = nn.Dense(
            1,
            kernel_init=nn.initializers.variance_scaling(
                self.initial_scale, mode="fan_avg", distribution="normal"
            ),
            bias_init=nn.initializers.zeros,
        )(x)

        return jnp.squeeze(x, axis=-1)

    def initialize(self, key):
        x_rand = jax.random.normal(key, (216 + 10,))
        return self.init(key, x_rand)
