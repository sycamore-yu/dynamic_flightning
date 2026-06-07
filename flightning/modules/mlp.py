from typing import NamedTuple
from typing import Union

import chex
import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron.

    Example creating a 2-layer MLP with 2 input features, 3 hidden units,
    and 1 output unit, and weights initialized with small variance scaling.
    Note that the bias is always initialized with zeros.

    >>> network = MLP([2, 3, 1])
    >>> key = jax.random.key(0)
    >>> key0, key1, key2 = jax.random.split(key, 3)
    >>> x_rand = jax.random.normal(key0, (2,))
    >>> params = network.init(key1, x_rand)

    Using the MLP:

    >>> x = jnp.zeros(2)
    >>> y = network.apply(params, x)
    """

    feature_list: list
    nonlinearity: callable = nn.relu
    initial_scale: float = 1.0
    action_bias: Union[float, jnp.ndarray] = 0.0

    @nn.compact
    def __call__(self, x):
        # Define the forward pass
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
        return x + self.action_bias

    def initialize(self, key):
        """
        Initialize the model with random weights. Shorthand for `init`.
        :param key: random key
        :return: initial parameters
        """
        x_rand = jax.random.normal(key, (self.feature_list[0],))
        return self.init(key, x_rand)


class OrthogonalMLP(MLP):
    @nn.compact
    def __call__(self, x):
        # Define the forward pass
        for feature in self.feature_list[1:-1]:
            x = nn.Dense(
                feature,
                kernel_init=nn.initializers.orthogonal(
                    scale=self.initial_scale
                ),
                bias_init=nn.initializers.zeros,
            )(x)
            x = self.nonlinearity(x)
        x = nn.Dense(
            self.feature_list[-1],
            kernel_init=nn.initializers.orthogonal(scale=self.initial_scale),
            bias_init=nn.initializers.zeros,
        )(x)
        return x + self.action_bias


class PiValue(NamedTuple):
    pi: distrax.Distribution
    value: jnp.ndarray


class SHACActionSample(NamedTuple):
    action: jnp.ndarray
    mean: jnp.ndarray
    std: jnp.ndarray


class SHACActor(nn.Module):
    """Gaussian policy for SHAC.

    Forward pass (``__call__``) returns the action mean. The companion
    ``sample_action`` method draws a reparameterised Gaussian sample using
    a learnable per-action ``log_std`` parameter (clamped from below so that
    the standard deviation stays positive).

    ``@compact`` is used for the trunk so the dense layers are auto-named
    ``Dense_0``, ``Dense_1``, ... matching the MLP predictor used by the
    vision pretraining pipeline (see
    ``examples/train_shac_vision.ipynb``). ``log_std`` is declared in
    ``setup()`` so that the non-compact ``sample_action`` can read it.
    """

    feature_list: list
    nonlinearity: callable = nn.relu
    initial_scale: float = 1.0
    action_bias: Union[float, jnp.ndarray] = 0.0
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
        x = obs
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
        return x + self.action_bias

    def sample_action(
        self, obs: jnp.ndarray, key: chex.PRNGKey, deterministic: bool = False
    ) -> "SHACActionSample":
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
        x_rand = jax.random.normal(key, (self.feature_list[0],))
        # Initialise via __call__ so the @compact trunk is materialised;
        # log_std is declared in setup() and thus present automatically.
        return self.init(key, x_rand)


class SHACCritic(nn.Module):
    """Scalar value head for SHAC.

    The output is squeezed along the last axis so that downstream code can
    use ``critic.apply(params, obs)`` directly as a per-env scalar value.
    """

    feature_list: list
    nonlinearity: callable = nn.relu
    initial_scale: float = 1.0

    def setup(self):
        kernel_init = nn.initializers.variance_scaling(
            self.initial_scale, mode="fan_avg", distribution="normal"
        )
        bias_init = nn.initializers.zeros
        self.hidden_layers = [
            nn.Dense(f, kernel_init=kernel_init, bias_init=bias_init)
            for f in self.feature_list[1:-1]
        ]
        self.output_layer = nn.Dense(
            1, kernel_init=kernel_init, bias_init=bias_init
        )

    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        for layer in self.hidden_layers:
            x = self.nonlinearity(layer(x))
        x = self.output_layer(x)
        return jnp.squeeze(x, axis=-1)

    def initialize(self, key):
        x_rand = jax.random.normal(key, (self.feature_list[0],))
        return self.init(key, x_rand)


class ActorCriticPPO(MLP):
    initial_log_std: float = 0.0

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        # actor
        x = obs
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
        action_mean = x + self.action_bias
        # action_mean = nn.tanh(action_mean)

        action_logtstd = self.param(
            "log_std",
            nn.initializers.constant(self.initial_log_std),
            (self.feature_list[-1],),
        )
        action_std = jnp.maximum(jnp.exp(action_logtstd), 0.05)
        # create distribution object
        pi = distrax.MultivariateNormalDiag(action_mean, action_std)

        # critic
        x = obs
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
        value = jnp.squeeze(x, axis=-1)

        return PiValue(pi, value)
