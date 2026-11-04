from typing import Any, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from utils.loss_utils import HL_GAUSS_NUM_BINS, hl_gauss_from_logits


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = True
    dtype: Any = None

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init, dtype=self.dtype)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm(dtype=self.dtype)(x)
        return x


class ResidualMLP(nn.Module):
    """Residual MLP."""

    hidden_dims: Sequence[int]
    output_dim: int
    activations: Any = nn.gelu
    kernel_init: Any = default_init()
    layer_norm: bool = True
    dtype: Any = None

    @nn.compact
    def __call__(self, x: Any) -> Any:
        if len(self.hidden_dims) % 2:
            raise ValueError(f'ResidualMLP requires an even number of hidden layers; got {len(self.hidden_dims)}.')
        if not self.hidden_dims:
            return nn.Dense(self.output_dim, kernel_init=self.kernel_init, dtype=self.dtype)(x)

        hidden_dim = self.hidden_dims[0]
        if any(dim != hidden_dim for dim in self.hidden_dims):
            raise ValueError(f'ResidualMLP requires constant hidden widths; got {self.hidden_dims}.')

        x = nn.Dense(hidden_dim, kernel_init=self.kernel_init, dtype=self.dtype)(x)
        x = self.activations(x)
        if self.layer_norm:
            x = nn.LayerNorm(dtype=self.dtype)(x)

        for _ in range(len(self.hidden_dims) // 2):
            residual = x
            x = nn.Dense(hidden_dim, kernel_init=self.kernel_init, dtype=self.dtype)(x)
            x = self.activations(x)
            if self.layer_norm:
                x = nn.LayerNorm(dtype=self.dtype)(x)
            x = nn.Dense(hidden_dim, kernel_init=self.kernel_init, dtype=self.dtype)(x)
            x = residual + x
            x = self.activations(x)
            if self.layer_norm:
                x = nn.LayerNorm(dtype=self.dtype)(x)

        return nn.Dense(self.output_dim, kernel_init=self.kernel_init, dtype=self.dtype)(x)


MLP_CLASSES = dict(
    mlp=MLP,
    resmlp=ResidualMLP,
)


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class CategoricalValue(nn.Module):
    """Categorical value/critic network for HL-Gauss losses."""

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    num_bins: int = HL_GAUSS_NUM_BINS
    mlp_class: Any = MLP
    encoder: nn.Module = None
    dtype: Any = None

    def setup(self):
        use_residual = self.mlp_class is ResidualMLP
        mlp_class = self.mlp_class
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        if use_residual:
            self.value_net = mlp_class(self.hidden_dims, self.num_bins, layer_norm=self.layer_norm, dtype=self.dtype)
        else:
            self.value_net = mlp_class(
                (*self.hidden_dims, self.num_bins),
                activate_final=False,
                layer_norm=self.layer_norm,
                dtype=self.dtype,
            )

    def __call__(self, observations, actions=None, output_logits=False):
        """Return scalar values or categorical logits."""
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        logits = self.value_net(inputs).astype(jnp.float32)
        if output_logits:
            return logits
        return hl_gauss_from_logits(logits)


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    mlp_class: Any = MLP
    layer_norm: bool = True
    encoder: nn.Module = None
    dtype: Any = None

    def setup(self) -> None:
        if self.mlp_class is ResidualMLP:
            self.mlp = self.mlp_class(self.hidden_dims, self.action_dim, layer_norm=self.layer_norm, dtype=self.dtype)
        else:
            self.mlp = self.mlp_class(
                (*self.hidden_dims, self.action_dim),
                activate_final=False,
                layer_norm=self.layer_norm,
                dtype=self.dtype,
            )

    @nn.compact
    def __call__(self, observations, actions, times=None, conditions=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            conditions: Additional conditioning variables (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        inputs = [observations, actions]
        if conditions is not None:
            inputs.append(conditions)
        if times is not None:
            freqs = jnp.arange(1, 33, dtype=times.dtype)
            angles = 2 * jnp.pi * times * freqs
            inputs.append(jnp.concatenate([times, jnp.cos(angles), jnp.sin(angles)], axis=-1))
        inputs = jnp.concatenate(inputs, axis=-1)
        v = self.mlp(inputs)

        return v
