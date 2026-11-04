import functools
from typing import Any, Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP


class ResnetStack(nn.Module):
    """ResNet stack module."""

    num_features: int
    num_blocks: int
    max_pooling: bool = True
    dtype: Any = None

    @nn.compact
    def __call__(self, x):
        initializer = nn.initializers.xavier_uniform()
        conv_out = nn.Conv(
            features=self.num_features,
            kernel_size=(3, 3),
            strides=1,
            kernel_init=initializer,
            padding='SAME',
            dtype=self.dtype,
        )(x)

        if self.max_pooling:
            conv_out = nn.max_pool(
                conv_out,
                window_shape=(3, 3),
                padding='SAME',
                strides=(2, 2),
            )

        for _ in range(self.num_blocks):
            block_input = conv_out
            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                dtype=self.dtype,
                kernel_init=initializer,
            )(conv_out)

            conv_out = nn.relu(conv_out)
            conv_out = nn.Conv(
                features=self.num_features,
                kernel_size=(3, 3),
                strides=1,
                padding='SAME',
                dtype=self.dtype,
                kernel_init=initializer,
            )(conv_out)
            conv_out += block_input

        return conv_out


class ImpalaEncoder(nn.Module):
    """IMPALA encoder."""

    width: int = 1
    stack_sizes: tuple = (16, 32, 32)
    num_blocks: int = 2
    dropout_rate: float = None
    mlp_hidden_dims: Sequence[int] = (512,)
    layer_norm: bool = True
    dtype: Any = None

    def setup(self):
        stack_sizes = self.stack_sizes
        self.stack_blocks = [
            ResnetStack(
                num_features=stack_sizes[i] * self.width,
                num_blocks=self.num_blocks,
                dtype=self.dtype,
            )
            for i in range(len(stack_sizes))
        ]
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, train=True, cond_var=None):
        x = x.astype(jnp.float32) / 255.0

        conv_out = x

        for idx in range(len(self.stack_blocks)):
            conv_out = self.stack_blocks[idx](conv_out)
            if self.dropout_rate is not None:
                conv_out = self.dropout(conv_out, deterministic=not train)

        conv_out = nn.relu(conv_out)
        if self.layer_norm:
            conv_out = nn.LayerNorm(dtype=self.dtype)(conv_out)
        out = conv_out.reshape((*x.shape[:-3], -1))

        out = MLP(self.mlp_hidden_dims, activate_final=True, layer_norm=self.layer_norm, dtype=self.dtype)(out)

        return out


class ConvNeXtBlock(nn.Module):
    """ConvNeXt V2 residual block."""

    features: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        residual = x
        init = nn.initializers.truncated_normal(0.02)
        x = nn.Conv(
            self.features,
            kernel_size=(7, 7),
            padding='SAME',
            feature_group_count=self.features,
            kernel_init=init,
            dtype=self.dtype,
        )(x)
        x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype)(x)
        x = nn.Dense(4 * self.features, kernel_init=init, dtype=self.dtype)(x)
        x = nn.gelu(x, approximate=False)

        # Global response normalization.
        norm = jnp.sqrt(jnp.sum(x.astype(jnp.float32) ** 2, axis=(1, 2), keepdims=True) + 1e-6)
        norm = norm / (jnp.mean(norm, axis=-1, keepdims=True) + 1e-6)
        gamma = self.param('grn_gamma', nn.initializers.zeros, (4 * self.features,))
        beta = self.param('grn_beta', nn.initializers.zeros, (4 * self.features,))
        x = x + gamma.astype(x.dtype) * x * norm.astype(x.dtype) + beta.astype(x.dtype)

        x = nn.Dense(self.features, kernel_init=init, dtype=self.dtype)(x)
        return residual + x


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt V2 with shared weights for images shaped (..., cameras, H, W, C)."""

    depths: tuple
    dims: tuple
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        batch_shape = x.shape[:-4]
        num_cameras = x.shape[-4]
        x = x.reshape((-1, *x.shape[-3:])).astype(jnp.float32) / 255.0
        init = nn.initializers.truncated_normal(0.02)
        x = nn.Conv(
            self.dims[0], kernel_size=(4, 4), strides=(4, 4), padding='VALID', kernel_init=init, dtype=self.dtype
        )(x)
        x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype)(x)
        for i, (depth, dim) in enumerate(zip(self.depths, self.dims)):
            if i > 0:
                x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype)(x)
                x = nn.Conv(
                    dim, kernel_size=(2, 2), strides=(2, 2), padding='VALID', kernel_init=init, dtype=self.dtype
                )(x)
            for _ in range(depth):
                x = ConvNeXtBlock(dim, dtype=self.dtype)(x)
        x = nn.LayerNorm(epsilon=1e-6, dtype=self.dtype)(jnp.mean(x.astype(jnp.float32), axis=(1, 2)))
        return x.reshape((*batch_shape, num_cameras * self.dims[-1]))


encoder_modules = {
    'impala': ImpalaEncoder,
    'impala_debug': functools.partial(ImpalaEncoder, num_blocks=1, stack_sizes=(4, 4)),
    'impala_small': functools.partial(ImpalaEncoder, num_blocks=1),
    'impala_large': functools.partial(ImpalaEncoder, stack_sizes=(64, 128, 128), mlp_hidden_dims=(1024,)),
    'convnext_atto': functools.partial(ConvNeXtEncoder, depths=(2, 2, 6, 2), dims=(40, 80, 160, 320)),
    'convnext_femto': functools.partial(ConvNeXtEncoder, depths=(2, 2, 6, 2), dims=(48, 96, 192, 384)),
    'convnext_pico': functools.partial(ConvNeXtEncoder, depths=(2, 2, 6, 2), dims=(64, 128, 256, 512)),
    'convnext_nano': functools.partial(ConvNeXtEncoder, depths=(2, 2, 8, 2), dims=(80, 160, 320, 640)),
    'convnext_tiny': functools.partial(ConvNeXtEncoder, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768)),
    'convnext_base': functools.partial(ConvNeXtEncoder, depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024)),
    'convnext_large': functools.partial(ConvNeXtEncoder, depths=(3, 3, 27, 3), dims=(192, 384, 768, 1536)),
    'convnext_huge': functools.partial(ConvNeXtEncoder, depths=(3, 3, 27, 3), dims=(352, 704, 1408, 2816)),
}
