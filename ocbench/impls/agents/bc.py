from typing import Any, ClassVar

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP_CLASSES, ResidualMLP


class DeterministicActor(nn.Module):
    """Deterministic actor network."""

    hidden_dims: Any
    action_dim: int
    mlp_class: Any
    layer_norm: bool = True
    encoder: nn.Module = None
    ob_ndim: int = None
    dtype: Any = None

    @nn.compact
    def __call__(self, observations):
        if self.encoder is not None:
            batch_shape = observations.shape[: observations.ndim - self.ob_ndim]
            observations = self.encoder(observations)
            # Multi-camera observations (e.g., (num_cameras, H, W, C)) are encoded per camera; concatenate the features.
            observations = observations.reshape((*batch_shape, -1))
        if self.mlp_class is ResidualMLP:
            return self.mlp_class(self.hidden_dims, self.action_dim, layer_norm=self.layer_norm, dtype=self.dtype)(
                observations
            )
        return self.mlp_class(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
            dtype=self.dtype,
        )(observations)


class BCAgent(flax.struct.PyTreeNode):
    """Deterministic behavioral cloning agent."""

    sample_kwargs: ClassVar[dict] = dict(keys=('observations', 'actions'))

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params):
        """Compute the deterministic BC loss."""
        pred_actions = self.network.select('actor')(batch['observations'], params=grad_params)
        pred_actions = pred_actions.astype(jnp.float32)
        if self.config['loss'] == 'l1':
            actor_loss = jnp.mean(jnp.abs(pred_actions - batch['actions']))
        else:
            actor_loss = jnp.mean((pred_actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        info = {f'actor/{k}': v for k, v in actor_info.items()}

        return actor_loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, _ = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
    ):
        """Sample action chunks from the deterministic policy."""
        actions = self.network.select('actor')(observations)
        return actions.astype(jnp.float32)

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        dtype = jnp.dtype(config['compute_dtype'])

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor'] = encoder_module(dtype=dtype)

        if config['mlp_class'] not in MLP_CLASSES:
            raise ValueError(f'Invalid mlp_class: {config["mlp_class"]}')
        actor_mlp_class = MLP_CLASSES[config['mlp_class']]

        actor_def = DeterministicActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            mlp_class=actor_mlp_class,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor'),
            ob_ndim=len(ob_dims),
            dtype=dtype,
        )

        network_info = dict(
            actor=(actor_def, (ex_observations,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='bc',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=1e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            compute_dtype='float32',  # Network computation dtype; parameters and Adam stay float32.
            actor_hidden_dims=(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096),  # Actor network hidden dimensions.
            mlp_class='resmlp',  # MLP class ('mlp' or 'resmlp').
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            loss='l2',  # BC loss type ('l1' or 'l2').
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
