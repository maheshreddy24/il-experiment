from typing import Any, ClassVar

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.loss_utils import sample_beta_times
from utils.networks import MLP_CLASSES, ActorVectorField


class FBCAgent(flax.struct.PyTreeNode):
    """Flow behavioral cloning (FBC) agent."""

    sample_kwargs: ClassVar[dict] = dict(keys=('observations', 'actions'))

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params, rng):
        """Compute the flow-matching BC loss."""
        batch_size, action_dim = batch['actions'].shape
        x_rng, t_rng = jax.random.split(rng, 2)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        beta_a, beta_b = self.config['betas']
        if self.config['grid_snap']:
            ts = jnp.linspace(0.0, 1.0, self.config['flow_steps'] + 1)
            weights = ts[1:-1] ** (beta_a - 1) * (1 - ts[1:-1]) ** (beta_b - 1)
            left_weight = 1.0 if beta_a == 1.0 else (0.0 if beta_a > 1.0 else jnp.inf)
            right_weight = 1.0 if beta_b == 1.0 else (0.0 if beta_b > 1.0 else jnp.inf)
            weights = jnp.concatenate([jnp.array([left_weight]), weights, jnp.array([right_weight])])
            inf_mask = jnp.isinf(weights)
            weights = jnp.where(jnp.any(inf_mask), inf_mask.astype(weights.dtype), weights)
            t = ts[jax.random.categorical(t_rng, jnp.log(weights), shape=(batch_size, 1))]
        else:
            t = sample_beta_times(t_rng, (batch_size, 1), self.config['betas'])
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)
        actor_loss = jnp.mean((pred.astype(jnp.float32) - vel) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        rng = rng if rng is not None else self.rng

        actor_loss, actor_info = self.actor_loss(batch, grad_params, rng)
        info = {f'actor/{k}': v for k, v in actor_info.items()}

        return actor_loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def compute_metrics(self, batch, rng):
        """Compute additional metrics."""
        actions = self.sample_actions(batch['observations'], seed=rng)
        errors = actions - batch['actions']
        return {
            'actor/mse': jnp.mean(errors**2),
            'actor/mae': jnp.mean(jnp.abs(errors)),
        }

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
    ):
        """Sample actions from the flow policy."""
        noises = jax.random.normal(
            seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        return self.compute_flow_actions(observations, noises)

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_flow_encoder')(observations)
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_flow')(observations, actions, t, is_encoded=True)
            actions = actions + vels.astype(jnp.float32) / self.config['flow_steps']
        return actions

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

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        dtype = jnp.dtype(config['compute_dtype'])

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor_flow'] = encoder_module(dtype=dtype)

        if config['mlp_class'] not in MLP_CLASSES:
            raise ValueError(f'Invalid mlp_class: {config["mlp_class"]}')
        actor_mlp_class = MLP_CLASSES[config['mlp_class']]

        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            mlp_class=actor_mlp_class,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_flow'),
            dtype=dtype,
        )

        network_info = dict(
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times)),
        )
        if encoders.get('actor_flow') is not None:
            network_info['actor_flow_encoder'] = (encoders.get('actor_flow'), (ex_observations,))
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
            agent_name='fbc',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=1e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            compute_dtype='float32',  # Network computation dtype; parameters and Adam stay float32.
            actor_hidden_dims=(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096),  # Actor network hidden dimensions.
            mlp_class='resmlp',  # MLP class ('mlp' or 'resmlp').
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            flow_steps=10,  # Number of flow steps.
            betas=(1.0, 1.5),  # Beta parameters for time sampling.
            grid_snap=False,  # Whether to snap sampled times to the flow grid.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
