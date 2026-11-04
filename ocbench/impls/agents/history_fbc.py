from typing import Any, ClassVar

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.loss_utils import sample_beta_times
from utils.networks import MLP_CLASSES, ActorVectorField


class HistoryFBCAgent(flax.struct.PyTreeNode):
    """History-conditioned flow behavioral cloning agent."""

    sample_kwargs: ClassVar[dict] = dict(keys=('observations', 'actions', 'histories'))

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def actor_loss(self, batch, grad_params, rng):
        """Compute the flow-matching BC loss."""
        batch_size, action_dim = batch['actions'].shape
        x_rng, t_rng = jax.random.split(rng, 2)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = sample_beta_times(t_rng, (batch_size, 1), self.config['betas'])
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        observations = jnp.concatenate([batch['observations'], batch['histories']], axis=-1)
        pred = self.network.select('actor_flow')(observations, x_t, t, params=grad_params)
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
        actions = self.sample_actions(batch['observations'], histories=batch['histories'], seed=rng)
        errors = actions - batch['actions']
        return {
            'actor/mse': jnp.mean(errors**2),
            'actor/mae': jnp.mean(jnp.abs(errors)),
        }

    @jax.jit
    def sample_actions(
        self,
        observations,
        histories=None,
        seed=None,
    ):
        """Sample action chunks from the history-conditioned flow policy."""
        if histories is None:
            raise ValueError('HistoryFBCAgent.sample_actions requires histories.')
        noises = jax.random.normal(
            seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        return self.compute_flow_actions(observations, histories, noises)

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        histories,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        observations = jnp.concatenate([observations, histories], axis=-1)
        actions = noises
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_flow')(observations, actions, t)
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
        """Create a new agent."""
        if config['encoder'] is not None:
            raise ValueError('HistoryFBCAgent does not support encoders.')

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape[1:]
        if len(ob_dims) != 1:
            raise ValueError(f'HistoryFBCAgent only supports flat observations, got {ob_dims}.')
        action_dim = ex_actions.shape[-1]
        history_dim = ob_dims[-1] * config['history_length']
        dtype = jnp.dtype(config['compute_dtype'])

        if config['mlp_class'] not in MLP_CLASSES:
            raise ValueError(f'Invalid mlp_class: {config["mlp_class"]}')
        actor_mlp_class = MLP_CLASSES[config['mlp_class']]

        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            mlp_class=actor_mlp_class,
            layer_norm=config['actor_layer_norm'],
            dtype=dtype,
        )

        ex_histories = jnp.zeros((*ex_observations.shape[:-1], history_dim))
        ex_actor_observations = jnp.concatenate([ex_observations, ex_histories], axis=-1)
        ex_times = ex_actions[..., :1]
        network_info = dict(
            actor_flow=(actor_flow_def, (ex_actor_observations, ex_actions, ex_times)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['history_dim'] = history_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='history_fbc',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            history_dim=ml_collections.config_dict.placeholder(int),
            lr=1e-4,
            batch_size=1024,
            compute_dtype='float32',  # Network computation dtype; parameters and Adam stay float32.
            actor_hidden_dims=(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096),
            mlp_class='resmlp',
            actor_layer_norm=True,
            flow_steps=10,
            betas=(1.0, 1.5),
            history_interval=1,
            history_length=24,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
