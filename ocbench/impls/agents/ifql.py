import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.loss_utils import hl_gauss_cross_entropy, hl_gauss_from_logits, sample_beta_times
from utils.networks import MLP_CLASSES, ActorVectorField, CategoricalValue


class IFQLAgent(flax.struct.PyTreeNode):
    """Implicit flow Q-learning (IFQL) agent.

    IFQL is the flow variant of implicit diffusion Q-learning (IDQL).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def value_loss(self, batch, grad_params):
        """Compute the IQL value loss."""
        qs = self.network.select('target_critic')(batch['observations'], actions=batch['actions'])
        if self.config['q_agg'] == 'min':
            q = qs.min(axis=0)
        else:
            q = qs.mean(axis=0)
        v_logits = self.network.select('value')(batch['observations'], output_logits=True, params=grad_params)
        v = hl_gauss_from_logits(v_logits)
        weight = jnp.where(q - v >= 0, self.config['expectile'], 1 - self.config['expectile'])
        value_loss = (weight * hl_gauss_cross_entropy(v_logits, q)).mean()

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def critic_loss(self, batch, grad_params):
        """Compute the IQL critic loss."""
        next_v = self.network.select('value')(batch['next_observations'])
        discounts = batch['discounts'] if 'discounts' in batch else self.config['discount'] * batch['masks']
        q = batch['rewards'] + discounts * next_v

        q_logits = self.network.select('critic')(
            batch['observations'], actions=batch['actions'], output_logits=True, params=grad_params
        )
        critic_loss = hl_gauss_cross_entropy(q_logits, q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the behavioral flow-matching actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = sample_beta_times(t_rng, (batch_size, 1), self.config['betas'])
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(batch['observations'], x_t, t, params=grad_params)
        actor_loss = (pred - vel) ** 2
        if self.config.get('bc_success_only', False):
            bc_masks = batch['bc_masks']
            actor_loss = (actor_loss.mean(axis=-1) * bc_masks).sum() / jnp.maximum(bc_masks.sum(), 1)
        else:
            actor_loss = actor_loss.mean()

        return actor_loss, {
            'actor_loss': actor_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor."""
        orig_observations = observations
        batch_shape = orig_observations.shape[: -len(self.config['ob_dims'])]
        if self.config['encoder'] is not None:
            observations = self.network.select('actor_flow_encoder')(observations)
        action_seed, _ = jax.random.split(seed)

        # Sample `num_samples` noises and propagate them through the flow.
        sample_shape = (*batch_shape, self.config['num_samples'])
        actions = jax.random.normal(
            action_seed,
            (
                *sample_shape,
                self.config['action_dim'],
            ),
        )
        sample_axis = len(batch_shape)
        n_observations = jnp.repeat(
            jnp.expand_dims(observations, sample_axis), self.config['num_samples'], axis=sample_axis
        )
        n_orig_observations = jnp.repeat(
            jnp.expand_dims(orig_observations, sample_axis), self.config['num_samples'], axis=sample_axis
        )
        for i in range(self.config['flow_steps']):
            t = jnp.full((*sample_shape, 1), i / self.config['flow_steps'])
            vels = self.network.select('actor_flow')(n_observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)

        # Pick the action with the highest Q-value.
        qs = self.network.select('critic')(n_orig_observations, actions=actions)
        if self.config['q_agg'] == 'min':
            q = qs.min(axis=0)
        else:
            q = qs.mean(axis=0)
        best_idxs = jnp.argmax(q, axis=-1)
        gather_idxs = jnp.broadcast_to(best_idxs[..., None, None], (*best_idxs.shape, 1, self.config['action_dim']))
        actions = jnp.take_along_axis(actions, gather_idxs, axis=-2).squeeze(axis=-2)
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

        if config['mlp_class'] not in MLP_CLASSES:
            raise ValueError(f'Invalid mlp_class: {config["mlp_class"]}')
        mlp_class = MLP_CLASSES[config['mlp_class']]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor_flow'] = encoder_module()

        # Define networks.
        value_def = CategoricalValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=1,
            mlp_class=mlp_class,
            encoder=encoders.get('value'),
        )
        critic_def = CategoricalValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            mlp_class=mlp_class,
            encoder=encoders.get('critic'),
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            mlp_class=mlp_class,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_flow'),
        )

        network_info = dict(
            value=(value_def, (ex_observations,)),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_actions, ex_times)),
        )
        if encoders.get('actor_flow') is not None:
            # Add actor_flow_encoder to ModuleDict to make it separately callable.
            network_info['actor_flow_encoder'] = (encoders.get('actor_flow'), (ex_observations,))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='ifql',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=1e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096),  # Actor network hidden dimensions.
            value_hidden_dims=(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096),  # Value network hidden dimensions.
            mlp_class='resmlp',  # MLP class ('mlp' or 'resmlp').
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            discount=0.9999,  # Discount factor.
            tau=0.001,  # Target network update rate.
            expectile=0.5,  # IQL expectile.
            n_step=200,  # Number of steps for bootstrapped returns.
            q_agg='mean',  # Aggregation method for Q values.
            num_samples=32,  # Number of action samples for rejection sampling.
            flow_steps=10,  # Number of flow steps.
            bc_success_only=False,  # Train the actor only on successful trajectories.
            betas=(1.0, 1.5),  # Beta parameters for time sampling.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
