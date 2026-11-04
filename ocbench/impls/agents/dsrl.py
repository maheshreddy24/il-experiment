import copy
from typing import Any

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.loss_utils import hl_gauss_cross_entropy, hl_gauss_from_logits, sample_beta_times
from utils.networks import (
    MLP_CLASSES,
    ActorVectorField,
    CategoricalValue,
    LogParam,
    ResidualMLP,
    TransformedWithMode,
    default_init,
)


class LatentActor(nn.Module):
    """Gaussian actor over flow noise."""

    hidden_dims: Any
    action_dim: int
    mlp_class: Any
    layer_norm: bool = True
    log_std_min: float = -20
    log_std_max: float = 2
    encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, temperature=1.0):
        if self.encoder is not None:
            observations = self.encoder(observations)

        if self.mlp_class is ResidualMLP:
            features = self.mlp_class(self.hidden_dims, self.hidden_dims[-1], layer_norm=self.layer_norm)(observations)
        else:
            features = self.mlp_class(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(observations)

        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(features)
        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(features)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds) * jnp.maximum(temperature, 1e-6),
        )
        return TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))


class DSRLAgent(flax.struct.PyTreeNode):
    """Diffusion steering via reinforcement learning (DSRL-NA) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def squash_noises(self, noises):
        """Squash latent noises to the configured support."""
        return noises * self.config['noise_scale']

    def critic_loss(self, batch, grad_params, rng):
        """Compute action critic and latent critic losses."""
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_actions = jnp.clip(next_actions, -1, 1)

        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        discounts = batch['discounts'] if 'discounts' in batch else self.config['discount'] * batch['masks']
        target_q = batch['rewards'] + discounts * next_q

        q_logits = self.network.select('critic')(
            batch['observations'], actions=batch['actions'], output_logits=True, params=grad_params
        )
        critic_loss = hl_gauss_cross_entropy(q_logits, target_q).mean()
        q = hl_gauss_from_logits(q_logits)

        # Distill the action critic onto random latent noises.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch['actions'].shape[0], self.config['action_dim']))
        actions = self.compute_flow_actions(batch['observations'], noises=noises)
        target_z_qs = self.network.select('critic')(batch['observations'], actions=actions)

        z_q_logits = self.network.select('z_critic')(
            batch['observations'], actions=noises, output_logits=True, params=grad_params
        )
        distill_loss = hl_gauss_cross_entropy(z_q_logits, target_z_qs).mean()
        z_q = hl_gauss_from_logits(z_q_logits)

        total_loss = critic_loss + distill_loss

        return total_loss, {
            'total_loss': total_loss,
            'critic_loss': critic_loss,
            'distill_loss': distill_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'target_q_mean': target_q.mean(),
            'z_q_mean': z_q.mean(),
            'z_q_max': z_q.max(),
            'z_q_min': z_q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute latent actor, entropy, and BC flow losses."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, noise_rng = jax.random.split(rng, 4)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = sample_beta_times(t_rng, (batch_size, 1), self.config['betas'])
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # SAC-style latent actor loss.
        dist = self.network.select('actor')(batch['observations'], params=grad_params)
        noises, log_probs = dist.sample_and_log_prob(seed=noise_rng)
        scaled_noises = self.squash_noises(noises)

        qs = self.network.select('z_critic')(batch['observations'], actions=scaled_noises)
        if self.config['q_agg'] == 'min':
            q = qs.min(axis=0)
        else:
            q = qs.mean(axis=0)

        alpha = self.network.select('alpha')()
        actor_loss = (alpha * log_probs - q).mean()

        grad_alpha = self.network.select('alpha')(params=grad_params)
        entropy = -jax.lax.stop_gradient(log_probs).mean()
        alpha_loss = (grad_alpha * (entropy - self.config['target_entropy'])).mean()

        total_loss = bc_flow_loss + actor_loss + alpha_loss

        return total_loss, {
            'total_loss': total_loss,
            'bc_flow_loss': bc_flow_loss,
            'actor_loss': actor_loss,
            'alpha': alpha,
            'alpha_loss': alpha_loss,
            'entropy': -log_probs.mean(),
            'q': q.mean(),
            'noise_abs_mean': jnp.abs(scaled_noises).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
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
        self.target_update(new_network, 'actor_bc_flow')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions by steering the BC flow policy with latent noise."""
        if seed is None:
            seed = jax.random.PRNGKey(0)
        dist = self.network.select('actor')(observations, temperature=temperature)
        sampled_noises = dist.sample(seed=seed)
        noises = jax.lax.cond(
            temperature <= 0,
            lambda: dist.mode(),
            lambda: sampled_noises,
        )
        noises = self.squash_noises(noises)
        return self.compute_flow_actions(observations, noises=noises)

    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        actions = noises
        model_name = 'target_actor_bc_flow' if self.config['use_target_latent'] else 'actor_bc_flow'
        for i in range(self.config['flow_steps']):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config['flow_steps'])
            vels = self.network.select(model_name)(observations, actions, t)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)
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
        if config['target_entropy'] is None:
            config['target_entropy'] = -config['target_entropy_multiplier'] * action_dim

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['z_critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define networks.
        critic_def = CategoricalValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            mlp_class=mlp_class,
            encoder=encoders.get('critic'),
        )
        z_critic_def = CategoricalValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=config['num_qs'],
            mlp_class=mlp_class,
            encoder=encoders.get('z_critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            mlp_class=mlp_class,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_def = LatentActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            mlp_class=mlp_class,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor'),
        )
        alpha_def = LogParam(init_value=config['alpha'])

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            z_critic=(z_critic_def, (ex_observations, ex_actions)),
            actor_bc_flow=(actor_bc_flow_def, (ex_observations, ex_actions, ex_times)),
            target_actor_bc_flow=(copy.deepcopy(actor_bc_flow_def), (ex_observations, ex_actions, ex_times)),
            actor=(actor_def, (ex_observations,)),
            alpha=(alpha_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_actor_bc_flow'] = params['modules_actor_bc_flow']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='dsrl',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            lr=1e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048),  # Actor network hidden dimensions.
            value_hidden_dims=(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048),  # Value network hidden dimensions.
            mlp_class='resmlp',  # MLP class ('mlp' or 'resmlp').
            layer_norm=True,  # Whether to use layer normalization.
            actor_layer_norm=True,  # Whether to use layer normalization for the actor.
            discount=0.9999,  # Discount factor.
            tau=0.001,  # Target network update rate.
            n_step=200,  # Number of steps for bootstrapped returns.
            q_agg='mean',  # Aggregation method for Q values ('mean' or 'min').
            num_qs=2,  # Number of Q ensemble members.
            flow_steps=10,  # Number of flow steps.
            betas=(1.0, 1.5),  # Beta parameters for time sampling.
            alpha=1.0,  # Initial entropy coefficient.
            target_entropy=ml_collections.config_dict.placeholder(float),  # Target latent-policy entropy.
            target_entropy_multiplier=0.5,  # Multiplier to action dimension for automatic target entropy.
            noise_scale=0.5,  # Scale of the bounded latent noise.
            use_target_latent=True,  # Whether to use the target BC flow as the latent-noise map.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
        )
    )
    return config
