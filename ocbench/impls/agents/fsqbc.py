import math
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP_CLASSES, ResidualMLP


def round_ste(x):
    """Round with straight-through gradients."""
    return x + jax.lax.stop_gradient(jnp.round(x) - x)


class FSQActionTokenizer(nn.Module):
    """FSQ tokenizer for flattened action chunks."""

    hidden_dims: Any
    action_dim: int
    num_tokens: int
    fsq_levels: Any
    mlp_class: Any
    layer_norm: bool = True
    state_cond: bool = False

    @property
    def fsq_dim(self):
        return len(self.fsq_levels)

    def setup(self):
        if self.mlp_class is ResidualMLP:
            self.encoder = self.mlp_class(
                self.hidden_dims,
                self.num_tokens * self.fsq_dim,
                layer_norm=self.layer_norm,
            )
            self.decoder = self.mlp_class(
                self.hidden_dims,
                self.action_dim,
                layer_norm=self.layer_norm,
            )
        else:
            self.encoder = self.mlp_class(
                (*self.hidden_dims, self.num_tokens * self.fsq_dim),
                activate_final=False,
                layer_norm=self.layer_norm,
            )
            self.decoder = self.mlp_class(
                (*self.hidden_dims, self.action_dim),
                activate_final=False,
                layer_norm=self.layer_norm,
            )

    def levels(self, dtype):
        return jnp.asarray(self.fsq_levels, dtype=dtype)

    def basis(self):
        levels = jnp.asarray(self.fsq_levels, dtype=jnp.int32)
        return jnp.concatenate([jnp.ones((1,), dtype=jnp.int32), jnp.cumprod(levels[:-1])])

    def bound(self, z):
        levels_i = jnp.asarray(self.fsq_levels, dtype=jnp.int32)
        levels = levels_i.astype(z.dtype)
        eps = 1e-3
        half_l = (levels - 1) * (1 - eps) / 2
        offset = jnp.where(levels_i % 2 == 1, 0.0, 0.5).astype(z.dtype)
        shift = jnp.tan(offset / half_l)
        return jnp.tanh(z + shift) * half_l - offset

    def codes_to_idxs(self, codes):
        levels = jnp.asarray(self.fsq_levels, dtype=jnp.int32)
        half_width = levels // 2
        level_idxs = codes.astype(jnp.int32) + half_width
        level_idxs = jnp.clip(level_idxs, 0, levels - 1)
        token_idxs = jnp.sum(level_idxs * self.basis(), axis=-1)
        return token_idxs.astype(jnp.int32)

    def idxs_to_codes(self, token_idxs):
        levels = jnp.asarray(self.fsq_levels, dtype=jnp.int32)
        half_width = levels // 2
        level_idxs = jnp.mod(jnp.floor_divide(token_idxs[..., None], self.basis()), levels)
        return (level_idxs.astype(jnp.float32) - half_width.astype(jnp.float32)) / half_width.astype(jnp.float32)

    def quantize(self, z_e):
        codes = round_ste(self.bound(z_e))
        token_idxs = self.codes_to_idxs(codes)
        z_q = codes / (self.levels(z_e.dtype) // 2)
        return z_q, token_idxs

    def decode(self, observations, z_q):
        z_q = z_q.reshape(*z_q.shape[:-2], self.num_tokens * self.fsq_dim)
        if self.state_cond:
            z_q = jnp.concatenate([observations, z_q], axis=-1)
        return self.decoder(z_q)

    def __call__(self, observations, actions=None, token_idxs=None):
        if token_idxs is None:
            inputs = actions
            if self.state_cond:
                inputs = jnp.concatenate([observations, actions], axis=-1)
            z_e = self.encoder(inputs)
            z_e = z_e.reshape(*z_e.shape[:-1], self.num_tokens, self.fsq_dim)
            z_q, token_idxs = self.quantize(z_e)
            recon_actions = self.decode(observations, z_q)
            return recon_actions, token_idxs, z_e, z_q

        z_q = self.idxs_to_codes(token_idxs)
        actions = self.decode(observations, z_q)
        return actions


class AutoregressiveTokenActor(nn.Module):
    """Categorical actor over packed FSQ action tokens."""

    hidden_dims: Any
    num_tokens: int
    num_codes: int
    token_embed_dim: int
    mlp_class: Any
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self):
        if self.mlp_class is ResidualMLP:
            self.net = self.mlp_class(self.hidden_dims, self.num_codes, layer_norm=self.layer_norm)
        else:
            self.net = self.mlp_class(
                (*self.hidden_dims, self.num_codes),
                activate_final=False,
                layer_norm=self.layer_norm,
            )
        self.token_embed = self.param(
            'token_embed',
            nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform'),
            (self.num_codes + 1, self.token_embed_dim),
        )
        self.pos_embed = self.param(
            'pos_embed',
            nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform'),
            (self.num_tokens, self.token_embed_dim),
        )

    def __call__(self, observations, token_idxs):
        if self.encoder is not None:
            observations = self.encoder(observations)

        start_idxs = jnp.full((*token_idxs.shape[:-1], 1), self.num_codes, dtype=token_idxs.dtype)
        prefix_idxs = jnp.concatenate([start_idxs, token_idxs[..., :-1]], axis=-1)
        prefix_embs = self.token_embed[prefix_idxs]

        slots = jnp.arange(self.num_tokens)
        pred_pos = jnp.arange(self.num_tokens)
        prefix_mask = (slots[None, :] <= pred_pos[:, None])[..., None]
        prefix_embs = prefix_embs[..., None, :, :] * prefix_mask
        prefix_embs = prefix_embs.reshape(*token_idxs.shape[:-1], self.num_tokens, -1)

        observations = jnp.broadcast_to(
            observations[..., None, :],
            (*observations.shape[:-1], self.num_tokens, observations.shape[-1]),
        )
        pos_embed = jnp.broadcast_to(
            self.pos_embed,
            (*token_idxs.shape[:-1], self.num_tokens, self.token_embed_dim),
        )
        inputs = jnp.concatenate([observations, prefix_embs, pos_embed], axis=-1)
        logits = self.net(inputs.reshape(-1, inputs.shape[-1]))
        return logits.reshape(*token_idxs.shape[:-1], self.num_tokens, self.num_codes)


class FSQBCAgent(flax.struct.PyTreeNode):
    """FSQ behavioral cloning agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def tokenizer_loss(self, batch, grad_params):
        """Compute the FSQ action-tokenizer loss."""
        recon_actions, token_idxs, z_e, z_q = self.network.select('tokenizer')(
            batch['observations'],
            batch['actions'],
            params=grad_params,
        )
        if self.config['recon_loss'] == 'l2':
            recon_loss = jnp.mean((recon_actions - batch['actions']) ** 2)
        else:
            recon_loss = jnp.mean(jnp.abs(recon_actions - batch['actions']))
        recon_mse = jnp.mean((recon_actions - batch['actions']) ** 2)

        one_hot = jax.nn.one_hot(token_idxs, self.config['num_codes'])
        code_probs = jnp.mean(one_hot, axis=tuple(range(one_hot.ndim - 1)))
        code_perplexity = jnp.exp(-jnp.sum(code_probs * jnp.log(code_probs + 1e-8)))
        code_usage = jnp.mean(code_probs > 0)

        return recon_loss, token_idxs, {
            'tokenizer_loss': recon_loss,
            'recon_loss': recon_loss,
            'recon_mse': recon_mse,
            'code_perplexity': code_perplexity,
            'code_usage': code_usage,
            'z_mean': jnp.mean(z_e),
            'z_abs_mean': jnp.mean(jnp.abs(z_e)),
            'z_q_abs_mean': jnp.mean(jnp.abs(z_q)),
        }

    def actor_loss(self, batch, token_idxs, grad_params):
        """Compute the autoregressive categorical actor loss."""
        token_idxs = jax.lax.stop_gradient(token_idxs)
        logits = self.network.select('actor')(batch['observations'], token_idxs, params=grad_params)
        actor_loss = optax.softmax_cross_entropy_with_integer_labels(logits, token_idxs).mean()
        pred_token_idxs = jnp.argmax(logits, axis=-1)
        token_accuracy = jnp.mean(pred_token_idxs == token_idxs)

        return actor_loss, {
            'actor_loss': actor_loss,
            'token_accuracy': token_accuracy,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        tokenizer_loss, token_idxs, tokenizer_info = self.tokenizer_loss(batch, grad_params)
        actor_loss, actor_info = self.actor_loss(batch, token_idxs, grad_params)

        info = {f'tokenizer/{k}': v for k, v in tokenizer_info.items()}
        info.update({f'actor/{k}': v for k, v in actor_info.items()})

        loss = self.config['fsq_loss_weight'] * tokenizer_loss + self.config['actor_loss_weight'] * actor_loss
        return loss, info

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
        temperature=0.0,
    ):
        """Sample action chunks from the categorical actor and FSQ decoder."""
        batch_shape = observations.shape[: -len(self.config['ob_dims'])]
        token_idxs = jnp.zeros((*batch_shape, self.config['num_tokens']), dtype=jnp.int32)
        token_rngs = jax.random.split(seed, self.config['num_tokens'])

        for i in range(self.config['num_tokens']):
            logits = self.network.select('actor')(observations, token_idxs)
            cur_logits = logits[..., i, :]

            def sample_token():
                return jax.random.categorical(
                    token_rngs[i],
                    cur_logits / jnp.maximum(temperature, 1e-6),
                    axis=-1,
                )

            def greedy_token():
                return jnp.argmax(cur_logits, axis=-1)

            token_idx = jax.lax.cond(temperature > 0, sample_token, greedy_token)
            token_idxs = token_idxs.at[..., i].set(token_idx)

        actions = self.network.select('tokenizer')(observations, token_idxs=token_idxs)
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
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        if config['mlp_class'] not in MLP_CLASSES:
            raise ValueError(f'Invalid mlp_class: {config["mlp_class"]}')
        if min(config['fsq_levels']) < 2:
            raise ValueError(f'Invalid fsq_levels: {config["fsq_levels"]}')
        mlp_class = MLP_CLASSES[config['mlp_class']]
        num_codes = math.prod(config['fsq_levels'])

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['actor'] = encoder_module()
            if config['state_cond']:
                raise ValueError('FSQBCAgent does not support state_cond tokenizers with visual encoders.')

        tokenizer_def = FSQActionTokenizer(
            hidden_dims=config['fsq_hidden_dims'],
            action_dim=action_dim,
            num_tokens=config['num_tokens'],
            fsq_levels=config['fsq_levels'],
            mlp_class=mlp_class,
            layer_norm=config['layer_norm'],
            state_cond=config['state_cond'],
        )
        actor_def = AutoregressiveTokenActor(
            hidden_dims=config['actor_hidden_dims'],
            num_tokens=config['num_tokens'],
            num_codes=num_codes,
            token_embed_dim=config['actor_token_embed_dim'],
            mlp_class=mlp_class,
            layer_norm=config['layer_norm'],
            encoder=encoders.get('actor'),
        )
        ex_token_idxs = jnp.zeros((*ex_actions.shape[:-1], config['num_tokens']), dtype=jnp.int32)

        network_info = dict(
            tokenizer=(tokenizer_def, (ex_observations, ex_actions)),
            actor=(actor_def, (ex_observations, ex_token_idxs)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['num_codes'] = num_codes
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fsqbc',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            num_codes=ml_collections.config_dict.placeholder(int),
            lr=1e-4,
            batch_size=256,
            fsq_hidden_dims=(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048),
            actor_hidden_dims=(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048),
            mlp_class='resmlp',
            layer_norm=True,
            recon_loss='l1',
            num_tokens=4,
            fsq_levels=(8, 5, 5, 5),
            actor_token_embed_dim=256,
            fsq_loss_weight=1.0,
            actor_loss_weight=1.0,
            state_cond=True,
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
